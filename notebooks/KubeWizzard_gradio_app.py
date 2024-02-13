import re
import torch
import time
import pinecone
import pickle
import os
import numpy as np
import hashlib
import gradio as gr
from transformers.generation.stopping_criteria import StoppingCriteria, StoppingCriteriaList
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from torch import nn
from sentence_transformers.cross_encoder import CrossEncoder
from sentence_transformers import SentenceTransformer
from peft import PeftModel
from bs4 import BeautifulSoup
import requests
import logging

logging.basicConfig(format='[%(asctime)s] %(message)s', datefmt='%d-%b-%y %H:%M:%S', level=logging.INFO)

headers = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_9_5) AppleWebKit 537.36 (KHTML, like Gecko) Chrome",
    "Accept": "text/html,application/xhtml+xml,application/xml; q=0.9,image/webp,*/*;q=0.8",
    "Cookie": "CONSENT=YES+cb.20210418-17-p0.it+FX+917; ",
}


def google_search(text):
    logging.info(f"Google search on: {text}")
    try:
        site = requests.get(f"https://www.google.com/search?hl=en&q={text}", headers=headers)
        main = (
            BeautifulSoup(site.text, features="html.parser").select_one("#main").select(".VwiC3b.lyLwlc.yDYNvb.W8l4ac")
        )
        res = []
        for m in main:
            t = m.get_text()
            if "—" in t:
                t = t[len("—") + t.index("—") :].strip()

            res.append(t)

        ans = "  \n".join(res)
    except Exception as ex:
        logging.error(f"Error: {ex}")
        ans = ""

    logging.info(f"The result of the google search is: {ans}")

    return ans

PINECONE_API_KEY = os.environ.get("PINECONE_API_KEY")

pinecone.init(api_key=PINECONE_API_KEY, environment="gcp-starter")

sentencetransformer_model = SentenceTransformer('sentence-transformers/multi-qa-mpnet-base-cos-v1')

CACHE_DIR = "./.cache"
INDEX_NAME = "k8s-semantic-search"

if not os.path.exists(CACHE_DIR):
    os.makedirs(CACHE_DIR)


def cached(func):
    def wrapper(*args, **kwargs):
        SEP = "$|$"
        cache_token = (
            f"{func.__name__}{SEP}"
            f"{SEP.join(str(arg) for arg in args)}{SEP}"
            f"{SEP.join( str(key) + SEP * 2 + str(val) for key, val in kwargs.items())}"
        )

        hex_hash = hashlib.sha256(cache_token.encode()).hexdigest()
        cache_filename: str = os.path.join(CACHE_DIR, f"{hex_hash}")

        if os.path.exists(cache_filename):
            with open(cache_filename, "rb") as cache_file:
                return pickle.load(cache_file)

        result = func(*args, **kwargs)
        with open(cache_filename, "wb") as cache_file:
            pickle.dump(result, cache_file)

        return result

    return wrapper


@cached
def create_embedding(text: str):
    embed_text = sentencetransformer_model.encode(text)
    
    return embed_text.tolist()


index = pinecone.Index(INDEX_NAME)


def query_from_pinecone(query, top_k=3):
    embedding = create_embedding(query)
    if not embedding:
        return None

    return index.query(vector=embedding, top_k=top_k, include_metadata=True).get("matches")  # gets the metadata (text)


cross_encoder = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-12-v2")


def get_results_from_pinecone(query, top_k=3, re_rank=True, verbose=True):
    results_from_pinecone = query_from_pinecone(query, top_k=top_k)
    if not results_from_pinecone:
        return []

    if verbose:
        logging.info(f"Query: {query}")

    final_results = []

    if re_rank:
        if verbose:
            logging.info("Document ID (Hash)\t\tRetrieval Score\tCE Score\tText")

        sentence_combinations = [
            [query, result_from_pinecone["metadata"]["text"]] for result_from_pinecone in results_from_pinecone
        ]

        # Compute the similarity scores for these combinations
        similarity_scores = cross_encoder.predict(sentence_combinations, activation_fct=nn.Sigmoid())

        # Sort the scores in decreasing order
        sim_scores_argsort = reversed(np.argsort(similarity_scores))

        # Print the scores
        for idx in sim_scores_argsort:
            result_from_pinecone = results_from_pinecone[idx]
            final_results.append(result_from_pinecone)
            if verbose:
                logging.info(
                    f"{result_from_pinecone['id']:<4}\t{result_from_pinecone['score']:.2f}\t{similarity_scores[idx]:.2f}\t{result_from_pinecone['metadata']['text'][:50]}"
                )
        return final_results

    if verbose:
        logging.info("Document ID (Hash)\t\tRetrieval Score\tText")
    for result_from_pinecone in results_from_pinecone:
        final_results.append(result_from_pinecone)
        if verbose:
            logging.info(
                f"{result_from_pinecone['id']}\t{result_from_pinecone['score']:.2f}\t{result_from_pinecone['metadata']['text'][:50]}"
            )

    return final_results


def semantic_search(prompt):
    final_results = get_results_from_pinecone(prompt, top_k=9, re_rank=True, verbose=True)
    if not final_results:
        return ""

    return "\n\n".join(res["metadata"]["text"].strip() for res in final_results[:3])


base_model_id = "mistralai/Mistral-7B-Instruct-v0.1"
lora_model_id = "ComponentSoft/mistral-kubectl-instruct"

tokenizer = AutoTokenizer.from_pretrained(
    lora_model_id,
    padding_side="left",
    add_eos_token=False,
    add_bos_token=True,
)
tokenizer.pad_token = tokenizer.eos_token

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16
)

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    quantization_config=bnb_config,
    use_cache=True,
    trust_remote_code=True,
)

model = PeftModel.from_pretrained(base_model, lora_model_id)
model.eval()


def create_stop_criterion(*args):
    term_tokens = [torch.tensor(tokenizer.encode(term, add_special_tokens=False)).to("cuda") for term in args]

    class CustomStopCriterion(StoppingCriteria):
        def __call__(self, input_ids: torch.LongTensor, score: torch.FloatTensor, **kwargs):
            return any(torch.equal(e, input_ids[0][-len(e) :]) for e in term_tokens)

    return CustomStopCriterion()


eval_stop_criterion = create_stop_criterion("</s>", "#End")
category_stop_criterion = create_stop_criterion("</s>", "\n")

start_template = "### Answer:"
command_template = "# Command:"
end_template = "#End"

def str_to_md(text):
    def escape_hash(line):
        i = 0
        while i < len(line) and line[i] == ' ':
            i+=1

        if i == len(line):
            return line
        
        if line[i] == '#':
            line = line[:i] + '\\' + line[i:]
        
        return line

    lines = text.split('\n')
    lines = [escape_hash(line) for line in lines]
    return '  \n'.join(l if not all(c == '-' for c in l) else '_'*len(l) for l in lines)

def text_to_text_generation(verbose, prompt):
    prompt = prompt.strip()

    is_kubectl_prompt = (
        f"You are a helpful assistant who classifies prompts into three categories. [INST] Respond with 0 if it pertains to a 'kubectl' operation. This is an instruction that can be answered with a 'kubectl' action. Look for keywords like 'get', 'list', 'create', 'show', 'view', and other command-like words. This category is an instruction instead of a question. Respond with 1 only if the prompt is a question, and is about a definition related to Kubernetes, or non-action inquiries. Respond with 2 every other scenario, for example if the question is a general question, not related to Kubernetes or 'kubectl'.\n"
        f"Here are some examples:\n"
        f"text: List all pods in Kubernetes\n"
        f"response (0/1/2): 0 \n"
        f"text: What is a headless service and how to create one?\n"
        f"response (0/1/2): 1 \n"
        f"text: What is the capital of Hungary?\n"
        f"response (0/1/2): 2 \n"
        f"text: Display detailed information about the pod 'web-app-pod-1'\n"
        f"response (0/1/2): 0 \n"
        f"text: What are some typical foods in Germany?\n"
        f"response (0/1/2): 2 \n"
        f"text: What is a LoadBalancer in Kubernetes?\n"
        f"response (0/1/2): 1 \n"
        f"text: How can I enhance the performance of a k8s cluster?\n"
        f"response (0/1/2): 1 \n"
        f'Classify the following: [/INST] \ntext: "{prompt}\n"'
        f"response (0/1/2): "
    )

    model_input = tokenizer(is_kubectl_prompt, return_tensors="pt").to("cuda")

    with torch.no_grad():
        response = tokenizer.decode(
            model.generate(
                **model_input,
                max_new_tokens=8,
                pad_token_id=tokenizer.eos_token_id,
                repetition_penalty=1.15,
                stopping_criteria=StoppingCriteriaList([category_stop_criterion]),
            )[0],
            skip_special_tokens=True,
        )
    response = response[len(is_kubectl_prompt) :]


    response_num = 0 if "0" in response else (1 if "1" in response else 2)

    def create_generation_prompt(response_num, prompt, retriever):
        md = ""
        match response_num:
            case 0:
                prompt = f"[INST] {prompt}\n Lets think step by step. [/INST] {start_template}"
                logging.info('Kubectl command prompt:')
                logging.info(prompt)
            case 1:
                if retriever == "semantic_search":
                    question = prompt
                    logging.info('Semantic search prompt:')
                    logging.info(
                        (
                        f"You are a helpful kubernetes professional. [INST] Use the following documentation, if it is relevant to answer the question below. [/INST]\nDocumentation: [RETRIEVED_RESULTS_FROM_BOOK] [INST] Answer the following question: {question} [/INST]\nAnswer:  \n")

                    )
                    retrieved_results = semantic_search(prompt)
                    prompt = f"You are a helpful kubernetes professional. [INST] Use the following documentation, if it is relevant to answer the question below. [/INST]\nDocumentation: {retrieved_results} </s>\n<s> [INST] Answer the following question: {prompt} [/INST]\nAnswer:\n\n"

                    md = (
                        f"### Step 1: Preparing prompt for additional documentation  \n\n"
                        f"You are a helpful kubernetes professional. [INST] Use the following documentation, if it is relevant to answer the question below. [/INST]\nDocumentation:  \n\n"
                        f"### Step 2: Retrieving documentation from a book.  \n\n"
                        f"{str_to_md(retrieved_results)}  \n\n"
                        f"### Step 3: Creating full prompt given to model  \n\n"
                        f"You are a helpful kubernetes professional. [INST] Use the following documentation, if it is relevant to answer the question below. [/INST]\nDocumentation: [RETRIEVED_RESULTS_FROM_BOOK] [INST] Answer the following question: {question} [/INST]\nAnswer:"
                    )
                elif retriever == "google_search":
                    retrieved_results = google_search(prompt)
                    question = prompt
                    prompt = f"You are a helpful kubernetes professional. [INST] Use the following documentation, if it is relevant to answer the question below. [/INST]\nDocumentation: {retrieved_results} </s>\n<s> [INST] Answer the following question: {prompt} [/INST]\nAnswer: "
                    
                    logging.info('Google search prompt:')
                    logging.info(
                        (
                            f"You are a helpful kubernetes professional. [INST] Use the following documentation, if it is relevant to answer the question below. [/INST]\nDocumentation: [RETRIEVED_RESULTS_FROM_GOOGLE] [INST] Answer the following question: {question} [/INST]\nAnswer:\n\n" 
                        )
                    )

                    md = (
                        f"### Step 1: Preparing prompt for additional documentation  \n\n"
                        f"You are a helpful kubernetes professional. [INST] Use the following documentation, if it is relevant to answer the question below. [/INST]\nDocumentation:  \n\n"
                        f"### Step 2: Retrieving documentation from Google.  \n\n"
                        f"{str_to_md(retrieved_results)}  \n\n"
                        f"### Step 3: Creating full prompt given to model  \n\n"
                        f"You are a helpful kubernetes professional. [INST] Use the following documentation, if it is relevant to answer the question below. [/INST]\nDocumentation: [RETRIEVED_RESULTS_FROM_GOOGLE] [INST] Answer the following question: {question} [/INST]\nAnswer:"
                    )
                else:
                    prompt = f"[INST] Answer the following question: {prompt} [/INST]\nAnswer: "
                    logging.info('No retriever question prompt:')
                    logging.info(prompt)

            case _:
                prompt = f"[INST] {prompt} [/INST]"
                logging.info('Other question prompt:')
                logging.info(prompt)

        return prompt, md

    def generate_batch(*prompts):
        tokenized_inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")

        with torch.no_grad():
            responses = tokenizer.batch_decode(
                model.generate(
                    **tokenized_inputs,
                    max_new_tokens=256,
                    pad_token_id=tokenizer.eos_token_id,
                    repetition_penalty=1.15,
                    stopping_criteria=StoppingCriteriaList([eval_stop_criterion]),
                ),
                skip_special_tokens=True,
            )

        decoded_prompts = tokenizer.batch_decode(tokenized_inputs.input_ids, skip_special_tokens=True)

        return [(prompt, answer) for prompt, answer in zip(decoded_prompts, responses)]

    def cleanup(prompt, answer):
        start = answer.index(start_template) + len(start_template) if start_template in answer else len(prompt)
        start = answer.index(command_template) + len(command_template) if command_template in answer else start
        end = answer.index(end_template) if end_template in answer else len(answer)

        return (prompt, answer[start:end].strip())

    modes = ["Kubectl command", "Kubernetes related", "Other"]

    logging.info(f'{" Query Start ":-^40}')
    logging.info(f"Classified as: {modes[response_num]}")

    modes[response_num] = f"**{modes[response_num]}**"
    modes = " / ".join(modes)


    if response_num == 2:
        prompt, md = create_generation_prompt(response_num, prompt, False)
        original, new = generate_batch(prompt)[0]
        prompt, response = cleanup(original, new)
        if verbose:
            return (
                f"# 📚KubeWizard📚\n"
                f"#### A helpful Kubernetes Assistant powered by Component Soft\n"
                f"--------------------------------------------\n"
                f"# Classified your prompt as:\n"
                f"{modes}\n\n" 
                f"# Prompt given to the model:\n" 
                f"{str_to_md(prompt)}\n"
                f"# Model's answer:\n" f"{str_to_md(response)}\n"
            )
        else:
            return (
                f"# 📚KubeWizard📚\n"
                f"#### A helpful Kubernetes Assistant powered by Component Soft\n"
                f"--------------------------------------------\n"
                f"# Classified your prompt as:\n"
                f"{modes}\n\n" 
                f"# Answer:\n" f"{str_to_md(response)}"
            )

    if response_num == 0:
        prompt, md = create_generation_prompt(response_num, prompt, False)
        original, new = generate_batch(prompt)[0]
        prompt, response = cleanup(original, new)
        model_response = new[len(original):].strip()
        if verbose:
            return (
                f"# 📚KubeWizard📚\n"
                f"#### A helpful Kubernetes Assistant powered by Component Soft\n"
                f"--------------------------------------------\n"
                f"# Classified your prompt as:\n"
                f"{modes}\n\n"
                f"# Prompt given to the model:\n"
                f"{str_to_md(prompt)}\n"
                f"# Model's answer:\n"
                f"{str_to_md(model_response)}\n"
                f"# Processed answer:\n"
                f"```bash\n{str_to_md(response)}\n```\n"
            )
        else:
            return (
                f"# 📚KubeWizard📚\n"
                f"#### A helpful Kubernetes Assistant powered by Component Soft\n"
                f"--------------------------------------------\n"
                f"# Classified your prompt as:\n"
                f"{modes}\n\n"
                f"# Answer:\n" f"```bash\n{str_to_md(response)}\n```\n"
            )

    res_prompt, res_md = create_generation_prompt(response_num, prompt, False)
    res_semantic_search_prompt, res_semantic_search_md = create_generation_prompt(response_num, prompt, "semantic_search")
    res_google_search_prompt, res_google_search_md = create_generation_prompt(response_num, prompt, "google_search")

    gen_normal, gen_semantic_search, gen_google_search = generate_batch(
        res_prompt, res_semantic_search_prompt, res_google_search_prompt
    )

    logging.info(f"SEMANTIC BEFORE CLEANUP: {str(gen_semantic_search)}")
    logging.info(f"GOOGLE BEFORE CLEANUP: {str(gen_google_search)}")


    res_prompt, res_normal = cleanup(*gen_normal)
    res_semantic_search_prompt, res_semantic_search = cleanup(*gen_semantic_search)
    res_google_search_prompt, res_google_search = cleanup(*gen_google_search)

    logging.info(f"SEMANTIC AFTER CLEANUP: {str(res_semantic_search)}")
    logging.info(f"GOOGLE AFTER CLEANUP: {str(res_google_search)}")

    if verbose:
        return (
            f"# 📚KubeWizard📚\n"
            f"#### A helpful Kubernetes Assistant powered by Component Soft\n"
            f"--------------------------------------------\n"
            f"# Classified your prompt as:\n"
            f"{modes}\n\n"
            f"--------------------------------------------\n"
            f"# Answer with finetuned model\n"
            f"## Prompt given to the model:\n"
            f"{str_to_md(res_prompt)}\n\n"
            f"## Model's answer:\n"
            f"{str_to_md(res_normal)}\n\n"
            f"--------------------------------------------\n"
            f"# Answer with RAG\n"
            f"## Section 1: Preparing for generation  \n\n{res_semantic_search_md}  \n\n"
            f"## Section 2: Generating answer  \n\n{str_to_md(res_semantic_search.strip())}  \n\n"
            f"--------------------------------------------\n"
            f"# Answer with Google search\n"
            f"## Section 1: Preparing for generation  \n\n{res_google_search_md}  \n\n"
            f"## Section 2: Generating answer  \n\n{str_to_md(res_google_search.strip())}  \n\n"
        )
    else:
        return (
            f"# 📚KubeWizard📚\n"
            f"#### A helpful Kubernetes Assistant powered by Component Soft\n"
            f"--------------------------------------------\n"
            f"# Classified your prompt as:\n"
            f"{modes}\n\n"
            f"# Answer with finetuned model  \n\n{str_to_md(res_normal)}  \n\n"
            f"# Answer with RAG  \n\n{str_to_md(res_semantic_search.strip())}  \n\n"
            f"# Answer with Google search  \n\n{str_to_md(res_google_search)}  \n\n"
        )


iface = gr.Interface(
    fn=text_to_text_generation,
    inputs=[
        gr.components.Checkbox(label="Verbose"),
        gr.components.Text(placeholder="prompt here ...", label="Prompt"),
    ],
    outputs=gr.components.Markdown(label="Answer"),
    allow_flagging="never",
    title="📚KubeWizard📚",
)

iface.launch()
