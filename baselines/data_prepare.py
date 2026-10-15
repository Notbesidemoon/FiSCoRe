from transformers import AutoTokenizer,AutoModelForCausalLM, AutoModelForSequenceClassification
import torch
import json
from tqdm import tqdm
import random
from argparse import ArgumentParser
from scipy.stats import entropy
import math
import os
import numpy as np
import torch.nn.functional as F
import logging
from vllm import LLM, SamplingParams



BRIEF_PROMPTS = {
    'default': "Answer the following questions as briefly as possible. ",
    'chat': 'Answer the following questions in a single brief but complete sentence.\n'}

def make_prompt(question, answer):
    prompt = ''
    
    prompt += f"Question: {question}\n"
    if answer:
        prompt += f"Answer: {answer}.\n\n"
    else:
        prompt += 'Answer:'
    return prompt

def construct_fewshot_prompt_from_indices(dataset, example_indices, make_prompt):
    """Given a dataset and indices, construct a fewshot prompt."""
    prompt = BRIEF_PROMPTS['default']

    for example_index in example_indices:

        example = dataset[example_index]
        # context = example["context"]
        question = example[0]
        answer = example[1]

        prompt = prompt + make_prompt(question, answer)

    return prompt

def inference(model, input_text, sampling_params):
    outputs = model.generate(
        input_text,
        sampling_params=sampling_params
    )
    n = len(outputs[0].outputs)
    output_texts = []
    for i in range(n):
        output_text = outputs[0].outputs[i].text
        idx = output_text.find('.')
        output_text = output_text[:idx]
        output_texts.append(output_text)
    return output_texts

def inference_t(input_text):
    # full_input = "Question:" + input_text + " Answer:"
    #full_input = input_text
    inputs = tokenizer(input_text,return_tensors="pt").to(2)
    ids = inputs['input_ids']
    length = len(ids[0])
    if args.method == "uncertain" or args.method == "semantic_uncertain":
        outputs = model.generate(
                ids,
                temperature=0.7,
                do_sample = True,
                max_new_tokens = 15,
                pad_token_id=tokenizer.eos_token_id

            )
    else:       
        outputs = model.generate(
                ids,
                #temperature=0.7,
                #do_sample = True,
                max_new_tokens = 15,
            )
    output_text = tokenizer.decode(outputs[0][length:])
    idx = output_text.find('.')
    output_text = output_text[:idx]
    return output_text

def get_semantic_ids(strings_list, model, strict_entailment=True, example=None):
    """Group list of predictions into semantic meaning."""

    def are_equivalent(text1, text2):

        implication_1 = model.check_implication(text1, text2, example=example)
        implication_2 = model.check_implication(text2, text1, example=example)  # pylint: disable=arguments-out-of-order
        assert (implication_1 in [0, 1, 2]) and (implication_2 in [0, 1, 2])

        if strict_entailment:
            semantically_equivalent = (implication_1 == 2) and (implication_2 == 2)

        else:
            implications = [implication_1, implication_2]
            # Check if none of the implications are 0 (contradiction) and not both of them are neutral.
            semantically_equivalent = (0 not in implications) and ([1, 1] != implications)

        return semantically_equivalent

    # Initialise all ids with -1.
    semantic_set_ids = [-1] * len(strings_list)
    # Keep track of current id.
    next_id = 0
    for i, string1 in enumerate(strings_list):
        # Check if string1 already has an id assigned.
        if semantic_set_ids[i] == -1:
            # If string1 has not been assigned an id, assign it next_id.
            semantic_set_ids[i] = next_id
            for j in range(i+1, len(strings_list)):
                # Search through all remaining strings. If they are equivalent to string1, assign them the same id.
                if are_equivalent(string1, strings_list[j]):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids
    semantic_set_counts = [semantic_set_ids.count(semantic_set_ids[i]) for i in range(len(semantic_set_ids))]
    return semantic_set_ids, semantic_set_counts

def cluster_assignment_entropy(semantic_ids):
    n_generations = len(semantic_ids)
    counts = np.bincount(semantic_ids)
    probabilities = counts/n_generations
    assert np.isclose(probabilities.sum(), 1)
    entropy = - (probabilities * np.log(probabilities)).sum()
    return entropy

class BaseEntailment:
    def save_prediction_cache(self):
        pass

class EntailmentDeberta(BaseEntailment):
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-v2-xlarge-mnli")
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "microsoft/deberta-v2-xlarge-mnli").to(device)
        self.device = device

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(self.device)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(F.softmax(logits, dim=1))
        prediction = largest_index.cpu().item()
        if os.environ.get('DEBERTA_FULL_LOG', False):
            logging.info('Deberta Input: %s -> %s', text1, text2)
            logging.info('Deberta Prediction: %s', prediction)

        return prediction

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default="pararel")
    parser.add_argument('--model', type=str, required=True)
    parser.add_argument('--output_dir',type=str, default="pararel_llama3_8b") 
    parser.add_argument('--alpha',type=float,default=0.5)
    parser.add_argument("--num_try",type=int,default="10")
    parser.add_argument("--device",type=str,default="0")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="JSON file containing [question, answer] pairs",
    )
    
    args = parser.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.device
    model = LLM(
        model=args.model,
        gpu_memory_utilization=0.8,
        
    )

    sampling_params_high_temp = SamplingParams(
        n = args.num_try,
        temperature=1.0,
    )

    sampling_params_low_temp = SamplingParams(
        max_tokens=15,
        temperature=0
    )

    LMFlow_data_R_Tuning = {"type": "text_only", "instances": []}
    LMFlow_data_R_Tuning_U = {"type": "text_only", "instances": []}
    LMFlow_data_SE_Tuning = {"type": "text_only", "instances": []}

    LMFlow_data_R_Tuning_refusal = {"type": "text_only", "instances": []}
    LMFlow_data_R_Tuning_U_refusal = {"type": "text_only", "instances": []}
    LMFlow_data_SE_Tuning_refusal = {"type": "text_only", "instances": []}

    original_results = {"samples":[], "stats":[]}
    # R-Tuning, R-Tuning_U, SE_Tuning
    R_Tuing_data, R_Tuning_U_data, SE_Tuning_data = [], [], []
    R_Tuning_refusal_data, R_Tuning_U_refusal_data, SE_Tuning_refusal_data = [], [], []
    semantic_uncertain_data, uncertain_data = [], []

    data = []
    with open(args.input_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    
    entailment_model = EntailmentDeberta()
    all_answers = []
    answerable_indices = [i for i in range(len(data))]
    
    # set random seed
    right_num = 0
    random.seed(32)
    prompt_indices = random.sample(answerable_indices, 5)

    cluster_nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    prefix_nums = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    remaining_answerable = list(set(answerable_indices) - set(prompt_indices))
    prompt = construct_fewshot_prompt_from_indices(data, prompt_indices, make_prompt)
    for sample_indice in tqdm(remaining_answerable):
        sample = data[sample_indice]
        qa_text = f"Question: {sample[0]} Answer: {sample[1]}. Are you sure you accurately answered the question based on your internal knowledge?"
        refusal_text = f"Question: {sample[0]} Answer: "
        input_prompt = prompt + make_prompt(sample[0], None)
        
        # R-Tuning, only use one answer in low temperature
        acc = 0
        low_t_ans = inference(model, input_prompt, sampling_params_low_temp)[0].strip().lower()
        if sample[1].lower() in low_t_ans:
            acc = 1
            right_num += 1
            low_t_text = qa_text + " I am sure."
            low_t_refusal_text = refusal_text + sample[1]
        else:
            low_t_text = qa_text + " I am unsure."  
            low_t_refusal_text = refusal_text + "I do not know the answer."
        R_Tuing_data.append({"text":low_t_text})
        R_Tuning_refusal_data.append({"text":low_t_refusal_text})
        
        # R-Tuning_U, use all answers in high temperature
        high_t_anses = []
        occurance = {}
        high_t_anses = inference(model, input_prompt, sampling_params_high_temp)

        for ans in high_t_anses:
            if ans in occurance:
                occurance[ans] += 1
            else:
                occurance[ans] = 1
        freq_list = list(occurance.values())
        answer_entropy = entropy(freq_list)
        uncertain_data.append((answer_entropy,sample))
        # SE_Tuning
        qa_texts =  [f'{sample[0]} {r}' for r in high_t_anses]
        semantic_ids, semantic_set_counts = get_semantic_ids(
            qa_texts, model=entailment_model,
            strict_entailment=True , example=None)
        for num in semantic_set_counts:
            cluster_nums[num] += 1
        semantic_entropy = cluster_assignment_entropy(semantic_ids)
        semantic_uncertain_data.append((semantic_entropy,sample))

        original_results["samples"].append({"text":input_prompt, "question":sample[0], "answer":sample[1], "acc":acc,
                                 "low_t_ans":low_t_ans,
                                 "high_t_anses":high_t_anses,
                                 "semantic_entropy":semantic_entropy,
                                 "answer_entropy":answer_entropy,
                                 "semantic_ids":semantic_ids,
                                 "semantic_set_counts":semantic_set_counts})

    prefix_nums = [sum(cluster_nums[:i + 1]) for i in range(len(cluster_nums))]
    original_results["stats"].append({"right_num":right_num, "total_num":len(remaining_answerable), "acc":right_num/len(remaining_answerable),
                                      "cluster_nums":cluster_nums, "prefix_nums":prefix_nums})
    # R-Tuning_data
    random.shuffle(R_Tuing_data)
    random.shuffle(R_Tuning_refusal_data)
    LMFlow_data_R_Tuning['instances'] = R_Tuing_data
    LMFlow_data_R_Tuning_refusal['instances'] = R_Tuning_refusal_data
    
    # sort uncertain data by entropy
    uncertain_data.sort(key=lambda x: x[0])
    split_half = math.floor(len(uncertain_data)*args.alpha)
    for (answer_entropy,sample) in uncertain_data[:split_half]:
        text = f"Question: {sample[0]} Answer: {sample[1]}. Are you sure you accurately answered the question based on your internal knowledge?"
        refusal_text = f"Question: {sample[0]} Answer: "
        R_Tuning_U_data.append({"text":f"{text} I am sure."})
        R_Tuning_U_refusal_data.append({"text":refusal_text + sample[1]})
        
    for (answer_entropy,sample) in uncertain_data[split_half:]:
        text = f"Question: {sample[0]} Answer: {sample[1]}. Are you sure you accurately answered the question based on your internal knowledge?"
        refusal_text = f"Question: {sample[0]} Answer: "
        R_Tuning_U_data.append({"text":f"{text} I am unsure."})
        R_Tuning_U_refusal_data.append({"text":refusal_text + "I do not know the answer."})

    random.shuffle(R_Tuning_U_data)
    random.shuffle(R_Tuning_U_refusal_data)
    LMFlow_data_R_Tuning_U['instances'] = R_Tuning_U_data
    LMFlow_data_R_Tuning_U_refusal['instances'] = R_Tuning_U_refusal_data

    # sort semantic uncertain data by entropy
    semantic_uncertain_data.sort(key=lambda x: x[0])
    split_half = math.floor(len(semantic_uncertain_data)*args.alpha)
    for (semantic_entropy,sample) in semantic_uncertain_data[:split_half]:
        text = f"Question: {sample[0]} Answer: {sample[1]}. Are you sure you accurately answered the question based on your internal knowledge?"
        refusal_text = f"Question: {sample[0]} Answer: "
        SE_Tuning_data.append({"text":f"{text} I am sure."})
        SE_Tuning_refusal_data.append({"text":refusal_text + sample[1]})

    for (answer_entropy,sample) in semantic_uncertain_data[split_half:]:
        text = f"Question: {sample[0]} Answer: {sample[1]}. Are you sure you accurately answered the question based on your internal knowledge?"
        refusal_text = f"Question: {sample[0]} Answer: "
        SE_Tuning_data.append({"text":f"{text} I am unsure."})
        SE_Tuning_refusal_data.append({"text":refusal_text + "I do not know the answer."})

    random.shuffle(SE_Tuning_data)
    random.shuffle(SE_Tuning_refusal_data)
    LMFlow_data_SE_Tuning['instances'] = SE_Tuning_data
    LMFlow_data_SE_Tuning_refusal['instances'] = SE_Tuning_refusal_data

    os.makedirs(args.output_dir, exist_ok=True)
    with open(f"{args.output_dir}/{args.dataset}_{args.model.split('/')[-1]}.json",'w') as f:
        json.dump(original_results,f,indent=4)

    with open(f"{args.output_dir}/R_Tuning.json",'w') as f:
        json.dump(LMFlow_data_R_Tuning,f,indent=4)
    with open(f"{args.output_dir}/R_Tuning_U.json",'w') as f:
        json.dump(LMFlow_data_R_Tuning_U,f,indent=4)
    with open(f"{args.output_dir}/SE_Tuning.json",'w') as f:
        json.dump(LMFlow_data_SE_Tuning,f,indent=4)
    with open(f"{args.output_dir}/R_Tuning_refusal.json",'w') as f:
        json.dump(LMFlow_data_R_Tuning_refusal,f,indent=4)
    with open(f"{args.output_dir}/R_Tuning_U_refusal.json",'w') as f:
        json.dump(LMFlow_data_R_Tuning_U_refusal,f,indent=4)
    with open(f"{args.output_dir}/SE_Tuning_refusal.json",'w') as f:
        json.dump(LMFlow_data_SE_Tuning_refusal,f,indent=4)
