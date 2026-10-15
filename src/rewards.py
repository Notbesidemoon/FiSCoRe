"""Reward functions for GRPO training."""
import os
import re

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForSequenceClassification

# Global variable used to load one NLI model per training process on demand.
nli_model = None

class BaseEntailment:
    def save_prediction_cache(self):
        pass

class EntailmentDeberta(BaseEntailment):
    def __init__(self):
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        self.device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
        print(f"Initializing EntailmentDeberta on device {self.device}")
        model_name = os.environ.get(
            "FISCORE_NLI_MODEL", "microsoft/deberta-v2-xlarge-mnli"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name
        ).to(self.device)
        self.model.eval()

    def check_implication(self, text1, text2, *args, **kwargs):
        inputs = self.tokenizer(text1, text2, return_tensors="pt").to(self.device)
        # The model checks if text1 -> text2, i.e. if text2 follows from text1.
        # check_implication('The weather is good', 'The weather is good and I like you') --> 1
        # check_implication('The weather is good and I like you', 'The weather is good') --> 2
        outputs = self.model(**inputs)
        logits = outputs.logits
        # Deberta-mnli returns `neutral` and `entailment` classes at indices 1 and 2.
        largest_index = torch.argmax(F.softmax(logits, dim=1))  # pylint: disable=no-member
        prediction = largest_index.cpu().item()
        # if os.environ.get('DEBERTA_FULL_LOG', False):
        #     logging.info('Deberta Input: %s -> %s', text1, text2)
        #     logging.info('Deberta Prediction: %s', prediction)

        return prediction

# initialize the global nli_model
def initialize_nli_model():
    global nli_model
    if nli_model is None:
        nli_model = EntailmentDeberta()
    return nli_model

# get the global nli_model
def get_nli_model():
    global nli_model
    if nli_model is None:
        initialize_nli_model()
    return nli_model

def are_equivalent(text1, text2, strict_entailment=True, example=None):
    """check if two texts are semantically equivalent"""
    model = get_nli_model()

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

def get_semantic_ids(strings_list, strict_entailment=True, example=None):
    """group the predictions into semantic groups"""
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
                if are_equivalent(string1, strings_list[j], strict_entailment=strict_entailment, example=example):
                    semantic_set_ids[j] = next_id
            next_id += 1

    assert -1 not in semantic_set_ids
    # for any string, calculate the number of strings in the same semantic set
    semantic_set_counts = [semantic_set_ids.count(semantic_set_ids[i]) for i in range(len(semantic_set_ids))]
    return semantic_set_ids, semantic_set_counts

def extract_content(text, tag_type="answer"):
    """Extract content based on tag type and model type.

    Args:
        text (str): The text to extract content from
        tag_type (str): Type of content to extract ("answer" or "confidence")
        model_type (str): Type of model format ("gemma" or "default")

    Returns:
        str: Extracted content
    """

    if tag_type == "answer":
        pattern = r"<answer>(.*?)</answer>"
    else:  # confidence
        pattern = r"<confidence>(.*?)</confidence>"


    matches = re.findall(pattern, text, re.DOTALL)
    return matches[0].strip() if matches else ""


def has_valid_format(text):
    """Return whether a response follows the required answer-confidence format."""
    pattern = (
        r"^\s*<answer>.*?</answer>\s*"
        r"<confidence>\s*(?:sure|unsure)\s*</confidence>\s*$"
    )
    return re.match(pattern, text, re.DOTALL | re.IGNORECASE) is not None

def confidence_reward(completions, question, **kwargs):
    """calculate the confidence reward function based on the semantic groups"""
    contents = [completion[0]["content"] for completion in completions]
    valid_formats = [has_valid_format(content) for content in contents]
    answer_contents = [extract_content(content, "answer") for content in contents]
    qas = [q + " " + a for q, a in zip(question, answer_contents)]
    _, semantic_set_counts = get_semantic_ids(qas, strict_entailment=True)
    confidence_words = [extract_content(content, "confidence") for content in contents]
    rewards = []
    for confidence_word, semantic_set_count, valid_format in zip(
        confidence_words, semantic_set_counts, valid_formats
    ):
        if not valid_format:
            rewards.append(0.0)
            continue
        p_confidence_word = confidence_word.strip().lower()
        if semantic_set_count < (len(contents) / 2):
            if "unsure" == p_confidence_word:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
        else:
            if "sure" == p_confidence_word:
                rewards.append(1.0)
            else:
                rewards.append(0.0)
    return rewards


# def confidence_reward(completions, question, **kwargs):
#     """calculate the certainty reward function based on the semantic groups"""
#     contents = [completion[0]["content"] for completion in completions]
#     answer_contents = [extract_content(content, "answer") for content in contents]
#     qas = [q + " " + a for q, a in zip(question, answer_contents)]
#     semantic_set_ids, semantic_set_counts = get_semantic_ids(qas, strict_entailment=True)
#     # print("semantic_set_ids", semantic_set_ids)
#     print("semantic_set_counts", semantic_set_counts)

#     confidence_words = [extract_content(content, "confidence") for content in contents]
#     rewards = []
#     for i, (confidence_word, semantic_set_count) in enumerate(zip(confidence_words, semantic_set_counts)):
#         p_confidence_word = confidence_word.strip().lower()
#         # ensure that the format of p_confidence_word is ending with % and the range is between 1% and 100%, if not, the reward is 0
#         valid_confidence = False
#         if p_confidence_word.endswith("%"):
#             try:
#                 num = int(p_confidence_word.rstrip("%"))
#                 if 1 <= num <= 100:
#                     p_confidence_word = f"{num}%"
#                     valid_confidence = True
#             except Exception:
#                 valid_confidence = False
#         else:
#             match = re.search(r"(\d{1,3})", p_confidence_word)
#             if match:
#                 num = int(match.group(1))
#                 if 1 <= num <= 100:
#                     p_confidence_word = f"{num}%"
#                     valid_confidence = True
#         if not valid_confidence:
#             rewards.append(0.0)
#             continue
#         # let the confidence score be the semantic_set_count / len(contents) * 100
#         # 10, 20, 30, 40, 50, 60, 70, 80, 90, 100
#         se_confidence_score = semantic_set_count / len(contents) * 100
#         model_confidence_score = int(p_confidence_word.rstrip("%"))
#         diff = abs(se_confidence_score - model_confidence_score)
#         rewards.append(1 - diff / 100)

#     return rewards

def accuracy_reward(completions, question, solution, **kwargs):
    """check if the completion is the same as the ground truth answer"""
    contents = [completion[0]["content"] for completion in completions]
    rewards = []
    for content, sol, q in zip(contents, solution, question):
        if not has_valid_format(content):
            rewards.append(0.0)
            continue
        answer = extract_content(content, "answer")
        model_qa = q + " " + answer
        std_qa = q + " " + sol
        if are_equivalent(model_qa, std_qa, strict_entailment=True):
            rewards.append(1.0)
        else:
            rewards.append(0.0)
    return rewards

def format_reward(completions, **kwargs):
    """Reward responses whose tags are complete and correctly ordered."""
    completion_contents = [completion[0]["content"] for completion in completions]
    return [1.0 if has_valid_format(content) else 0.0 for content in completion_contents]

def tag_count_reward(completions, **kwargs) -> list[float]:
    """Reward function that checks if we produce the desired number of tags."""
    def count_tags(text: str) -> float:
        count = 0.0
        if text.count("<answer>") == 1:
            count += 0.125
        if text.count("</answer>") == 1:
            count += 0.125
        if text.count("<confidence>") == 1:
            count += 0.125
        if text.count("</confidence>") == 1:
            count += 0.125

        return count * 2.0

    contents = [completion[0]["content"] for completion in completions]
    return [count_tags(c) for c in contents]





