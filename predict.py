import os
import json
import argparse
from typing import List, Dict

from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from lib import read_jsonl
from constants import ANS_DELIMITER
from digit_tokenization import enable_digit_tokenization


def prepare_input_text_in_nt5_format(
    question_text: str,
    context_text: str,
    max_context_length: int = 600,
    max_question_length: int = 100,
) -> str:
    # Format taken from NT5's codebase.
    context_text = "context: " + context_text.strip()
    context_text = " ".join(context_text.split(" ")[:max_context_length])
    question_text = "answer_me: " + question_text.strip()  # To act as separator
    question_text = " ".join(question_text.split(" ")[:max_question_length])
    input_text = question_text + " " + context_text
    return input_text


def prepare_input_text_in_unifiedqa_format(
    question_text: str,
    context_text: str,
    max_context_length: int = 600,
    max_question_length: int = 100,
) -> str:
    # Format taken from UnifiedQA and PreaSM's codebase.
    context_text = context_text.replace("\n", "").strip()
    context_text = " ".join(context_text.split(" ")[:max_context_length])
    question_text = question_text.replace("\n", "").strip()
    question_text = " ".join(question_text.split(" ")[:max_question_length])
    input_text = question_text + "\n" + context_text
    return input_text


def prepare_input_text(
    tokenizer: AutoTokenizer,
    question_text: str,
    context_text: str,
    max_context_length: int = 600,
    max_question_length: int = 100,
    format_type: str = None,
) -> str:
    if format_type is not None:
        # Pass format_type explicitly.
        assert format_type in ("nt5", "unifiedqa")
    else:
        # Or we'll try to infer it automatically.
        if "nt5" in tokenizer.name_or_path.lower():
            format_type = "nt5"
        elif "preasm" in tokenizer.name_or_path.lower():
            format_type = "unifiedqa"
        elif "poet" in tokenizer.name_or_path.lower():
            format_type = "nt5"
        elif "t5" in tokenizer.name_or_path.lower():
            format_type = "nt5"
        elif "bart" in tokenizer.name_or_path.lower():
            format_type = "nt5"
        else:
            raise Exception(
                "The input format_type couldn't be inferred. Please pass it explicitly."
            )

    if format_type == "nt5":
        function = prepare_input_text_in_nt5_format
    elif format_type == "unifiedqa":
        function = prepare_input_text_in_unifiedqa_format

    return function(
        question_text, context_text, max_context_length, max_question_length
    )


def _generate_predictions(
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    instances: List[Dict],
    device: torch.device("cpu"),
    max_context_length: int = 600,
    max_question_length: int = 100,
) -> List[Dict]:

    prepared_input_texts = [
        prepare_input_text(
            tokenizer,
            instance["question_text"],
            instance["context_text"],
            max_context_length=max_context_length,
            max_question_length=max_question_length,
        )
        for instance in instances
    ]
    if model.device != device:
        model.to(device)
    input_ids = tokenizer(
        prepared_input_texts,
        return_tensors="pt",
        truncation=True,
        max_length=800,
        add_special_tokens=True,
        padding=True,
    )["input_ids"].to(device)
    generated_ids = model.generate(
        input_ids,
        min_length=1,
        max_length=50,
        num_beams=1,
    )
    generated_predictions = tokenizer.batch_decode(
        generated_ids, skip_special_tokens=False
    )
    generated_predictions = [
        tokenizer.fix_decoded_text(generated_prediction)
        for generated_prediction in generated_predictions
    ]
    return generated_predictions


def generate_predictions(
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    instances: List[Dict],
    device: torch.device("cpu"),
    batch_size: int = 8,
    max_context_length: int = 600,
    max_question_length: int = 100,
) -> List[Dict]:

    model.to(device)

    predictions = []
    for index in tqdm(range(0, len(instances), batch_size)):
        batch_of_instances = instances[index : index + batch_size]
        batch_of_predictions = _generate_predictions(
            tokenizer,
            model,
            batch_of_instances,
            device=device,
            max_context_length=max_context_length,
            max_question_length=max_question_length,
        )
        predictions += batch_of_predictions

    return predictions


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions with one of the HF models on one of the datasets."
    )
    parser.add_argument("hf_model_name_or_path", type=str, help="hf_model_name_or_path")
    parser.add_argument("evaluation_path", type=str, help="evaluation_path")
    parser.add_argument("output_path", type=str, help="output_path")
    parser.add_argument("--batch_size", type=int, help="batch_size", default=32)
    parser.add_argument(
        "--max_context_length", type=int, help="max_context_length", default=600
    )
    parser.add_argument(
        "--max_question_length", type=int, help="max_question_length", default=100
    )
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.hf_model_name_or_path, use_fast=False
    )
    enable_digit_tokenization(tokenizer)

    model = AutoModelForSeq2SeqLM.from_pretrained(args.hf_model_name_or_path)

    instances = read_jsonl(args.evaluation_path)
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    generated_predictions = generate_predictions(
        tokenizer,
        model,
        instances,
        device=device,
        batch_size=args.batch_size,
        max_context_length=args.max_context_length,
        max_question_length=args.max_question_length,
    )

    for instance, generated_prediction in zip(instances, generated_predictions):
        generated_prediction = generated_prediction.strip()
        instance["predicted_text"] = generated_prediction
        instance["predicted_answers"] = [
            predicted_answer.strip()
            for predicted_answer in generated_prediction.split(ANS_DELIMITER)
        ]

    output_directory = os.path.dirname(args.output_path)
    os.makedirs(output_directory, exist_ok=True)
    print(f"Saving {len(instances)} predictions in {args.output_path}.")
    with open(args.output_path, "w") as file:
        for instance in instances:
            file.write(json.dumps(instance) + "\n")


if __name__ == "__main__":
    main()
