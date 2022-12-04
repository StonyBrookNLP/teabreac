from collections import defaultdict
import random
import os

from lib import read_json, write_jsonl


random.seed(13370)


def main():

    raw_data_directory = "raw_target_datasets"
    processed_data_directory = "processed_target_datasets"

    input_directory = os.path.join(raw_data_directory, "numglue")
    output_directory = os.path.join(processed_data_directory, "numglue")
    os.makedirs(output_directory, exist_ok=True)

    reasoning_name_to_type = {
        "math_application_chemistry": "type_2",
        "math_application_physics": "type_2",
        "nli_stresstest": "type_7",
        "nli_awpnli": "type_7",
        "nli_newsnli": "type_7",
        "nli_rte_quant": "type_7",
        "nli_redditnli": "type_7",
        "missing_numerical_knowledge": "type_1",
        "arithmetic_word_problem": "type_8",
        "quantitative_comparison": "type_3",
        "completion": "type_4",
        "trainrc_explicit": "type_5",
        "devrc_explicit": "type_5",
        "trainrc_implicit": "type_6",
        "devrc_implicit": "type_6",
    }

    set_names = ["train", "dev", "test"]
    for set_name in set_names:
        print(f"Processing {set_name}")

        processed_instances = []
        reasoning_type_to_processed_instances = defaultdict(list)

        input_filepath = os.path.join(input_directory, f"drop_format_{set_name}.json")
        output_filepath = os.path.join(output_directory, f"{set_name}.jsonl")

        data_object = read_json(input_filepath)

        for passage_id, passage_object in data_object.items():
            passage = passage_object["passage"]
            qa_pairs = passage_object["qa_pairs"]
            for qa_pair in qa_pairs:
                question_text = qa_pair["question"]
                answers_object = qa_pair["answer"]
                question_id = qa_pair["query_id"]

                answers_objects = [answers_object]
                if "validated_answers" in qa_pair:
                    answers_objects.extend(qa_pair["validated_answers"])

                reasoning_name = "_".join(passage_id.split("_")[:-1]).lower()
                reasoning_type = reasoning_name_to_type[reasoning_name]
                processed_instance = {
                    "reasoning_type": reasoning_type,
                    "passage_id": passage_id,
                    "question_id": question_id,
                    "question_text": question_text,
                    "context_text": passage,
                    "answers_objects": answers_objects,
                }

                processed_instances.append(processed_instance)
                reasoning_type_to_processed_instances[reasoning_type].append(
                    processed_instance
                )

        random.shuffle(processed_instances)
        write_jsonl(processed_instances, output_filepath)

        for (
            reasoning_type,
            _processed_instances,
        ) in reasoning_type_to_processed_instances.items():
            random.shuffle(_processed_instances)
            _output_filepath = os.path.join(
                output_directory, f"{reasoning_type}_{set_name}.jsonl"
            )
            write_jsonl(_processed_instances, _output_filepath)


if __name__ == "__main__":
    main()
