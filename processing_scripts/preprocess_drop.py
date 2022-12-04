import os

from lib import read_json, write_jsonl


def main():

    raw_data_directory = "raw_target_datasets"
    processed_data_directory = "processed_target_datasets"

    if not os.path.exists(raw_data_directory):
        raise Exception(
            f"Raw data directory ({raw_data_directory}) not found. Please download it first."
        )

    input_directory = os.path.join(raw_data_directory, "drop")
    output_directory = os.path.join(processed_data_directory, "drop")
    os.makedirs(output_directory, exist_ok=True)

    set_names = ["train", "dev", "test"]
    for set_name in set_names:
        print(f"Processing {set_name}")

        processed_instances = []

        input_filepath = os.path.join(input_directory, f"drop_dataset_{set_name}.json")
        output_filepath = os.path.join(output_directory, f"{set_name}.jsonl")

        data_object = read_json(input_filepath)

        for passage_id, passage_object in data_object.items():
            passage = passage_object["passage"]
            qa_pairs = passage_object["qa_pairs"]
            for qa_pair in qa_pairs:
                question_text = qa_pair["question"]
                question_id = qa_pair["query_id"]

                if set_name != "test":
                    answers_object = qa_pair["answer"]
                else:
                    # answers are not available.
                    answers_object = {
                        "number": "",
                        "date": {"day": "", "month": "", "year": ""},
                        "spans": [],
                    }

                answers_objects = [answers_object]
                if "validated_answers" in qa_pair:
                    answers_objects.extend(qa_pair["validated_answers"])

                processed_instance = {
                    "passage_id": passage_id,
                    "question_id": question_id,
                    "question_text": question_text,
                    "context_text": passage,
                    "answers_objects": answers_objects,
                }

                processed_instances.append(processed_instance)

        write_jsonl(processed_instances, output_filepath)


if __name__ == "__main__":
    main()
