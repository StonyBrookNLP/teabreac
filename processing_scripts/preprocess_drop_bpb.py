import os

from lib import read_json, write_jsonl


def main():

    raw_data_directory = "raw_target_datasets"
    processed_data_directory = "processed_target_datasets"

    input_filepath = os.path.join(
        raw_data_directory, "drop_bpb", "drop_dev_contrast_set_sample_validated.json"
    )

    output_filepath = os.path.join(
        processed_data_directory, "drop_bpb", "validated_dev.jsonl"
    )

    os.makedirs(os.path.dirname(output_filepath), exist_ok=True)

    processed_instances = []
    data_object = read_json(input_filepath)
    for passage_id, passage_object in data_object.items():
        passage = passage_object["passage"]
        qa_pairs = passage_object["qa_pairs"]
        for qa_pair in qa_pairs:
            question_text = qa_pair["question"]
            answers_object = qa_pair["answer"]
            question_id = qa_pair["query_id"]

            if answers_object["spans"] in (["yes"], ["no"]):
                # Original DROP doesn't have yes/no, so asking yes/no questions
                # directly on the test set is not fair.
                continue

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
