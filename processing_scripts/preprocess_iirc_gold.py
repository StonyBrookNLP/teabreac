# Mostly taken from iirc_dataset.py from PreaSM code.
import uuid

import os
from lib import read_json, write_jsonl


def main():

    raw_data_directory = "raw_target_datasets"
    processed_data_directory = "processed_target_datasets"

    input_directory = os.path.join(raw_data_directory, "iirc")
    output_directory = os.path.join(processed_data_directory, "iirc_gold")
    os.makedirs(output_directory, exist_ok=True)

    set_names = ["train", "dev", "test"]
    for set_name in set_names:
        print(f"Processing {set_name}")

        processed_instances = []

        input_filepath = os.path.join(input_directory, f"{set_name}.json")
        output_filepath = os.path.join(output_directory, f"{set_name}.jsonl")

        data_objects = read_json(input_filepath)

        for passage_index, data_object in enumerate(data_objects):
            for question_object in data_object["questions"]:

                # The test set doesn't have qids, so we generate a random unique
                # id at runtime.
                question_id = question_object.get("qid", uuid.uuid4().hex)

                contexts = question_object["context"]
                passage_id = str(passage_index)

                context_text = " \n ".join(
                    [
                        context["passage"] + ": " + context["text"]
                        for context in contexts
                    ]
                )

                question_text = question_object["question"]
                answer_object = question_object["answer"]

                if answer_object["type"] == "none":
                    answer_list = ["none"]
                elif answer_object["type"] == "span":
                    answer_list = [
                        "#".join(
                            [a["text"] for a in answer_object["answer_spans"]]
                        ).strip()
                    ]
                elif answer_object["type"] in ["binary", "value"]:
                    answer_list = [answer_object["answer_value"].strip()]
                else:
                    raise Exception("Unknown answer type.")

                answer_type = answer_object["type"]
                processed_instance = {
                    "passage_id": passage_id,
                    "question_id": question_id,
                    "question_text": question_text,
                    "context_text": context_text,
                    "answer_list": answer_list,
                    "answer_type": answer_type,
                }

                processed_instances.append(processed_instance)

        write_jsonl(processed_instances, output_filepath)


if __name__ == "__main__":
    main()
