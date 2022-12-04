# Mostly taken from iirc_retrieval_dataset.py from PreaSM code.
import uuid
import os

from lib import read_json, read_jsonl, write_jsonl


def main():

    raw_data_directory = "raw_target_datasets"
    processed_data_directory = "processed_target_datasets"

    input_directory = os.path.join(raw_data_directory, "iirc")
    output_directory = os.path.join(processed_data_directory, "iirc_retrieved")
    os.makedirs(output_directory, exist_ok=True)

    set_names = ["train", "dev", "test"]
    for set_name in set_names:
        print(f"Processing {set_name}")

        if set_name in ["dev", "test"]:
            dev_test_retrieval_filepath = os.path.join(
                raw_data_directory, "iirc", f"{set_name}_retrieved.jsonl"
            )
            all_question_retrieval_data = read_jsonl(dev_test_retrieval_filepath)
            all_question_retrieval_data_dict = {}
            for index, question_retrieval_data in enumerate(
                all_question_retrieval_data
            ):
                all_question_retrieval_data_dict[index] = question_retrieval_data

        processed_instances = []

        input_filepath = os.path.join(input_directory, f"{set_name}.json")
        output_filepath = os.path.join(output_directory, f"{set_name}.jsonl")

        data_objects = read_json(input_filepath)

        global_index = 0
        for passage_index, data_object in enumerate(data_objects):
            for question_object in data_object["questions"]:

                question_id = question_object.get("qid", uuid.uuid4().hex)
                question_text = question_object["question"]

                contexts = question_object["context"]
                passage_id = str(passage_index)

                main_passage = data_object["title"] + ": " + data_object["text"]

                if set_name == "train":

                    gold_sentences = [
                        c["passage"] + ": " + c["text"]
                        for c in contexts
                        if c["passage"] != "main"
                    ]
                    context_text = (
                        "Links: \n "
                        + "\n".join(gold_sentences)
                        + " \n Main: \n "
                        + main_passage
                    )

                else:

                    question_retrieval_data = all_question_retrieval_data_dict[
                        global_index
                    ]
                    assert (
                        question_retrieval_data["question"]
                        == question_object["question"]
                    )

                    retrieved_sentences = []
                    for retrieved_context in question_retrieval_data[
                        "context_retrieval"
                    ]["predicted_link_name_sent_list"]:
                        sentence_text = retrieved_context["sent"]
                        if sentence_text[:12] == "Introduction":
                            # remove the prefix
                            sentence_text = "\n\n".join(sentence_text.split("\n\n")[1:])
                        sentence = {
                            "passage": retrieved_context["title"],
                            "text": sentence_text,
                        }
                        retrieved_sentences.append(sentence)

                    links = {
                        l["target"].lower(): l["target"] for l in data_object["links"]
                    }
                    retrieved_sentences = [
                        links[r["passage"]] + ": " + r["text"].replace("\n", " ")
                        for r in retrieved_sentences
                        if "NULL" not in r["text"]
                    ]
                    context_text = (
                        "Links: \n "
                        + "\n".join(retrieved_sentences)
                        + " \n Main: \n "
                        + main_passage
                    )

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
                global_index += 1

        write_jsonl(processed_instances, output_filepath)


if __name__ == "__main__":
    main()
