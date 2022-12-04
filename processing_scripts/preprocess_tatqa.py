# Taken mostly from TAT-QA code.
import os
from typing import List, Dict

from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel

from lib import read_json, write_jsonl, hash_object
from constants import DF_COL_DELIMITER, DF_ROW_DELIMITER


def get_order_by_tf_idf(
    question: str, order_to_paragraph_texts: Dict[str, List]
) -> List[str]:
    sorted_order = []
    corpus = [question]
    for order, text in order_to_paragraph_texts.items():
        corpus.append(text)
        sorted_order.append(order)
    tf_idf = TfidfVectorizer().fit_transform(corpus)
    cosine_similarities = linear_kernel(tf_idf[0:1], tf_idf).flatten()[1:]
    sorted_similarities = sorted(enumerate(cosine_similarities), key=lambda x: x[1])
    idx = [i[0] for i in sorted_similarities][::-1]
    return [sorted_order[index] for index in idx]


def main():

    raw_data_directory = "raw_target_datasets"
    processed_data_directory = "processed_target_datasets"

    input_directory = os.path.join(raw_data_directory, "tatqa")
    output_directory = os.path.join(processed_data_directory, "tatqa")
    os.makedirs(output_directory, exist_ok=True)

    set_names = ["train", "dev", "test"]
    for set_name in set_names:
        print(f"Processing {set_name}")

        processed_instances = []

        input_filepath = os.path.join(input_directory, f"tatqa_dataset_{set_name}.json")
        output_filepath = os.path.join(output_directory, f"{set_name}.jsonl")

        data_objects = read_json(input_filepath)

        for data_object in tqdm(data_objects):

            table = data_object["table"]
            paragraphs = data_object["paragraphs"]
            questions = data_object["questions"]

            table_text = f" {DF_COL_DELIMITER} ".join(
                [
                    f" {DF_ROW_DELIMITER} ".join([e.strip() for e in row_object])
                    for row_object in table["table"]
                ]
            )

            order_to_paragraph_texts = {
                paragraph["order"]: paragraph["text"] for paragraph in paragraphs
            }

            passage_id = hash_object(
                table["uid"]
                + " ".join(
                    [e["uid"] for e in sorted(paragraphs, key=lambda e: e["order"])]
                )
            )
            for question in questions:

                question_id = question["uid"]
                question_text = question["question"]

                sorted_order = get_order_by_tf_idf(
                    question_text, order_to_paragraph_texts
                )
                local_paragraph_text = " ".join(
                    [order_to_paragraph_texts[order] for order in sorted_order]
                )

                context_text = f"TABLE: {table_text}  PARAGRAPH: {local_paragraph_text}"

                if set_name != "test":

                    answer_type = question["answer_type"]
                    scale = question["scale"]

                    if isinstance(question["answer"], (list, tuple)):
                        answer_list = [
                            str(answer).strip() + " " + scale.strip()
                            for answer in question["answer"]
                        ]
                    elif isinstance(question["answer"], (int, float, str)):
                        answer_list = [
                            str(question["answer"]).strip() + " " + scale.strip()
                        ]
                    else:
                        raise Exception("Unknown answer type.")

                    answer_list = [e.strip() for e in answer_list]

                else:

                    answer_type = "span"
                    answer_list = []

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
