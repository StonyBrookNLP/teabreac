from typing import List, Dict
from collections import Counter, defaultdict
import argparse
import json
import os
import subprocess
import uuid

from allennlp.common.util import import_module_and_submodules
import_module_and_submodules("allennlp_lib")
from allennlp_lib.training.metrics.list_squad_em_and_f1 import ListSquadEmAndF1
from allennlp_lib.tools.drop import answer_json_to_strings

from lib import read_jsonl
from predictions_to_official_format import (
    predictions_to_drop_format,
    predictions_to_tatqa_format,
)


def compute_answer_scores(prediction_instances: List[Dict]) -> Dict:

    answer_text_metrics = defaultdict(lambda: ListSquadEmAndF1(keep_whitespace=True))

    category_overall = ("overall", "overall")

    categories_overall = set([category_overall])
    categories_num_steps = set()
    categories_answer_type = set()
    categories_task_name = set()
    categories_program_type = set()

    category_counter = Counter()

    for instance in prediction_instances:

        answer_type = instance.get("answer_type", None)
        if answer_type is None:
            answer_type = instance.get("reasoning_type", None)

        all_answer_texts = None
        if "all_answer_texts" in instance:
            all_answer_texts = instance["all_answer_texts"]
            for e in all_answer_texts:
                assert isinstance(e, tuple) or isinstance(e, list)
            all_answer_texts = [tuple(e) for e in all_answer_texts]

        if "answers_objects" in instance:
            all_answer_texts = [
                answer_json_to_strings(answer_object)[0]
                for answer_object in instance["answers_objects"]
            ]

            answers_object = instance["answers_objects"][0]
            if answer_type is not None:
                pass
            elif "number" in answers_object and answers_object["number"]:
                answer_type = "number"
            elif "spans" in answers_object and answers_object["spans"]:
                answer_type = "spans"
            elif "date" in answers_object:
                answer_type = "date"
            else:
                raise Exception("Answer type couldn't be determined.")

        if "answers" in instance: # It's a list of validated answers.
            for answer in instance["answers"]:
                assert isinstance(answer, str)
            all_answer_texts = [(answer,) for answer in instance["answers"]]

        if "answer_list" in instance: # It's a single "answer" (which is a list of entities/etc)
            assert isinstance(instance["answer_list"], tuple) or isinstance(instance["answer_list"], list)
            all_answer_texts = [tuple(instance["answer_list"])]

        assert all_answer_texts is not None

        predicted_answers = instance["predicted_answers"]

        if "program_modules" not in instance:
            program_type = "unknown"
            num_steps = "unknown"
            task_name = "real_qa"
        else:
            program_modules = instance["program_modules"]
            program_type = "__".join(program_modules)
            num_steps = len(program_modules)
            task_name = "teabreac_primitive_qa" if num_steps == 1 else "teabreac_multistep_qa"

        category_num_steps = ("num_steps", num_steps)
        category_answer_type = ("answer_type", answer_type)
        category_task_name = ("task_name", task_name)
        category_program_type = ("program_type", program_type)

        categories_num_steps.add(category_num_steps)
        categories_answer_type.add(category_answer_type)
        categories_task_name.add(category_task_name)
        categories_program_type.add(category_program_type)

        key_tuples = [
            category_overall, category_num_steps,
            category_answer_type, category_task_name,
            category_program_type
        ]

        for key_tuple in key_tuples:
            if predicted_answers is not None:
                answer_text_metrics[key_tuple](predicted_answers, all_answer_texts)
            category_counter[key_tuple] += 1

    all_key_tuples = (
        sorted(categories_overall, key=lambda e: e[1]) +
        sorted(categories_num_steps, key=lambda e: e[1]) +
        sorted(categories_answer_type, key=lambda e: e[1]) +
        sorted(categories_task_name, key=lambda e: e[1]) +
        sorted(categories_program_type, key=lambda e: e[1])
    )

    answer_text_em_metric_values = {key_tuple: round(metric.get_metric(reset=False)["ans_em"], 3)
                                    for key_tuple, metric in answer_text_metrics.items()}
    answer_text_f1_metric_values = {key_tuple: round(metric.get_metric(reset=False)["ans_f1"], 3)
                                    for key_tuple, metric in answer_text_metrics.items()}
    data = {
        "category": [e[0] for e in all_key_tuples],
        "subcategory": [e[1] for e in all_key_tuples],
        "answer_em": [answer_text_em_metric_values.get(key_tuple, 0.0) for key_tuple in all_key_tuples],
        "answer_f1": [answer_text_f1_metric_values.get(key_tuple, 0.0) for key_tuple in all_key_tuples],
        "counts": [category_counter[key_tuple] for key_tuple in all_key_tuples]
    }
    # You can load this data in pandas like
    # print(pd.from_dict(data).to_string())
    # and get a detailed summary of results in various categories, subcategories.

    ans_em = answer_text_em_metric_values[category_overall]
    ans_f1 = answer_text_f1_metric_values[category_overall]

    result = {"ans_em": ans_em, "ans_f1": ans_f1, "data": data}
    return result


def compute_answer_scores_with_official_scripts(
    original_prediction_instances: List[Dict], dataset: str
) -> Dict:
    official_prediction_path = os.path.join(".tmp", uuid.uuid4().hex + ".txt")
    official_metrics_path = os.path.join(".tmp", uuid.uuid4().hex + ".txt")

    metrics = {}
    if dataset == "drop_dev":
        predictions_to_drop_format(
            original_prediction_instances, official_prediction_path
        )

        run_command = (
            f"python official_evaluation_scripts/drop_eval.py "
            f"--gold_path raw_target_datasets/drop/drop_dataset_dev.json "
            f"--prediction_path {official_prediction_path} "
            f"--output_path {official_metrics_path}"
        )
        subprocess.run(run_command.split())

        with open(official_metrics_path) as file:
            metrics = json.loads(file.read())
            metrics["ans_em"] = round(metrics.pop("global_em") * 100, 1)
            metrics["ans_f1"] = round(metrics.pop("global_f1") * 100, 1)

    elif dataset == "tatqa_dev":
        predictions_to_tatqa_format(
            original_prediction_instances, official_prediction_path
        )

        run_command = (
            f"python official_evaluation_scripts/tatqa_eval.py "
            f"--gold_path raw_target_datasets/tatqa/tatqa_dataset_dev.json "
            f"--pred_path {official_prediction_path} "
            f"--output_path {official_metrics_path}"
        )
        subprocess.run(run_command.split())

        with open(official_metrics_path) as file:
            metrics = json.loads(file.read())
            metrics["ans_em"] = round(metrics.pop("global_em") * 100, 1)
            metrics["ans_f1"] = round(metrics.pop("global_f1") * 100, 1)

    elif dataset in (
        "iirc_gold_dev",
        "iirc_gold_test",
        "iirc_retrieved_dev",
        "iirc_retrieved_test",
    ):

        metrics = compute_answer_scores(original_prediction_instances)
        metrics["ans_em"] = metrics["ans_em"]
        metrics["ans_f1"] = metrics["ans_f1"]

    elif dataset in ("numglue_dev", "numglue_test"):

        metrics_data = compute_answer_scores(original_prediction_instances)["data"]
        type_to_em = {}
        type_to_f1 = {}
        for num in range(1, 8 + 1):
            for subcategory, ans_em, ans_f1 in zip(
                metrics_data["subcategory"], metrics_data["answer_em"], metrics_data["answer_f1"]
            ):
                if subcategory.lower() == f"type_{num}":
                    type_to_em[f"type_{num}"] = ans_em
                    type_to_f1[f"type_{num}"] = ans_f1
            assert f"type_{num}" in type_to_em

        metrics = {}
        metrics["ans_em"] = round(100 * sum(type_to_em.values()) / len(type_to_em), 1)
        metrics["ans_f1"] = round(100 * sum(type_to_f1.values()) / len(type_to_f1), 1)

    elif dataset in ("drop_cs", "drop_bpb"):

        metrics = compute_answer_scores(original_prediction_instances)

    elif dataset in ("tatqa_test", "drop_test"):
        raise Exception(
            f"The dataset of type {dataset} needs to be evaluated on the leaderboard."
        )

    else:
        raise Exception(f"No matching dataset found for {dataset}")

    return metrics


def main():

    parser = argparse.ArgumentParser(description="Evaluate predictions.")
    parser.add_argument("prediction_path", type=str, help="prediction file path")
    parser.add_argument(
        "dataset",
        type=str,
        default=None,
        choices={
            "drop_dev",
            "drop_test",
            "drop_cs",
            "drop_bpb",
            "tatqa_dev",
            "tatqa_test",
            "iirc_gold_dev",
            "iirc_gold_test",
            "iirc_retrieved_dev",
            "iirc_retrieved_test",
            "numglue_dev",
            "numglue_test",
        },
        help="dataset for official eval.",
    )
    parser.add_argument("--output_file_path", type=str, help="file path to save metrics in.")
    args = parser.parse_args()

    prediction_instances = read_jsonl(args.prediction_path)
    print(f"Number of prediction_instances: {len(prediction_instances)}")
    result = compute_answer_scores_with_official_scripts(
        prediction_instances, args.dataset
    )

    print("\n---------------------------------------")

    print("Answer Metrics:")
    for key, value in result.items():
        print(f"{key}: {value}")

    if args.output_file_path:
        print(f"Saving metrics in {args.output_file_path}")
        os.makedirs(os.path.dirname(args.output_file_path), exist_ok=True)
        with open(args.output_file_path, "w") as file:
            json.dump(result, file)


if __name__ == "__main__":
    main()
