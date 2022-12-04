from typing import List, Dict
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


os.environ["TOKENIZERS_PARALLELISM"] = "false"


def compute_answer_scores(prediction_instances: List[Dict]) -> Dict:

    answer_text_metrics = ListSquadEmAndF1(keep_whitespace=True)

    for instance in prediction_instances:

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

        if "answers" in instance:  # It's a list of validated answers.
            for answer in instance["answers"]:
                assert isinstance(answer, str)
            all_answer_texts = [(answer,) for answer in instance["answers"]]

        if (
            "answer_list" in instance
        ):  # It's a single "answer" (which is a list of entities/etc)
            assert isinstance(instance["answer_list"], tuple) or isinstance(
                instance["answer_list"], list
            )
            all_answer_texts = [tuple(instance["answer_list"])]

        assert all_answer_texts is not None

        predicted_answers = instance["predicted_answers"]
        answer_text_metrics(predicted_answers, all_answer_texts)

    metric_values = answer_text_metrics.get_metric(reset=False)

    result = {
        "ans_em": round(metric_values["ans_em"], 3),
        "ans_f1": round(metric_values["ans_f1"], 3),
    }
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
            for _, subdata in metrics_data.iterrows():
                if subdata["subcategory"].lower() == f"type_{num}":
                    type_to_em[f"type_{num}"] = subdata["answer_em"]
                    type_to_f1[f"type_{num}"] = subdata["answer_f1"]
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
        os.makedirs(os.path.dirname(args.output_file_path))
        with open(args.output_file_path, "w") as file:
            json.dump(result, file)


if __name__ == "__main__":
    main()
