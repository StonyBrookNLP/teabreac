from typing import List, Dict
import json
import os


def predictions_to_drop_format(prediction_instances: List[Dict], output_path: str):

    official_predictions_json = {}
    for prediction_instance in prediction_instances:
        question_id = prediction_instance["question_id"]
        predicted_answers = prediction_instance["predicted_answers"]
        predicted_answers = [predicted_answer for predicted_answer in predicted_answers]
        if len(predicted_answers[0]) == 1:
            prediction_obj = predicted_answers[0]
        else:
            prediction_obj = predicted_answers
        official_predictions_json[question_id] = prediction_obj

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(official_predictions_json, file, indent=4)


def predictions_to_tatqa_format(prediction_instances: List[Dict], output_path: str):

    official_predictions_json = {}
    for prediction_instance in prediction_instances:
        question_id = prediction_instance["question_id"]
        predicted_answers = prediction_instance["predicted_answers"]
        predicted_answers = [predicted_answer for predicted_answer in predicted_answers]

        metric_clipped_predicted_answers = []
        for predicted_answer in predicted_answers:
            metric = ""
            possible_metrics = ["thousand", "million", "billion", "percent"]
            for possible_metric in possible_metrics:
                if predicted_answer.strip().endswith(possible_metric):
                    metric = possible_metric
                    predicted_answer = predicted_answer.strip().replace(
                        possible_metric, ""
                    )
                    break
            metric_clipped_predicted_answers.append(predicted_answer)

        prediction_obj = [metric_clipped_predicted_answers, metric]
        official_predictions_json[question_id] = prediction_obj

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w") as file:
        json.dump(official_predictions_json, file, indent=4)
