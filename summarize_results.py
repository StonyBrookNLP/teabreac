import os
import json

import pandas as pd


def main():

    metric_type = "ans_f1" # choices: ans_em or ans_f1.

    dataframe_dict = { # Order is according to Table 1.
        "model": [],
        "drop_dev": [],
        # "drop_dev": [], # needs to be evaluated on the leaderboard
        "tatqa_dev": [],
        # "tatqa_test": [], # needs to be evaluated on the leaderboard
        "iirc_gold_dev": [],
        "iirc_gold_test": [],
        "iirc_retrieved_dev": [],
        "iirc_retrieved_test": [],
        "numglue_dev": [],
        "numglue_test": [],
        "drop_cs": [],
        "drop_bpb": [],
    }

    ordered_model_names = [ # Order is according to Table 1.
        "bart-large",
        "teabreac-bart-large",
        "t5-large",
        "teabreac-t5-large",
        "t5-3b",
        "teabreac-t5-3b",
        "nt5-small",
        "teabreac-nt5-small",
        "preasm-large",
        "teabreac-preasm-large",
        "poet-large"
        "teabreac-poet-large"
    ]

    for model_name in ordered_model_names:

        evaluation_names = list(dataframe_dict.keys())[1:]

        dataframe_dict["model"].append(model_name)
        for evaluation_name in evaluation_names:

            model_name += "-"
            model_name += evaluation_name.replace("_dev", "").replace("_test", "")

            metrics_file_path = os.path.join(
                "evaluations", model_name + "__" + evaluation_name + ".json"
            )
            if not os.path.exists(metrics_file_path):
                metric_value = "n/a"
            else:
                with open(metrics_file_path, "r") as file:
                    metrics = json.load(file)

                if metric_type not in metrics:
                    raise Exception(
                        f"The metric_type {metric_type} not present"
                        f"in metrics found at {metrics_file_path}"
                    )
                metric_value = metrics[metric_type]
            dataframe_dict[evaluation_name].append(metric_value)

    dataframe = pd.DataFrame.from_dict(dataframe_dict)

    report_path = "results_report.txt"
    print(
        f"Saving report in {report_path} "
        f"(Best viewed in an editor with wordwrapping disabled)."
    )
    with open(report_path, "w") as file:
        file.write(dataframe.to_string())


if __name__ == "__main__":
    main()
