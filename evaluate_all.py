import os
import subprocess

from constants import ALL_MODEL_NAMES, EVALUATION_NAME_TO_FILEPATH


def main():

    for index, model_name in enumerate(ALL_MODEL_NAMES):
        print(f"\n\nWorking on model {index+1}/{len(ALL_MODEL_NAMES)} [{model_name}].")

        for evaluation_name in EVALUATION_NAME_TO_FILEPATH.keys():

            if evaluation_name in ("drop_test", "tatqa_test"):
                print(f"Skipping {evaluation_name} as it needs to be evaluated on the leaderboard.")
                continue

            model_with_data_name = model_name
            model_with_data_name += "-"
            model_with_data_name += (
                evaluation_name.replace("_dev", "").replace("_test", "").replace(
                    "_cs", ""
                ).replace("_bpb", "").replace("_", "-")
            )

            prediction_file_path = os.path.join(
                "predictions", model_with_data_name + "__" + evaluation_name + ".jsonl"
            )
            output_file_path = os.path.join(
                "evaluations", model_with_data_name + "__" + evaluation_name + ".json"
            )

            if not os.path.exists(prediction_file_path):
                print(
                    f"The prediction file path {prediction_file_path} doesn't exist yet. "
                    f"So skipping evaluation for now."
                )
                continue

            command = " ".join(
                [
                    "python",
                    "evaluate.py",
                    prediction_file_path,
                    evaluation_name,
                    f"--output_file_path {output_file_path}"
                ]
            )
            print(command)
            subprocess.call(command.split())


if __name__ == "__main__":
    main()
