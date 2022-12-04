import os
import subprocess

from constants import ALL_MODEL_NAMES, EVALUATION_NAME_TO_FILEPATH


def main():

    for index, model_name in enumerate(ALL_MODEL_NAMES):
        model_path = f"StonyBrookNLP/{model_name}"
        print(f"\n\nWorking on model {index+1}/{len(ALL_MODEL_NAMES)} [{model_name}].")

        for (
            evaluation_name,
            evaluation_file_path,
        ) in EVALUATION_NAME_TO_FILEPATH.items():
            output_file_path = os.path.join(
                "predictions", model_name + "__" + evaluation_name + ".jsonl"
            )
            command = " ".join(
                [
                    "python",
                    "predict.py",
                    model_path,
                    evaluation_file_path,
                    output_file_path,
                ]
            )
            print(command)
            subprocess.call(command.split())


if __name__ == "__main__":
    main()
