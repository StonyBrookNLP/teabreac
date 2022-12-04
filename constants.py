import itertools

DF_COL_DELIMITER = "[COLD]"
DF_ROW_DELIMITER = "[ROWD]"
ANS_DELIMITER = "<ss>" # To keep it consistent with nt5


EVALUATION_NAME_TO_FILEPATH = {
    "drop_dev": "processed_target_datasets/drop/dev.jsonl",
    "drop_test": "processed_target_datasets/drop/test.jsonl",
    "drop_cs": "processed_target_datasets/drop_cs/test.jsonl",
    "drop_bpb": "processed_target_datasets/drop_bpb/validated_dev.jsonl",
    "tatqa_dev": "processed_target_datasets/tatqa/dev.jsonl",
    "tatqa_test": "processed_target_datasets/tatqa/test.jsonl",
    "iirc_gold_dev": "processed_target_datasets/iirc_gold/dev.jsonl",
    "iirc_gold_test": "processed_target_datasets/iirc_gold/test.jsonl",
    "iirc_retrieved_dev": "processed_target_datasets/iirc_retrieved/dev.jsonl",
    "iirc_retrieved_test": "processed_target_datasets/iirc_retrieved/test.jsonl",
    "numglue_dev": "processed_target_datasets/numglue/dev.jsonl",
    "numglue_test": "processed_target_datasets/numglue/test.jsonl",
}

model_name_prefixes = ["", "teabreac-"]
model_name_middles = [
    "bart-large",
    "t5-large",
    "t5-3b",
    "nt5-small",
    "preasm-large",
    "poet-large",
]
model_name_suffixes = [
    "-drop",
    "-iirc-gold",
    "-iirc-retrieved",
    "-numglue",
    "-tatqa",
]
ALL_MODEL_NAMES = [
    model_name_prefix + model_name_middle + model_name_suffix
    for model_name_prefix, model_name_middle, model_name_suffix in itertools.product(
        model_name_prefixes, model_name_middles, model_name_suffixes
    )
]
