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

ALL_MODEL_NAMES = [ # Order is according to Table 1.
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