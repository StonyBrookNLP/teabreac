# <h2 align="center"> Teaching Broad Reasoning Skills for Multi-Step QA by Generating Hard Contexts </h2>

This is the repository for our EMNLP 2022 paper ["Teaching Broad Reasoning Skills for Multi-Step QA by Generating Hard Contexts"](https://arxiv.org/abs/2205.12496).

## Data

:tea: TeaBReaC dataset is distributed under a [CC BY 4.0 License](https://creativecommons.org/licenses/by/4.0/).

You can downoad it manually from [here](https://drive.google.com/file/d/1DLap7BsrwEon6vJQZdtr84Ii5rr2pt8y/view?usp=sharing) or run `./download_teabreac_data.sh` and it'll available in `data/` directory.

## Models

All the models in our paper are released on the Huggingface hub over [here](https://huggingface.co/StonyBrookNLP). In all, we release the following models:

- **A**: Base Models finetuned on target datasets: `{base_model}-{target_dataset}`
- **B**: Base models pretrained on TeaBReaC: `teabreac-{base_model}`
- **C**: Base models pretrained on TeaBReaC and then finetuned on target datasets: `teabreac-{base_model}-{target_dataset}`

The base_model above can be from: `bart-large`, `t5-large`, `t5-3b`, `nt5-small`, `preasm-large`. The target_dataset above can be from: `drop`, `tatqa`, `iirc-gold`, `iirc-retrieved`, `numglue`.

The **A** models are only released for completeness / reproducibility. In your end application you probably just want to use either **B** or **C**.

You can use any of the models in the following way:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from digit_tokenization import enable_digit_tokenization # from digit_tokenization.py

model_name = "teabreac-t5-3b-drop"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False) # Fast doesn't work with digit tokenization
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
enable_digit_tokenization(tokenizer)
input_texts = [
    "answer_me: Who scored the first touchdown of the game?" +
    "context: ... Oakland would get the early lead in the first quarter as quarterback JaMarcus Russell completed a 20-yard touchdown pass to rookie wide receiver Chaz Schilens..."
    # Note: some models have slightly different qn/ctxt format. See the predict.py
]
input_ids = tokenizer(
    input_texts, return_tensors="pt",
    truncation=True, max_length=800,
    add_special_tokens=True, padding=True,
)
generated_ids = model.generate(input_ids, min_length=1,  max_length=50)
generated_predictions = tokenizer.batch_decode(generated_ids, skip_special_tokens=False)
generated_predictions = [
    tokenizer.fix_decoded_text(generated_prediction) for generated_prediction in generated_predictions
]
# => ["Chaz Schilens"]
```
The above has been verified with python=3.9.0, transformers=4.24.0 and PyTorch=1.12.1.

If you want to run a model on one of the datasets we've evaluated on directly, see ## Experiments.

## Experiments

You can generate predictions and/or evaluations for all the models/datasets by the following steps.

#### Prepare processed datasets:

For this, you can either download them using
```bash
./download_processed_target_datasets.sh
```

or download the raw data and process them yourself:
```bash
./download_raw_target_datasets.sh # each folder in it has source.txt saying where we got it from.
pip install -r requirements/preprocess.txt
python processing_scripts/preprocess_drop.py
python processing_scripts/preprocess_drop_bpb.py
python processing_scripts/preprocess_drop_cs.py
python processing_scripts/preprocess_tatqa.py
python processing_scripts/preprocess_iirc_gold.py
python processing_scripts/preprocess_iirc_retrieved.py
python processing_scripts/preprocess_numglue.py
```

#### Run predictions

Install dependencies: `pip install -r requirements/predict.txt`. You can then generate predictions with the model and dataset combination of your choice:

```bash
python predict.py teabreac-t5-3b-drop processed_data/drop/dev.jsonl predictions/teabreac-t5-3b-drop__drop_dev.jsonl
#                 ^ model_name        ^ evaluation path             ^ (output) prediction path
```
You can also generate predictions for all model-data combinations with `python predict_all.py`.

#### Run evaluations

Install dependencies: `pip install -r requirements/evaluate.txt` (you may want to upgrade/reinstall pytorch, transformers here as installing allennlp would downgrade their versions). You can then evaluate these predictions with:
```bash
python evaluate.py predictions/teabreac-t5-3b-drop__drop_dev.jsonl drop_dev
#                  ^ prediction_path                               ^ dataset_name
```
You can also generate evaluation metrics for all model-data combinations with `python evaluate_all.py`.

#### Summarize results

If you've generated all predictions and evaluations, you can also generate the full summary of results on all model/dataset combinations (like Table 1) by running:

```bash
python summarize_results.py
```

Note that all our experiments (training, prediction, evaluation) were done in allennlp, and we ported the models and prediction scripts to huggingface posthoc. So there may be slight difference (expectedly < 0.2 F1 points) in the numbers.

## Citation

If you use this work, please consider citing us:
```
@article{trivedi2022teaching,
  title={Teaching Broad Reasoning Skills for Multi-Step QA by Generating Hard Contexts},
  author={Trivedi, Harsh and Balasubramanian, Niranjan and Khot, Tushar and Sabharwal, Ashish},
  journal={arXiv preprint arXiv:2205.12496},
  year={2022}
}
```