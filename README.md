# BIONER - Named Entity Recognition for BioMedical datasets

## Experiment

```shell script
python run_bert_crf_ner.py \
    --data_path ./data \
    --dataset bc2gm \
    --train_fn train.txt \
    --test_fn test.txt \
    --additional_data True \
    --use_bilstm True
```

List of datasets:
- bc2gm
- bc4chemdner
- bc5cdr_chem
- bc5cdr_disease
- jnlpba
- ncbi_disease

## Installation

```shell script
pip install -r requirements.txt
```
