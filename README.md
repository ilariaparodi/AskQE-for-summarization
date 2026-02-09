# AskQE-for-summarization
 Adaptation of the AskQE framework to the task of summarization, rather than machine translation. We aim to evaluate whether the key facts contained in a document are retained in the summary. 

## PubMed Summarization with DistilBART
This script generates abstractive summaries for PubMed articles using `sshleifer/distilbart-cnn-12-6`.

## Requirements
```bash
pip install -r requirements.txt
```
## Usage
```bash
python summarize_pubmed.py \
  --input data/pubmed_500.json \
  --output outputs/pubmed_distilbart_summaries.jsonl
```
