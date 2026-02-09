# AskQE-for-summarization
 Adaptation of the AskQE framework to the task of summarization, rather than machine translation. We aim to evaluate whether the key facts contained in a document are retained in the summary. 

## PubMed Summarization with DistilBART
This script generates abstractive summaries for PubMed articles using `sshleifer/distilbart-cnn-12-6`.

### Requirements
```bash
pip install -r requirements.txt
```
### Usage
```bash
python summarize_pubmed.py \
  --input data/pubmed_500.json \
  --output outputs/pubmed_distilbart_summaries.jsonl
```
## Fact extraction
```bash
python extract_atomic_facts.py \
  --input data/pubmed_500.json \
  --output outputs/pubmed_atomic_facts.jsonl
```
## NLI filtering
```bash
python nli_filter_facts.py \
  --pubmed data/pubmed_500.json \
  --atomic_facts outputs/pubmed_atomic_facts.jsonl \
  --output outputs/pubmed_facts_entailed.jsonl \
  --threshold 0.5
```
