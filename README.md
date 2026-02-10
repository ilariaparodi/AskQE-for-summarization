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
  --input dpubmed_data.json \
  --output pubmed_distilbart_summaries.jsonl
```
## Fact extraction
```bash
python extract_atomic_facts.py \
  --input pubmed_data.json \
  --output pubmed_atomic_facts.jsonl
```
## NLI filtering
```bash
python nli_filter_facts.py \
  --pubmed data/pubmed_data.json \
  --atomic_facts pubmed_atomic_facts.jsonl \
  --output pubmed_facts_entailed.jsonl \
  --threshold 0.5
```
## Question Generation
```bash
python generate_questions.py \
  --input pubmed_facts_entailed.jsonl \
  --output pubmed_questions_generation.jsonl
```
# Question Answering
```bash
python question_answering.py \
  --pubmed pubmed_data.json \
  --summaries pubmed_distilbart_summaries.jsonl \
  --questions pubmed_questions_generation.jsonl \
  --output pubmed_qa_results.jsonl
```

