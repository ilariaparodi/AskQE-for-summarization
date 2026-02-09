import json
import os
import argparse
import ast
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate questions from entailed atomic facts using Qwen2.5-3B-Instruct"
    )
    parser.add_argument("--input", type=str, required=True,
                        help="JSONL file with entailed atomic facts")
    parser.add_argument("--output", type=str, required=True,
                        help="Output JSONL file with generated questions")
    parser.add_argument("--max_new_tokens", type=int, default=256,
                        help="Maximum number of tokens to generate")
    return parser.parse_args()


QG_PROMPT = """Task: You will be given an English article and a list of atomic facts, which are short sentences conveying one piece of information. Your goal is to generate a list of relevant questions based on the sentence. Output the list of questions in Python list format without giving any additional explanation. Do not output as code format (```python```).

*** Example Starts ***
Article: Tardive dystonia is characterized by sustained involuntary muscle contractions. It occurs in approximately 3% of patients with long-term antipsychotic exposure.
Atomic facts: ["Tardive dystonia is characterized by sustained involuntary muscle contractions.", "Tardive dystonia occurs in approximately 3% of patients with long-term antipsychotic exposure."]
Questions: ["How is tardive dystonia characterized?", "In approximately what percentage of patients with long-term antipsychotic exposure does tardive dystonia occur?"]
*** Example Ends ***

Article: {article}
Atomic facts: {atomic_facts}
Questions:
"""


def generate_questions(article, facts, tokenizer, model, max_new_tokens):
    prompt = QG_PROMPT.format(
        article=article.strip(),
        atomic_facts=json.dumps(facts, ensure_ascii=False)
    )

    messages = [
        {"role": "system", "content": "You are a careful scientific evaluator."},
        {"role": "user", "content": prompt},
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_new_tokens=max_new_tokens,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id
        )

    raw = tokenizer.decode(
        outputs[0][inputs["input_ids"].shape[-1]:],
        skip_special_tokens=True
    ).strip()

    # Robust parsing: try Python list first
    try:
        parsed = ast.literal_eval(raw)
        if isinstance(parsed, list):
            return [
                q.strip()
                for q in parsed
                if isinstance(q, str) and "?" in q
            ]
    except Exception:
        pass

    # Fallback: one question per line
    return [line.strip() for line in raw.split("\n") if "?" in line]


def main():
    args = parse_args()

    model_id = "Qwen/Qwen2.5-3B-Instruct"

    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )

    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.input, "r", encoding="utf-8") as fin, \
         open(args.output, "w", encoding="utf-8") as fout:

        for line in tqdm(fin):
            obj = json.loads(line)

            sid = obj.get("stringID")
            source = obj.get("source")
            facts = obj.get("facts_entailed", [])

            if not source or not facts:
                continue

            questions = generate_questions(
                article=source,
                facts=facts,
                tokenizer=tokenizer,
                model=model,
                max_new_tokens=args.max_new_tokens
            )

            fout.write(json.dumps({
                "stringID": sid,
                "questions": questions
            }, ensure_ascii=False) + "\n")

            print(sid, "→", len(questions), "questions")

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
