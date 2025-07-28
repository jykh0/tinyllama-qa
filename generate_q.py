import os
import pdfplumber
import json
import re
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM

PDF_PATH = "trainingdata/TrainingPDF_SolarSystem.pdf"
OUTPUT_PATH = "trainingdata/QqA_Pairs.jsonl"
CHUNK_SIZE = 500  # characters per chunk
MODEL_NAME = "google/flan-t5-base"

# 1. Read and chunk the PDF
def read_and_chunk_pdf(pdf_path, chunk_size=CHUNK_SIZE):
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks

# 2. Load model and tokenizer
print("Loading model and tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
gen_pipe = pipeline("text2text-generation", model=model, tokenizer=tokenizer)

# 3. Generate Q&A pairs for each chunk
def generate_qa(chunk):
    prompt = (
        "Based only on the following text, write 1 or 2 question and answer pairs like this:\n"
        "Q: What is ...?\nA: ...\n\n"
        "Text:\n" + chunk
    )
    result = gen_pipe(prompt, max_new_tokens=256, do_sample=False)[0]["generated_text"]
    print("MODEL OUTPUT:", result)  # Debug: See what the model returns

    # Parse Q:/A: pairs
    qa_pairs = []
    lines = [line.strip() for line in result.splitlines() if line.strip()]
    q, a = None, None
    for line in lines:
        if line.startswith("Q:"):
            if q and a:
                qa_pairs.append({"instruction": q, "output": a})
            q = line[2:].strip()
            a = None
        elif line.startswith("A:"):
            a = line[2:].strip()
    # Add last pair if present
    if q and a:
        qa_pairs.append({"instruction": q, "output": a})
    # Filter out incomplete or bad pairs
    qa_pairs = [pair for pair in qa_pairs if pair["instruction"] and pair["output"] and len(pair["instruction"]) > 5 and len(pair["output"]) > 3 and "?" in pair["instruction"]]
    return qa_pairs

# 4. Process all chunks and save to JSONL
chunks = read_and_chunk_pdf(PDF_PATH)
all_qa = []
# Clear the output file at the start
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    pass
for i, chunk in enumerate(chunks):
    print(f"Processing chunk {i+1}/{len(chunks)}...")
    qa_pairs = generate_qa(chunk)
    all_qa.extend(qa_pairs)
    # Save incrementally
    with open(OUTPUT_PATH, "a", encoding="utf-8") as f:
        for qa in qa_pairs:
            f.write(json.dumps(qa, ensure_ascii=False) + "\n")

print(f"Done! Saved {len(all_qa)} Q&A pairs to {OUTPUT_PATH}")
