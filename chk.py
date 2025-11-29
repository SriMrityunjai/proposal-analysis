# Proposal_analysis_backend_app.py
"""
FastAPI backend for RFP Proposal Analysis POC
--------------------------------------------
Endpoints:
- POST /extract_rfp: Extracts summary and RFP rules from uploaded RFP file.
- POST /evaluate_proposal: Evaluates proposal documents against RFP rules.
- POST /compare_reports: Compares multiple evaluation reports.

Run backend:
    pip install fastapi uvicorn python-docx PyPDF2 openai pandas
    uvicorn Proposal_analysis_backend_app:app --reload --port 8000

This backend interacts with the frontend Streamlit app (Proposal_analysis_frontend_app.py).
Ensure your OpenRouter API key is set in the environment:
    export OPENROUTER_API_KEY="sk-or-v1-..."
"""

import io
import os
import json
import re
from typing import List
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from openai import OpenAI
import pandas as pd

# ---------- Initialize ----------
app = FastAPI(title="Proposal Analysis Backend API", version="1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Helper Functions ----------
try:
    from docx import Document
except Exception:
    Document = None

try:
    from PyPDF2 import PdfReader
except Exception:
    PdfReader = None

def extract_text_from_docx(file_bytes: bytes) -> str:
    if Document is None:
        raise ImportError('python-docx not installed')
    doc = Document(io.BytesIO(file_bytes))
    return "\n".join(p.text for p in doc.paragraphs if p.text.strip())

def extract_text_from_pdf(file_bytes: bytes) -> str:
    if PdfReader is None:
        raise ImportError('PyPDF2 not installed')
    reader = PdfReader(io.BytesIO(file_bytes))
    text = []
    for page in reader.pages:
        try:
            t = page.extract_text()
            if t:
                text.append(t)
        except Exception:
            continue
    return "\n".join(text)

def extract_text_from_txt(file_bytes: bytes) -> str:
    return file_bytes.decode(errors='ignore')

def extract_text(filename: str, file_bytes: bytes) -> str:
    fname = filename.lower()
    if fname.endswith('.pdf'):
        return extract_text_from_pdf(file_bytes)
    if fname.endswith('.docx'):
        return extract_text_from_docx(file_bytes)
    if fname.endswith('.txt'):
        return extract_text_from_txt(file_bytes)
    return file_bytes.decode(errors='ignore')

# ---------- OpenRouter Client ----------
def make_openrouter_client():
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    api_key = os.getenv("OPENROUTER_API_KEY", "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")
    if not api_key:
        raise RuntimeError("Missing OPENROUTER_API_KEY environment variable")
    return OpenAI(base_url=base_url, api_key=api_key)

def call_openrouter_chat(prompt: str, model: str, system: str = None):
    client = make_openrouter_client()
    messages = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})
    completion = client.chat.completions.create(model=model, messages=messages)
    try:
        return completion.choices[0].message.content
    except Exception:
        return str(completion)

# ---------- Prompt Templates ----------
def build_summary_prompt(rfp_text: str, max_words: int = 300) -> str:
    return f"""
Provide a concise summary (no more than {max_words} words) of the following RFP.
Highlight project purpose, scope, key deliverables, and scoring sections.
Return only plain text.
RFP text:
```{rfp_text}```
"""

def build_rule_extraction_prompt(rfp_text: str) -> str:
    return f"""
Extract proposal requirements, submission criteria, and evaluation guidelines.
Return strict JSON array only. Normalize keys as:
Requirement, Description, Mandatory, Max_Points, Reasoning.
RFP text:
```{rfp_text}```
"""

def build_eval_prompt(rule, proposal_text):
    requirement = rule.get("RFP_Requirement_or_Criteria", rule.get("Requirement", ""))
    description = rule.get("Description", "")
    mandatory = rule.get("Mandatory", "")
    max_points = rule.get("Max_Points", "")
    reasoning = rule.get("Reasoning", "")

    return f"""
You are an expert RFP evaluator. Evaluate whether the proposal meets this rule.
Requirement: {requirement}
Description: {description}
Mandatory: {mandatory}
Max Points: {max_points}
Reasoning: {reasoning}
Proposal:
{proposal_text[:4000]}
Output JSON:
{{
  "Requirement": "{requirement}",
  "Satisfied": "<Yes|No|Partially>",
  "Reasoning": "<your reasoning>",
  "Evidence": "<proposal phrases>"
}}
"""

# ---------- API Endpoints ----------
@app.post("/extract_rfp")
async def extract_rfp(file: UploadFile = File(...)):
    file_bytes = await file.read()
    text = extract_text(file.filename, file_bytes)

    model = "minimax/minimax-m2:free"
    summary_prompt = build_summary_prompt(text)
    rule_prompt = build_rule_extraction_prompt(text)

    summary = call_openrouter_chat(summary_prompt, model)
    rules_raw = call_openrouter_chat(rule_prompt, model)

    try:
        m = re.search(r"```(?:json)?\s*(.*?)\s*```", rules_raw, flags=re.S)
        if m:
            rules_raw = m.group(1)
        rules = json.loads(rules_raw)
        # normalize keys to expected ones
        normalized_rules = []
        for r in rules:
            normalized_rules.append({
                "Requirement": r.get("Requirement") or r.get("RFP_Requirement_or_Criteria", ""),
                "Description": r.get("Description", ""),
                "Mandatory": r.get("Mandatory", ""),
                "Max_Points": r.get("Max_Points", ""),
                "Reasoning": r.get("Reasoning", "")
            })
        rules = normalized_rules
    except Exception:
        rules = []

    return {"summary": summary.strip(), "rules": rules}

@app.post("/evaluate_proposal")
async def evaluate_proposal(file: UploadFile = File(...), rules_json: str = Form(None)):
    file_bytes = await file.read()
    proposal_text = extract_text(file.filename, file_bytes)

    if not rules_json:
        return {"error": "Missing rules JSON"}

    try:
        rules = json.loads(rules_json)
    except Exception:
        return {"error": "Invalid rules JSON"}

    model = "minimax/minimax-m2:free"
    client = make_openrouter_client()
    evaluations = []

    for rule in rules:
        prompt = build_eval_prompt(rule, proposal_text)

        # debug log
        print("---- Prompt for rule ----")
        print(prompt[:500] + "...\n")  # first 500 chars

        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}]
            )
            response_text = completion.choices[0].message.content.strip()

            # debug log
            print("---- LLM response ----")
            print(response_text[:500] + "...\n")

            # Extract JSON from backticks if present
            m = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", response_text, flags=re.S)
            json_text = m.group(1) if m else response_text

            try:
                result_json = json.loads(json_text)
            except Exception:
                result_json = {"Requirement": rule.get("Requirement", ""), "Raw": response_text}

        except Exception as e:
            result_json = {"Requirement": rule.get("Requirement", ""), "Error": str(e)}

        evaluations.append(result_json)

    return {"evaluations": evaluations}

@app.post("/compare_reports")
async def compare_reports(files: List[UploadFile] = File(...)):
    evals = []
    for f in files:
        content = json.loads((await f.read()).decode())
        evals.append({"filename": f.filename, "data": content})

    combined = {}
    for e in evals:
        for ev in e['data'].get('evaluations', []):
            req = ev.get('Requirement', 'Unknown')
            sat = ev.get('Satisfied', 'Unknown')
            if req not in combined:
                combined[req] = {}
            combined[req][e['filename']] = sat

    df = pd.DataFrame.from_dict(combined, orient='index')
    csv_data = df.to_csv(index=True)

    return {"comparison_table": df.to_dict(), "csv": csv_data}

# ---------- End of backend file ----------
