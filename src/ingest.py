# contains how to load and parse all files into a uniform format described in the schema

import fitz
import json
import os
from bs4 import BeautifulSoup
from markdown import markdown
import re

def ingest_file(file_path):
    if file_path.endswith(".pdf"):
        return ingest_pdf(file_path)
    elif file_path.endswith(".md"):
        return ingest_markdown(file_path)
    elif file_path.endswith(".html"):
        return ingest_html(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

def clean_text(text):
    # Remove invisible and problematic unicode characters, but preserve \n, \r, and \t
    return re.sub(r'[\u200b\u200c\u200d\ufeff\xa0\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\x9f]', '', text)

def pdf_to_text(file_path):
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return clean_text(text).strip()

def markdown_to_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    html_ver = markdown(text)
    return ''.join(BeautifulSoup(html_ver).findAll(text=True))

def html_to_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return ''.join(BeautifulSoup(text).findAll(text=True))

def ingest_pdf(file_path):
    dictionaryReturn = {}
    dictionaryReturn["text"] = pdf_to_text(file_path)
    dictionaryReturn["source"] = file_path
    dictionaryReturn["file_type"] = "pdf"
    save_ingested_json(json.dumps(dictionaryReturn), file_path)

def ingest_markdown(file_path):
    dictionaryReturn = {}
    dictionaryReturn["text"] = markdown_to_text(file_path)
    dictionaryReturn["source"] = file_path
    dictionaryReturn["file_type"] = "md"
    save_ingested_json(json.dumps(dictionaryReturn), file_path)

def ingest_html(file_path):
    dictionaryReturn = {}
    dictionaryReturn["text"] = html_to_text(file_path)
    dictionaryReturn["source"] = file_path
    dictionaryReturn["file_type"] = "html"
    save_ingested_json(json.dumps(dictionaryReturn), file_path)

def save_ingested_json(ingested_json, file_path):
    ingested_dir = os.path.join(os.path.dirname(__file__), '..', '@ingested')
    os.makedirs(ingested_dir, exist_ok=True)

    base_name = os.path.basename(file_path)
    json_filename = os.path.splitext(base_name)[0] + ".json"
    json_path = os.path.join(ingested_dir, json_filename)

    with open(json_path, 'w', encoding='utf-8') as f:
        f.write(ingested_json)
    return json_path