# Catalan Dialect Classifier

This project provides a Python script to classify Catalan text into three major dialects: **Central**, **Valencian**, and **Balearic**. The classification is based on morphological and grammatical heuristic rules found in the input text. The script processes PARQUE/JSONL/TSV/CSV files, and generates at least three separate JSONL files, each corresponding to one of the dialects. Sentences that do not fall under any of the three dialects will be saved into a separate JSONL file under the primary language as identified by FastText.

## Installation

Clone this repository:
```bash
git clone https://github.com/your-username/catalan-dialect-classifier.git
cd catalan-dialect-classifier
```

## Usage 

python classify_dialects.py input.jsonl