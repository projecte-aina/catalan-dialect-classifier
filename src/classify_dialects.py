import json
import csv
import sys
import os
import re
import nltk
import fasttext
from pathlib import Path
import spacy
from nltk.tokenize import sent_tokenize
import pandas as pd  # Required for handling Parquet files
from tqdm import tqdm

# python -m spacy download ca_core_news_sm

#nltk.download("punkt")
#nltk.download('punkt_tab')

# Load models
nlp = spacy.load("ca_core_news_sm")
lang_model = fasttext.load_model(os.path.join(os.getcwd(), "lid.176.bin"))  # Download from https://fasttext.cc/docs/en/language-identification.html

# Rules implemented in this classifier:
# Demostrativos: aquest/aquel (CAT); eixe, este, ese (VAL)
# 3a conjugacion acabados en -ir: incoatius -isc "preferisc" (VAL);  -eixo "prefereixo" (CAT)
# Pronombres cliticos: pro-cliticos (antes del verbo) son la forma plana: me, mos, te, vos, se, ne. (VAL); em, ens, et, us, en (CAT)
# Articulo salado (determinante): Solo en transcripciones orales, no en escrito: es, sa (BALEAR); el, la (CAT)


# Helper functions
def detect_language(text):
    """
    Detects the language of the given text using FastText.
    Returns a tuple of (language_code, confidence).
    """
    text = text.replace("\n", " ") 
    predictions = lang_model.predict(text, k=1)
    lang, confidence = predictions[0][0].replace("__label__", ""), predictions[1][0]
    return lang, confidence

def is_catalan(text):
    """
    Checks if the given text is in Catalan using FastText language identification.
    """
    lang, confidence = detect_language(text)
    return lang == "ca" and confidence > 0.7  # Adjust threshold

def is_central(sentence):
    """
    Determines if the sentence belongs to the Central Catalan dialect.
    """
    if re.search(r"\b(aquest|aquell)\b", sentence, re.IGNORECASE):
        return True
    if re.search(r"\b(em|et|ens|us)\b", sentence, re.IGNORECASE):
        return True
    if re.search(r"\b(el|la)\b", sentence, re.IGNORECASE):
        return True
    return False

def is_valencian(sentence):
    """
    Determines if the sentence belongs to the Valencian Catalan dialect.
    Matches specific grammatical and lexical cues:
    - Demonstratives: 'eixe', 'este', 'ese'
    - Inchoative verbs ending in -isc (only if the word is a verb)
    - Proclitic pronouns: 'me', 'nos', 'te', 'vos', 'se', 'ne'
    """
    doc = nlp(sentence)
    valencian_demonstratives = {"eixe", "este", "ese"}
    valencian_pronouns = {"me", "te", "nos", "vos"}

    for token in doc:
        if token.text.lower() in valencian_demonstratives:
            return True
        if token.pos_ == "VERB" and token.text.lower().endswith("isc"):
            return True
        if token.text.lower() in valencian_pronouns:
            return True

    return False

def is_balearic(sentence):
    """
    Determines if the sentence belongs to the Balearic Catalan dialect.
    Matches 'es' or 'sa' only when followed by a noun.
    """
    doc = nlp(sentence)
    for i, token in enumerate(doc[:-1]):
        if token.text.lower() in ["es", "sa"] and doc[i + 1].pos_ == "NOUN":
            return True

    return False

def classify_dialect(sentence):
    """
    Classifies the sentence into one of the three Catalan dialects.
    """
    if is_central(sentence):
        return "central"
    elif is_valencian(sentence):
        print("valencian")
        return "valencian"
    elif is_balearic(sentence):
        print("balearic")
        return "balearic"
    else:
        return "unknown"

def process_text_file(file_path, file_type, output_folder, output_files, unknown_files, stats):
    """
    Processes the input text file (JSONL, CSV, or Parquet), annotates each sentence, and
    classifies sentences by dialect.
    """
    if file_type == "jsonl":
        with open(file_path, "r", encoding="utf-8") as infile:
            #for doc_id, line in tqdm(enumerate(infile), desc="Processing JSONL", unit="line"):
            for doc_id, line in enumerate(infile):
                try:
                    data = json.loads(line)
                    text = data.get("text", data.get("content", ""))
                    classify_and_save_sentences(text, doc_id, output_folder, output_files, unknown_files, stats, file_path)
                except json.JSONDecodeError:
                    print(f"Error decoding JSON line: {line.strip()}")
    elif file_type == "csv":
        with open(file_path, "r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile)
            #for doc_id, row in tqdm(enumerate(reader), desc="Processing CSV", unit="row"):
            for doc_id, row in enumerate(reader):
                text = row.get("text", row.get("content", ""))
                classify_and_save_sentences(text, doc_id, output_folder, output_files, unknown_files, stats, file_path)
    elif file_type == "tsv":
        with open(file_path, "r", encoding="utf-8") as infile:
            reader = csv.DictReader(infile, delimiter= "\t")
            #for doc_id, row in tqdm(enumerate(reader), desc="Processing TSV", unit="row"):
            for doc_id, row in enumerate(reader):
                text = row.get("text", row.get("content", row.get("sentence", "")))
                classify_and_save_sentences(text, doc_id, output_folder, output_files, unknown_files, stats, file_path)
    elif file_type == "parquet":
        df = pd.read_parquet(file_path)
        #for doc_id, row in tqdm(df.iterrows(), desc="Processing Parquet", unit="row", total=len(df)):
        for doc_id, row in df.iterrows():
            #text = row.get("text", row.get("content", ""))
            text = row.text
            classify_and_save_sentences(text, doc_id, output_folder, output_files, unknown_files, stats, file_path)

def classify_and_save_sentences(text, doc_id, output_folder, output_files, unknown_files, stats, file_name):
    """
    Tokenizes text into sentences, classifies each sentence, and saves it to the appropriate output file.
    """
    sentences = sent_tokenize(text)
    for sent_id, sentence in enumerate(sentences):
        if is_catalan(sentence):
            dialect = classify_dialect(sentence)
            identifier = f"{file_name}::doc{doc_id}::sent{sent_id}"
            stats[dialect] += 1
            if dialect in output_files:
                output_files[dialect].write(json.dumps({"id": identifier, "text": sentence}, ensure_ascii=False) + "\n")
                output_files[dialect].flush()
            elif dialect == "unknown":
                lang, _ = detect_language(sentence)
                if lang not in unknown_files:
                    input_file_name = Path(file_name).stem
                    unknown_files[lang] = open(os.path.join(output_folder, f"{input_file_name}_{lang}.jsonl"), "w", encoding="utf-8")
                unknown_files[lang].write(json.dumps({"id": identifier, "text": sentence}, ensure_ascii=False) + "\n")
                unknown_files[lang].flush()

def process_file(input_file, output_folder):
    """
    Determines the file type and processes the input file.
    """
    input_file_name = Path(input_file).stem

    output_files = {
        "central": open(os.path.join(output_folder, f"{input_file_name}_central.jsonl"), "w", encoding="utf-8"),
        "valencian": open(os.path.join(output_folder, f"{input_file_name}_valencian.jsonl"), "w", encoding="utf-8"),
        "balearic": open(os.path.join(output_folder, f"{input_file_name}_balearic.jsonl"), "w", encoding="utf-8")
    }

    unknown_files = {}

    stats = {"central": 0, "valencian": 0, "balearic": 0, "unknown": 0}

    try:
        _, ext = os.path.splitext(input_file.lower())
        if ext == ".jsonl":
            process_text_file(input_file, "jsonl", output_folder, output_files, unknown_files, stats)
        elif ext == ".csv":
            process_text_file(input_file, "csv", output_folder, output_files, unknown_files, stats)
        elif ext == ".parquet":
            process_text_file(input_file, "parquet", output_folder, output_files, unknown_files, stats)
        elif ext == ".tsv":
            process_text_file(input_file, "tsv", output_folder, output_files, unknown_files, stats)
        else:
            print(f"Unsupported file type: {ext}")
            sys.exit(1)
    finally:
        print("[INFO] Closing files")
        for f in output_files.values():
            f.close()
        for f in unknown_files.values():
            f.close()
        print("[INFO] Finished closing files")
        
        # Save statistics
        stats_file_path = os.path.join(output_folder, f"{input_file_name}_stats.json")
        with open(stats_file_path, "w", encoding="utf-8") as stats_file:
            json.dump(stats, stats_file, indent=4, ensure_ascii=False)
        print("[INFO] Finished saving stats")
        

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python3 classify_dialects.py <input_file> <output_folder>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_folder = sys.argv[2]
    print(f"Processing file {input_file}")
    process_file(input_file, output_folder)
