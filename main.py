import os
import json
import fitz  # PyMuPDF
import numpy as np
import joblib
import re
import faiss
import torch
from datetime import datetime
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import warnings
warnings.filterwarnings("ignore")

torch.set_num_threads(4)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------------ Config ------------------
INPUT_DIR = "PDFs/"
OUTPUT_FILE = "output/result.json"
INPUT_JSON_FILE = "challenge1b_input.json"
LABELS_TO_INCLUDE = {"H1", "H2", "Body"}
MIN_BODY_LENGTH = 200
SUMMARY_MODEL_NAME = "google/flan-t5-small"
RETURN_TOP_K = 20
BATCH_SIZE = 8

# ---------------------- PDF Feature Extraction ----------------------
def extract_pdf_features(pdf_path):
    result = []
    doc = fitz.open(pdf_path)
    all_font_sizes = []
    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    all_font_sizes.append(round(span["size"], 2))

    sizes_np = np.array(all_font_sizes)
    mean = np.mean(sizes_np)
    std = np.std(sizes_np) if np.std(sizes_np) != 0 else 1

    for page_num, page in enumerate(doc):
        spans = []
        for block in page.get_text("dict")["blocks"]:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                for span in line["spans"]:
                    normalized_size = round((span["size"] - mean) / std, 3)
                    spans.append({
                        "text": span["text"],
                        "size": normalized_size,
                        "flags": span["flags"],
                        "font": span["font"],
                        "bbox": span["bbox"],
                        "origin": (span["bbox"][0], span["bbox"][1])
                    })

        spans.sort(key=lambda s: (round(s["origin"][1]), s["origin"][0]))

        lines, current_line, current_y = [], [], None
        y_threshold = 2
        for span in spans:
            y = round(span["origin"][1])
            if current_y is None or abs(y - current_y) <= y_threshold:
                current_line.append(span)
                current_y = y
            else:
                lines.append(merge_line(current_line))
                current_line = [span]
                current_y = y
        if current_line:
            lines.append(merge_line(current_line))

        paragraphs = []
        if lines:
            current_para = lines[0].copy()
            for i in range(1, len(lines)):
                same_font = lines[i]["font"] == lines[i - 1]["font"]
                same_size = lines[i]["size"] == lines[i - 1]["size"]
                if same_font and same_size:
                    current_para["text"] += " " + lines[i]["text"]
                    current_para["bbox"][3] = max(current_para["bbox"][3], lines[i]["bbox"][3])
                else:
                    paragraphs.append(current_para)
                    current_para = lines[i].copy()
            paragraphs.append(current_para)

        result.append({
            "page_number": page_num,
            "width": page.rect.width,
            "height": page.rect.height,
            "text_blocks": paragraphs
        })
    return result

def merge_line(spans):
    if not spans:
        return {}
    spans.sort(key=lambda s: s["origin"][0])
    full_text = "".join(span["text"] for span in spans).strip()
    full_text = re.sub(r'\d+$', '', full_text).rstrip()
    full_text = re.sub(r'[.-]{4,}$', '', full_text).rstrip()
    return {
        "text": full_text,
        "size": round(max(s["size"] for s in spans), 3),
        "flags": spans[0]["flags"],
        "font": spans[-1]["font"],
        "bbox": [
            spans[0]["bbox"][0],
            spans[0]["bbox"][1],
            spans[-1]["bbox"][2],
            max(s["bbox"][3] for s in spans)
        ]
    }

# ------------------ RAG Helper Functions ------------------
def batch_summarize(texts):
    summaries = []
    for i in range(0, len(texts), BATCH_SIZE):
        batch = texts[i:i + BATCH_SIZE]
        inputs = gen_tokenizer(batch, return_tensors="pt", padding=True, truncation=True).to(dev)
        with torch.inference_mode():
            outputs = gen_model.generate(**inputs, max_new_tokens=50)
        summaries.extend(gen_tokenizer.batch_decode(outputs, skip_special_tokens=True))
    return summaries

def batch_embed(texts):
    return embedder.encode(texts, convert_to_numpy=True, batch_size=BATCH_SIZE, show_progress_bar=False)

def generate_query_variants_with_flan(query):
    prompt = f"Generate 5 diverse paraphrased versions of the query: '{query}'"
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True).to(dev)
    outputs = gen_model.generate(**inputs, max_new_tokens=128, num_return_sequences=5, do_sample=True)
    return list(set(gen_tokenizer.decode(o, skip_special_tokens=True) for o in outputs)) + [query]

def build_faiss_index(vectors):
    index = faiss.IndexFlatL2(vectors.shape[1])
    index.add(vectors)
    return index

def search_faiss(index, query_vecs, top_k):
    sims, indices = index.search(query_vecs, top_k)
    flat_indices = list(set(indices.flatten()))
    return flat_indices

def rerank_with_flan_t5(query, texts):
    prompt = f"""You are given a query: \"{query}\"\n\nAmong the following texts, which is the most relevant to the query? Rank them from most to least relevant and explain briefly why.\n\nTexts:\n""" + "\n\n".join([f"{i+1}. {t}" for i, t in enumerate(texts)])
    inputs = gen_tokenizer(prompt, return_tensors="pt", truncation=True, max_length=1024).to(dev)
    with torch.no_grad():
        outputs = gen_model.generate(inputs["input_ids"], max_length=256)
    return gen_tokenizer.decode(outputs[0], skip_special_tokens=True)

def parse_rerank_indices(flan_output, original_count):
    # Find all occurrences of "<number>. " at the start or inside lines
    matches = re.findall(r"\b(\d+)\.", flan_output)
    seen = set()
    indices = []
    for m in matches:
        idx = int(m) - 1
        if 0 <= idx < original_count and idx not in seen:
            indices.append(idx)
            seen.add(idx)

    return indices if indices else list(range(original_count))

# ---------------------- Main Pipeline ----------------------
def main():
    with open(INPUT_JSON_FILE, 'r') as f:
        data = json.load(f)

    docs = [os.path.join(INPUT_DIR, d["filename"]) for d in data["documents"]]
    persona = data["persona"]["role"]
    job = data["job_to_be_done"]["task"]

    global gen_tokenizer, gen_model, embedder
    gen_tokenizer = AutoTokenizer.from_pretrained(SUMMARY_MODEL_NAME)
    gen_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARY_MODEL_NAME).to(dev)
    embedder = SentenceTransformer("sentence-transformers/multi-qa-distilbert-cos-v1", device=dev)

    clf = joblib.load("best_model.pkl")
    font_enc = joblib.load("font_encoder.pkl")
    layout_scaler = joblib.load("layout_scaler.pkl")
    label_enc = joblib.load("label_encoder.pkl")
    text_model = joblib.load("sentence_transformer.pkl")

    headers, bodies = [], []
    meta_headers, meta_bodies = [], []

    for doc_path in docs:
        pdf_blocks = extract_pdf_features(doc_path)
        blocks = []
        for page in pdf_blocks:
            for b in page["text_blocks"]:
                b["page"] = page["page_number"]
                blocks.append(b)

        texts = [b["text"] for b in blocks]
        sizes = [b["size"] for b in blocks]
        bboxes = [b["bbox"] for b in blocks]
        fonts = [b["font"] for b in blocks]

        text_embeddings = text_model.encode(texts, batch_size=32)
        font_features = font_enc.transform(np.array(fonts).reshape(-1, 1)).toarray()
        layout_raw = np.hstack([np.array(sizes).reshape(-1, 1), np.array(bboxes), font_features])
        layout_features = layout_scaler.transform(layout_raw)
        X = np.hstack([text_embeddings, layout_features])
        y_pred = clf.predict(X)
        y_labels = label_enc.inverse_transform(y_pred)

        for block, label in zip(blocks, y_labels):
            if label in {"H1", "H2"}:
                headers.append(block["text"])
                meta_headers.append({"document": os.path.basename(doc_path), "page": block["page"], "text": block["text"]})
            elif label.lower() == "body" and len(block["text"]) >= MIN_BODY_LENGTH:
                bodies.append(block["text"])
                meta_bodies.append({"document": os.path.basename(doc_path), "page": block["page"], "text": block["text"]})

    query = persona + " wants to " + job
    queries = generate_query_variants_with_flan(query)
    query_vectors = batch_embed(queries)

    header_summaries = batch_summarize(headers)
    header_summary_vectors = batch_embed(header_summaries)
    header_index = build_faiss_index(header_summary_vectors)
    matched_header_indices = search_faiss(header_index, query_vectors, RETURN_TOP_K)

    top_header_texts = [meta_headers[idx]["text"] for idx in matched_header_indices]
    reranked_result = rerank_with_flan_t5(query, top_header_texts)
    reranked_order = parse_rerank_indices(reranked_result, len(top_header_texts))

    section_analysis = []
    for new_rank, rel_idx in enumerate(reranked_order[:6], 1):
        actual_idx = matched_header_indices[rel_idx]
        info = meta_headers[actual_idx]
        section_analysis.append({
            "document": info["document"],
            "section_title": info["text"],
            "importance_rank": new_rank,
            "page_number": info["page"] + 1
        })

    summaries = batch_summarize(bodies)
    summary_vectors = batch_embed(summaries)
    index = build_faiss_index(summary_vectors)
    matched_indices = search_faiss(index, query_vectors, RETURN_TOP_K)

    top_body_texts = [meta_bodies[idx]["text"] for idx in matched_indices]
    reranked_body = rerank_with_flan_t5(query, top_body_texts)
    reranked_body_order = parse_rerank_indices(reranked_body, len(top_body_texts))

    subsection_analysis = []
    for rel_idx in reranked_body_order[:6]:
        actual_idx = matched_indices[rel_idx]
        info = meta_bodies[actual_idx]
        subsection_analysis.append({
            "document": info["document"],
            "refined_text": info["text"],
            "page_number": info["page"] + 1
        })

    output = {
        "metadata": {
            "input_documents": [os.path.basename(d) for d in docs],
            "persona": persona,
            "job_to_be_done": job,
            "processing_timestamp": datetime.now().isoformat()
        },
        "section_analysis": section_analysis,
        "subsection_analysis": subsection_analysis,
    }

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    with open(OUTPUT_FILE, 'w', encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()