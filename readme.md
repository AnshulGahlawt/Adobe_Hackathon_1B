# Adobe Hackathon 2025 — Round 1B

## Challenge: Persona-Driven Document Intelligence

---

## Problem Statement

In Round 1B of the Adobe India Hackathon 2025 — themed *"Connect What Matters — For the User Who Matters"*, the challenge was to build an intelligent system that extracts and ranks the most relevant sections from a set of PDFs based on:

- A given **Persona** (e.g., student, analyst, researcher)
- A **Job-to-be-Done** (e.g., summarize financials, prepare literature review)

The goal was to personalize document analysis, producing a structured output with:

- Relevant **sections and sub-sections**
- Metadata, titles, page numbers, and importance ranks
- All within strict **offline**, **CPU-only**, and **size/time-limited** constraints

---

## Our Approach

### 1. PDF Parsing and Feature Extraction

We used **PyMuPDF** to extract structured text from PDFs. This included:
- Font size, font name, bold/italic flags
- Bounding boxes and paragraph alignment
- Page number and reading order

We reconstructed lines and paragraphs from raw spans using geometric heuristics and grouped blocks with similar styling.

---

### 2. Heading Classification

To distinguish between headings (H1–H3) and body text:
- We used a pretrained **sentence-transformer** (`multi-qa-distilbert-cos-v1`) to generate semantic embeddings of text blocks.
- We extracted layout-based features like font size, bounding boxes, and font type.
- We trained multiple classifiers (MLP, XGBoost, LightGBM) using a labeled dataset.
- The best model (an **MLPClassifier** fine-tuned with RandomizedSearchCV) was selected based on validation accuracy and persisted using `joblib`.

This allowed our system to infer heading types (H1, H2, H3, Body) during inference with high precision.

---

### 3. Semantic Retrieval & Reranking

After classification into headings and bodies, we perform a **RAG-style** retrieval pipeline:

1. **Query Generation**  
   - Compose a single prompt:  
     ```
     Generate 5 diverse paraphrased versions of: "<Persona> wants to <Job>"
     ```
   - Model: `google/flan-t5-small` (via HuggingFace Transformers)  
2. **Embedding & Faiss Indexing**  
   - Embed both the paraphrased queries and candidate text blocks (headings or long body paragraphs) with the same Sentence-Transformer  
   - Build FAISS L2 indexes for headings and for bodies  
3. **Nearest-Neighbor Search & Rerank**  
   - Retrieve top K candidates for headings and bodies separately  
   - Rerank with FLAN-T5 by prompting:  
     ```
     Among the following texts, rank them from most to least relevant to "<Persona> wants to <Job>" and explain.
     ```
   - Parse the ranked list to produce final `importance_rank`  

### 4. Summarization of Body Text (New)

- If any sub-sections remain after reranking, their raw text is batched (size 8) and summarized with the same `google/flan-t5-small` model (max 50 new tokens each).  
- These concise summaries are included under `"generated_summary"` in the final JSON.

---

### 4. Output Generation

The final output (`result.json`) included:
- Metadata: documents, persona, job, timestamp
- Ranked list of extracted sections (document, title, page number, importance)
- Subsection analysis (document, refined paragraph, page number)

---


## Input Format

Place the following in the `Pdf/` folder:

- 3 to 10 PDFs
- A JSON file named `challenge1b_input.json`:

```json
{
  "documents": [
    { "filename": "doc1.pdf" },
    { "filename": "doc2.pdf" }
  ],
  "persona": {
    "role": "PhD Researcher in Computational Biology"
  },
  "job_to_be_done": {
    "task": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks"
  }
}
```
## Output Format

The result will be saved as `output/result.json` and should follow this structure:

```json
{
  "metadata": {
    "input_documents": ["doc1.pdf", "doc2.pdf"],
    "persona": "PhD Researcher in Computational Biology",
    "job_to_be_done": "Prepare a comprehensive literature review focusing on methodologies, datasets, and performance benchmarks",
    "processing_timestamp": "2025-07-28T13:00:00"
  },
  "extracted_sections": [
    {
      "document": "doc1.pdf",
      "section_title": "Graph Neural Network Architectures",
      "importance_rank": 1,
      "page_number": 4
    },
    {
      "document": "doc2.pdf",
      "section_title": "Molecular Interaction Analysis",
      "importance_rank": 2,
      "page_number": 2
    }
  ],
  "subsection_analysis": [
    {
      "document": "doc2.pdf",
      "refined_text": "Recent studies show that drug-target interaction networks can be modeled with GNNs.",
      "page_number": 2
    },
    {
      "document": "doc1.pdf",
      "refined_text": "The benchmark dataset for evaluating molecular GNNs includes Tox21 and SIDER.",
      "page_number": 4
    }
  ]
}
```

## Libraries & Models (updated)

| Library / Package                 | Purpose                                                        |
|-----------------------------------|----------------------------------------------------------------|
| `PyMuPDF (fitz)`                  | PDF parsing & layout feature extraction                        |
| `sentence-transformers`           | Embedding model for text blocks (`multi-qa-distilbert-cos-v1`) |
| `transformers`                    | FLAN-T5 model for paraphrasing, reranking, and summarization   |
| `faiss`                           | Fast vector similarity search                                  |
| `torch`                           | Inference backend for FLAN-T5                                  |
| `tqdm`                            | Progress bars                                                  |
| `joblib`                          | Model & encoder persistence                                    |
| `numpy`, `json`, `os`, `re`       | Core utilities                                                 |
| `datetime`                        | Timestamp generation                                           |
| `warnings`                        | Suppressing benign warnings                                    |

**Models loaded at runtime**:
- **Sentence-Transformer**: `multi-qa-distilbert-cos-v1`
- **Summarization/Reranking**: `google/flan-t5-small`
- **Heading Classifier**: `best_model.pkl`
- **Pre-fit Encoders/Scalers**: `font_encoder.pkl`, `layout_scaler.pkl`, `label_encoder.pkl`, `sentence_transformer.pkl`

---

## Execution Instructions

###  Build the Docker Image

Make sure you're in the root directory of the project, then run:

```bash
docker build --platform linux/amd64 -t mysolutionname:somerandomidentifier .

```

### Run the Docker Container
```bash
docker run --rm -v $(pwd)/PDFs:/app/PDFs -v $(pwd)/output:/app/output --network none mysolutionname:somerandomidentifier

```

## Potential Improvements

If constraints such as model size, runtime, or offline execution were relaxed, the following enhancements could significantly improve performance and flexibility:

| Area                      | Suggested Enhancement                                                                 |
|---------------------------|----------------------------------------------------------------------------------------|
| Section Summarization     | Use models like `T5-small` or `BART` for generating concise summaries of sections     |
| Layout Awareness          | Integrate layout-aware models like `LayoutLMv3` for better document structure parsing |
| Ranking Optimization      | Use learning-to-rank models such as `LambdaMART` to improve relevance ranking         |
| TOC Utilization           | Leverage document Table of Contents (if present) to aid section segmentation          |
| Persona Context Expansion | Expand persona understanding using structured taxonomies or pretrained intent models  |
| Multilingual Support      | Incorporate multilingual models to handle non-English documents                       |
| Visualization Layer       | Integrate with a frontend to present ranked sections in a user-centric dashboard       |
