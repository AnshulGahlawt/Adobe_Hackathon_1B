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

### 3. Semantic Ranking with Persona and Task Context

At inference time:
- The **persona** and **job-to-be-done** input were combined into a semantic query.
- We embedded both the query and all heading/body blocks using `all-MiniLM-L6-v2`.
- Using **FAISS** vector indexing, we performed nearest-neighbor searches to identify:
  - Most relevant sections (headings)
  - Most relevant body paragraphs (sub-sections)

Sections were scored based on cosine similarity and ranked accordingly.

---

### 4. Output Generation

The final output (`result.json`) included:
- Metadata: documents, persona, job, timestamp
- Ranked list of extracted sections (document, title, page number, importance)
- Subsection analysis (document, refined paragraph, page number)

---


## Input Format

Place the following in the `input/` folder:

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

## Libraries & Models Used

### Python Libraries

| Library                  | Purpose                                           |
|--------------------------|---------------------------------------------------|
| `PyMuPDF (fitz)`         | PDF parsing and layout reconstruction             |
| `sentence-transformers`  | Semantic embedding of text                        |
| `faiss`                  | Fast vector similarity search for ranking         |
| `scikit-learn`           | Classification, encoders, scalers                 |
| `xgboost`                | Gradient boosting classifier (for evaluation)     |
| `lightgbm`               | Lightweight boosting model (for evaluation)       |
| `numpy`, `pandas`        | Data handling and numerical operations            |
| `joblib`                 | Model saving/loading                              |
| `json`, `os`, `re`, `datetime` | Standard libraries for file I/O, processing |

### Models Used

| Model Name                    | Purpose                                            | Final Use |
|------------------------------|----------------------------------------------------|-----------|
| `multi-qa-distilbert-cos-v1` | Embedding model used during training               | Yes       |
| `all-MiniLM-L6-v2`           | Embedding model used at inference time             | Yes       |
| `MLPClassifier`              | Final heading classifier (via RandomizedSearchCV)  | Yes       |
| `XGBoost`                    | Evaluated during training, not used finally        | No        |
| `LightGBM`                   | Evaluated during training, not used finally        | No        |

---

## Execution Instructions

Follow the steps below to build and run the solution using Docker:

### 1. Build the Docker Image

Ensure you're in the root directory of the project (where the Dockerfile is located), then run:

```bash
docker build --platform linux/amd64 -t adobe1b:submission .
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
