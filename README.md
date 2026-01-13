# SBERT Question Similarity (all-MiniLM-L6-v2)

[![Python Version](https://img.shields.io/badge/python-3.10-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Issues](https://img.shields.io/github/issues/FatimaM05/sbert-question-similarity)](https://github.com/FatimaM05/sbert-question-similarity/issues)

---

## 📌 Overview

This project implements a **semantic duplicate question detection system** that identifies whether two questions convey the same meaning, even if phrased differently.  
It leverages **Sentence-BERT (SBERT)** embeddings and **cosine similarity** to capture deep semantic relationships between questions.

The system is evaluated using the **Quora Question Pairs dataset**, a widely used benchmark for duplicate question detection.

---

## 📖 Table of Contents

- [Problem Statement](#-problem-statement)  
- [Methodology](#-methodology)  
- [Dataset](#-dataset)  
- [Evaluation Metrics](#-evaluation-metrics)  
- [Tech Stack](#-tech-stack)  
- [Installation](#-installation)  
- [Usage](#-usage)  
- [Limitations](#-limitations)  
- [Future Improvements](#-future-improvements)  
- [Use Cases](#-use-cases)  

---

## 🎯 Problem Statement

Online platforms often face the issue of **duplicate questions**, where users ask the same question using different wording.  
Traditional keyword-based approaches fail to capture semantic similarity.

This project aims to:

* Detect semantically similar (duplicate) questions  
* Go beyond surface-level word matching  
* Provide accurate similarity-based retrieval  

---

## 🧠 Methodology

The system follows a **modular pipeline**:

### Data Cleaning & Preprocessing
* Remove missing values  
* Normalize text  
* Extract unique questions  

### Sentence Embedding Generation
* Model used: `all-MiniLM-L6-v2` (SBERT)  
* Each question converted into a dense vector  
* Batch processing for efficiency  

### Vector Storage
* Embeddings stored in `.npy` format  
* Metadata and mappings stored for traceability  

### Similarity Computation
* Cosine similarity between query and stored embeddings  
* Top-k most similar questions retrieved  

### Duplicate Classification
* Similarity threshold used to classify duplicates vs non-duplicates  

---

## 📂 Dataset

**Quora Question Pairs Dataset**  

* Contains labeled question pairs (`duplicate / non-duplicate`)  
* Diverse real-world question phrasing  

**Example:**
Q1: How can I increase internet speed while using VPN?
Q2: How do I make internet faster when VPN is connected?
Label: Duplicate


---

## 📊 Evaluation Metrics

### Classification Metrics
* Precision  
* Recall  
* F1-score  
* Accuracy  
* ROC-AUC  

### Retrieval Metrics
* Recall@K  
* Mean Reciprocal Rank (MRR)  

### Key Results
* Best performance at similarity threshold ≈ **0.5**  
* High AUC (~0.89)  
* Recall@5 ≈ **0.91**  

---

## 🛠️ Tech Stack

* Python  
* Sentence-Transformers (SBERT)  
* NumPy  
* Scikit-learn  
* Pandas  

## 💻 Installation

1. **Clone the repository:**

  git clone https://github.com/FatimaM05/sbert-question-similarity.git
  cd sbert-question-similarity

2. **Install required packages:**

```bash
pip install -r requirements.txt
```

> **Note:** Make sure `requirements.txt` includes `sentence-transformers`, `numpy`, `pandas`, `scikit-learn`.

---

## 🚀 Usage

1. **Generate embeddings for the dataset:**

```bash
python generate_embeddings.py --dataset questions.csv
```

2. **Check for duplicates given a query:**

```bash
python query_duplicate.py --question "How do I speed up my VPN connection?"
```

3. **Adjust similarity threshold** in `config.py` to tune detection sensitivity.

---

## ⚠️ Limitations

* Dataset-specific (Quora only)
* Brute-force cosine similarity (not scalable for very large datasets)
* Contextual metadata not used (e.g., tags, user info)

---

## 🔮 Future Improvements

* Integrate **FAISS / ANN** for fast similarity search
* Use larger SBERT models (e.g., MPNet)
* Extend to multilingual datasets
* Add contextual and metadata-based features

---

## 📚 Use Cases

* Q&A platforms (duplicate prevention)
* Search engines
* Knowledge base optimization
* Content moderation systems


