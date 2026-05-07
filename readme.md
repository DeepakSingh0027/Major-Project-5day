# Clinical RAG Decision System

A comprehensive Retrieval-Augmented Generation (RAG) pipeline designed for clinical decision support. This system ingests raw synthetic FHIR patient bundles, processes clinical text through an advanced hybrid vector/lexical retrieval engine, and surfaces dual-mode LLM summaries (Clinician and Patient-facing) in an interactive Streamlit dashboard.

Built over a 5-day sprint with an emphasis on **latency tuning**, **clinical safety (hallucination verifiers)**, and **human-in-the-loop auditability**.

---

## 🎯 Key Features

- **FHIR Parsing Pipeline:** Extracts patient demographics, conditions, lab results, and medications from raw FHIR JSON bundles into a structured, relational CSV format.
- **Hybrid Retrieval Engine:** Combines **FAISS** (dense semantic vector search using `sentence-transformers`) and **BM25** (sparse keyword search) to find the most relevant clinical notes for any query.
- **Dual-Mode Summarization:** Utilizes a locally hosted LLM (via **Ollama**, e.g., `llama3`) to generate specialized summaries:
  - 🩺 **Clinician Mode:** High-density, medically precise summaries with highlighted abnormal labs.
  - 👤 **Patient Mode:** Accessible, plain-language explanations of medical conditions.
- **Safety Verifier:** A local validation layer checks generated summaries for:
  - Numeric hallucination (comparing generated numbers against retrieved chunks).
  - Semantic divergence (ensuring clinical claims are backed by context).
- **System Testing & Analytics Dashboard:**
  - Track **generation latency** against a strict 2.8-second SLA.
  - Interactive **Latency Tuning** panel (hot-swap models, throttle `top_k` chunks dynamically).
  - Human-in-the-loop **Clinician Override** workflow for accountability.

---

## 🛠️ Technology Stack

- **Backend & RAG:** Python, FAISS, rank_bm25, sentence-transformers, Ollama (Llama 3)
- **Data Engineering:** Pandas, JSON Parsing (FHIR-like syntax)
- **Frontend / Dashboard:** Streamlit
- **Infrastructure:** Local CUDA/CPU LLM inference (no external API calls for strict privacy)

---

## 🚀 Setup & Installation

### 1. Prerequisites
- Python 3.10+
- [Ollama](https://ollama.com/) installed and running locally
- GPU (optional, but highly recommended for fast LLM inference)

### 2. Environment Setup
Clone the repository and install the dependencies (assuming a virtual environment):
```bash
pip install pandas faiss-cpu sentence-transformers rank-bm25 requests streamlit
```

### 3. Start the LLM Backend
Ensure Ollama is running and pull the default model:
```bash
ollama serve
ollama pull llama3
```
*(For faster inference, consider a quantized model: `ollama pull llama3:8b-instruct-q4_0`)*

---

## 🏃 Workflow & Execution

The system must be built sequentially to ensure the data pipeline is populated before the frontend starts.

### Step 1: Data Extraction
Parse the raw FHIR JSON bundles into structured CSVs.
```bash
python extract_data.py
```
*Outputs are saved to `datasets/`: `patients.csv`, `conditions.csv`, `medications.csv`, `labs.csv`.*

### Step 2: Build Vector Database
Process the clinical notes and build the FAISS and BM25 index artifacts.
```bash
python build_database.py
```
*Creates `artifacts/faiss_index.bin`, `artifacts/bm25_index.pkl`, etc.*

### Step 3: Run the Dashboard
Launch the interactive Streamlit interface.
```bash
streamlit run app.py
```

---

## 📊 Dashboard Usage

1. **Patient Selection:** Select a patient from the sidebar to load their profile.
2. **Clinician / Patient Tabs:** Enter a clinical query and generate a summary. The LLM will use the active model and current context window settings.
3. **Audit & Analytics Tab (Day 4):**
   - Monitor the **Health Scorecard** (Performance %, Grounding %, Override %).
   - Use the **Latency Tuning Panel** to adjust `top_k` chunks or swap the active Ollama model if your average response time exceeds the 2.8-second target.
   - Review the full audit log of all system queries and safety verifier flags.

---

## 🏗️ Architecture Design (Day 1 - 5)

- **Day 1:** RAG Infrastructure setup (Hybrid Retriever + Local LLM orchestration).
- **Day 2:** Data Pipeline (FHIR JSON Parsing & Relational Integrity).
- **Day 3:** Application Layer (Streamlit Dual-Output UI & Verifier Integration).
- **Day 4:** System Testing (Latency Analytics, Clinician Overrides, CUDA/VRAM stabilization).
- **Day 5:** Delivery & Handover (E2E Verification, Documentation).
