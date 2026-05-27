# RAG Chatbot - Computer Vision Research Q&A

A Retrieval-Augmented Generation (RAG) chatbot for querying a local collection of computer vision research papers, with a full evaluation suite covering **RAGAS metrics** and **LLM-as-a-Judge** scoring across multiple model backends.

---

## Table of Contents

1. [Architecture](#architecture)
2. [Installation](#installation)
3. [Quick Start](#quick-start)
4. [Running the Chatbot](#running-the-chatbot)
5. [Evaluation Suite](#evaluation-suite)
   - [RAGAS Metrics](#ragas-metrics)
   - [LLM-as-a-Judge](#llm-as-a-judge)
   - [Supported Models](#supported-models)
   - [CLI Reference](#cli-reference)
   - [Evaluation Dataset Format](#evaluation-dataset-format)
6. [Project Structure](#project-structure)
7. [Output Files](#output-files)

---

## Architecture

```
PDF papers
    └── ingest.py  ──►  Chroma vector store (embeddings)
                                │
                         rag.py (RAG chain)
                                │
                    ┌───────────┴───────────┐
               Retriever              LLM (Phi-3 / GPT-4o / Claude …)
                  (MMR, k=5)
                                │
                         app.py (Gradio UI)

Evaluation
    └── evaluation/
            ├── ragas_eval.py    ─── faithfulness, precision, recall, relevancy
            ├── llm_judge.py     ─── faithfulness, relevance, correctness, coherence
            ├── models.py        ─── HuggingFace / OpenAI / Anthropic factory
            ├── dataset.py       ─── EvalDataset I/O + RAGAS format conversion
            ├── config.py        ─── ModelConfig, EvaluationConfig, presets
            └── run_evaluation.py── CLI for single-model and multi-model runs
```

### RAG data flow

```
Question
  → Chroma retriever (MMR, k=5 chunks)
  → format_docs()
  → PromptTemplate (context + question)
  → LLM
  → post-processing (strip loops)
  → Answer + Sources
```

---

## Installation

```bash
git clone https://github.com/caraevangeline/PaperQA-RAG-over-Research-Corpus
cd PaperQA-RAG-over-Research-Corpus
pip install -r requirements.txt
```

For **cloud model providers** install the relevant extras:

```bash
# OpenAI (GPT-4o, gpt-4o-mini)
pip install openai langchain-openai

# Anthropic (Claude Haiku, Claude Sonnet)
pip install anthropic langchain-anthropic
```

---

## Quick Start

### 1. Add papers

Place PDF research papers in the `papers/` directory.

### 2. Build the vector store

```bash
python ingest.py
```

This creates the `vectorstore/` directory.

### 3. Launch the chatbot

```bash
python app.py
```

Open the Gradio URL shown in the terminal (default: `http://localhost:7860`).

---

## Running the Chatbot

The Gradio UI supports:
- Free-form question input
- Example prompts for quick testing
- Source citations with every answer
- Conversation history with a clear button

The underlying RAG chain uses:
- **Embedding model:** `sentence-transformers/all-mpnet-base-v2`
- **Retrieval:** Maximal Marginal Relevance, top 5 chunks
- **Default LLM:** `microsoft/Phi-3-mini-4k-instruct`

---

## Evaluation Suite

### RAGAS Metrics

[RAGAS](https://docs.ragas.io) measures four dimensions of RAG quality:

| Metric | What it measures | Range |
|---|---|-------|
| **Faithfulness** | Are all answer claims supported by the retrieved context? | 0-1   |
| **Context Precision** | Are the retrieved chunks relevant to the question? | 0-1   |
| **Context Recall** | Does the retrieved context cover the ground-truth answer? | 0-1   |
| **Answer Relevancy** | Does the answer actually address the question? | 0-1   |

RAGAS needs an LLM and embeddings internally. By default it uses OpenAI; use
`--judge` to route evaluation through any supported model (see below).

### LLM-as-a-Judge

A separate judge LLM scores each answer on a 1–5 rubric across four criteria:

| Criterion | Description |
|---|---|
| **Faithfulness** | Every claim grounded in the retrieved context |
| **Relevance** | Answer directly addresses the question |
| **Correctness** | Factual agreement with the reference answer |
| **Coherence** | Well-structured and grammatically sound |

The judge returns a JSON `{"score": 1-5, "explanation": "..."}` for each
criterion. Scores of 0 indicate a parse failure.

### Supported Models

| Preset | Provider | Model ID | Notes |
|---|---|---|---|
| `phi3` | HuggingFace | `microsoft/Phi-3-mini-4k-instruct` | Default, runs locally |
| `mistral` | HuggingFace | `mistralai/Mistral-7B-Instruct-v0.3` | Requires GPU recommended |
| `llama3` | HuggingFace | `meta-llama/Meta-Llama-3-8B-Instruct` | Requires access token |
| `gpt4o-mini` | OpenAI | `gpt-4o-mini` | Requires `OPENAI_API_KEY` |
| `gpt4o` | OpenAI | `gpt-4o` | Requires `OPENAI_API_KEY` |
| `claude-haiku` | Anthropic | `claude-haiku-4-5-20251001` | Requires `ANTHROPIC_API_KEY` |
| `claude-sonnet` | Anthropic | `claude-sonnet-4-6` | Requires `ANTHROPIC_API_KEY` |

Set API keys before running:

```bash
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."
```

### CLI Reference

#### Single model evaluation

```bash
# Phi-3 as RAG model, RAGAS default judge (OpenAI)
python -m evaluation.run_evaluation --model phi3

# Phi-3 as RAG model, Claude Sonnet as judge
python -m evaluation.run_evaluation --model phi3 --judge claude-sonnet

# Custom dataset, specific metrics, custom output directory
python -m evaluation.run_evaluation \
    --model gpt4o-mini \
    --judge gpt4o-mini \
    --dataset evaluation/sample_dataset.json \
    --metrics faithfulness context_precision \
    --output-dir results/gpt4o-mini/

# RAGAS only (skip judge)
python -m evaluation.run_evaluation --model phi3 --skip-judge

# LLM judge only (skip RAGAS)
python -m evaluation.run_evaluation --model phi3 --skip-ragas
```

#### Multi-model comparison

```bash
python -m evaluation.run_evaluation \
    --compare phi3 gpt4o-mini claude-haiku \
    --judge claude-sonnet \
    --output-dir results/comparison/
```

Produces a `model_comparison.csv` with mean RAGAS and judge scores per model.

#### All options

```
--model PRESET          Single model preset to evaluate
--compare PRESET ...    Two or more presets for side-by-side comparison
--judge PRESET          Model to use as evaluator/judge
--dataset PATH          Path to custom JSON dataset (default: built-in sample)
--metrics METRIC ...    RAGAS metrics to compute
--output-dir DIR        Output directory (default: evaluation_results/)
--vectorstore-dir DIR   Chroma vector store path (default: vectorstore/)
--skip-ragas            Skip RAGAS computation
--skip-judge            Skip LLM-as-a-Judge scoring
```

### Evaluation Dataset Format

Create a JSON file following this schema to use your own Q&A pairs:

```json
{
  "samples": [
    {
      "question": "What is the main contribution of the YOLO paper?",
      "ground_truth": "YOLO reframes detection as a single regression problem ...",
      "expected_sources": ["yolo.pdf"]
    }
  ]
}
```

`expected_sources` is optional and reserved for future source-accuracy audits.

A ready-to-use example with ten CV questions is provided at
`evaluation/sample_dataset.json`.

---

## Project Structure

```
RAG-Chatbot/
├── papers/                         # Place PDF research papers here
├── vectorstore/                    # Auto-generated by ingest.py
├── evaluation_results/             # Auto-generated by run_evaluation.py
│
├── ingest.py                       # PDF loading, chunking, vector store build
├── rag.py                          # RAG chain (LLM + retriever + prompt)
├── app.py                          # Gradio web UI
├── requirements.txt
│
└── evaluation/
    ├── __init__.py
    ├── config.py                   # ModelConfig, EvaluationConfig, presets
    ├── dataset.py                  # EvalSample, EvalDataset
    ├── models.py                   # LLM + embeddings factory (multi-provider)
    ├── ragas_eval.py               # RAGASEvaluator
    ├── llm_judge.py                # LLMJudge, LLMJudgeEvaluator
    ├── run_evaluation.py           # CLI entry point
    └── sample_dataset.json         # Built-in 10-question CV dataset
```

---

## Output Files

After a run, the configured `--output-dir` (default `evaluation_results/`) contains:

| File | Contents |
|---|---|
| `<model>_ragas_results.csv` | Per-sample RAGAS scores + mean row |
| `<model>_judge_results.csv` | Per-sample judge scores + explanations + mean row |
| `model_comparison.csv` | Mean scores side-by-side for every compared model |

### Example RAGAS output

```
question                           faithfulness  context_precision  context_recall  answer_relevancy
What is the main contribution …    0.92          0.80               0.88            0.91
How does self-attention work …     0.85          0.75               0.82            0.87
…
MEAN                               0.88          0.77               0.85            0.89
```

### Example judge output

```
question                          faithfulness  relevance  correctness  coherence
What is the main contribution …   4             5          4            5
How does self-attention work …    3             4          3            4
…
MEAN                              3.5           4.5        3.5          4.5
```
