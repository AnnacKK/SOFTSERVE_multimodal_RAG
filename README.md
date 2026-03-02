# The Batch RAG: Multimodal RAG from THE BATCH 
    V 0.1: Containers without Kuberflow, manual run (02/03/2026)


![RAG](documentation/RAG.gif)



The Batch RAG is a hardware-optimized, Multimodal Retrieval-Augmented Generation (RAG) system designed to analyze and synthesize news from **The Batch**. 

Unlike standard RAG pipelines, Oracle features a high-fidelity **Autonomous Evaluation Loop** that audits every response across 17 metrics to detect hallucinations and data drift in real-time.
With usage of double ML system: main query on OLLAMA 4 from GROK and local OLLAMA Qwen model for optimization.

---

## 🧐 Project Overview
This project is final project from SoftServe course, and the main project is study how to build, validate and evaluate robust RAG system.
The goal of this project was to build a system that can:
1. **Ingest** complex textual and visual data from newsletters.
2. **Retrieve** relevant content using a high-performance vector database (Qdrant).
3. **Generate** context-aware answers using local, open-source LLMs.
4. **Audit** results automatically using a multi-framework evaluation strategy.

---

## 🛠️ Core Mechanics & Features
### 1. Multimodal Integration
Utilizes a vision-aware pipeline to link text chunks with their original imagery. This ensures that when a user asks about a specific diagram or figure from *The Batch*, the system retrieves both the explanation and the visual asset.
System retrieves:
- **AI Summarization** with OLLAMA 4 (2-3 second) with using of stream - user see result pop up in batch, no need to wait until all result is generated.
- **Related diagrams** - OLLAMA 4 checks do display only valuable images: diagrams, documentation etc, but not display logos and another non-valuable content.
- **Diagrams description** - since not all usefull images from The Batch contain image description as metadata, I use OLLAMA 4 also to generate descriprion for all usefull images.
- **Links to related articles** - retrieves links for atricles that was picked up from retriever and used for generation.
- **Answear Verying** - After user got result, system srart evaluation process, if faithfulness is low, system called **refine_answer()** to generate more robust answear.
### 2. The 11 metrics Audit Framework
Every response is subjected to a "Judge LLM" audit using:
- **Ragas**: Measuring faithfulness and relevancy.
- **FactCC**: Checking logical consistency against source text.
- **NeuLab InterpretEval**: Tracking Named Entity (NER) hallucinations.
- **Microsoft RAI**: Ensuring safety and harm mitigation.

### 3. Hardware Optimization (GTX 1050 Ti)
The system is engineered for low-VRAM environments:
- **Ollama Parallelism**: Locked to `1` to prevent VRAM spikes.
- **Sequential Awaits**: Logic ensures the Embedding, Synthesis, and Audit steps don't compete for the GPU simultaneously.

---
### Full technical documentation can be found in /documentation/Report.md

---

## 🚀 Getting Started

### Prerequisites
- Docker Desktop (WSL2 Backend)
- Ollama (installed on host)

### Installation
1. **Clone the repo**:
   ```bash
   git clone [https://github.com/YourUsername/MultimodalRAG.git](https://github.com/YourUsername/MultimodalRAG.git)
   cd MultimodalRAG
2. **Crate docker containers:**
    ``` bash 
   docker compose up -d
3. **Ingest data**
docker-compose.yml created not to check if Qdrant is ready, meaning when running ``docker compose up`` process dont run data ingestion by default.
To run data ingestion you can:
   - change ``docker_compose.yml``: unmarked this line: 
    ``` 
   data_processing:
          condition: service_completed_successfully`` 
with this every time you run ``docker compose up`` data ingestion will running at first. Dont recomended.
- **RECOMENDED APPROACH** run next line to ingest data from the BATCH:
``` bash
    docker compose --profile ingest up data_processing
```


### Evaluation Dashboard: how to run and model accuracy
Evaluation dashboard runs alltogether with api, so to acess go to ``http://localhost:8501``
- View Scores: Track metrics and tradeoffs over time.

- Drift Analysis: Monitor if new newsletters are causing performance drops.

- Audit Logs: Inspect the full 17-column SQLite report for every query.

```
🏁 Background Audit Finished. Scores: {'faithfulness': 1.0, 'answer_relevancy': 0.8, 'context_utilization': 0.9999999999666667, 'bleu': 0.033779448904876434, 'rouge_l': 0.226787181
59408381, 'factcc_consistency': 0.5, 'ner_coverage': 0.29850746268656714, 'ner_hallucination': 0.3103448275862069, 'ner_density': 0.10943396226415095, 'harm_score': 0.0, 'harm_category': 'none', 'is_corrected': 0}
```

Results overview: models (OLLAMA and Qwen) can summarize (OLLAMA) and judging (Qwen).
But OLLAMA has problem with yapping (NLP and factcc scores): it can return much more text that needed about not highly related topics,
based on this Judge Qwen gives lower relevance score.
NER Hallucination: 0.31 -  31% of the names/entities in answer weren't found in the source text. This might be due to the model using its internal "world knowledge" instead of just the provided context.