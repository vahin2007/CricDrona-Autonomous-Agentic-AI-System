# 🏏 CricDrona: Autonomous Text-to-SQL Agentic Workflow

> 🚧 **STATUS: ACTIVE DEVELOPMENT (SPRINT 3)**
> *The ETL pipeline and SQLite database architectures are fully deployed. Currently actively engineering the LangChain ReAct framework and Episodic Memory loops for the Llama-3 agent. Expect frequent commits.*

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg) ![LangChain](https://img.shields.io/badge/LangChain-Integration-black) ![SQLite](https://img.shields.io/badge/SQLite-Optimized-lightgrey)

An enterprise-grade autonomous AI system that translates natural language queries into deterministic SQL to analyze complex sports datasets. 

### ⚙️ System Architecture
1. **High-Performance ETL Pipeline:** Utilizes `pandas` to process 300,000+ rows of raw ball-by-ball data, normalizing it into a highly optimized 7-table SQLite relational database. Achieves millisecond query latency.
2. **Out-of-Distribution (OOD) Resolution:** Implements a Cosine Similarity clustering algorithm (`scikit-learn`) to dynamically map uncapped/unknown entities to their closest statistical matches, enabling zero-shot fallback.
3. **Agentic Orchestration:** Integrates **Llama-3** (via Ollama) with the **LangChain ReAct framework**. Utilizes Few-Shot In-Context Learning to autonomously generate, execute, and self-correct SQL queries.
4. **Episodic Memory:** Features a feedback loop that dynamically logs past tactical mistakes, surfacing raw SQL provenance and mathematical confidence scores for Explainable AI (XAI).

### 🚀 How to Run Locally
1. Clone the repository and install dependencies: `pip install -r requirements.txt`
2. Run the ETL pipeline to generate the SQLite database: `python etl_pipeline.py`
3. Ensure Ollama is running locally with the Llama-3 model.
4. Initialize the Agent: `python agent.py`
