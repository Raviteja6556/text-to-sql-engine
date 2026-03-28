# 🧠 Text-to-SQL Conversational Engine
### Natural Language → Schema-Aware SQL via RAG + Gemini 2.5 Flash + FastAPI + DuckDB

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat-square&logo=python)
![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green?style=flat-square&logo=fastapi)
![DuckDB](https://img.shields.io/badge/DuckDB-0.10+-yellow?style=flat-square)
![ChromaDB](https://img.shields.io/badge/ChromaDB-0.4+-red?style=flat-square)
![Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-orange?style=flat-square&logo=google)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32+-lightgrey?style=flat-square&logo=streamlit)

---

## 🎯 What This System Does

Most data exploration requires writing SQL — a skill barrier that locks non-technical stakeholders out of their own data. This project eliminates that barrier entirely.

**Ask a question in plain English. Get the answer from your database. No SQL required.**

The system translates any natural language query into a syntactically correct, schema-grounded DuckDB SQL query using a **Retrieval-Augmented Generation (RAG)** pipeline — ensuring the LLM never hallucinates table names or column structures.

---

## 🏗️ System Architecture

```
User Query (Natural Language)
        │
        ▼
┌─────────────────────┐
│  Sentence Transformer│  ← all-MiniLM-L6-v2
│  Query Embedding     │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│     ChromaDB         │  ← schema_embeddings collection
│  Schema Retrieval    │  ← top-k similarity search
│  (RAG Context)       │
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   Gemini 2.5 Flash   │  ← schema-injected prompt
│   SQL Generation     │  ← prompt-engineered for DuckDB
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│   FastAPI Backend    │  ← /chat /generate-sql /execute-sql
│   SQL Execution      │  ← in-memory DuckDB
│   JSON Serialization │  ← custom LIST/ARRAY handler
└────────┬────────────┘
         │
         ▼
┌─────────────────────┐
│  Streamlit Frontend  │  ← conversational UI
│  Result Display      │
└─────────────────────┘
```

---

## ⚙️ Tech Stack

| Layer | Technology |
|---|---|
| **Data Storage** | DuckDB (in-memory, columnar) |
| **Embeddings** | Sentence Transformers (`all-MiniLM-L6-v2`) |
| **Vector Store** | ChromaDB (`schema_embeddings` collection) |
| **LLM** | Google Gemini 2.5 Flash via LangChain |
| **Backend API** | FastAPI + Uvicorn |
| **Frontend** | Streamlit |
| **Data Layer** | Pandas, KaggleHub |
| **Dataset** | E-commerce Order Dataset (Kaggle) |

---

## 📂 Dataset Schema

5 relational tables loaded into DuckDB:

```
customers     → customer_id, name, location, ...
orders        → order_id, customer_id, order_date, status, ...
order_items   → item_id, order_id, product_id, quantity, price, ...
payments      → payment_id, order_id, amount, method, ...
products      → product_id, name, category, price, ...
```

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/schema` | GET | Returns all extracted schema descriptions |
| `/generate-sql` | POST | Translates NL query → DuckDB SQL via RAG + LLM |
| `/execute-sql` | POST | Executes raw SQL against in-memory DuckDB |
| `/chat` | POST | End-to-end: NL → SQL → Execute → Return results |

---

## 🧩 Key Engineering Decisions

### 1. RAG for Schema Grounding
Rather than dumping the entire schema into every LLM prompt (expensive, noisy), the system embeds each table's schema description independently and retrieves only the **most relevant schema context** per query using cosine similarity. This keeps prompts lean and SQL generation accurate.

### 2. Prompt Engineering for Hallucination Prevention
The LLM prompt includes explicit directives:
- Use **only** table and column names from the provided schema
- Use `order_items` as the bridge table when joining `payments` and `products`
- Return **only** the SQL query — no markdown, no explanation

### 3. Markdown Stripping Logic
Gemini wraps SQL in ` ```sql ``` ` blocks by default. The `/generate-sql` endpoint implements post-processing to strip all markdown delimiters before passing SQL to DuckDB — eliminating `400 Bad Request` and `Parser Error` failures.

### 4. Custom JSON Serialization
DuckDB `LIST` aggregate results return Python lists/NumPy arrays that FastAPI's default JSON encoder cannot handle. A custom serialization layer converts all complex types to string representations before response formatting — ensuring zero `ValueError` crashes.

---

## 🚧 Challenges & Solutions

| Challenge | Root Cause | Solution |
|---|---|---|
| `429 RESOURCE_EXHAUSTED` errors | `gemini-pro-latest` quota limits | Switched to `gemini-2.5-flash` |
| `400 Bad Request` on SQL execution | LLM wrapping SQL in markdown blocks | Implemented markdown stripping in `/generate-sql` |
| Hallucinated table names | LLM inferring non-existent tables | Enhanced prompt with strict schema-adherence directives |
| `ValueError` on JSON response | DuckDB LIST types not JSON-serializable | Custom serialization converting arrays to strings |

---

## 🚀 Getting Started

### Prerequisites
```bash
Python 3.10+
Google Gemini API Key
Kaggle API credentials
```

### Installation
```bash
git clone https://github.com/Raviteja6556/text-to-sql-engine.git
cd text-to-sql-engine
pip install -r requirements.txt
```

### Environment Setup
```bash
# Set your Gemini API key
export GOOGLE_API_KEY="your_gemini_api_key_here"
```

### Run
```bash
# Start FastAPI backend
uvicorn app:app --reload --port 8000

# Launch Streamlit frontend (separate terminal)
streamlit run streamlit_app.py
```

---

## 💬 Example Queries

```
"What are the top 5 customers by total order value?"
"How many orders were placed in each month of 2023?"
"Which product categories generate the highest revenue?"
"List all orders with payment method 'credit card' above $500."
"Show me customers who placed more than 3 orders."
```
