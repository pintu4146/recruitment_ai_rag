# Recruitment AI RAG (Production-Ready Setup)

This project uses FastAPI and AI to match resumes with job descriptions using LLMs and vector similarity.

---

## Features

- FastAPI-based backend
- LLM-powered JD generation (Gemini + LangChain prompts)
- Flexible JD generation accepting comma-separated tech stacks
- Single resume upload & parsing
- Sentence-transformer embeddings
- ChromaDB (local vector store)
- Cosine similarity scoring out of 100
- Basic regex-based skill extraction
- Modular, scalable architecture (LLD + HLD ready)

---

## Folder Structure

```
app/
├── routes/       # API endpoints
├── utils/        # Helpers: file parsers, validators
├── core/         # Config management
├── llm/          # LLM API wrappers (Gemini/OpenAI)
├── prompts/      # Prompt templates for LangChain
├── services/     # Business logic (resume scoring etc.)
├── models/       # Pydantic schemas
├── db/           # Vector store clients (Chroma/FAISS)
scripts/          # Setup, maintenance
tests/            # Unit + integration tests
.github/          # CI/CD via GitHub Actions
examples/         # Sample JD/resume files
```

---

## Setup Instructions

### 1. Install Poetry (if not already)

```bash
pip install poetry
```

### 2. Install Dependencies

```bash
poetry install
```

### 3. Activate Environment

```bash
poetry shell
```

### 4. Create `.env` file

```bash
cp .env.example .env
```

### 5. Run Locally (dev mode)

```bash
uvicorn main:app --reload
```

---

## .env File

```env
GEMINI_API_KEY=your_gemini_api_key
CHROMA_DIR=.chromadb
```

---

## Run Tests

```bash
pytest
```

---

## Roadmap (Phases)

- Project Setup
- Single Resume Parsing + JD Matching
- Score + Remarks + Skill Gap
- Email Generation
- Frontend with Jinja2
- Deployment (Docker/GitHub Actions)

### Web Interface

The application exposes simple HTML forms rendered via Jinja2 templates. Visit `/web/resume` to analyze resumes and `/web/jd` to generate job descriptions interactively.

---
