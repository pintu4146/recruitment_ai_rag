"""Simple benchmarking script for embedding encoders."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

import mlflow

from app.embedding.factory import get_embedding_encoder
from app.llm.factory import get_retriever_llm
from app.core.config import settings
from app.services.matching import cosine_score


def load_text(file_path: Path) -> str:
    return file_path.read_text()


def benchmark(jd_file: Path, resumes_dir: Path, models: list[str]):
    jd_text = load_text(jd_file)
    resume_files = list(resumes_dir.glob("*.txt"))
    resumes = [load_text(f) for f in resume_files]

    for model_name in models:
        encoder = get_embedding_encoder(model_name)
        start = time.time()
        jd_emb = encoder.encode(jd_text)
        resume_embs = [encoder.encode(r) for r in resumes]
        latency = time.time() - start

        sims = [cosine_score(e, jd_emb) for e in resume_embs]
        avg_sim = sum(sims) / len(sims) if sims else 0.0

        mlflow.log_metric(f"{model_name}_latency", latency)
        mlflow.log_metric(f"{model_name}_avg_similarity", avg_sim)

        retriever = get_retriever_llm(settings.RETRIEVAL_MODEL)
        output = retriever.generate(jd_text, resumes[0] if resumes else "")
        mlflow.evaluate({"output": output, "expected": jd_text}, model_type="text_generation")
        print(f"Model {model_name}: latency={latency:.2f}s avg_sim={avg_sim:.2f}")


def main():
    parser = argparse.ArgumentParser(description="Benchmark embedding models")
    parser.add_argument("--jd", type=Path, required=True, help="Path to JD text file")
    parser.add_argument("--resumes", type=Path, required=True, help="Path to directory with resume txt files")
    parser.add_argument("--models", type=str, default="MiniLM,E5", help="Comma separated list of encoders")
    args = parser.parse_args()

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    benchmark(args.jd, args.resumes, models)


if __name__ == "__main__":
    main()

