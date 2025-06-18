"""Evaluation utilities leveraging MLflow evaluators."""
from __future__ import annotations

from typing import Any, Dict

import mlflow


def evaluate_generated_output(generated: str, reference: str) -> Dict[str, Any]:
    """Score generated text against a reference using built-in evaluators."""
    results = mlflow.evaluate(
        data={"output": generated, "expected": reference},
        model_type="text_generation",
        evaluators=["mosaic_llm_judge"],
    )
    return results

