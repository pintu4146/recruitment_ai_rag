"""MLflow configuration for Recruitment AI RAG."""
from mlflow import set_tracking_uri, set_experiment

# Use local directory to store mlruns
set_tracking_uri("file:./mlruns")

# Register experiment name
set_experiment("RecruitmentAI-RAG")

