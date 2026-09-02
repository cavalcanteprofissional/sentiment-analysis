"""Carregamento do modelo de sentiment analysis.

Reaproveitado por app.py, mlflow_tracking.py e api/main.py.
"""

from transformers import pipeline

DEFAULT_MODEL = None  # None = usa o modelo default do pipeline (distilbert-sst2-english)


def load_classifier(model_name: str | None = DEFAULT_MODEL):
    if model_name:
        return pipeline("sentiment-analysis", model=model_name)
    return pipeline("sentiment-analysis")
