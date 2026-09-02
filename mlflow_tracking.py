"""
Exercicio obrigatorio: Rastreamento de Experimentos com MLflow.

Reaproveita o mesmo pipeline de sentiment-analysis do app.py/model.py
para comparar diferentes modelos pre-treinados em um pequeno conjunto
de frases de teste, registrando parametros, metricas e artefatos.

TODO (aluno): adicione um terceiro modelo a lista MODELS_TO_COMPARE
(pode ser um dos sugeridos no Challenge 3 do README) e rode novamente.

Rodar: python mlflow_tracking.py
"""

import json
import os
import sys
import time

import mlflow

# Melhoria: garante o encoding UTF-8 no console (o MLflow 3.x imprime emojis
# no log de fim de run; no Windows cp1252 isso lancaria UnicodeEncodeError).
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from model import load_classifier

# Melhoria (TASK.md §1): respeitar o MLFLOW_TRACKING_URI do ambiente
# (ex.: http://localhost:5000 apontando para o servidor MLflow via Docker),
# com fallback para o SQLite local caso a variavel nao esteja definida.
mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db"))
mlflow.set_experiment("sentiment-analysis-comparacao-modelos")

TEST_SENTENCES = [
    {"text": "I love this course! The teacher is amazing.", "expected": "POSITIVE"},
    {"text": "This was a terrible experience, I'm very disappointed.", "expected": "NEGATIVE"},
    {"text": "Eu amei esse curso, muito bom!", "expected": "POSITIVE"},
    {"text": "Que produto horrivel, nao recomendo.", "expected": "NEGATIVE"},
]

MODELS_TO_COMPARE = [
    None,  # None = modelo default do pipeline (distilbert-sst2-english)
    "cardiffnlp/twitter-xlm-roberta-base-sentiment",
    "nlptown/bert-base-multilingual-uncased-sentiment",
]


def normalize_label(label: str) -> str:
    """Modelos diferentes usam esquemas de rotulo diferentes (ex.: POSITIVE/NEGATIVE
    vs Positive/Negative/Neutral vs "1 star".."5 stars"). Aqui normalizamos os casos
    mais comuns para poder calcular uma accuracy comparavel entre modelos.

    Divergencia/melhoria (MLOps_45 TODO.md §3): o modelo nlptown retorna rotulos
    "1 star".."5 stars" em vez de POSITIVE/NEGATIVE; sem este mapeamento a sua
    accuracy no conjunto de teste seria zerada.
    """
    label = label.upper()
    if "POS" in label:
        return "POSITIVE"
    if "NEG" in label:
        return "NEGATIVE"
    for stars, sentiment in (
        ("4", "POSITIVE"),
        ("5", "POSITIVE"),
        ("1", "NEGATIVE"),
        ("2", "NEGATIVE"),
    ):
        if f"{stars} STARS" in label or label.startswith(stars):
            return sentiment
    return label


def run_experiment(model_name):
    run_name = model_name or "default-distilbert-sst2"
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("model_name", run_name)
        classifier = load_classifier(model_name)

        results = []
        start = time.time()
        for item in TEST_SENTENCES:
            pred = classifier(item["text"])[0]
            results.append({**item, "predicted": pred["label"], "score": pred["score"]})
        latency = (time.time() - start) / len(TEST_SENTENCES)

        avg_confidence = sum(r["score"] for r in results) / len(results)
        correct = sum(normalize_label(r["predicted"]) == r["expected"] for r in results)
        accuracy = correct / len(results)

        mlflow.log_metric("avg_confidence", avg_confidence)
        mlflow.log_metric("accuracy_frases_teste", accuracy)
        mlflow.log_metric("latencia_media_seg", latency)

        with open("resultados.json", "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        mlflow.log_artifact("resultados.json")

        print(f"{run_name}: accuracy={accuracy:.2f} avg_confidence={avg_confidence:.3f}")


if __name__ == "__main__":
    for model_name in MODELS_TO_COMPARE:
        run_experiment(model_name)

    print("\nComparando execucoes:")
    df = mlflow.search_runs(order_by=["metrics.accuracy_frases_teste DESC"])
    cols = [
        "params.model_name",
        "metrics.accuracy_frases_teste",
        "metrics.avg_confidence",
        "metrics.latencia_media_seg",
    ]
    print(df[cols].to_string(index=False))
