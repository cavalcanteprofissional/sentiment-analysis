"""Gera os artefatos de comparacao das runs MLflow (tabela CSV/MD) para o entregavel."""

import pathlib

import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://localhost:5000")

client = MlflowClient()
experiment = client.get_experiment_by_name("sentiment-analysis-comparacao-modelos")
print("experiment id:", experiment.experiment_id if experiment else "NAO ACHOU")

df = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["metrics.accuracy_frases_teste DESC"],
)
cols = [
    "params.model_name",
    "metrics.accuracy_frases_teste",
    "metrics.avg_confidence",
    "metrics.latencia_media_seg",
]
df = df[cols]

out_dir = pathlib.Path("docs/prints")
out_dir.mkdir(parents=True, exist_ok=True)
df.to_csv(out_dir / "01_mlflow_comparacao.csv", index=False)
md_path = out_dir / "01_mlflow_comparacao.md"
md_path.write_text(df.to_csv(index=False))

print(df.to_string(index=False))
print("\nArtefatos salvos em docs/prints/")
