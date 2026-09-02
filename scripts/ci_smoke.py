"""Smoke test de CI: valida que o modulo mlflow_tracking importa e expoe a lista de modelos."""
import os

os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///ci_check.db")

import mlflow_tracking  # noqa: E402

if __name__ == "__main__":
    count = mlflow_tracking.MODELS_TO_COMPARE
    print(f"smoke test ok: {len(count)} modelos em MODELS_TO_COMPARE")
