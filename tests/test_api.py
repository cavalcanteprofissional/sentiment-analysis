"""Testes dos endpoints da API do Exercício B (FastAPI).

Valida GET /health, POST /predict e POST /predict/batch (incluindo casos de
erro: lista vazia e item vazio). Usam fastapi.testclient.
"""

import os

# Garante tracking local (SQLite) durante os testes, sem depender do servidor.
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///mlruns.db")

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


def test_predict():
    resp = client.post("/predict", json={"text": "I love this course!"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "I love this course!"
    assert body["label"] in {"POSITIVE", "NEGATIVE"}
    assert 0.0 <= body["score"] <= 1.0


def test_predict_batch_mesma_ordem():
    textos = ["I love this course!", "This is terrible"]
    resp = client.post("/predict/batch", json={"texts": textos})
    assert resp.status_code == 200
    body = resp.json()
    assert isinstance(body, list)
    assert [item["text"] for item in body] == textos


def test_predict_batch_lista_vazia():
    resp = client.post("/predict/batch", json={"texts": []})
    assert resp.status_code == 400


def test_predict_batch_item_vazio():
    resp = client.post("/predict/batch", json={"texts": ["ok", "   "]})
    assert resp.status_code == 400
    assert "invalido" in resp.json()["detail"].lower()
