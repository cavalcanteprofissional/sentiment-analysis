"""Exercicio obrigatorio: Endpoint de inferencia via FastAPI.

Rodar localmente (a partir da raiz do repositorio, fora do Colab):
    uvicorn api.main:app --reload

Testar /health e /predict:
    curl -X POST http://localhost:8000/predict -H "Content-Type: application/json" \
         -d '{"text": "I love this course!"}'
    ou abrir http://localhost:8000/docs (Swagger UI)

TODO (aluno): implementar /predict/batch.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from model import load_classifier

app = FastAPI(title="Sentiment Analysis API")
classifier = load_classifier()


class TextInput(BaseModel):
    text: str


class BatchInput(BaseModel):
    texts: list[str]


def _predict_one(text: str) -> dict:
    """Reaproveita a logica de predicao individual usada em /predict e /predict/batch."""
    result = classifier(text)[0]
    return {"text": text, "label": result["label"], "score": result["score"]}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict")
def predict(payload: TextInput):
    try:
        return _predict_one(payload.text)
    except Exception as exc:  # noqa: BLE001 - erro generico de inferencia
        raise HTTPException(status_code=500, detail=f"Erro na inferencia: {exc}") from exc


@app.post("/predict/batch")
def predict_batch(payload: BatchInput):
    """Recebe uma lista de textos e retorna uma predicao para cada um, na mesma ordem.

    - Lista vazia ou itens vazios/invalidos -> HTTP 400.
    - Erro do modelo -> HTTP 500 com mensagem clara.
    """
    if not payload.texts:
        raise HTTPException(status_code=400, detail="A lista 'texts' nao pode ser vazia.")
    for i, text in enumerate(payload.texts):
        if text is None or not str(text).strip():
            raise HTTPException(
                status_code=400,
                detail=f"Item na posicao {i} e invalido (texto vazio ou nulo).",
            )
    try:
        return [_predict_one(text) for text in payload.texts]
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=500, detail=f"Erro na inferencia em batch: {exc}") from exc
