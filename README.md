# Sentiment Analyzer — MLOps com MLflow

> **Discente:** Lucas Cavalcante dos Santos | cavalcantesidi@outlook.com
> **Docente:** Diego Luis Pires | dl.pires@sidi.org.br
> **Disciplina:** MLOps (aula 3) — rastreamento de experimentos, deploy e monitoramento
> **Instituição:** SiDi SOFTEX
> **Repositório (fork):** `cavalcanteprofissional/sentiment-analysis` (base: `profdiegoluispires/sentiment-analysis`)

Analisador de sentimentos de texto com Hugging Face Transformers. O projeto demonstra o ciclo MLOps completo: **rastreamento de experimentos com MLflow**, **deploy de API com FastAPI**, uma **demo em Gradio** e a documentação conceitual de **monitoramento em produção**.

---

## Índice

1. [Estrutura do repositório](#estrutura)
2. [Setup do ambiente (Poetry)](#setup)
3. [Exercício A — Rastreamento com MLflow](#mlflow)
4. [Exercício B — API FastAPI](#api)
5. [App Gradio](#gradio)
6. [Monitoramento em produção (conceitual)](#monitoramento)
7. [Divergências / melhorias aplicadas](#divergencias)
8. [Como reproduzir os entregáveis](#reproduzir)
9. [Bônus e extensões possíveis](#bonus)

---

## 1. Estrutura do repositório <a name="estrutura"></a>

```
sentiment-analysis/
├── app.py                    # Demo Gradio
├── model.py                  # Carregamento compartilhado do modelo (pipeline Hugging Face)
├── mlflow_tracking.py        # Exercício A: rastreamento de experimentos (MLflow)
├── api/
│   ├── __init__.py
│   └── main.py               # Exercício B: API de inferência (FastAPI)
├── scripts/
│   ├── generate_comparison.py  # Gera a tabela de comparação de runs (CSV/MD)
│   ├── capture_screen.ps1      # Captura de tela (PowerShell, sem dependências)
│   └── shoot_web.mjs           # Web-shot via Playwright (ferramenta dev, fora do runtime)
├── docs/prints/              # Evidências (prints) das etapas entregáveis
├── train_model.py            # Bônus: fine-tuning com IMDB (opcional)
├── MODEL_CARD.md             # Model card
├── tests/                    # Testes (pytest + fastapi.testclient) para o CI
├── pyproject.toml            # Gerenciamento de dependências com Poetry
├── poetry.lock               # Lockfile reproduzível
├── requirements.txt          # Compat: export `poetry export --without-hashes -o requirements.txt`
├── requirements_train.txt    # Deps extras para o bônus de fine-tuning
├── .env.example              # Variáveis de ambiente (template; .env não é versionado)
└── .gitignore
```

---

## 2. Setup do ambiente (Poetry) <a name="setup"></a>

O projeto gerencia dependências com **Poetry** (instalado via **pipx**). O venv é criado pelo Poetry em `.venv/` (isolado por projeto).

```bash
# 1. Clone do fork
git clone https://github.com/cavalcanteprofissional/sentiment-analysis.git
cd sentiment-analysis

# 2. Instalar as dependências
poetry install

# 3. Opcional: grupos de desenvolvimento e treino
poetry install --with dev   # pytest + ruff
poetry install --with train # scikit-learn (bônus fine-tuning)

# 4. Variável de ambiente do MLflow (ver seção MLflow)
cp .env.example .env        # edite se preciso
```

> **Nota sobre o ambiente Windows:** o projeto reside em um caminho com espaços e acentos (`D:\BACK UP\...\RESIDÊNCIA PRÁTICA\...`). Nesse cenário o venv do Poetry é redirecionado para um diretório sem acentos para evitar erros de encoding (ver [Divergências](#divergencias)).
> Uso alternativo sem Poetry: `pip install -r requirements.txt` (arquivo gerado via `poetry export`).

Execute qualquer comando com `poetry run` (ex.: `poetry run python mlflow_tracking.py`).

---

## 3. Exercício A — Rastreamento com MLflow <a name="mlflow"></a>

Compara **três modelos** de análise de sentimentos sobre um pequeno conjunto de frases de teste, registrando no MLflow **parâmetros**, **métricas** e **artefatos** por execução (run).

### Modelos comparados

| Run | Modelo | Esquema de rótulo |
|-----|--------|-------------------|
| 1 | `default-distilbert-sst2` (distilbert-base-uncased-finetuned-sst-2-english) | POSITIVE / NEGATIVE |
| 2 | `cardiffnlp/twitter-xlm-roberta-base-sentiment` | POSITIVE / NEGATIVE (multilíngue) |
| 3 | `nlptown/bert-base-multilingual-uncased-sentiment` | "1 star".."5 stars" (multilíngue) |

Métricas logadas por run: `accuracy_frases_teste`, `avg_confidence`, `latencia_media_seg`. Artefato: `resultados.json` com a predição detalhada de cada frase.

### Subir o servidor MLflow (UI)

```bash
docker run -p 5000:5000 \
  -v "$PWD/mlruns:/mlflow/mlruns" \
  ghcr.io/mlflow/mlflow \
  mlflow server --host 0.0.0.0 --port 5000
```

A UI fica em `http://localhost:5000`. O tracking URI é controlado pela variável `MLFLOW_TRACKING_URI` (template em `.env.example`):

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000   # UI + persistência via Docker
# ou, sem servidor (fallback do script):
export MLFLOW_TRACKING_URI=sqlite:///mlruns.db
```

### Rodar

```bash
poetry run python mlflow_tracking.py
```

Ao final, o próprio script imprime a tabela de comparação via `mlflow.search_runs()`. Também é possível gerar os artefatos de comparação (CSV/MD) em `docs/prints/`:

```bash
poetry run python scripts/generate_comparison.py
```

> A 3ª run usa o modelo `nlptown`, cujos rótulos são "1 star".."5 stars". A função `normalize_label()` foi estendida para mapeá-los a POSITIVE/NEGATIVE (ver [Divergências](#divergencias)); sem isso a acurácia desse modelo seria zerada.

---

## 4. Exercício B — API FastAPI <a name="api"></a>

Endpoint de inferência com FastAPI, reutilizando o mesmo modelo via `model.py`.

```bash
poetry run uvicorn api.main:app --reload
```

Endpoints (Swagger UI em `http://localhost:8000/docs`):

| Método | Rota | Descrição |
|--------|------|-----------|
| GET | `/health` | Saúde da aplicação |
| POST | `/predict` | Predição de um único texto |
| POST | `/predict/batch` | Predição de uma lista de textos (mesma ordem) |

Exemplos:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" -d '{"text": "I love this course!"}'

curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["I love this course!", "This is terrible"]}'
```

Tratamento de erros: lista vazia ou itens inválidos → HTTP 400; falha na inferência → HTTP 500 com mensagem clara. A lógica de predição individual foi extraída para `_predict_one()` e é reutilizada por `/predict` e `/predict/batch`.

---

## 5. App Gradio <a name="gradio"></a>

Demo interativa:

```bash
poetry run python app.py
```

Interface carrega o modelo e retorna a predição de sentimento do texto (POSITIVE/NEGATIVE). Publicação temporária grátis: `demo.launch(share=True)`.

---

## 6. Monitoramento em produção <a name="monitoramento"></a>

> Nível conceitual (documentado aqui e via parâmetros/métricas no MLflow).

### Data drift (deriva de dados)
Mudança na **distribuição dos dados de entrada** em relação ao conjunto de treino. Ex.: o modelo treinado com reviews de filmes em inglês passa a receber mensagens curtas em português — a distribuição de comprimento, vocabulário e idioma muda, mesmo que a relação entrada→saída continue a mesma.

### Model / concept drift (deriva de modelo/conceito)
Mudança na **relação entre entrada e saída esperada** ao longo do tempo. Ex.: o significado de "tremendo" muda conforme o domínio/época; a mesma frase passa a ser rotulada de forma incorreta mesmo com dados semelhantes aos do treino.

### Métricas de acompanhamento sugeridas
- **Volume de predições** por unidade de tempo (detecta mudanças na demanda/uso).
- **Distribuição das classes previstas** (proporção POSITIVE/NEGATIVE — deslocamentos sinalizam drift de dados ou de conceito).
- **Confiança média** das predições (logada como `avg_confidence` no MLflow).

### Sinal de alerta
Queda da **confiança média** ou da **acurácia** sobre amostras rotuladas manualmente ao longo do tempo. Um limiar de confiança decrescente, ou um drift estatístico na distribuição das classes, deve acionar a revisão.

### Ação diante de drift
**Retreino periódico** com dados mais recentes (recoletados/rotulados), reavaliando o modelo com o pipeline de tracking do MLflow para comparar a nova versão (accuracy, confiança, latência) antes do deploy.

---

## 7. Divergências / melhorias aplicadas <a name="divergencias"></a>

Ajustes feitos sobre o repositório-base, todos sinalizados aqui por exigência da atividade:

1. **Tracking URI via variável de ambiente** (`mlflow_tracking.py`) — o base fixava `sqlite:///mlruns.db`. Passou a ler `MLFLOW_TRACKING_URI` (com fallback SQLite) para funcionar com o servidor Docker (`:5000`) e com o SQLite local, sem hardcodar.
2. **Terceiro modelo** (`MLFLOW_MODELS`) — adicionado `nlptown/bert-base-multilingual-uncased-sentiment` como 3ª run.
3. **`normalize_label()` estendida** (`mlflow_tracking.py`) — o base só mapeava POS/NEG; a partir do modelo `nlptown` passou a mapear "1-2 stars"→NEGATIVE, "4-5 stars"→POSITIVE (e é neutro para "3 stars"). Sem isso a acurácia da 3ª run seria 0.
4. **Encoding UTF-8 no console** (`mlflow_tracking.py`) — o MLflow 3.x imprime um emoji no fim de cada run; no console Windows (cp1252) isso lançaria `UnicodeEncodeError`. Adicionado `sys.stdout.reconfigure(encoding="utf-8")`.
5. **`/predict/batch` implementado** (`api/main.py`) — o base deixava `raise NotImplementedError`. Validação Pydantic (`BatchInput`), lista vazia/itens inválidos → HTTP 400, erro de inferência → HTTP 500, e lógica extraída para `_predict_one()` reutilizada.
6. **Poetry + pyproject.toml** — migração de `requirements.txt` (mantido via `poetry export` para compat) para Poetry; grupos `dev` (pytest, ruff) e `train` (scikit-learn).
7. **Ruff (lint + format)** — aplicado de forma consistente (inclui ajuste de whitespace/linhas longas em arquivos do base, como `model.py` e `train_model.py`).
8. **`.gitignore` ampliado** — cobertura de `mlruns/`, `.env`, caches, `.venv/`, `.pytest_cache`, etc.
9. **Venv do Poetry em diretório sem acentos** — contorno de erro de encoding do Poetry no Windows para caminhos não-ASCII (`D:\BACK UP\...\RESIDÊNCIA PRÁTICA\...`).

---

## 8. Como reproduzir os entregáveis <a name="reproduzir"></a>

| Entregável | Passos | Evidência gerada |
|---|---|---|
| Tabela de comparação MLflow | `export MLFLOW_TRACKING_URI=http://localhost:5000` + `poetry run python mlflow_tracking.py` | `docs/prints/01_mlflow_comparacao.csv` / `.md` + UI |
| Teste `/predict/batch` | `poetry run uvicorn api.main:app --reload` + curl/Swagger | `docs/prints/04_swagger_predict_batch.png` |
| App Gradio | `poetry run python app.py` | `docs/prints/06_gradio_app.png` |
| CI/CD (bônus) | Git push no fork → GitHub Actions | link do run (README) |

Os prints ficam em `docs/prints/` (mantidos apenas local, ignorados no remoto): `00_mlflow_ui.png` (UI do servidor), `01_mlflow_comparacao.csv/.md` (tabela), `02_mlflow_ui_runs.png` (UI com as 3 runs), `03_mlflow_console.png` (console), `04_swagger_predict_batch.png` (Swagger), `05_api_batch_console.png` (curl), `06_gradio_app.png` (interface Gradio), `07_ci_actions_run.txt`/`07_ci_actions_success.png` (CI bônus).

**CI/CD (bônus):** o workflow `.github/workflows/ci.yml` roda lint (ruff), testes pytest (`tests/test_api.py`) e smoke test do `mlflow_tracking`. Run de exemplo (sucesso): https://github.com/cavalcanteprofissional/sentiment-analysis/actions/runs/33592516367

---

## 9. Bônus e extensões possíveis <a name="bonus"></a>

- **Fine-tuning** com IMDb (`train_model.py` + `poetry install --with train`).
- **Deploy Hugging Face Spaces** (ZeroGPU gratuito ou `share=True`); CPU dedicada exige plano PRO.
- **CI/CD** via GitHub Actions — o workflow `.github/workflows/ci.yml` roda em cada push para `main`:
  1. `ruff check` (lint);
  2. `pytest` (5 testes em `tests/test_api.py` cobrindo `/health`, `/predict` e `/predict/batch` incluindo casos de erro);
  3. smoke test do `mlflow_tracking` (SQLite, sem servidor) via `scripts/ci_smoke.py`.
  - Run de sucesso: https://github.com/cavalcanteprofissional/sentiment-analysis/actions/runs/33592516367
  - Evidência textual: `docs/prints/07_ci_actions_run.txt` (mantido apenas local).

---

## Licença

MIT — uso educacional. Modelos do Hugging Face Hub são públicos/abertos; nenhum token é necessário.