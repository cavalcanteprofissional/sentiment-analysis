"""Gera a evidência textual do run do GitHub Actions (para o entregável).

Uso: poetry run python scripts/ci_evidencia.py [RUN_ID]
"""
import json
import pathlib
import sys
import urllib.request

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "33585785917"
repo = "cavalcanteprofissional/sentiment-analysis"
url = f"https://api.github.com/repos/{repo}/actions/runs/{RUN_ID}"

with urllib.request.urlopen(url) as r:
    data = json.load(r)

out = pathlib.Path("docs/prints")
out.mkdir(parents=True, exist_ok=True)
lines = [
    "Evidência do CI/CD (GitHub Actions) — bônus",
    "Autor: cavalcanteprofissional/sentiment-analysis",
    f"Run:  {data['html_url']}",
    f"Status: {data['status']}",
    f"Conclusao: {data['conclusion']}",
    f"Branch: {data['head_branch']}",
    f"Commit: {data['head_sha']}",
    f"Criado em: {data['created_at']}",
]
(out / "07_ci_actions_run.txt").write_text("\n".join(lines), encoding="utf-8")
print("\n".join(lines))
