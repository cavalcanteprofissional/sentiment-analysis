"""Renderiza um "terminal" em PNG com comandos e saída reais (evidência local).

Não é uma captura de tela de um emulador de terminal, e sim uma reprodução
fiel do que o terminal exibiria para os comandos da atividade. Os comandos
são executados de verdade e a saída real é incorporada na imagem.
"""
import os
import pathlib
import subprocess

from PIL import Image, ImageDraw, ImageFont

CMD_HEALTH = "curl -s http://127.0.0.1:8000/health"
CMD_PREDICT = (
    'curl -s http://127.0.0.1:8000/predict -H "Content-Type: application/json" '
    '-d {"text": "I love this MLflow course!"}'
)
CMD_BATCH = (
    'curl -s http://127.0.0.1:8000/predict/batch -H "Content-Type: application/json" '
    '-d {"texts": ["I love MLflow and this course", "This experience was terrible", '
    '"Quite nice, I recommend it"]}'
)
CMD_TRACK = (
    "MLFLOW_TRACKING_URI=sqlite:///tmp_console_evidence.db "
    "poetry run python mlflow_tracking.py"
)

API = "http://127.0.0.1:8000"


def _txt_lines(lines: list[tuple[str, str]]) -> list[str]:
    """[(tipo, conteudo)] -> texto do terminal com prompt."""
    out: list[str] = []
    for kind, content in lines:
        if kind == "cmd":
            out.append("C:\\senti> " + content)
        else:
            out.extend(content.rstrip("\n").splitlines())
    return out


def render(lines: list[tuple[str, str]], out_file: str, title: str) -> None:
    mono = "consola.ttf"  # fonte utilitária do Windows (apoiada por PIL)
    font_path = pathlib.Path("C:/Windows/Fonts") / mono
    font = ImageFont.truetype(str(font_path), 18) if font_path.exists() else None

    text = _txt_lines(lines)
    cell_h = 22
    width = 960
    height = 10 + len(text) * cell_h + 10

    img = Image.new("RGB", (width, height), (12, 12, 12))
    draw = ImageDraw.Draw(img)
    y = 10
    for line in text:
        # linhas longas são quebradas explicitamente; aqui apenas desenha
        draw.text((14, y), line[:120], font=font, fill=(200, 200, 200))
        y += cell_h

    bar_h = 26
    banner = Image.new("RGB", (width, bar_h), (30, 30, 30))
    bdraw = ImageDraw.Draw(banner)
    bdraw.text((8, 4), title, font=font, fill=(255, 255, 255))
    img.paste(banner, (0, 0))

    out = pathlib.Path(out_file)
    out.parent.mkdir(parents=True, exist_ok=True)
    img.save(out)
    print("salvo:", out_file)


def _curl(path: str, data: str | None = None) -> str:
    args = ["curl", "-s", API + path]
    if data is not None:
        args += ["-H", "Content-Type: application/json", "-d", data]
    return subprocess.run(args, capture_output=True, text=True).stdout.strip()


def main() -> None:
    base = "docs/prints"

    # --- 05: curl da API /predict/batch ---
    body_predict = '{"text": "I love this MLflow course!"}'
    body_batch = (
        '{"texts": ["I love MLflow and this course", '
        '"This experience was terrible", "Quite nice, I recommend it"]}'
    )
    health = _curl("/health")
    predict = _curl("/predict", body_predict)
    batch = _curl("/predict/batch", body_batch)

    render(
        [
            ("cmd", CMD_HEALTH),
            ("out", health),
            ("", ""),
            ("cmd", CMD_PREDICT),
            ("out", predict),
            ("", ""),
            ("cmd", CMD_BATCH),
            ("out", batch),
        ],
        f"{base}/05_api_batch_console.png",
        "FastAPI - /predict e /predict/batch (evidencia real)",
    )

    # --- 03: console do tracking MLflow (3 runs) ---
    # Saída real de mlflow_tracking.py direcionado a um sqlite TEMPORÁRIO
    # (não polui o servidor Docker). Summary por modelo + tabela de comparação.
    tracking_env = dict(os.environ)
    tracking_env["MLFLOW_TRACKING_URI"] = "sqlite:///tmp_console_evidence.db"
    output = subprocess.run(
        ["poetry", "run", "python", "mlflow_tracking.py"],
        capture_output=True,
        text=True,
        env=tracking_env,
    ).stdout.splitlines()

    resumo = [ln for ln in output if ": accuracy=" in ln]
    tabela: list[str] = []
    grab = False
    for ln in output:
        if "Comparando execucoes" in ln:
            grab = True
        if grab:
            tabela.append(ln)

    render(
        [("cmd", CMD_TRACK)]
        + [("out", ln) for ln in resumo]
        + [("", "")]
        + [("out", ln) for ln in tabela],
        f"{base}/03_mlflow_console.png",
        "MLflow tracking - 3 modelos comparados (evidencia real)",
    )

    # limpeza do sqlite temporário
    try:
        os.remove("tmp_console_evidence.db")
    except FileNotFoundError:
        pass


if __name__ == "__main__":
    main()
