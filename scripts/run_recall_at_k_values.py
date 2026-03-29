"""Recall@k (3, 5 e 10 por defeito): avalia com o índice actual, grava docs/dados/recall_por_k.json
e gera um PNG por k — em cada um, Denso / Esparso / Híbrido com duas barras (MiniLM vs BERT legal).

Os modelos comparados no gráfico são sempre estes dois (strings iguais às de `embeddings.model` no config / JSON):
  - sentence-transformers/all-MiniLM-L6-v2
  - stjiris/bert-large-portuguese-cased-legal-tsdae

Use sempre o mesmo identificador no `config.yaml` para o MiniLM (evita dois runs duplicados no JSON).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JSON_PATH = ROOT / "docs" / "dados" / "recall_por_k.json"
IMG_DIR = ROOT / "docs" / "imagens"

MODEL_MINILM = "sentence-transformers/all-MiniLM-L6-v2"
MODEL_BERT = "stjiris/bert-large-portuguese-cased-legal-tsdae"

MODES = ["dense", "sparse", "hybrid"]
LABELS_PT = ["Denso", "Esparso", "Híbrido"]


def _run_by_model(data: dict, model_name: str) -> dict | None:
    for run in data.get("runs") or []:
        if (run.get("embedding_model") or "").strip() == model_name:
            return run.get("by_k") or {}
    return None


def _plot_one_k(
    k: int,
    pct_minilm: list[float],
    pct_bert: list[float],
    out_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    from matplotlib.patches import Patch

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    x = np.arange(len(MODES))
    w = 0.36
    ax.bar(
        x - w / 2,
        pct_minilm,
        width=w,
        label=MODEL_MINILM,
        color="#4472C4",
        edgecolor="#333",
        linewidth=0.5,
    )
    ax.bar(
        x + w / 2,
        pct_bert,
        width=w,
        label=MODEL_BERT,
        color="#ED7D31",
        edgecolor="#333",
        linewidth=0.5,
    )
    for off, vals in [(-w / 2, pct_minilm), (w / 2, pct_bert)]:
        for xi, v in zip(x + off, vals):
            ax.annotate(
                f"{v:.1f}%",
                xy=(xi, v),
                xytext=(0, 2),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS_PT)
    ax.set_ylabel("Recall (%)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Recall@{k}")
    ax.legend(loc="upper right", fontsize=8)
    ax.axhline(0, color="gray", linewidth=0.4)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight", pad_inches=0.2)
    plt.close(fig)


def _write_plots_from_json(k_list: list[int]) -> None:
    if not JSON_PATH.is_file():
        print(f"Aviso: {JSON_PATH} não existe; não há gráficos por k.")
        return
    data = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    by_m = _run_by_model(data, MODEL_MINILM)
    by_b = _run_by_model(data, MODEL_BERT)
    if not by_m or not by_b:
        print(
            "Aviso: no JSON faltam entradas para MiniLM e/ou BERT legal; "
            "gráficos só saem quando ambos estiverem em runs[]."
        )
        return
    for k in k_list:
        sk = str(k)
        if sk not in by_m or sk not in by_b:
            print(f"Aviso: k={k} em falta para um dos modelos; saltado.")
            continue
        try:
            pm = [float(by_m[sk][m]["recall_pct"]) for m in MODES]
            pb = [float(by_b[sk][m]["recall_pct"]) for m in MODES]
        except (KeyError, TypeError, ValueError):
            print(f"Aviso: dados inválidos para k={k}; saltado.")
            continue
        out = IMG_DIR / f"recall_k{k}.png"
        _plot_one_k(k, pm, pb, out)
        print(f"Gráfico: {out}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Recall@k + JSON + um PNG por k (MiniLM vs BERT).")
    parser.add_argument("--golden-file", default=None, help="Golden set (opcional).")
    parser.add_argument("--k-values", default="3,5,10", help="Ex.: 3,5,10")
    args = parser.parse_args()

    os.chdir(ROOT)
    for p in (str(ROOT), str(ROOT / "src")):
        if p not in sys.path:
            sys.path.insert(0, p)

    if not (ROOT / "data" / "index" / "chroma_db").exists():
        print("ERRO: falta índice. Rode: python src/pipeline.py --step index")
        sys.exit(1)

    from src.pipeline import run_evaluate

    k_list = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    for k in k_list:
        print(f"\n--- Recall@{k} ---\n")
        if run_evaluate(
            golden_file=args.golden_file,
            k=k,
            append_recall_json=str(JSON_PATH),
        ) is None:
            sys.exit(1)

    _write_plots_from_json(k_list)


if __name__ == "__main__":
    main()
