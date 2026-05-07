from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
ROOT = Path(__file__).resolve().parents[2]

FIGURE_DIR = ROOT / "workspace/train/research_docs/benchmark_results/figures"

DATASETS = {
    "mimic": {
        "scr_json": ROOT
        / "data/realworld_results/mimic_low_urine_random5000_seed111_20260506_060453/scr_tpp/scr_tpp_seed111.json",
        "manifest": ROOT / "data/realworld_prepared/mimic_low_urine_random_5000/rule_inputs/rule_input_manifest.json",
        "time_unit": "hours",
        "target": "low_urine_output",
        "out_stem": "mimic_low_urine_scr_tpp_learned_kernel_distributions",
    },
    "bpi": {
        "scr_json": ROOT
        / "data/realworld_results/bpi2017_o_accepted_mixed_random5000_seed111_20260507_040210/scr_tpp/scr_tpp_seed111.json",
        "manifest": ROOT
        / "data/realworld_prepared/bpi2017_o_accepted_mixed_random_5000/rule_inputs/rule_input_manifest.json",
        "time_unit": "days",
        "target": "O_Accepted",
        "out_stem": "bpi2017_o_accepted_scr_tpp_learned_kernel_distributions",
    },
}

COLORS = [
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#56B4E9",
    "#E69F00",
]


def _load_labels(manifest_path: Path) -> dict[int, str]:
    payload = json.loads(manifest_path.read_text())
    id_to_event = payload["source_metadata"]["id_to_event"]
    return {int(k): str(v) for k, v in id_to_event.items()}


def _format_sign(sign: str) -> str:
    if str(sign).lower().startswith("exc"):
        return "excitation"
    if str(sign).lower().startswith("inh"):
        return "inhibition"
    return str(sign)


def _plot_dataset(
    *,
    scr_json: Path,
    manifest: Path,
    time_unit: str,
    target: str,
    out_path: Path,
    cols: int,
) -> None:
    payload = json.loads(scr_json.read_text())
    labels = _load_labels(manifest)
    details = list(payload["learned_rule_parameter_details"])

    n_rules = len(details)
    ncols = min(cols, max(1, n_rules))
    nrows = int(math.ceil(n_rules / ncols))
    fig_width = 3.45 * ncols
    fig_height = 2.35 * nrows

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 8.2,
            "axes.labelsize": 8.4,
            "legend.fontsize": 7.0,
            "xtick.labelsize": 7.8,
            "ytick.labelsize": 7.8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    fig, axes = plt.subplots(nrows, ncols, figsize=(fig_width, fig_height), squeeze=False)
    axes_flat = list(axes.ravel())
    for idx, (ax, detail) in enumerate(zip(axes_flat, details), start=1):
        source_distributions = detail["kernel_distribution_by_source"]
        for source_idx, source_dist in enumerate(source_distributions):
            source_id = int(source_dist["source"])
            knots = np.asarray(source_dist["knots"], dtype=np.float64)
            density = np.asarray(source_dist["area_normalized_density"], dtype=np.float64)
            source_name = labels[source_id]
            ax.plot(
                knots,
                density,
                lw=1.9,
                marker="o",
                ms=2.8,
                color=COLORS[source_idx % len(COLORS)],
                label=source_name,
            )

        ax.text(0.02, 0.96, f"R{idx} ({_format_sign(detail['sign'])})", transform=ax.transAxes, ha="left", va="top", fontsize=7.5)
        ax.grid(True, color="#E6E6E6", lw=0.55)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlabel(f"lag to target ({time_unit})")
        ax.set_ylabel("normalized density")
        ax.set_xlim(0.0, 30.0)
        ymax = max(
            1e-8,
            *[float(np.max(np.asarray(d["area_normalized_density"], dtype=np.float64))) for d in source_distributions],
        )
        ax.set_ylim(0.0, ymax * 1.22)
        ax.legend(
            loc="upper right",
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.72,
            handlelength=1.55,
            handletextpad=0.45,
            borderpad=0.25,
            labelspacing=0.25,
        )

    for ax in axes_flat[n_rules:]:
        ax.axis("off")

    fig.tight_layout(w_pad=1.0, h_pad=1.0)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot real-world SCR-TPP learned source-kernel distributions.")
    parser.add_argument("--datasets", default="mimic,bpi", help="Comma-separated keys: mimic,bpi")
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--out-dir", default=str(FIGURE_DIR))
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    selected = [x.strip() for x in str(args.datasets).split(",") if x.strip()]
    for key in selected:
        cfg = DATASETS[key]
        out_path = out_dir / f"{cfg['out_stem']}.pdf"
        _plot_dataset(
            scr_json=cfg["scr_json"],
            manifest=cfg["manifest"],
            time_unit=str(cfg["time_unit"]),
            target=str(cfg["target"]),
            out_path=out_path,
            cols=int(args.cols),
        )
        print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
