from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import rule_dependent_kernel_active_set as rd
import tmp_inhibition_profile_block_validate as proto


PAPER_SUITE = REPO_ROOT / "data" / "paper_suite"
BENCHMARKS = [
    {"name": "logical_clean_plus", "config": PAPER_SUITE / "paper_logical_clean_plus.yaml"},
    {"name": "logical_shared", "config": PAPER_SUITE / "paper_logical_shared.yaml"},
    {"name": "logical_context", "config": PAPER_SUITE / "paper_logical_context.yaml"},
    {"name": "kernel_triangular", "config": PAPER_SUITE / "paper_kernel_robustness_triangular.yaml"},
    {"name": "kernel_exponential", "config": PAPER_SUITE / "paper_kernel_robustness_exponential.yaml"},
    {"name": "kernel_gaussian", "config": PAPER_SUITE / "paper_kernel_robustness_gaussian.yaml"},
    {"name": "num_predicates_10", "config": PAPER_SUITE / "paper_num_predicates_10.yaml"},
    {"name": "num_predicates_20", "config": PAPER_SUITE / "paper_num_predicates_20.yaml"},
    {"name": "num_predicates_30", "config": PAPER_SUITE / "paper_num_predicates_30.yaml"},
    {"name": "ablation_excitation_only", "config": PAPER_SUITE / "paper_ablation_excitation_only.yaml"},
    {"name": "ablation_inhibition_only", "config": PAPER_SUITE / "paper_ablation_inhibition_only.yaml"},
    {"name": "ablation_mixed_sign", "config": PAPER_SUITE / "paper_ablation_mixed_sign.yaml"},
]


def local_data_path(config_path: Path) -> Path:
    return config_path.with_suffix(".local.pkl")


def format_rules(rules, target: int) -> list[str]:
    return [rd.format_rule(r, int(target)) for r in sorted(rules)]


def summarize_prediction(problem, exc_params: dict[int, float], inh_params: dict[int, float]) -> list[str]:
    return sorted(
        rd.summarize_results(
            subsets=problem["subsets"],
            exc_params=exc_params,
            inh_params=inh_params,
            target=int(problem["target"]),
        )
    )


def run_exact_inhibition_correction(
    *,
    problem,
    anchor,
    opt_steps: int,
    lr: float,
    include_singletons: bool,
    max_add: int,
    device: torch.device,
):
    subsets = problem["subsets"]
    active_idx_set = {int(ar.idx) for ar in anchor["active_rules"]}
    n_eff = float(max(anchor["arrays_all"][0][2].size + problem["grid_weights_val"].size, 2))
    penalty_per_rule = 0.5 * np.log(n_eff)

    candidate_ids = []
    for idx, subset in enumerate(subsets):
        order = len(tuple(subset))
        if order not in (1, 2, 3) or int(idx) in active_idx_set:
            continue
        if order == 1 and not bool(include_singletons):
            continue
        gain, sign, _coef0 = rd.rule_score(
            feat_event=anchor["arrays_all"][int(idx)][0],
            feat_grid=anchor["arrays_all"][int(idx)][1],
            mu=float(anchor["mu"]),
            exc_event=anchor["exc_ev"],
            inh_event=anchor["inh_ev"],
            exc_grid=anchor["exc_gr"],
            inh_grid=anchor["inh_gr"],
            grid_weights_train=problem["grid_weights_train"],
            penalty=float(penalty_per_rule),
        )
        if str(sign) != "inh" or float(gain) <= 1e-12:
            continue
        candidate_ids.append(int(idx))

    event_sum_map = {int(idx): float(np.sum(anchor["arrays_all"][int(idx)][0])) for idx in candidate_ids}
    grid_col_map = {int(idx): np.asarray(anchor["arrays_all"][int(idx)][1], dtype=np.float64) for idx in candidate_ids}

    best_state, node_count, _cache = proto.search_best_additions(
        candidate_ids=candidate_ids,
        event_sum_map=event_sum_map,
        grid_col_map=grid_col_map,
        base_grid_mass=np.asarray(anchor["base_grid_mass"], dtype=np.float64),
        penalty_per_rule=float(penalty_per_rule),
        max_add=int(max_add),
    )

    if not best_state.support:
        return {
            "accepted": False,
            "candidate_count": int(len(candidate_ids)),
            "node_count": int(node_count),
            "selected_additions": [],
            "bic": float(anchor["bic"]),
            "mu": float(anchor["mu"]),
            "exc_params": dict(anchor["exc_params"]),
            "inh_params": dict(anchor["inh_params"]),
            "active_rules": list(anchor["active_rules"]),
        }

    added_rules = [rd.ActiveRule(idx=int(idx), sign="inh") for idx in best_state.support]
    final_active = list(anchor["active_rules"]) + added_rules
    init_coef_map = rd.build_init_coef_map(final_active, anchor["exc_params"], anchor["inh_params"], fallback=0.1)
    for idx, beta in zip(best_state.support, best_state.beta.tolist()):
        init_coef_map[(int(idx), "inh")] = float(max(beta, 1e-4))

    torch_basis_cache = rd.TorchBasisCache(problem["basis_cache"], device)
    refit_bic, refit_mu, refit_exc, refit_inh, _rule_heights, _arrays_fit = rd.optimize_active_set_torch(
        active_rules=final_active,
        subsets=subsets,
        basis_cache=problem["basis_cache"],
        rule_heights=problem["rule_heights"],
        init_mu=float(anchor["mu"]),
        init_coef_map=init_coef_map,
        grid_weights_train=problem["grid_weights_train"],
        grid_weights_val=problem["grid_weights_val"],
        device=device,
        torch_basis_cache=torch_basis_cache,
        steps=max(30, int(opt_steps)),
        lr=float(lr),
        penalize_kernel_df=False,
    )

    accepted = float(refit_bic) + 1e-8 < float(anchor["bic"])
    if not accepted:
        return {
            "accepted": False,
            "candidate_count": int(len(candidate_ids)),
            "node_count": int(node_count),
            "selected_additions": [int(idx) for idx in best_state.support],
            "bic": float(anchor["bic"]),
            "mu": float(anchor["mu"]),
            "exc_params": dict(anchor["exc_params"]),
            "inh_params": dict(anchor["inh_params"]),
            "active_rules": list(anchor["active_rules"]),
        }

    return {
        "accepted": True,
        "candidate_count": int(len(candidate_ids)),
        "node_count": int(node_count),
        "selected_additions": [int(idx) for idx in best_state.support],
        "bic": float(refit_bic),
        "mu": float(refit_mu),
        "exc_params": {int(k): float(v) for k, v in refit_exc.items()},
        "inh_params": {int(k): float(v) for k, v in refit_inh.items()},
        "active_rules": final_active,
    }


def evaluate_benchmark(
    *,
    name: str,
    config_path: Path,
    opt_steps: int,
    lr: float,
    include_singletons: bool,
    max_add: int,
    cleanup_data: bool,
):
    data_path = local_data_path(config_path)
    started = time.time()
    proto.ensure_local_dataset(config_path, data_path)
    try:
        problem = proto.setup_problem(data_path, config_path)
        device = torch.device("cpu")
        anchor = proto.run_anchor(
            problem,
            opt_steps=int(opt_steps),
            lr=float(lr),
            device=device,
            use_postprune_anchor=True,
        )
        corrected = run_exact_inhibition_correction(
            problem=problem,
            anchor=anchor,
            opt_steps=int(opt_steps),
            lr=float(lr),
            include_singletons=bool(include_singletons),
            max_add=int(max_add),
            device=device,
        )
        target = int(problem["target"])
        gt = sorted(problem["gt"])
        preds = summarize_prediction(problem, corrected["exc_params"], corrected["inh_params"])
        gt_set = set(gt)
        pred_set = set(preds)
        matched = sorted(gt_set & pred_set)
        missing = sorted(gt_set - pred_set)
        extra = sorted(pred_set - gt_set)
        return {
            "name": str(name),
            "config_path": str(config_path),
            "elapsed_sec": float(time.time() - started),
            "anchor_bic": float(anchor["bic"]),
            "bic": float(corrected["bic"]),
            "correction_accepted": bool(corrected["accepted"]),
            "candidate_count": int(corrected["candidate_count"]),
            "search_nodes": int(corrected["node_count"]),
            "selected_additions": format_rules(
                {(tuple(int(s) for s in problem["subsets"][int(idx)]), "inh", target) for idx in corrected["selected_additions"]},
                target,
            ),
            "true_rule": format_rules(gt, target),
            "matched_rule": format_rules(matched, target),
            "missing_rule": format_rules(missing, target),
            "extra_rule": format_rules(extra, target),
            "recall": float(len(matched)) / max(len(gt_set), 1),
            "precision": float(len(matched)) / max(len(pred_set), 1),
        }
    finally:
        if bool(cleanup_data):
            try:
                data_path.unlink(missing_ok=True)
            except Exception:
                pass


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmarks", default="all")
    ap.add_argument("--opt_steps", type=int, default=40)
    ap.add_argument("--lr", type=float, default=0.05)
    ap.add_argument("--max_add", type=int, default=4)
    ap.add_argument("--no_singletons", action="store_true")
    ap.add_argument("--keep_data", action="store_true")
    ap.add_argument("--json_out", default="data/paper_suite/tmp_exact_correction_benchmark_results.json")
    args = ap.parse_args()

    thread_cap = int(os.environ.get("OMP_NUM_THREADS", "1"))
    torch.set_num_threads(max(thread_cap, 1))
    torch.set_num_interop_threads(1)

    wanted = None if str(args.benchmarks).strip().lower() == "all" else {x.strip() for x in str(args.benchmarks).split(",") if x.strip()}
    items = [item for item in BENCHMARKS if wanted is None or item["name"] in wanted]
    if not items:
        raise ValueError("no benchmarks selected")

    results = {}
    out_path = REPO_ROOT / str(args.json_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for item in items:
        name = str(item["name"])
        print(f"RUNNING {name}", flush=True)
        try:
            result = evaluate_benchmark(
                name=name,
                config_path=Path(item["config"]),
                opt_steps=int(args.opt_steps),
                lr=float(args.lr),
                include_singletons=not bool(args.no_singletons),
                max_add=int(args.max_add),
                cleanup_data=not bool(args.keep_data),
            )
        except Exception as exc:
            result = {
                "name": name,
                "error": f"{type(exc).__name__}: {exc}",
            }
        results[name] = result
        out_path.write_text(json.dumps(results, indent=2))
        print(json.dumps({name: result}, indent=2), flush=True)
    print(f"WROTE {out_path}", flush=True)


if __name__ == "__main__":
    main()
