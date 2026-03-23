"""Unified driver for overlap-synthetic research runs."""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

import yaml


MANIFEST_PATH = Path(__file__).with_name("overlap_research_manifest.yaml")


def load_manifest() -> dict:
    with MANIFEST_PATH.open("r") as f:
        return yaml.safe_load(f)


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def print_methods(manifest: dict):
    current_best = str(manifest["current_best"]["method"])
    print("Methods:")
    for name, spec in manifest["methods"].items():
        best_tag = " [current best]" if name == current_best else ""
        print(f"  - {name}: status={spec['status']}{best_tag}")
        print(f"    {spec['description']}")


def print_datasets(manifest: dict):
    print("Datasets:")
    for name, spec in manifest["datasets"].items():
        print(f"  - {name}: data={spec['data']}")
        print(f"    config={spec['config']}")


def print_best(manifest: dict):
    current_best = manifest["current_best"]
    print("Current best method:")
    print(f"  - method: {current_best['method']}")
    print(f"  - summary: {current_best['summary']}")
    print("  - observed:")
    for dataset, spec in current_best.get("observed", {}).items():
        print(f"    - {dataset}: recall={spec['recall']}")
        print(f"      notes={spec.get('notes', '')}")
        args = spec.get("args", [])
        if args:
            print(f"      args={' '.join(args)}")


def resolve_dataset(manifest: dict, dataset: str) -> tuple[str, str, int | None]:
    spec = manifest["datasets"].get(dataset)
    if spec is None:
        raise SystemExit(f"unknown dataset preset: {dataset}")
    fixed_target = spec.get("fixed_target")
    return str(spec["data"]), str(spec["config"]), (None if fixed_target is None else int(fixed_target))


def resolve_method(manifest: dict, method: str) -> dict:
    spec = manifest["methods"].get(method)
    if spec is None:
        raise SystemExit(f"unknown method: {method}")
    return spec


def build_command(
    manifest: dict,
    *,
    method: str,
    dataset: str | None,
    data: str | None,
    config: str | None,
    fixed_target: int | None,
    use_best_preset: bool,
    extra_args: list[str],
) -> list[str]:
    method_spec = resolve_method(manifest, method)
    if dataset is not None:
        data_path, config_path, dataset_fixed_target = resolve_dataset(manifest, dataset)
    else:
        if not data or not config:
            raise SystemExit("either --dataset or both --data/--config are required")
        data_path, config_path = data, config
        dataset_fixed_target = None

    resolved_fixed_target = int(dataset_fixed_target if fixed_target is None else fixed_target)

    cmd = [
        sys.executable,
        str(method_spec["script"]),
        "--data",
        str(data_path),
        "--config",
        str(config_path),
        "--fixed_target",
        str(int(resolved_fixed_target)),
    ]
    cmd.extend(str(x) for x in method_spec.get("default_args", []))
    if use_best_preset:
        current_best = manifest["current_best"]
        if method != str(current_best["method"]):
            raise SystemExit("--use-best-preset is only valid for the current best method")
        if dataset is None:
            raise SystemExit("--use-best-preset requires --dataset")
        observed = current_best.get("observed", {}).get(dataset)
        if observed is None:
            raise SystemExit(f"no recorded best preset for dataset: {dataset}")
        cmd.extend(str(x) for x in observed.get("args", []))
    cmd.extend(extra_args)
    return cmd


def run_command(cmd: list[str]):
    print("Running:")
    print(f"  {' '.join(shlex.quote(x) for x in cmd)}")
    sys.stdout.flush()
    subprocess.run(cmd, cwd=str(repo_root()), check=True)


def main():
    manifest = load_manifest()

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="subcmd", required=True)

    sub.add_parser("list")
    sub.add_parser("show-best")

    ap_run = sub.add_parser("run")
    ap_run.add_argument("--method", required=True, choices=sorted(manifest["methods"].keys()))
    ap_run.add_argument("--dataset", choices=sorted(manifest["datasets"].keys()))
    ap_run.add_argument("--data")
    ap_run.add_argument("--config")
    ap_run.add_argument("--fixed_target", type=int)
    ap_run.add_argument("--use-best-preset", action="store_true")
    ap_run.add_argument("--extra-arg", action="append", default=[])

    ap_best = sub.add_parser("run-best")
    ap_best.add_argument("--dataset", required=True, choices=sorted(manifest["datasets"].keys()))
    ap_best.add_argument("--fixed_target", type=int)
    ap_best.add_argument("--extra-arg", action="append", default=[])

    args = ap.parse_args()

    if args.subcmd == "list":
        print(f"Evaluation: {manifest['evaluation_protocol']['selection']}")
        print_methods(manifest)
        print_datasets(manifest)
        return

    if args.subcmd == "show-best":
        print(f"Evaluation: {manifest['evaluation_protocol']['selection']}")
        print_best(manifest)
        return

    if args.subcmd == "run":
        cmd = build_command(
            manifest,
            method=str(args.method),
            dataset=args.dataset,
            data=args.data,
            config=args.config,
            fixed_target=args.fixed_target,
            use_best_preset=bool(args.use_best_preset),
            extra_args=list(args.extra_arg),
        )
        run_command(cmd)
        return

    if args.subcmd == "run-best":
        current_best = str(manifest["current_best"]["method"])
        cmd = build_command(
            manifest,
            method=current_best,
            dataset=str(args.dataset),
            data=None,
            config=None,
            fixed_target=args.fixed_target,
            use_best_preset=True,
            extra_args=list(args.extra_arg),
        )
        run_command(cmd)
        return

    raise SystemExit(f"unknown subcommand: {args.subcmd}")


if __name__ == "__main__":
    main()
