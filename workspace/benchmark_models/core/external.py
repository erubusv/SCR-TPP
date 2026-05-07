from __future__ import annotations

import json
import os
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .schema import NormalizedRule, load_json_or_yaml, normalize_rules


@dataclass(frozen=True)
class ExternalBaselineSpec:
    name: str
    source_url: str
    implementation_type: str
    command: str | list[str]
    output_format: str = "normalized_rules_json"
    working_dir: str | None = None
    env: dict[str, str] | None = None

    @staticmethod
    def from_obj(obj: dict[str, Any]) -> "ExternalBaselineSpec":
        required = ["name", "source_url", "implementation_type", "command"]
        missing = [key for key in required if key not in obj]
        if missing:
            raise ValueError(f"baseline spec missing keys: {missing}")
        return ExternalBaselineSpec(
            name=str(obj["name"]),
            source_url=str(obj["source_url"]),
            implementation_type=str(obj["implementation_type"]),
            command=obj["command"],
            output_format=str(obj.get("output_format", "normalized_rules_json")),
            working_dir=obj.get("working_dir"),
            env={str(k): str(v) for k, v in dict(obj.get("env", {})).items()},
        )


def load_baseline_manifest(path: str | Path) -> dict[str, ExternalBaselineSpec]:
    data = load_json_or_yaml(path)
    specs = data.get("baselines", data if isinstance(data, list) else [])
    out = {}
    for spec_obj in specs:
        spec = ExternalBaselineSpec.from_obj(spec_obj)
        out[spec.name] = spec
    return out


def _format_command(command: str | list[str], values: dict[str, Any]) -> list[str]:
    if isinstance(command, str):
        formatted = command.format(**values)
        return shlex.split(formatted)
    return [str(part).format(**values) for part in command]


def run_external_rule_baseline(
    *,
    spec: ExternalBaselineSpec,
    config_path: str | Path,
    dataset_path: str | Path,
    output_path: str | Path,
    seed: int,
    extra_values: dict[str, Any] | None = None,
) -> tuple[list[NormalizedRule], dict[str, Any]]:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "config_path": str(config_path),
        "dataset_path": str(dataset_path),
        "output_path": str(output_path),
        "seed": int(seed),
    }
    if extra_values:
        values.update(extra_values)
    cmd = _format_command(spec.command, values)
    env = os.environ.copy()
    if spec.env:
        env.update(spec.env)
    start = time.perf_counter()
    proc = subprocess.run(
        cmd,
        cwd=spec.working_dir,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    runtime_sec = float(time.perf_counter() - start)
    metadata = {
        "source_url": spec.source_url,
        "implementation_type": spec.implementation_type,
        "command": cmd,
        "returncode": int(proc.returncode),
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "runtime_sec": runtime_sec,
    }
    if proc.returncode != 0:
        raise RuntimeError(f"baseline {spec.name} failed with code {proc.returncode}: {proc.stderr[-2000:]}")
    if not output_path.exists():
        raise FileNotFoundError(f"baseline {spec.name} did not create output file: {output_path}")
    payload = load_json_or_yaml(output_path)
    metadata["raw_output_path"] = str(output_path)
    if isinstance(payload, dict) and isinstance(payload.get("metadata"), dict):
        metadata["payload_metadata"] = payload["metadata"]
    if spec.output_format == "normalized_rules_json":
        rules_obj = payload.get("predicted_rules", payload.get("rules", payload))
        rules = normalize_rules(rules_obj)
    elif spec.output_format == "runner_strings_json":
        default_target = int(payload.get("target", -1))
        if default_target < 0:
            raise ValueError("runner_strings_json requires target")
        rules = normalize_rules(payload.get("predicted", []), default_target=default_target)
    else:
        raise ValueError(f"unsupported output_format: {spec.output_format}")
    return rules, metadata


def write_example_manifest(path: str | Path) -> None:
    path = Path(path)
    baseline_rows = [
        ("TELLER", "https://openreview.net/forum?id=P07dq7iSAGr", "/path/to/teller/run.py"),
        ("CLNN", "https://research.ibm.com/publications/weighted-clock-logic-point-process", "/path/to/clnn/run.py"),
        ("NSTPP", "https://proceedings.mlr.press/v235/yang24ag.html", "/path/to/nstpp/run.py"),
        ("CLUSTER", "https://proceedings.mlr.press/v238/kuang24a.html", "/path/to/cluster/run.py"),
    ]
    example = {
        "baselines": [
            {
                "name": name,
                "source_url": source_url,
                "implementation_type": "official_external",
                "command": [
                    "python",
                    script_path,
                    "--config",
                    "{config_path}",
                    "--data",
                    "{dataset_path}",
                    "--seed",
                    "{seed}",
                    "--output",
                    "{output_path}",
                ],
                "output_format": "normalized_rules_json",
            }
            for name, source_url, script_path in baseline_rows
        ]
    }
    path.write_text(yaml.safe_dump(example, sort_keys=False))
