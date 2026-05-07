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

from .schema import RealWorldPredictionResult, load_json_or_yaml, normalize_rules


@dataclass(frozen=True)
class PredictionBaselineSpec:
    name: str
    source_url: str
    implementation_type: str
    command: str | list[str]
    output_format: str = "prediction_metrics_json"
    working_dir: str | None = None
    env: dict[str, str] | None = None

    @staticmethod
    def from_obj(obj: dict[str, Any]) -> "PredictionBaselineSpec":
        required = ["name", "source_url", "implementation_type", "command"]
        missing = [key for key in required if key not in obj]
        if missing:
            raise ValueError(f"prediction baseline spec missing keys: {missing}")
        return PredictionBaselineSpec(
            name=str(obj["name"]),
            source_url=str(obj["source_url"]),
            implementation_type=str(obj["implementation_type"]),
            command=obj["command"],
            output_format=str(obj.get("output_format", "prediction_metrics_json")),
            working_dir=obj.get("working_dir"),
            env={str(k): str(v) for k, v in dict(obj.get("env", {})).items()},
        )


def load_prediction_manifest(path: str | Path) -> dict[str, PredictionBaselineSpec]:
    data = load_json_or_yaml(path)
    specs = data.get("baselines", data if isinstance(data, list) else [])
    out = {}
    for spec_obj in specs:
        spec = PredictionBaselineSpec.from_obj(spec_obj)
        out[spec.name] = spec
    return out


def _format_command(command: str | list[str], values: dict[str, Any]) -> list[str]:
    if isinstance(command, str):
        return shlex.split(command.format(**values))
    return [str(part).format(**values) for part in command]


def _optional_float(payload: dict[str, Any], key: str) -> float | None:
    value = payload.get(key)
    if value is None:
        return None
    return float(value)


def run_external_prediction_baseline(
    *,
    spec: PredictionBaselineSpec,
    dataset: str,
    target: str,
    train_path: str | Path,
    dev_path: str | Path,
    test_path: str | Path,
    output_path: str | Path,
    seed: int,
    extra_values: dict[str, Any] | None = None,
) -> RealWorldPredictionResult:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    values = {
        "dataset": dataset,
        "target": target,
        "train_path": str(train_path),
        "dev_path": str(dev_path),
        "test_path": str(test_path),
        "output_path": str(output_path),
        "output_dir": str(output_path.parent),
        "seed": int(seed),
        "model": spec.name,
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
        raise RuntimeError(f"prediction baseline {spec.name} failed: {proc.stderr[-2000:]}")
    if not output_path.exists():
        raise FileNotFoundError(f"prediction baseline {spec.name} did not write {output_path}")
    payload = load_json_or_yaml(output_path)
    if spec.output_format != "prediction_metrics_json":
        raise ValueError(f"unsupported prediction output_format: {spec.output_format}")
    learned_rules_obj = payload.get("learned_rules", payload.get("rules", []))
    learned_rules = [rule.to_dict() for rule in normalize_rules(learned_rules_obj)] if learned_rules_obj else []
    result = RealWorldPredictionResult(
        model=spec.name,
        dataset=dataset,
        target=target,
        nll=_optional_float(payload, "nll"),
        time_mae=_optional_float(payload, "time_mae"),
        time_rmse=_optional_float(payload, "time_rmse"),
        type_acc=_optional_float(payload, "type_acc"),
        topk={str(k): float(v) for k, v in dict(payload.get("topk", {})).items()},
        runtime_sec=float(payload.get("runtime_sec", runtime_sec)),
        learned_rules=learned_rules,
        model_metadata={**metadata, **dict(payload.get("model_metadata", {}))},
    )
    result.validate()
    return result


def write_prediction_example_manifest(path: str | Path) -> None:
    path = Path(path)
    rows = [
        ("RMTPP", "https://github.com/ant-research/EasyTemporalPointProcess", "/path/to/run_easytpp.py"),
        ("NHP", "https://github.com/ant-research/EasyTemporalPointProcess", "/path/to/run_easytpp.py"),
        ("SAHP", "https://github.com/ant-research/EasyTemporalPointProcess", "/path/to/run_easytpp.py"),
        ("THP", "https://github.com/ant-research/EasyTemporalPointProcess", "/path/to/run_easytpp.py"),
        ("AttNHP", "https://github.com/ant-research/EasyTemporalPointProcess", "/path/to/run_easytpp.py"),
        ("IntensityFree", "https://github.com/ant-research/EasyTemporalPointProcess", "/path/to/run_easytpp.py"),
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
                    "--model",
                    "{model}",
                    "--train",
                    "{train_path}",
                    "--dev",
                    "{dev_path}",
                    "--test",
                    "{test_path}",
                    "--seed",
                    "{seed}",
                    "--output",
                    "{output_path}",
                ],
                "output_format": "prediction_metrics_json",
            }
            for name, source_url, script_path in rows
        ]
    }
    path.write_text(yaml.safe_dump(example, sort_keys=False))
