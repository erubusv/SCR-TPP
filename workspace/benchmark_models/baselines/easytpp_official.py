from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
import types
from pathlib import Path
from typing import Any

import yaml


ROOT = Path(__file__).resolve().parents[2]
EASYTPP_ROOT = ROOT / "external" / "official_baselines" / "EasyTemporalPointProcess"


def _infer_dim_process(path: str | Path) -> int:
    with Path(path).open("rb") as f:
        payload = pickle.load(f)
    dim = int(payload.get("dim_process", 0))
    if dim <= 0:
        raise ValueError(f"cannot infer dim_process from {path}")
    return dim


def _read_input_manifest(path: str | Path) -> dict[str, Any]:
    manifest_path = Path(path).resolve().parent / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text())
    except json.JSONDecodeError:
        return {}


def _target_mark_compute_loglikelihood(model: Any, target_event_id: int) -> Any:
    def compute_loglikelihood(
        self: Any,
        time_delta_seq: Any,
        lambda_at_event: Any,
        lambdas_loss_samples: Any,
        seq_mask: Any,
        type_seq: Any,
    ) -> tuple[Any, Any, int]:
        import torch

        target_mask = torch.logical_and(seq_mask.bool(), type_seq == int(target_event_id))
        target_mask_f = target_mask.to(dtype=time_delta_seq.dtype)
        interval_mask_f = seq_mask.to(dtype=time_delta_seq.dtype)
        lambda_at_event = lambda_at_event + self.eps
        lambdas_loss_samples = lambdas_loss_samples + self.eps

        target_lambda_at_event = lambda_at_event[..., int(target_event_id)]
        event_ll = torch.log(target_lambda_at_event) * target_mask_f
        target_sampled_lambdas = lambdas_loss_samples[..., int(target_event_id)]
        if self.use_mc_samples:
            interval_integral = target_sampled_lambdas.mean(dim=-1) * time_delta_seq
        else:
            interval_integral = (
                0.5
                * (target_sampled_lambdas[..., 1:] + target_sampled_lambdas[..., :-1]).mean(dim=-1)
                * time_delta_seq
            )
        non_event_ll = interval_integral * interval_mask_f
        return event_ll, non_event_ll, int(target_mask.sum().item())

    return types.MethodType(compute_loglikelihood, model)


def _target_mark_time_metrics_from_grid(model: Any, batch: Any, target_event_id: int, grid_size: int) -> tuple[float, float, int] | None:
    import math
    import torch

    if not hasattr(model, "compute_intensities_at_sample_times"):
        return None
    time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, _ = batch
    hist_time = time_seqs[:, :-1]
    hist_delta = time_delta_seqs[:, :-1]
    hist_type = type_seqs[:, :-1]
    label_delta = time_delta_seqs[:, 1:]
    label_type = type_seqs[:, 1:]
    label_mask = batch_non_pad_mask[:, 1:].bool()
    target_mask = torch.logical_and(label_mask, label_type == int(target_event_id))
    if not bool(target_mask.any()):
        return None

    dtime_max = float(getattr(getattr(model, "event_sampler", None), "dtime_max", 5.0))
    boundary = torch.maximum(label_delta * dtime_max, label_delta + dtime_max).clamp(min=1e-6)
    unit_grid = (torch.arange(int(grid_size), dtype=time_delta_seqs.dtype, device=time_delta_seqs.device) + 0.5) / max(int(grid_size), 1)
    sample_dtimes = boundary.unsqueeze(-1) * unit_grid.view(1, 1, -1)
    intensities = model.compute_intensities_at_sample_times(hist_time, hist_delta, hist_type, sample_dtimes)
    target_lam = intensities[..., int(target_event_id)].clamp(min=1e-12)
    dt = boundary / max(int(grid_size), 1)
    cumhaz_left = torch.cumsum(target_lam * dt.unsqueeze(-1), dim=-1) - target_lam * dt.unsqueeze(-1)
    density = target_lam * torch.exp(-cumhaz_left.clamp(min=0.0, max=80.0))
    denom = torch.sum(density * dt.unsqueeze(-1), dim=-1)
    pred_delta = torch.sum(sample_dtimes * density * dt.unsqueeze(-1), dim=-1) / denom.clamp(min=1e-12)
    fallback = 1.0 / target_lam[..., 0].clamp(min=1e-8)
    pred_delta = torch.where(denom > 1e-12, pred_delta, fallback)
    diff = pred_delta[target_mask] - label_delta[target_mask]
    rmse = float(torch.sqrt(torch.mean(diff * diff)).item())
    mae = float(torch.mean(torch.abs(diff)).item())
    count = int(target_mask.sum().item())
    if not (math.isfinite(rmse) and math.isfinite(mae)):
        return None
    return rmse, mae, count


def _intensity_free_target_mark_loss(model: Any, batch: Any, target_event_id: int) -> tuple[Any, int]:
    import torch
    from torch.distributions import Categorical
    from easy_tpp.model.torch_model.torch_intensity_free import (
        LogNormalMixtureDistribution,
        clamp_preserve_gradients,
    )

    time_seqs, time_delta_seqs, type_seqs, batch_non_pad_mask, _ = batch
    context = model.forward(time_delta_seqs[:, :-1], type_seqs[:, :-1])
    raw_params = model.linear(context)
    locs = raw_params[..., :model.num_mix_components]
    log_scales = raw_params[..., model.num_mix_components:(2 * model.num_mix_components)]
    log_weights = raw_params[..., (2 * model.num_mix_components):]

    log_scales = clamp_preserve_gradients(log_scales, -5.0, 3.0)
    log_weights = torch.log_softmax(log_weights, dim=-1)
    inter_time_dist = LogNormalMixtureDistribution(
        locs=locs,
        log_scales=log_scales,
        log_weights=log_weights,
        mean_log_inter_time=model.mean_log_inter_time,
        std_log_inter_time=model.std_log_inter_time,
    )
    target_mask = torch.logical_and(batch_non_pad_mask[:, 1:].bool(), type_seqs[:, 1:] == int(target_event_id))
    target_mask_f = target_mask.to(dtype=time_delta_seqs.dtype)
    inter_times = time_delta_seqs[:, 1:].clamp(min=1e-5)
    time_ll = inter_time_dist.log_prob(inter_times) * target_mask_f
    mark_logits = torch.log_softmax(model.mark_linear(context), dim=-1)
    mark_ll = Categorical(logits=mark_logits).log_prob(type_seqs[:, 1:]) * target_mask_f
    loss = -(time_ll + mark_ll).sum()
    return loss, int(target_mask.sum().item())


def _target_mark_loss(model: Any, batch: Any, target_event_id: int) -> tuple[Any, int]:
    if model.__class__.__name__ == "IntensityFree":
        return _intensity_free_target_mark_loss(model, batch, int(target_event_id))
    original_compute_loglikelihood = model.compute_loglikelihood
    model.compute_loglikelihood = _target_mark_compute_loglikelihood(model, int(target_event_id))
    try:
        return model.loglike_loss(batch)
    finally:
        model.compute_loglikelihood = original_compute_loglikelihood


def _run_target_mark_test_epoch(runner: Any, target_event_id: int, *, prediction_grid_size: int) -> dict[str, Any]:
    import math
    import torch

    device = runner.model_wrapper.device
    model = runner.model_wrapper.model
    if int(target_event_id) < 0 or int(target_event_id) >= int(model.num_event_types):
        raise ValueError(f"target_event_id={target_event_id} outside model event type range")

    total_loss = 0.0
    total_target_events = 0
    sq_error = 0.0
    abs_error = 0.0
    time_count = 0
    model.eval()
    with torch.no_grad():
        for batch_encoding in runner._data_loader.test_loader():
            batch = batch_encoding.to(device).values()
            loss, num_target_events = _target_mark_loss(model, batch, int(target_event_id))
            total_loss += float(loss.item())
            total_target_events += int(num_target_events)

            grid_metrics = None
            if model.__class__.__name__ != "IntensityFree":
                grid_metrics = _target_mark_time_metrics_from_grid(
                    model,
                    batch,
                    int(target_event_id),
                    grid_size=int(prediction_grid_size),
                )
            if grid_metrics is not None:
                batch_rmse, batch_mae, batch_count = grid_metrics
                sq_error += float(batch_rmse * batch_rmse * batch_count)
                abs_error += float(batch_mae * batch_count)
                time_count += int(batch_count)
            elif model.event_sampler:
                pred_dtime, pred_type = model.predict_one_step_at_every_event(batch=batch)
                label_dtime = batch[1][:, 1:]
                label_type = batch[2][:, 1:]
                label_mask = batch[3][:, 1:].bool()
                target_mask = torch.logical_and(label_mask, label_type == int(target_event_id))
                if bool(target_mask.any()):
                    diff = pred_dtime[target_mask] - label_dtime[target_mask]
                    sq_error += float(torch.sum(diff * diff).item())
                    abs_error += float(torch.sum(torch.abs(diff)).item())
                    time_count += int(target_mask.sum().item())

    nll = None if total_target_events == 0 else float(total_loss / float(total_target_events))
    rmse = None if time_count == 0 else float(math.sqrt(sq_error / float(time_count)))
    mae = None if time_count == 0 else float(abs_error / float(time_count))
    return {
        "loglike": None if nll is None else -float(nll),
        "num_events": int(total_target_events),
        "nll": nll,
        "rmse": rmse,
        "mae": mae,
        "acc": None,
        "time_count": int(time_count),
    }

def _write_config(
    *,
    config_path: Path,
    model: str,
    train_path: str,
    dev_path: str,
    test_path: str,
    output_dir: Path,
    seed: int,
    gpu: int,
    batch_size: int,
    max_epoch: int,
    learning_rate: float,
    hidden_size: int,
    time_emb_size: int,
    num_layers: int,
    num_heads: int,
    mc_samples: int,
    max_len: int | None,
) -> str:
    dim_process = _infer_dim_process(train_path)
    exp_id = f"{model}_realworld"
    data_specs: dict[str, Any] = {
        "num_event_types": int(dim_process),
        "pad_token_id": int(dim_process),
        "padding_side": "right",
        "truncation_side": "right",
    }
    if max_len is not None and int(max_len) > 0:
        data_specs["padding_strategy"] = "max_length"
        data_specs["truncation_strategy"] = "longest_first"
        data_specs["max_len"] = int(max_len)
    payload = {
        "pipeline_config_id": "runner_config",
        "data": {
            "realworld": {
                "data_format": "pkl",
                "train_dir": str(Path(train_path).resolve()),
                "valid_dir": str(Path(dev_path).resolve()),
                "test_dir": str(Path(test_path).resolve()),
                "data_specs": data_specs,
            }
        },
        exp_id: {
            "base_config": {
                "stage": "train",
                "backend": "torch",
                "dataset_id": "realworld",
                "runner_id": "std_tpp",
                "model_id": str(model),
                "base_dir": str((output_dir / "easytpp_checkpoints" / f"{model}_seed{seed}").resolve()),
            },
            "trainer_config": {
                "batch_size": int(batch_size),
                "max_epoch": int(max_epoch),
                "shuffle": False,
                "optimizer": "adam",
                "learning_rate": float(learning_rate),
                "valid_freq": 1,
                "use_tfb": False,
                "metrics": ["acc", "rmse"],
                "seed": int(seed),
                "gpu": int(gpu),
            },
            "model_config": {
                "hidden_size": int(hidden_size),
                "time_emb_size": int(time_emb_size),
                "num_layers": int(num_layers),
                "num_heads": int(num_heads),
                "sharing_param_layer": False,
                "loss_integral_num_sample_per_step": int(mc_samples),
                "dropout_rate": 0.0,
                "use_ln": False,
                "thinning": {
                    "num_seq": 10,
                    "num_sample": 1,
                    "num_exp": 500,
                    "look_ahead_time": 10,
                    "patience_counter": 5,
                    "over_sample_rate": 5,
                    "num_samples_boundary": 5,
                    "dtime_max": 5,
                },
            },
        },
    }
    if str(model) == "IntensityFree":
        payload[exp_id]["model_config"]["model_specs"] = {"num_mix_components": 3}
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False))
    return exp_id


def _import_easytpp() -> tuple[Any, Any, Any]:
    if str(EASYTPP_ROOT) not in sys.path:
        sys.path.insert(0, str(EASYTPP_ROOT))
    try:
        from easy_tpp.config_factory import Config
        from easy_tpp.runner import Runner
        from easy_tpp.utils import RunnerPhase
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "EasyTPP official wrapper could not import required dependencies. "
            "Install EasyTPP requirements in the active environment, e.g. "
            "`pip install -r external/official_baselines/EasyTemporalPointProcess/requirements.txt`."
        ) from exc
    return Config, Runner, RunnerPhase


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an official EasyTPP model and emit benchmark metrics JSON.")
    ap.add_argument("--model", required=True, choices=["RMTPP", "NHP", "SAHP", "THP", "AttNHP", "IntensityFree"])
    ap.add_argument("--train", required=True)
    ap.add_argument("--dev", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--seed", type=int, default=111)
    ap.add_argument("--output", required=True)
    ap.add_argument("--gpu", type=int, default=-1)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_epoch", type=int, default=20)
    ap.add_argument("--learning_rate", type=float, default=1e-3)
    ap.add_argument("--hidden_size", type=int, default=64)
    ap.add_argument("--time_emb_size", type=int, default=16)
    ap.add_argument("--num_layers", type=int, default=2)
    ap.add_argument("--num_heads", type=int, default=2)
    ap.add_argument("--mc_samples", type=int, default=20)
    ap.add_argument("--max_len", type=int)
    ap.add_argument("--evaluation_scope", default="marked_sequence")
    ap.add_argument("--target_event_id", type=int)
    ap.add_argument("--target_event_label")
    ap.add_argument("--target_prediction_grid_size", type=int, default=128)
    args = ap.parse_args()

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    config_path = output_path.parent / f"{args.model}_seed{int(args.seed)}_easytpp_config.yaml"
    exp_id = _write_config(
        config_path=config_path,
        model=str(args.model),
        train_path=args.train,
        dev_path=args.dev,
        test_path=args.test,
        output_dir=output_path.parent,
        seed=int(args.seed),
        gpu=int(args.gpu),
        batch_size=int(args.batch_size),
        max_epoch=int(args.max_epoch),
        learning_rate=float(args.learning_rate),
        hidden_size=int(args.hidden_size),
        time_emb_size=int(args.time_emb_size),
        num_layers=int(args.num_layers),
        num_heads=int(args.num_heads),
        mc_samples=int(args.mc_samples),
        max_len=args.max_len,
    )

    Config, Runner, RunnerPhase = _import_easytpp()
    total_start = time.perf_counter()
    config = Config.build_from_yaml_file(str(config_path), experiment_id=exp_id)
    runner = Runner.build_from_config(config)
    train_start = time.perf_counter()
    runner.run()
    train_runtime_sec = float(time.perf_counter() - train_start)
    saved_model_dir = runner.runner_config.base_config.specs.get("saved_model_dir")
    if saved_model_dir:
        try:
            runner._load_model(saved_model_dir)
        except Exception:
            pass
    inference_start = time.perf_counter()
    full_test_metrics = runner.run_one_epoch(runner._data_loader.test_loader(), RunnerPhase.VALIDATE)
    loglike = full_test_metrics.get("loglike")
    rmse = full_test_metrics.get("rmse")
    mae = None
    acc = full_test_metrics.get("acc")
    if args.target_event_id is not None:
        target_test_metrics = _run_target_mark_test_epoch(
            runner,
            int(args.target_event_id),
            prediction_grid_size=int(args.target_prediction_grid_size),
        )
        loglike = target_test_metrics.get("loglike")
        rmse = target_test_metrics.get("rmse")
        mae = target_test_metrics.get("mae")
        acc = target_test_metrics.get("acc")
    else:
        target_test_metrics = {}
    inference_runtime_sec = float(time.perf_counter() - inference_start)
    runtime_sec = float(time.perf_counter() - total_start)
    input_manifest = _read_input_manifest(args.train)
    payload = {
        "nll": None if loglike is None else -float(loglike),
        "time_mae": None if args.target_event_id is None else mae,
        "time_rmse": None if rmse is None else float(rmse),
        "type_acc": None if args.target_event_id is not None or acc is None else float(acc),
        "topk": {},
        "runtime_sec": runtime_sec,
        "training_time_sec": train_runtime_sec,
        "inference_time_sec": inference_runtime_sec,
        "model_metadata": {
            "source_url": "https://github.com/ant-research/EasyTemporalPointProcess",
            "implementation_type": "official_external",
            "evaluation_scope": str(args.evaluation_scope),
            "target_event_id": args.target_event_id,
            "target_event_label": args.target_event_label,
            "easytpp_root": str(EASYTPP_ROOT),
            "generated_config": str(config_path),
            "input_manifest": input_manifest,
            "training_time_sec": train_runtime_sec,
            "inference_time_sec": inference_runtime_sec,
            "raw_test_metrics": {str(k): (float(v) if isinstance(v, (int, float)) else str(v)) for k, v in full_test_metrics.items()},
            "target_mark_test_metrics": {
                str(k): (float(v) if isinstance(v, (int, float)) else str(v))
                for k, v in target_test_metrics.items()
            },
        },
    }
    output_path.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
