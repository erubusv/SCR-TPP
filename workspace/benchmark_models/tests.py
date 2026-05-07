from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from .adapters.easytpp import sequence_to_gatech_events, write_gatech_pickles
from .core.external import ExternalBaselineSpec, run_external_rule_baseline
from .scr_tpp.synthetic_results import import_ours_result_file
from .core.prediction_external import PredictionBaselineSpec, run_external_prediction_baseline
from .baselines.logical_rule_baselines import run_baseline as run_restricted_baseline
from .adapters.realworld import events_csv_to_sequences
from .core.schema import NormalizedRule, match_rule_sets, parse_rule_string, true_rules_from_config
from .core.summarize import realworld_summary, synthetic_summary
from .adapters.synthetic import SyntheticCase, write_all_as_train_dataset


class BenchmarkHarnessTests(unittest.TestCase):
    def test_rule_string_parser(self) -> None:
        rule = parse_rule_string("A and C and D -> T : inhibition", default_target=7)
        self.assertEqual(rule, NormalizedRule(sources=(0, 2, 3), sign="inhibition", target=7))

    def test_rule_matching(self) -> None:
        truth = [NormalizedRule(sources=(0, 2), sign="excitation", target=7)]
        pred = ["C and A -> T : excitation"]
        out = match_rule_sets(pred, truth, default_target=7)
        self.assertEqual(out["recall"], 1.0)
        self.assertEqual(out["precision"], 1.0)
        self.assertEqual(out["source_jaccard_f1"], 1.0)
        self.assertEqual(out["signed_source_jaccard_f1"], 1.0)

    def test_source_jaccard_partial_overlap(self) -> None:
        truth = [NormalizedRule(sources=(2, 6, 7), sign="excitation", target=7)]
        pred = [NormalizedRule(sources=(2, 3, 7), sign="excitation", target=7)]
        out = match_rule_sets(pred, truth)
        self.assertEqual(out["recall"], 0.0)
        self.assertEqual(out["precision"], 0.0)
        self.assertAlmostEqual(out["source_jaccard_recall"], 0.5)
        self.assertAlmostEqual(out["source_jaccard_precision"], 0.5)
        self.assertAlmostEqual(out["source_jaccard_f1"], 0.5)
        self.assertEqual(len(out["source_jaccard_pairs"]), 1)

    def test_source_jaccard_subset_shadow(self) -> None:
        truth = [NormalizedRule(sources=(2, 6, 7), sign="excitation", target=7)]
        pred = [NormalizedRule(sources=(2, 6), sign="excitation", target=7)]
        out = match_rule_sets(pred, truth)
        self.assertAlmostEqual(out["source_jaccard_f1"], 2.0 / 3.0)

    def test_signed_source_jaccard_requires_same_sign(self) -> None:
        truth = [NormalizedRule(sources=(2, 6, 7), sign="excitation", target=7)]
        pred = [NormalizedRule(sources=(2, 6, 7), sign="inhibition", target=7)]
        out = match_rule_sets(pred, truth)
        self.assertEqual(out["source_jaccard_f1"], 1.0)
        self.assertEqual(out["signed_source_jaccard_f1"], 0.0)

    def test_duplicate_predicted_rules_are_deduplicated_before_jaccard(self) -> None:
        truth = [NormalizedRule(sources=(0, 2), sign="excitation", target=7)]
        pred = [
            {"sources": [2, 0], "sign": "excitation", "target": 7},
            {"sources": [0, 2], "sign": "excitation", "target": 7},
        ]
        out = match_rule_sets(pred, truth)
        self.assertEqual(len(out["predicted_rules"]), 1)
        self.assertEqual(out["recall"], 1.0)
        self.assertEqual(out["precision"], 1.0)

    def test_true_rule_parser_matches_yaml_shape(self) -> None:
        config = Path("data/paper_suite/configs/hetero_source_2000_adjusted/logical_context.yaml")
        if not config.exists():
            self.skipTest("paper suite config not present")
        rules = true_rules_from_config(config)
        self.assertGreaterEqual(len(rules), 1)
        self.assertTrue(all(rule.target == 7 for rule in rules))

    def test_easytpp_gatech_conversion(self) -> None:
        seq = {"time": [1.0, 3.5, 5.0], "event": [2, 2, 1]}
        out = sequence_to_gatech_events(seq)
        self.assertEqual(out[0]["time_since_last_event"], 1.0)
        self.assertEqual(out[1]["time_since_last_event"], 2.5)
        self.assertEqual(out[1]["time_since_last_same_event"], 2.5)
        self.assertEqual(out[2]["type_event"], 1)

    def test_write_gatech_pickles(self) -> None:
        seq = {"time": [1.0], "event": [0]}
        with tempfile.TemporaryDirectory() as td:
            paths = write_gatech_pickles(train=[seq], dev=[], test=[], dim_process=2, output_dir=td)
            with open(paths["train"], "rb") as f:
                data = pickle.load(f)
            self.assertEqual(data["dim_process"], 2)
            self.assertEqual(data["train"][0][0]["type_event"], 0)

    def test_realworld_csv_adapter_keeps_target_under_topk(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            csv_path = Path(td) / "events.csv"
            csv_path.write_text(
                "seq,t,event\n"
                "p1,10,A\n"
                "p1,11,B\n"
                "p1,12,C\n"
                "p2,20,A\n"
                "p2,21,B\n"
                "p2,22,TARGET\n"
            )
            sequences, metadata = events_csv_to_sequences(
                csv_path=csv_path,
                sequence_col="seq",
                time_col="t",
                event_col="event",
                top_k_event_types=2,
                target_event_label="TARGET",
                min_events_per_sequence=1,
            )
            self.assertEqual(len(sequences), 2)
            self.assertIn("TARGET", metadata["event_to_id"])
            self.assertEqual(metadata["dim_process"], 2)

    def test_external_prediction_wrapper_reads_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "metrics.json"
            spec = PredictionBaselineSpec(
                name="DummyTPP",
                source_url="local",
                implementation_type="test_double",
                command=[
                    sys.executable,
                    "-c",
                    (
                        "import json,sys;"
                        "json.dump(dict(nll=1.2, time_mae=0.3, "
                        "time_rmse=0.4, type_acc=0.5, topk=dict(k5=0.9)), open(sys.argv[1], 'w'))"
                    ),
                    "{output_path}",
                ],
            )
            result = run_external_prediction_baseline(
                spec=spec,
                dataset="toy",
                target="target",
                train_path="train.pkl",
                dev_path="dev.pkl",
                test_path="test.pkl",
                output_path=output_path,
                seed=111,
            )
            self.assertEqual(result.nll, 1.2)
            self.assertEqual(result.time_rmse, 0.4)
            self.assertEqual(result.topk["k5"], 0.9)

    def test_external_rule_wrapper_reads_normalized_rules(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            output_path = Path(td) / "rules.json"
            spec = ExternalBaselineSpec(
                name="DummyRule",
                source_url="local",
                implementation_type="test_double",
                command=[
                    sys.executable,
                    "-c",
                    (
                        "import json,sys;"
                        "json.dump(dict(predicted_rules=[dict(sources=[0,2], "
                        "sign='excitation', target=7)]), open(sys.argv[1], 'w'))"
                    ),
                    "{output_path}",
                ],
            )
            rules, metadata = run_external_rule_baseline(
                spec=spec,
                config_path="config.yaml",
                dataset_path="data.pkl",
                output_path=output_path,
                seed=111,
            )
            self.assertEqual(rules, [NormalizedRule(sources=(0, 2), sign="excitation", target=7)])
            self.assertEqual(metadata["returncode"], 0)

    def test_restricted_baseline_writes_normalized_output(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_path = root / "toy.yaml"
            dataset_path = root / "toy.pkl"
            output_path = root / "out.json"
            config_path.write_text(
                "num_event_types: 3\n"
                "time_horizon: 6.0\n"
                "baseline_time_window: 2.0\n"
                "rules:\n"
                "  - target: 2\n"
                "    W_pos: 1.0\n"
                "    W_neg: 0.0\n"
                "    condition:\n"
                "      0: {}\n"
            )
            with dataset_path.open("wb") as f:
                pickle.dump(
                    {
                        "train": [
                            {"time": [0.5, 1.0, 2.0, 4.0], "event": [0, 2, 1, 2]},
                            {"time": [0.2, 0.9, 1.5, 3.0], "event": [1, 0, 2, 2]},
                        ],
                        "val": [],
                        "test": [],
                    },
                    f,
                )
            payload = run_restricted_baseline(
                Namespace(
                    model="CLNN",
                    config=str(config_path),
                    data=str(dataset_path),
                    seed=111,
                    output=str(output_path),
                    device="cpu",
                    grid_size=4,
                    masking="last",
                    epochs=1,
                    refine_epochs=1,
                    em_iters=1,
                    lr=0.03,
                    weight_decay=1e-4,
                    num_formulas=2,
                    max_rules=2,
                    max_rule_length=2,
                    alpha=0.5,
                    tau=0.5,
                    delta=0.0,
                    softmin_rho=20.0,
                    dummy_count=2,
                    laplace_scale=0.5,
                    gamma_threshold=1e-3,
                    cover_threshold=0.5,
                )
            )
            self.assertEqual(payload["model"], "CLNN")
            self.assertIn("predicted_rules", payload)
            self.assertEqual(payload["metadata"]["implementation_type"], "restricted_paper_based_reimplementation_weighted_clock_logic_tpp")

    def test_ours_adapter_reads_single_result_file(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_root = root / "configs"
            config_root.mkdir()
            (config_root / "toy.yaml").write_text(
                "num_event_types: 3\n"
                "rules:\n"
                "  - target: 2\n"
                "    W_pos: 1.0\n"
                "    W_neg: 0.0\n"
                "    condition:\n"
                "      0: {}\n"
                "      1: {}\n"
            )
            result_path = root / "toy_seed111.json"
            result_path.write_text(
                '{"benchmark":"toy","target":2,"dataset_seed":111,'
                '"predicted":["A and B -> T : excitation"],"elapsed_sec":1.5}'
            )
            rows = import_ours_result_file(result_path=result_path, config_root=config_root)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0].recall, 1.0)
            self.assertEqual(rows[0].source_jaccard_f1, 1.0)

    def test_realworld_summary_averages_metrics(self) -> None:
        rows = [
            {"model": "M", "dataset": "D", "target": "T", "nll": 1.0, "time_mae": 2.0, "time_rmse": 3.0, "type_acc": 0.5, "runtime_sec": 3.0},
            {"model": "M", "dataset": "D", "target": "T", "nll": 3.0, "time_mae": 4.0, "time_rmse": 5.0, "type_acc": 0.7, "runtime_sec": 5.0},
        ]
        out = realworld_summary(rows)
        self.assertEqual(out[0]["mean_nll"], 2.0)
        self.assertEqual(out[0]["mean_time_rmse"], 4.0)
        self.assertEqual(out[0]["mean_type_acc"], 0.6)

    def test_synthetic_summary_averages_jaccard_metrics(self) -> None:
        rows = [
            {
                "model": "M",
                "dataset": "D",
                "recall": 1.0,
                "precision": 1.0,
                "f1": 1.0,
                "source_jaccard_recall": 1.0,
                "source_jaccard_precision": 1.0,
                "source_jaccard_f1": 1.0,
                "signed_source_jaccard_recall": 1.0,
                "signed_source_jaccard_precision": 1.0,
                "signed_source_jaccard_f1": 1.0,
                "runtime_sec": 2.0,
                "missing": [],
                "extra": [],
            },
            {
                "model": "M",
                "dataset": "D",
                "recall": 0.0,
                "precision": 0.0,
                "f1": 0.0,
                "source_jaccard_recall": 0.5,
                "source_jaccard_precision": 0.5,
                "source_jaccard_f1": 0.5,
                "signed_source_jaccard_recall": 0.0,
                "signed_source_jaccard_precision": 0.0,
                "signed_source_jaccard_f1": 0.0,
                "runtime_sec": 4.0,
                "missing": [{}],
                "extra": [{}],
            },
        ]
        out = synthetic_summary(rows)
        self.assertEqual(out[0]["mean_source_jaccard_f1"], 0.75)
        self.assertEqual(out[0]["mean_signed_source_jaccard_f1"], 0.5)

    def test_synthetic_adapter_writes_all_sequences_as_train(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            dataset_path = Path(td) / "toy.pkl"
            with dataset_path.open("wb") as f:
                pickle.dump(
                    {
                        "train": [{"time": [1.0], "event": [0]}],
                        "val": [{"time": [2.0], "event": [1]}],
                        "test": [{"time": [3.0], "event": [2]}],
                        "metadata": {"num_types": 3},
                    },
                    f,
                )
            case = SyntheticCase(
                name="toy",
                seed=111,
                config_path=Path(td) / "toy.yaml",
                dataset_path=dataset_path,
            )
            out_path = write_all_as_train_dataset(case, Path(td) / "adapted.pkl")
            with out_path.open("rb") as f:
                adapted = pickle.load(f)
            self.assertEqual(len(adapted["train"]), 3)
            self.assertEqual(len(adapted["val"]), 0)
            self.assertEqual(len(adapted["test"]), 0)
            self.assertEqual(adapted["metadata"]["benchmark_split_strategy"], "all_sequences_as_train")

    def test_synthetic_runner_reuses_cases_for_multiple_external_models(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            root = Path(td)
            config_root = root / "configs"
            dataset_root = root / "datasets" / "seed_111"
            config_root.mkdir(parents=True)
            dataset_root.mkdir(parents=True)
            (config_root / "toy.yaml").write_text(
                "num_event_types: 3\n"
                "rules:\n"
                "  - target: 2\n"
                "    W_pos: 1.0\n"
                "    W_neg: 0.0\n"
                "    condition:\n"
                "      0: {}\n"
            )
            with (dataset_root / "toy.pkl").open("wb") as f:
                pickle.dump({"train": [{"time": [1.0, 2.0], "event": [0, 2]}], "val": [], "test": []}, f)
            manifest = root / "manifest.yaml"
            manifest.write_text(
                "baselines:\n"
                "  - name: A\n"
                "    source_url: local\n"
                "    implementation_type: test_double\n"
                "    command:\n"
                f"      - {sys.executable}\n"
                "      - -c\n"
                "      - \"import json,sys; json.dump({{'predicted_rules':[{{'sources':[0],'sign':'excitation','target':2}}]}}, open(sys.argv[1], 'w'))\"\n"
                "      - \"{output_path}\"\n"
                "  - name: B\n"
                "    source_url: local\n"
                "    implementation_type: test_double\n"
                "    command:\n"
                f"      - {sys.executable}\n"
                "      - -c\n"
                "      - \"import json,sys; json.dump({{'predicted_rules':[{{'sources':[0],'sign':'excitation','target':2}}]}}, open(sys.argv[1], 'w'))\"\n"
                "      - \"{output_path}\"\n"
            )
            out_jsonl = root / "rows.jsonl"
            proc = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "workspace.benchmark_models.runners.synthetic_rule_recovery",
                    "--models",
                    "A,B",
                    "--config_root",
                    str(config_root),
                    "--dataset_root",
                    str(root / "datasets"),
                    "--seeds",
                    "111",
                    "--baseline_manifest",
                    str(manifest),
                    "--out_jsonl",
                    str(out_jsonl),
                ],
                cwd=Path.cwd(),
                env={**dict(), **{"PYTHONPATH": str(Path.cwd())}},
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, proc.stderr)
            rows = [json.loads(line) for line in out_jsonl.read_text().splitlines() if line.strip()]
            self.assertEqual([row["model"] for row in rows], ["A", "B"])

if __name__ == "__main__":
    unittest.main()
