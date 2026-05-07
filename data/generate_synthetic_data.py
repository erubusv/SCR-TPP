import pickle
import sys
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from data.synthetic import create_rules_from_config, generate_canonical_loglink_data


DEFAULT_CONFIG = "./data/config_synthetic.yaml"
DEFAULT_OUTPUT = "./data/generated_synthetic.pkl"


def load_config(path: str | Path) -> dict:
    with Path(path).open() as f:
        return yaml.safe_load(f)


def main() -> None:
    config_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(DEFAULT_CONFIG)
    config = load_config(config_path)

    print("=" * 50)
    print("Generating Canonical Synthetic Data")
    print("=" * 50)
    print(f"Num samples: {config.get('num_samples', 5000)}")
    print(f"Time horizon: {config.get('time_horizon', 100.0)}")
    print(f"Num event types: {config.get('num_event_types', 4)}")
    print(f"Num rules: {len(config.get('rules', []))}")
    print(f"Seed: {config.get('seed', None)}")
    print()

    rules = create_rules_from_config(config)

    print("Rules created:")
    for rule in rules:
        print(f"  Rule {rule.rule_id}: target={rule.target}, W_pos={rule.W_pos:.2f}, W_neg={rule.W_neg:.2f}")
    print()

    print("Generating data...")
    data = generate_canonical_loglink_data(
        rules=rules,
        num_samples=config.get('num_samples', 5000),
        time_horizon=config.get('time_horizon', 100.0),
        base_intensities=config.get('base_intensity', {0: 0.1}),
        max_len=config.get('max_len', 1024),
        seed=config.get('seed'),
    )

    print(f"Generated {len(data)} sequences")

    seq_lens = [len(d['time']) for d in data]
    print(f"Sequence lengths: min={min(seq_lens)}, max={max(seq_lens)}, mean={np.mean(seq_lens):.1f}")

    all_events = []
    for d in data:
        all_events.extend(d['event'])

    print("Event type distribution:")
    for k in sorted(config.get('base_intensity', {}).keys()):
        count = all_events.count(k)
        print(f"  Type {k}: {count} ({100*count/len(all_events):.1f}%)")

    all_dts = []
    for d in data:
        times = d['time']
        if len(times) > 1:
            all_dts.extend([times[i+1] - times[i] for i in range(len(times)-1)])

    mean_dt = np.mean(all_dts)
    print(f"Mean time difference: {mean_dt:.4f}")

    np.random.shuffle(data)
    n = len(data)
    train_data = data[:int(0.8*n)]
    val_data = data[int(0.8*n):int(0.9*n)]
    test_data = data[int(0.9*n):]

    print(f"\nSplit: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")

    output_path = Path(config.get('path', {}).get('output_path', DEFAULT_OUTPUT))
    dataset = {
        'train': train_data,
        'val': val_data,
        'test': test_data,
        'metadata': {
            'num_types': config.get('num_event_types', 4),
            'mean_time_diff': mean_dt,
            'config': config
        }
    }

    with output_path.open('wb') as f:
        pickle.dump(dataset, f)

    print(f"\nSaved to {output_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
