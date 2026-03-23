"""Script to generate synthetic data for HNSTPP-Refactored (multiplicative inhibition)."""

import sys
import pickle
import yaml
import numpy as np

sys.path.insert(0, '/workspace')

from synthetic import create_rules_from_config, generate_exp_data


def load_config(path):
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def main():
    config_path = './data/config_synthetic.yaml'
    if len(sys.argv) > 1:
        config_path = sys.argv[1]
    config = load_config(config_path)
    
    print("="*50)
    print("Generating Synthetic Data for HNSTPP-Refactored")
    print("="*50)
    print(f"Num samples: {config.get('num_samples', 5000)}")
    print(f"Time horizon: {config.get('time_horizon', 100.0)}")
    print(f"Num event types: {config.get('num_event_types', 4)}")
    print(f"Num rules: {len(config.get('rules', []))}")
    print(f"Seed: {config.get('seed', None)}")
    print()
    
    # Create rules
    rules = create_rules_from_config(config)
    
    print("Rules created:")
    for rule in rules:
        print(f"  Rule {rule.rule_id}: target={rule.target}, W_pos={rule.W_pos:.2f}, W_neg={rule.W_neg:.2f}")
    print()
    
    # Generate data
    print("Generating data...")
    # Clean up kwargs generally used for old model
    data = generate_exp_data(
        rules=rules,
        num_samples=config.get('num_samples', 5000),
        time_horizon=config.get('time_horizon', 100.0),
        base_intensities=config.get('base_intensity', {0: 0.1}),
        max_len=config.get('max_len', 512),
        seed=config.get('seed'),
    )
    
    print(f"Generated {len(data)} sequences")
    
    # Compute statistics
    seq_lens = [len(d['time']) for d in data]
    print(f"Sequence lengths: min={min(seq_lens)}, max={max(seq_lens)}, mean={np.mean(seq_lens):.1f}")
    
    # Event type distribution
    all_events = []
    for d in data:
        all_events.extend(d['event'])
    
    print("Event type distribution:")
    for k in sorted(config.get('base_intensity', {}).keys()):
        count = all_events.count(k)
        print(f"  Type {k}: {count} ({100*count/len(all_events):.1f}%)")
    
    # Time differences
    all_dts = []
    for d in data:
        times = d['time']
        if len(times) > 1:
            all_dts.extend([times[i+1] - times[i] for i in range(len(times)-1)])
    
    mean_dt = np.mean(all_dts)
    print(f"Mean time difference: {mean_dt:.4f}")
    
    # Split into train/val/test
    np.random.shuffle(data)
    n = len(data)
    train_data = data[:int(0.8*n)]
    val_data = data[int(0.8*n):int(0.9*n)]
    test_data = data[int(0.9*n):]
    
    print(f"\nSplit: train={len(train_data)}, val={len(val_data)}, test={len(test_data)}")
    
    # Save
    output_path = config.get('path', {}).get('output_path', './data/synthetic_data.pkl')
    
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
    
    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
    
    print(f"\nSaved to {output_path}")
    print("="*50)


if __name__ == "__main__":
    main()
