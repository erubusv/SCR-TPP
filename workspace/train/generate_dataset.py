import pickle
import numpy as np
import os
import argparse
from data.synthetic import generate_complex_data, ComplexRule
import yaml


def load_config(config_path):
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)
    

def parse_rules(rules_config):
    rule_objects = []
    
    for r_conf in rules_config:
        source_configs = {}
        for src_id, params in r_conf['sources'].items():
            source_configs[int(src_id)] = params
            
        rule = ComplexRule(
            rule_id=int(r_conf['rule_id']),
            source_configs=source_configs,
            target=int(r_conf['target']),
            base_prob=float(r_conf.get('base_prob', 1.0))
        )
        rule_objects.append(rule)
        
    return rule_objects


def parse_interactions(inter_config):
    interactions = []
    if not inter_config:
        return interactions

    for inter in inter_config:
        interactions.append({
            'src': int(inter['src']),
            'tgt': int(inter['tgt']),
            'factor': float(inter['factor']),
            'beta': float(inter['beta']),
            'sources': [int(inter['src'])] 
        })
    return interactions


def create_ground_truth_rules(rules, interactions):
    if rules is None:
        rule1 = ComplexRule(
            rule_id=1,
            source_configs={0: {'mu': 2.0, 'std': 0.5}},
            target=1,
            base_prob=0.8
        )
        rule2 = ComplexRule(
            rule_id=2,
            source_configs={
                2: {'mu': 3.0, 'std': 0.5},
                3: {'mu': 1.0, 'std': 0.2}
            },
            target=4,
            base_prob=0.9
        )

        rules = [rule1, rule2]
    
    if interactions is None:
        interactions = [
            {'src': 1, 'tgt': 2, 'factor': 0.1, 'beta': 0.5} 
        ]
    
    return rules, interactions

def save_dataset(output_path, num_samples, time_horizon, base_intensities: dict, rules, interactions):
    rules, interactions = create_ground_truth_rules(rules, interactions)
    base_intensities = base_intensities

    print(f"Generating {num_samples} sequences...")
    
    raw_data = generate_complex_data(
        rules=rules,
        interactions=interactions,
        num_samples=num_samples,
        time_horizon=time_horizon,
        base_intensities=base_intensities
    )

    n = len(raw_data)
    n_train = int(n * 0.7)
    n_val = int(n * 0.1)
    
    dataset = {
        'train': raw_data[:n_train],
        'val': raw_data[n_train:n_train+n_val],
        'test': raw_data[n_train+n_val:],
        'metadata': {
            'num_types': 5,
            'time_horizon': time_horizon,
            'ground_truth_rules': str(rules),
            'ground_truth_interactions': str(interactions)
        }
    }

    with open(output_path, 'wb') as f:
        pickle.dump(dataset, f)
        
    print(f"Dataset saved to {output_path}")
    print(f"- Train: {len(dataset['train'])}")
    print(f"- Val:   {len(dataset['val'])}")
    print(f"- Test:  {len(dataset['test'])}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./workspace/train/config.yaml', help='Path to the configuration YAML file.')
    args = parser.parse_args()

    config = load_config(args.config)
    settings = config['settings']
    rules = parse_rules(config['rules'])
    interactions = parse_interactions(config.get('interactions', []))
    base_intensities = {int(k): v for k, v in settings['base_intensities'].items()}

    save_dataset(
        output_path=settings['output_file'],
        num_samples=settings['num_samples'],
        time_horizon=settings['time_horizon'],
        base_intensities=base_intensities,
        rules=rules,
        interactions=interactions
    )
