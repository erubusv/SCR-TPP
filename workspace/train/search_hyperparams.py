"""
Simple hyperparameter search script for HNSTPP and TransformerHawkes.
Searches over learning rates and regularizer weights.
Saves results to CSV for each parameter combination.

USAGE:
  python workspace/train/search_hyperparams.py --model hnstpp --epochs 5
  python workspace/train/search_hyperparams.py --model transformer --epochs 5
  
  With custom ranges:
  python workspace/train/search_hyperparams.py --model hnstpp --lrs 1e-4,1e-3 --min_lrs 1e-5
"""

import os
import sys
import argparse
import subprocess
import pandas as pd
import yaml
import re
import time
from pathlib import Path
from datetime import datetime

def create_config(model_name, lr, min_lr, lambda_ortho, lambda_sparse, lambda_interaction, 
                  epochs, batch_size, config_path, pad_token_id=5):
    """Create a temporary config file for training."""
    
    base_config = {
        'path': {
            'data_path': 'data/synthetic_data.pkl'
        },
        'num_types': 5,
        'num_event_types': 5,
        'pad_token_id': pad_token_id
    }
    
    if model_name.lower() == 'hnstpp':
        base_config['hnstpp'] = {
            'num_rules': 2,
            'embed_dim': 32,
            'num_heads': 3,
            'tau': 0.5,
            'mc_samples': 10,
            'lambda_ortho': lambda_ortho,
            'lambda_sparse': lambda_sparse,
            'lambda_interaction': lambda_interaction,
            'start_learning_rate': lr,
            'min_learning_rate': min_lr,
            'batch_size': batch_size,
            'epochs': epochs,
            'pad_token_id': pad_token_id,
            'model_save_path': './workspace/saved_models/synthetic/HNSTPP',
            'top_k': 2,
            'gpu': 0
        }
    else:  # TransformerHawkes
        base_config['transformer'] = {
            'embed_dim': 128,
            'num_heads': 4,
            'n_layers': 2,
            'start_learning_rate': lr,
            'min_learning_rate': min_lr,
            'batch_size': batch_size,
            'epochs': epochs,
            'pad_token_id': pad_token_id,
            'model_save_path': './workspace/saved_models/synthetic/transformer',
            'top_k': 2,
            'gpu': 1
        }
    
    with open(config_path, 'w') as f:
        yaml.dump(base_config, f)


def run_training(model_name, lr, min_lr, lambda_ortho, lambda_sparse, lambda_interaction, 
                 epochs=5, batch_size=32, pad_token_id=5):
    """Run training for a single hyperparameter combination."""
    
    timestamp = int(time.time() * 1000000) % 1000000
    config_path = f'/tmp/temp_config_{timestamp}.yaml'
    
    try:
        # Create temporary config
        create_config(model_name, lr, min_lr, lambda_ortho, lambda_sparse, lambda_interaction, 
                      epochs, batch_size, config_path, pad_token_id=pad_token_id)
        
        if model_name.lower() == 'hnstpp':
            script = 'workspace/train/train_hnstpp.py'
        else:
            script = 'workspace/train/train_transformer_hawkes.py'
        
        cmd = ['python', script, '--config', config_path]
        
        # Set environment to avoid MKL threading issues
        env = os.environ.copy()
        env['MKL_THREADING_LAYER'] = 'GNU'
        
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=3600, env=env)
        output = result.stdout + result.stderr
        
        # Look for known validation patterns in output
        val_loss = None
        for line in output.split('\n'):
            low = line.lower()
            # HNSTPP prints: "Dev NLL: <val>"
            if 'dev nll:' in low:
                numbers = re.findall(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', line)
                if numbers:
                    val_loss = float(numbers[-1])
            # Transformer prints: "... | val nll=4.2350 mae=..."
            elif 'val nll=' in low:
                m = re.search(r'val nll=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)', low)
                if m:
                    try:
                        val_loss = float(m.group(1))
                    except:
                        pass
        
        # Clean up
        if os.path.exists(config_path):
            os.remove(config_path)
        
        return val_loss
        
    except Exception as e:
        print(f"Error in run_training: {e}")
        if os.path.exists(config_path):
            os.remove(config_path)
        return None


def search_hyperparams(model_name='hnstpp', 
                       lr_values=None,
                       min_lr_values=None,
                       lambda_ortho_values=None,
                       lambda_sparse_values=None,
                       lambda_interaction_values=None,
                       epochs=5,
                       output_dir='./search_results',
                       pad_token_id=5):
    """Run grid search over hyperparameters."""
    
    if lr_values is None:
        lr_values = [1e-4, 5e-4, 1e-3, 5e-3]
    if min_lr_values is None:
        min_lr_values = [1e-6, 1e-5, 1e-4]
    if lambda_ortho_values is None:
        lambda_ortho_values = [0.0, 0.001, 0.01] if model_name.lower() == 'hnstpp' else [0.0]
    if lambda_sparse_values is None:
        lambda_sparse_values = [0.0, 0.001, 0.01] if model_name.lower() == 'hnstpp' else [0.0]
    if lambda_interaction_values is None:
        lambda_interaction_values = [0.0, 0.001, 0.01] if model_name.lower() == 'hnstpp' else [0.0]
    
    # Create output directory
    Path(output_dir).mkdir(parents=True, exist_ok=True)
    
    # Results list
    results = []
    total_combinations = (len(lr_values) * len(min_lr_values) * 
                         len(lambda_ortho_values) * len(lambda_sparse_values) * 
                         len(lambda_interaction_values))
    
    print(f"\n{'='*60}")
    print(f"Search: {model_name.upper()}")
    print(f"Total combinations: {total_combinations}")
    print(f"{'='*60}\n")
    
    count = 0
    for lr in lr_values:
        for min_lr in min_lr_values:
            for lambda_o in lambda_ortho_values:
                for lambda_s in lambda_sparse_values:
                    for lambda_i in lambda_interaction_values:
                        count += 1
                        print(f"[{count}/{total_combinations}] lr={lr:.0e} min_lr={min_lr:.0e} "
                              f"ortho={lambda_o} sparse={lambda_s} inter={lambda_i}", end="")
                        
                        val_loss = run_training(
                            model_name=model_name,
                            lr=lr,
                            min_lr=min_lr,
                            lambda_ortho=lambda_o,
                            lambda_sparse=lambda_s,
                            lambda_interaction=lambda_i,
                            epochs=epochs,
                            pad_token_id=pad_token_id
                        )
                        
                        if val_loss is not None:
                            results.append({
                                'learning_rate': lr,
                                'min_learning_rate': min_lr,
                                'lambda_ortho': lambda_o,
                                'lambda_sparse': lambda_s,
                                'lambda_interaction': lambda_i,
                                'val_loss': val_loss
                            })
                            print(f" → loss: {val_loss:.6f}")
                        else:
                            print(f" → FAILED")
    
    # Save results to CSV
    df = pd.DataFrame(results)
    output_file = os.path.join(output_dir, f'{model_name}_search_results.csv')
    df.to_csv(output_file, index=False)
    
    # Print summary
    if len(df) > 0:
        df_sorted = df.sort_values('val_loss')
        print(f"\n{'='*60}")
        print(f"Results saved to {output_file}")
        print(f"Top 5 configurations:")
        print(df_sorted.head(5).to_string(index=False))
        print(f"{'='*60}\n")
    
    return df


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Hyperparameter search for TPP models')
    parser.add_argument('--model', type=str, default='hnstpp', 
                       choices=['hnstpp', 'transformer'],
                       help='Model to search hyperparameters for')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs per run')
    parser.add_argument('--num_pad_id', type=int, default=5,
                       help='index of the padding token')
    parser.add_argument('--output_dir', type=str, default='./search_results',
                       help='Output directory for results')
    parser.add_argument('--lrs', type=str, default='1e-4,5e-4,1e-3,5e-3',
                       help='Comma-separated learning rates to search')
    parser.add_argument('--min_lrs', type=str, default='1e-6,1e-5,1e-4',
                       help='Comma-separated minimum learning rates')
    parser.add_argument('--lambda_ortho', type=str, default='0,0.001,0.01',
                       help='Comma-separated lambda_ortho values (HNSTPP only)')
    parser.add_argument('--lambda_sparse', type=str, default='0,0.001,0.01',
                       help='Comma-separated lambda_sparse values (HNSTPP only)')
    parser.add_argument('--lambda_inter', type=str, default='0,0.001,0.01',
                       help='Comma-separated lambda_interaction values (HNSTPP only)')
    
    args = parser.parse_args()
    
    # Parse comma-separated values
    lr_values = [float(x) for x in args.lrs.split(',')]
    min_lr_values = [float(x) for x in args.min_lrs.split(',')]
    lambda_ortho_values = [float(x) for x in args.lambda_ortho.split(',')]
    lambda_sparse_values = [float(x) for x in args.lambda_sparse.split(',')]
    lambda_inter_values = [float(x) for x in args.lambda_inter.split(',')]
    
    df = search_hyperparams(
        model_name=args.model,
        lr_values=lr_values,
        min_lr_values=min_lr_values,
        lambda_ortho_values=lambda_ortho_values,
        lambda_sparse_values=lambda_sparse_values,
        lambda_interaction_values=lambda_inter_values,
        epochs=args.epochs,
        output_dir=args.output_dir,
        pad_token_id=args.num_pad_id
    )
