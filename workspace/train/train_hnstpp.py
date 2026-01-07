import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
import pickle
import argparse
import pandas as pd

from workspace.models.HNSTPP import HNSTPP
from workspace.train.generate_dataset import load_config

class TPPDataset(Dataset):
    def __init__(self, data_list):
        self.data = data_list

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

def get_collate_fn(pad_id):
    def collate_fn(batch):
        max_len = max(len(x['time']) for x in batch)
        
        padded_input_ids = []
        padded_time_diffs = []
        attention_masks = []
        
        for item in batch:
            times = np.array(item['time'])
            events = np.array(item['event'])
            seq_len = len(times)
            
            if seq_len > 0:
                time_diffs = np.concatenate([[times[0]], np.diff(times)])
            else:
                time_diffs = np.array([])
            
            pad_len = max_len - seq_len
            
            ids = np.pad(events, (0, pad_len), 'constant', constant_values=pad_id)
            padded_input_ids.append(ids)

            diffs = np.pad(time_diffs, (0, pad_len), 'constant', constant_values=0.0)
            padded_time_diffs.append(diffs)

            mask = np.concatenate([np.ones(seq_len), np.zeros(pad_len)])
            attention_masks.append(mask)
            
        return {
            'input_ids': torch.LongTensor(np.array(padded_input_ids)),
            'time_diffs': torch.FloatTensor(np.array(padded_time_diffs)),
            'attention_mask': torch.FloatTensor(np.array(attention_masks))
        }
    return collate_fn

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='./workspace/train/train_config.yaml', help='Path to the configuration YAML file.')
    args = parser.parse_args()

    config = load_config(args.config)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if not os.path.exists(config['path']['data_path']):
        print(f"Error: File {config['path']['data_path']} not found.")
        return

    with open(config['path']['data_path'], 'rb') as f:
        dataset_dict = pickle.load(f)
        
    train_data = TPPDataset(dataset_dict['train'])
    val_data = TPPDataset(dataset_dict['val'])
    metadata = dataset_dict.get('metadata', {'num_types': 5})
    
    print(f"Train size: {len(train_data)}, Dev size: {len(val_data)}")
    
    collate_fn = get_collate_fn(config['pad_token_id'])
    
    train_loader = DataLoader(train_data, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_data, batch_size=config['batch_size'], shuffle=False, collate_fn=collate_fn)

    model = HNSTPP(config).to(device)
    optimizer = optim.Adam(model.parameters(), lr=float(config['learning_rate']))
    
    print("\nStarting Training...")
    best_val_loss = float('inf')
    
    for epoch in range(config['epochs']):
        model.train()
        total_loss = 0
        nll_loss_sum = 0

        scaler = torch.cuda.amp.GradScaler()
        
        for batch in train_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            optimizer.zero_grad()

            with torch.cuda.amp.autocast():
                output = model(batch)
                losses = model.compute_loss(batch, output)
                loss = losses['total_loss']

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            nll_loss_sum += losses['nll_loss']
            
        avg_train_loss = total_loss / len(train_loader)
        avg_nll = nll_loss_sum / len(train_loader)
        
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                batch = {k: v.to(device) for k, v in batch.items()}
                output = model(batch)
                losses = model.compute_loss(batch, output)
                val_loss += losses['nll_loss']
                
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"Epoch {epoch+1:02d} | Train Loss: {avg_train_loss:.4f} (NLL: {avg_nll:.4f}) | Dev NLL: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            os.makedirs(config["path"]["model_save_path"], exist_ok=True)
            torch.save(model.state_dict(), f'{config["path"]["model_save_path"]}/best_model.pt')

    print("\n" + "="*40)
    print("INTERPRETATION OF LEARNED RULES")
    print("="*40)
    
    model.eval()
    
    event_names = [f"Type {i}" for i in range(config['num_event_types'])]
    df_rules, df_interactions = model.explain_model_parameters(event_names=event_names)

    df_rules.to_pickle(f"{config['path']['model_save_path']}/rule_df.pkl")
    df_interactions.to_pickle(f"{config['path']['model_save_path']}/interaction_df.pkl")
    
    print("\n[Learned Logic Rules]")
    print(df_rules)
    
    print("\n[Learned Interactions]")
    if not df_interactions.empty:
        print(df_interactions.head())
    else:
        print("No significant interactions found.")

if __name__ == "__main__":
    main()