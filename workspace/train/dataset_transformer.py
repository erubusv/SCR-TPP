import pickle
import torch
from torch.utils.data import Dataset
import numpy as np


class EventSeqDataset(Dataset):
    def __init__(self, path, split='train'):
        with open(path, 'rb') as f:
            ds = pickle.load(f)
        self.data = ds[split]
        self.metadata = ds.get('metadata', {})

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        times = np.array(seq['time'], dtype=np.float32)
        types = np.array(seq['event'], dtype=np.int64)
        return {'times': times, 'types': types}


def collate_fn(batch, pad_id):
    batch_sizes = [len(x['times']) for x in batch]
    L = max(batch_sizes)
    B = len(batch)

    types = torch.full((B, L), pad_id, dtype=torch.long)
    dt = torch.zeros((B, L), dtype=torch.float)
    times = torch.zeros((B, L), dtype=torch.float)
    mask = torch.ones((B, L), dtype=torch.bool)  # True: padding
    tgt_dt = torch.zeros((B, L), dtype=torch.float)
    tgt_type = torch.full((B, L), pad_id, dtype=torch.long)

    for i, item in enumerate(batch):
        t = torch.from_numpy(item['times'])
        ty = torch.from_numpy(item['types'])
        ln = len(t)
        types[i, :ln] = ty
        dt_i = torch.zeros(ln)
        dt_i[1:] = t[1:] - t[:-1]
        dt[i, :ln] = dt_i
        # store absolute times (0 for padding positions)
        times[i, :ln] = t
        mask[i, :ln] = False

        if ln > 1:
            tgt_dt[i, :ln-1] = t[1:] - t[:-1]
            tgt_type[i, :ln-1] = ty[1:]
            tgt_type[i, ln-1] = pad_id
            tgt_dt[i, ln-1] = 0.0
        else:
            tgt_type[i, 0] = pad_id
            tgt_dt[i, 0] = 0.0

    return {
        'types': types,
        'dt': dt,
        'times': times,
        'mask': mask,
        'tgt_dt': tgt_dt,
        'tgt_type': tgt_type,
        'lengths': torch.tensor(batch_sizes, dtype=torch.long)
    }
