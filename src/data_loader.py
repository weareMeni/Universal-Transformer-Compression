import torch
from torch.utils.data import Dataset, DataLoader
import random
import numpy as np

class DyckDataset(Dataset):
    def __init__(self, num_samples, seq_len, is_train=True):
        self.num_samples = num_samples
        self.seq_len = seq_len
        self.is_train = is_train
        self.test_samples, self.test_labels = list(), list()
        if not is_train:
            for _ in range(num_samples):
                s, l = self._generate_one(seq_len)
                self.test_samples.append(s)
                self.test_labels.append(l)

    def _generate_one(self, length):
        is_balanced = random.random() > 0.5
        open_brackets = list((1, 3, 5))
        close_map = {1: 2, 3: 4, 5: 6}
        seq, stack = list(), list()
        for i in range(length):
            if is_balanced:
                if len(stack) == length - i:
                    seq.append(close_map.get(stack.pop()))
                elif not stack or random.random() > 0.5:
                    op = random.choice(open_brackets)
                    stack.append(op)
                    seq.append(op)
                else:
                    seq.append(close_map.get(stack.pop()))
            else:
                seq.append(random.randint(1, 6))
        return torch.tensor(seq, dtype=torch.long), (1 if is_balanced else 0)

    def __len__(self): return self.num_samples
    def __getitem__(self, idx): 
        return (self.test_samples[idx], self.test_labels[idx]) if not self.is_train else self._generate_one(self.seq_len)

class RecallDataset(Dataset):
    def __init__(self, num_samples, num_pairs, is_train=True):
        self.num_samples = num_samples
        self.num_pairs = num_pairs
        self.is_train = is_train
        self.key_start = 1
        self.val_start = self.key_start + num_pairs
        self.q_tok = self.val_start + num_pairs
        self.test_samples, self.test_labels = list(), list()
        if not is_train:
            for _ in range(num_samples):
                s, l = self._generate_one()
                self.test_samples.append(s)
                self.test_labels.append(l)

    def _generate_one(self):
        keys = random.sample(range(self.key_start, self.val_start), self.num_pairs)
        vals = random.sample(range(self.val_start, self.q_tok), self.num_pairs)
        seq = list()
        for k, v in zip(keys, vals): seq.extend(list((k, v)))
        idx = random.randint(0, self.num_pairs - 1)
        seq.extend(list((self.q_tok, keys[idx]))) 
        return torch.tensor(seq, dtype=torch.long), (vals[idx] - self.val_start)

    def __len__(self): return self.num_samples
    def __getitem__(self, idx): 
        return (self.test_samples[idx], self.test_labels[idx]) if not self.is_train else self._generate_one()

class ListOpsDataset(Dataset):
    def __init__(self, num_samples, max_seq_len=128, max_depth=5, is_train=True):
        self.num_samples = num_samples
        self.max_seq_len = max_seq_len
        self.max_depth = max_depth
        self.is_train = is_train
        self.test_samples, self.test_labels = list(), list()
        if not is_train:
            for _ in range(num_samples):
                tokens, label = self._generate_tree(0, max_depth)
                tokens = tokens[:max_seq_len] + list((0,)) * max(0, max_seq_len - len(tokens))
                self.test_samples.append(torch.tensor(tokens, dtype=torch.long))
                self.test_labels.append(label)

    def _generate_tree(self, depth, max_depth):
        terminal_prob = 0.1 + (depth / max_depth) * 0.8
        if depth >= max_depth or random.random() < terminal_prob:
            val = random.randint(0, 9)
            return list((val + 1,)), val
            
        op = random.choice(list((11, 12, 13, 14)))
        num_args = random.randint(2, 3)
        tokens = list((15, op)) 
        args_vals = list()
        for _ in range(num_args):
            arg_tokens, arg_val = self._generate_tree(depth + 1, max_depth)
            tokens.extend(arg_tokens)
            args_vals.append(arg_val)
        tokens.append(16) 
        
        if op == 11: res = min(args_vals)
        elif op == 12: res = max(args_vals)
        elif op == 13: res = int(np.median(args_vals))
        elif op == 14: res = sum(args_vals) % 10
        return tokens, res

    def __len__(self): return self.num_samples
    def __getitem__(self, idx):
        if not self.is_train: return self.test_samples[idx], self.test_labels[idx]
        tokens, label = self._generate_tree(0, self.max_depth)
        tokens = tokens[:self.max_seq_len] + list((0,)) * max(0, self.max_seq_len - len(tokens))
        return torch.tensor(tokens, dtype=torch.long), label

def get_dyck_loaders(batch_size, train_len=20, test_len=60):
    return DataLoader(DyckDataset(8000, train_len, True), batch_size, shuffle=True), DataLoader(DyckDataset(2000, test_len, False), batch_size), 7, 2

def get_recall_loaders(batch_size, train_len=20, test_len=60):
    vocab_size = (train_len * 2) + 2
    return DataLoader(RecallDataset(8000, train_len, True), batch_size, shuffle=True), DataLoader(RecallDataset(2000, test_len, False), batch_size), vocab_size, train_len

def get_listops_loaders(batch_size, train_len=20, test_len=60):
    return DataLoader(ListOpsDataset(8000, train_len, max_depth=3, is_train=True), batch_size, shuffle=True), DataLoader(ListOpsDataset(2000, test_len, max_depth=5, is_train=False), batch_size), 17, 10