import os, argparse
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler


def encode_and_save(corpus_path: str, tokenizer, out_dir: str, val_split: float = 0.1):
    with open(corpus_path, "r", encoding="utf-8") as f:
        text = f.read()
    ids = np.array(tokenizer.encode(text).ids, dtype=np.uint16)
    split_idx = int(len(ids) * (1 - val_split))
    os.makedirs(out_dir, exist_ok=True)
    ids[:split_idx].tofile(f"{out_dir}/train.bin")
    ids[split_idx:].tofile(f"{out_dir}/val.bin")
    print(f"Train: {split_idx:,} tokens | Val: {len(ids)-split_idx:,} tokens")


class TokenDataset(Dataset):
    def __init__(self, bin_path: str, context_length: int):
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")
        self.context_length = context_length

    def __len__(self):
        return len(self.data) - self.context_length - 1

    def __getitem__(self, idx):
        chunk = self.data[idx : idx + self.context_length + 1].astype(np.int64)
        return torch.from_numpy(chunk[:-1]), torch.from_numpy(chunk[1:])


def make_dataloader(dataset: TokenDataset, batch_size: int, ctx, shuffle: bool = True) -> DataLoader:
    if ctx.backend == "ddp":
        sampler = DistributedSampler(dataset, shuffle=shuffle,
                                    rank=ctx.rank, num_replicas=ctx.world_size)
        return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                         num_workers=2, pin_memory=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                     num_workers=0, pin_memory=(ctx.backend == "single_gpu"))


if __name__ == "__main__":
    from tokenizers import Tokenizer
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default="data/raw/corpus.md")
    p.add_argument("--tokenizer", default="tokenizer/rei_tokenizer/tokenizer.json")
    p.add_argument("--out_dir", default="data/processed")
    args = p.parse_args()
    tok = Tokenizer.from_file(args.tokenizer)
    encode_and_save(args.corpus, tok, args.out_dir)
