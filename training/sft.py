import json, os, argparse, torch
from torch.utils.data import Dataset, DataLoader
from tokenizers import Tokenizer
from model.config import ModelConfig, PRESETS
from model.transformer import GPT
from training.device import setup_device, cleanup_ddp, log, wrap_ddp, unwrap


def format_sft_sample(user: str, assistant: str, tokenizer, max_len: int):
    full = f"<|user|> {user} <|end|>\n<|assistant|> {assistant} <|end|>"
    prefix = f"<|user|> {user} <|end|>\n<|assistant|> "
    ids = tokenizer.encode(full).ids[:max_len]
    plen = len(tokenizer.encode(prefix).ids)
    labels = [-1] * plen + ids[plen:]
    labels = labels[:max_len]
    pad = max_len - len(ids)
    ids += [0] * pad
    labels += [-1] * pad
    return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


class SFTDataset(Dataset):
    def __init__(self, jsonl_path, tokenizer, context_length):
        self.samples = []
        with open(jsonl_path) as f:
            for line in f:
                item = json.loads(line.strip())
                self.samples.append(format_sft_sample(item["user"], item["assistant"],
                                                     tokenizer, context_length))
    def __len__(self): return len(self.samples)
    def __getitem__(self, i): return self.samples[i]


def sft_train(config: ModelConfig, pretrain_ckpt: str, sft_jsonl: str,
              tokenizer_path: str, checkpoint_dir: str = "checkpoints/sft"):
    ctx = setup_device()
    tokenizer = Tokenizer.from_file(tokenizer_path)
    config.vocab_size = tokenizer.get_vocab_size()

    model = GPT(config).to(ctx.device)
    ckpt = torch.load(pretrain_ckpt, map_location=ctx.device)
    state = {k.replace("module.", ""): v for k, v in ckpt["model"].items()}
    model.load_state_dict(state)
    log(ctx, f"Loaded pretrain weights from {pretrain_ckpt}")

    n_freeze = config.n_layers // 2
    for i, block in enumerate(model.blocks):
        if i < n_freeze:
            for p in block.parameters():
                p.requires_grad = False
    log(ctx, f"Froze layers 0–{n_freeze-1} | Training layers {n_freeze}–{config.n_layers-1}")

    model = wrap_ddp(model, ctx)
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, unwrap(model).parameters()),
                                  lr=config.sft_lr, weight_decay=0.01)

    ds = SFTDataset(sft_jsonl, tokenizer, config.context_length)
    dl = DataLoader(ds, batch_size=8, shuffle=True)
    amp_ctx = torch.autocast(device_type=ctx.device.type, dtype=ctx.amp_dtype, enabled=True)

    model.train()
    for step, (x, y) in enumerate(dl):
        if step >= config.sft_max_iters: break
        x, y = x.to(ctx.device), y.to(ctx.device)
        with amp_ctx:
            _, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), 1.0)
        optimizer.step()
        optimizer.zero_grad()
        if step % 100 == 0:
            log(ctx, f"SFT [{step}/{config.sft_max_iters}] loss={loss.item():.4f}")

    if ctx.is_main:
        os.makedirs(checkpoint_dir, exist_ok=True)
        torch.save({"model": unwrap(model).state_dict()}, f"{checkpoint_dir}/ckpt_final.pt")
        log(ctx, f"SFT complete. Checkpoint: {checkpoint_dir}/ckpt_final.pt")

    cleanup_ddp()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain_ckpt", default="checkpoints/pretrain/ckpt_best.pt")
    p.add_argument("--sft_data", default="data/raw/sft_data.jsonl")
    p.add_argument("--tokenizer", default="tokenizer/rei_tokenizer/tokenizer.json")
    p.add_argument("--preset", default="small", choices=PRESETS.keys())
    args = p.parse_args()
    sft_train(PRESETS[args.preset], args.pretrain_ckpt, args.sft_data, args.tokenizer)
