import math, os, argparse, torch
from tokenizers import Tokenizer
from model.config import ModelConfig, PRESETS
from model.transformer import GPT
from training.device import setup_device, cleanup_ddp, log, wrap_ddp, unwrap
from training.dataset import TokenDataset, make_dataloader


def cosine_with_warmup(step, warmup, total):
    if step < warmup:
        return step / max(1, warmup)
    progress = (step - warmup) / max(1, total - warmup)
    return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))


def train(config: ModelConfig, train_path: str, val_path: str, checkpoint_dir: str = "checkpoints/pretrain"):
    ctx = setup_device()
    log(ctx, f"Hardware: {ctx.backend.upper()} | World size: {ctx.world_size} | "
             f"Device: {ctx.device} | AMP dtype: {ctx.amp_dtype} | GradScaler: {ctx.use_scaler}")
    log(ctx, config.param_count())

    model = GPT(config).to(ctx.device)
    model = wrap_ddp(model, ctx)
    optimizer = torch.optim.AdamW(unwrap(model).parameters(), lr=config.learning_rate,
                                  weight_decay=config.weight_decay, betas=(0.9, 0.95))
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: cosine_with_warmup(s, config.warmup_iters, config.max_iters))

    train_ds = TokenDataset(train_path, config.context_length)
    val_ds = TokenDataset(val_path, config.context_length)
    train_dl = make_dataloader(train_ds, config.batch_size, ctx, shuffle=True)
    val_dl = make_dataloader(val_ds, config.batch_size, ctx, shuffle=False)

    scaler = torch.cuda.amp.GradScaler(enabled=ctx.use_scaler)
    amp_ctx = torch.autocast(device_type=ctx.device.type, dtype=ctx.amp_dtype, enabled=True)

    model.train()
    step, accum_loss = 0, 0.0
    optimizer.zero_grad()

    for epoch in range(9999):
        if ctx.backend == "ddp":
            train_dl.sampler.set_epoch(epoch)

        for x, y in train_dl:
            if step >= config.max_iters:
                break

            x = x.to(ctx.device, non_blocking=True)
            y = y.to(ctx.device, non_blocking=True)

            with amp_ctx:
                _, loss = model(x, y)
                loss = loss / config.grad_accum_steps

            scaler.scale(loss).backward()
            accum_loss += loss.item()

            if (step + 1) % config.grad_accum_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(unwrap(model).parameters(), config.grad_clip)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                optimizer.zero_grad()

            if step % config.eval_interval == 0 and ctx.is_main:
                val_loss = evaluate(model, val_dl, ctx, amp_ctx, config)
                effective_loss = accum_loss * config.grad_accum_steps
                log(ctx, f"[{step:5d}/{config.max_iters}] train={effective_loss:.4f} "
                         f"val={val_loss:.4f} lr={scheduler.get_last_lr()[0]:.2e}")
                accum_loss = 0.0
                save_checkpoint(unwrap(model), optimizer, step, checkpoint_dir, ctx)

            step += 1

        if step >= config.max_iters:
            break

    if ctx.is_main:
        save_checkpoint(unwrap(model), optimizer, step, checkpoint_dir, ctx, name="ckpt_best.pt")

    log(ctx, "Pretraining complete.")
    cleanup_ddp()


@torch.no_grad()
def evaluate(model, val_dl, ctx, amp_ctx, config):
    model.eval()
    total, count = 0.0, 0
    for i, (x, y) in enumerate(val_dl):
        if i >= config.eval_iters: break
        x, y = x.to(ctx.device), y.to(ctx.device)
        with amp_ctx:
            _, loss = model(x, y)
        total += loss.item(); count += 1
    model.train()
    return total / max(1, count)


def save_checkpoint(model, optimizer, step, directory, ctx, name=None):
    if not ctx.is_main: return
    os.makedirs(directory, exist_ok=True)
    fname = name or f"ckpt_step{step:06d}.pt"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "step": step},
               f"{directory}/{fname}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/processed/train.bin")
    p.add_argument("--val", default="data/processed/val.bin")
    p.add_argument("--preset", default="small", choices=PRESETS.keys())
    p.add_argument("--tokenizer", default="tokenizer/rei_tokenizer/tokenizer.json")
    args = p.parse_args()
    config = PRESETS[args.preset]
    tok = Tokenizer.from_file(args.tokenizer)
    config.vocab_size = tok.get_vocab_size()
    train(config, args.train, args.val)
