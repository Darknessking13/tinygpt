#!/usr/bin/env python3
import torch, argparse, json, re
from tokenizers import Tokenizer
from model.config import ModelConfig, PRESETS
from model.transformer import GPT
from rich.console import Console
from rich.prompt import Prompt

console = Console()


def detect_inference_device() -> torch.device:
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        console.print(f"[dim]GPU: {name}[/dim]")
        return torch.device("cuda:0")
    console.print("[dim]CPU mode[/dim]")
    return torch.device("cpu")


def load_model(ckpt_path: str, config: ModelConfig, device: torch.device) -> GPT:
    model = GPT(config)
    ckpt = torch.load(ckpt_path, map_location=device)
    state = ckpt.get("model", ckpt)
    state = {k.replace("module.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    return model.to(device).eval()


def chat(model: GPT, tokenizer: Tokenizer, config: ModelConfig, device: torch.device):
    console.print("\n[bold cyan]★ Rei Corpus — Local Chat ★[/bold cyan]")
    console.print("[dim]Commands: /temp 0.9 | /topk 50 | quit[/dim]\n")

    end_id = tokenizer.token_to_id("<|end|>")
    temperature = 0.8
    top_k = 40

    while True:
        user_input = Prompt.ask("[bold green]You[/bold green]")

        if user_input.lower() in {"quit", "exit", "q"}:
            break
        if user_input.startswith("/temp "):
            temperature = float(user_input.split()[1])
            console.print(f"[dim]Temperature → {temperature}[/dim]"); continue
        if user_input.startswith("/topk "):
            top_k = int(user_input.split()[1])
            console.print(f"[dim]Top-k → {top_k}[/dim]"); continue

        prompt = f"<|user|> {user_input} <|end|>\n<|assistant|> "
        ids = tokenizer.encode(prompt).ids
        idx = torch.tensor([ids], device=device)

        with torch.no_grad():
            out = model.generate(idx, max_new_tokens=256,
                               temperature=temperature, top_k=top_k,
                               stop_token_id=end_id)

        new_toks = out[0, len(ids):].tolist()
        response = tokenizer.decode(new_toks).replace("<|end|>", "").strip()

        m = re.search(r"<\|json\|>(.*?)<\|/json\|>", response, re.DOTALL)
        if m:
            try:
                parsed = json.loads(m.group(1).strip())
                response = response[:m.start()] + \
                          json.dumps(parsed, indent=2) + \
                          response[m.end():]
            except json.JSONDecodeError:
                pass

        console.print(f"[bold magenta]Rei[/bold magenta]: {response}\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default="checkpoints/sft/ckpt_final.pt")
    p.add_argument("--tokenizer", default="tokenizer/rei_tokenizer/tokenizer.json")
    p.add_argument("--preset", default="small", choices=PRESETS.keys())
    args = p.parse_args()

    tokenizer = Tokenizer.from_file(args.tokenizer)
    config = PRESETS[args.preset]
    config.vocab_size = tokenizer.get_vocab_size()
    device = detect_inference_device()
    model = load_model(args.checkpoint, config, device)
    chat(model, tokenizer, config, device)
