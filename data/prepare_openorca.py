from datasets import load_dataset
import subprocess

print("Loading OpenOrca (streaming)...")
ds = load_dataset("Open-Orca/OpenOrca", 
                  split="train", 
                  streaming=True)  # ← KEY FIX

print("Converting to plain text...")
count = 0
max_rows = 500_000

with open("data/corpus.md", "w") as f:
    for row in ds:
        if count >= max_rows:
            break
            
        q = row["question"].strip()
        r = row["response"].strip()
        
        # skip very long responses
        if len(r.split()) > 200:
            continue
        
        f.write(f"{q}\n{r}\n\n")
        count += 1
        
        if count % 10_000 == 0:
            print(f"Progress: {count}/{max_rows}")

print(f"Done! Wrote {count} examples")
result = subprocess.run(["wc", "-w", "data/corpus.md"], 
                       capture_output=True, text=True)
print(f"Word count: {result.stdout}")
