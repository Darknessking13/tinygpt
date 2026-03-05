import subprocess

with open('data/corpus.md', 'r') as f:
    content = f.read(500 * 1024 * 1024)  # 500MB 🔥

with open('data/corpus_small.md', 'w') as f:
    f.write(content)

r = subprocess.run(['wc', '-w', 'data/corpus_small.md'], 
                  capture_output=True, text=True)
print(r.stdout)