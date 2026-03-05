import subprocess

with open('data/corpus.md', 'r') as f:
    content = f.read(100 * 1024 * 1024)  # 100MB

with open('data/corpus_small.md', 'w') as f:
    f.write(content)

r = subprocess.run(['wc', '-w', 'data/corpus_small.md'], 
                  capture_output=True, text=True)
print(r.stdout)