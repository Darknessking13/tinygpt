with open('data/corpus.md', 'r') as f:
    content = f.read(50 * 1024 * 1024)  # 50MB only

with open('data/corpus_small.md', 'w') as f:
    f.write(content)
    
import subprocess
r = subprocess.run(['wc', '-w', 'data/corpus_small.md'], 
                  capture_output=True, text=True)
print(r.stdout)
