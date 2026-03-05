#!/usr/bin/env python3
"""Clean Project Gutenberg books by removing metadata and legal text."""

import re

def clean_gutenberg_text(text):
    """Remove Project Gutenberg headers, footers, and metadata."""
    
    # Remove everything before "START OF THE PROJECT GUTENBERG" or "START OF THIS PROJECT GUTENBERG"
    start_patterns = [
        r'\*\*\* START OF TH(IS|E) PROJECT GUTENBERG.*?\*\*\*',
        r'START OF THE PROJECT GUTENBERG',
    ]
    
    for pattern in start_patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.DOTALL)
        if match:
            text = text[match.end():]
            break
    
    # Remove everything after "END OF THE PROJECT GUTENBERG" or "END OF THIS PROJECT GUTENBERG"
    end_patterns = [
        r'\*\*\* END OF TH(IS|E) PROJECT GUTENBERG.*?\*\*\*',
        r'END OF THE PROJECT GUTENBERG',
    ]
    
    for pattern in end_patterns:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            text = text[:match.start()]
            break
    
    # Remove common Gutenberg metadata lines
    lines_to_remove = [
        r'.*Project Gutenberg.*',
        r'.*gutenberg\.org.*',
        r'.*eBook.*',
        r'.*License.*',
        r'.*Copyright.*',
        r'.*Public Domain.*',
        r'.*\*\*\*.*',
        r'.*Produced by.*',
        r'.*Transcriber.*',
    ]
    
    lines = text.split('\n')
    cleaned_lines = []
    
    for line in lines:
        # Skip lines matching removal patterns
        should_remove = False
        for pattern in lines_to_remove:
            if re.match(pattern, line, re.IGNORECASE):
                should_remove = True
                break
        
        if not should_remove and line.strip():
            cleaned_lines.append(line)
    
    return '\n'.join(cleaned_lines)


if __name__ == '__main__':
    print("Cleaning corpus.md...")
    
    with open('data/corpus.md', 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()
    
    print(f"Original size: {len(content)} characters")
    
    cleaned = clean_gutenberg_text(content)
    
    print(f"Cleaned size: {len(cleaned)} characters")
    print(f"Removed: {len(content) - len(cleaned)} characters")
    
    with open('data/corpus.md', 'w', encoding='utf-8') as f:
        f.write(cleaned)
    
    print("Done! Corpus cleaned.")
