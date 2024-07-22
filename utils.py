import re

def clean_code_snippet(snippet):
    snippet = re.sub(r'\W', ' ', snippet)
    snippet = re.sub(r'\d', ' ', snippet)
    snippet = snippet.lower()
    snippet = re.sub(r'\s+', ' ', snippet)
    return snippet
