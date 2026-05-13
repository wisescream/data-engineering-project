import re
import os

files = ['api/quantize.py', 'api/benchmark.py', 'api/main.py']
for f in files:
    if os.path.exists(f):
        with open(f, 'r', encoding='utf-8') as file:
            content = file.read()
        # Replace non-ASCII with '-'
        clean_content = re.sub(r'[^\x00-\x7F]+', '-', content)
        with open(f, 'w', encoding='utf-8') as file:
            file.write(clean_content)
        print(f"Cleaned {f}")
