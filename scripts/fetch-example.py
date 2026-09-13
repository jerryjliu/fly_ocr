from pathlib import Path
from flyocr.data.download import fetch
from flyocr.common import read_json
source = read_json('examples/microsoft-2025/source.json')
print(fetch(source['url'], Path('data/source/microsoft-2025.pdf'), source['sha256'], source['bytes']))
