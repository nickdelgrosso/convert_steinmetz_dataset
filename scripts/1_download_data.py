from pathlib import Path

import requests
from tqdm import tqdm


urls = [
  ("https://osf.io/agvxh/download", "steinmetz_part0.npz"),
  ("https://osf.io/uv3mw/download", "steinmetz_part1.npz"),
  ("https://osf.io/ehmw2/download", "steinmetz_part2.npz"),
]

base_path = Path("data/raw")
base_path.mkdir(parents=True, exist_ok=True)

for url, fname in tqdm(urls):
    path: Path = base_path / fname
    if path.exists():
        continue
  
    r = requests.get(url)
    r.raise_for_status()
    path.write_bytes(r.content)
