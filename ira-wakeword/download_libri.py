import os
import urllib.request
import tarfile
from pathlib import Path

urls = [
    "https://www.openslr.org/resources/12/dev-clean.tar.gz",
    "https://www.openslr.org/resources/12/test-clean.tar.gz",
    "https://www.openslr.org/resources/12/dev-other.tar.gz",
    "https://www.openslr.org/resources/12/test-other.tar.gz"
]

download_dir = Path("dataset/negative/downloads")
download_dir.mkdir(parents=True, exist_ok=True)
extract_dir = download_dir / "LibriSpeech"

for url in urls:
    filename = url.split("/")[-1]
    filepath = download_dir / filename
    if not filepath.exists():
        print(f"Downloading {filename}...")
        urllib.request.urlretrieve(url, filepath)
    else:
        print(f"{filename} already exists.")
        
    print(f"Extracting {filename}...")
    with tarfile.open(filepath, "r:gz") as tar:
        tar.extractall(path=download_dir)

print("Download and extraction complete.")
