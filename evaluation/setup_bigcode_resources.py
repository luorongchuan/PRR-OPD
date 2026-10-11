"""Download only public NLTK scoring resources into the private overlay."""
import os
from pathlib import Path
import sys

target = Path(sys.argv[1]).resolve() if len(sys.argv) > 1 else Path(__file__).resolve().parent / "runtime/bigcode_packages"
sys.path.insert(0, str(target))
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import nltk

for name in ("punkt", "stopwords"):
    if not nltk.download(name, download_dir=str(target / "nltk_data"), raise_on_error=True):
        raise RuntimeError(f"Failed to download scoring resource: {name}")
