"""Central configuration for the ontology-guided texture-embedding project."""
from pathlib import Path

DATA_DIR = Path("data")
CACHE_DIR = Path("cache")
RESULTS_DIR = Path("results")
for _d in (DATA_DIR, CACHE_DIR, RESULTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

DTD_PARTITION = 1
N_CLASSES = 47

BACKBONES = {
    "dinov2_vitb14": ("vision", "dinov2"),   # self-supervised, pure vision
    "clip_vitb32":   ("vlm",    "clip"),      # vision-language
    "siglip_vitb16": ("vlm",    "siglip"),    # stronger vision-language
}
VLMS = ("clip", "siglip")
VLM_CACHE = {"clip": "clip_vitb32", "siglip": "siglip_vitb16"}

TOP_K = (1, 5, 10)
KNN_K = 5
SEED = 42
