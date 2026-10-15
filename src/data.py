"""DTD loading via torchvision (downloads from the Oxford VGG site)."""
from typing import List, Tuple
import torchvision
import config

def _load_split(split: str):
    return torchvision.datasets.DTD(
        root=str(config.DATA_DIR), split=split,
        partition=config.DTD_PARTITION, download=True)

def _paths_labels(ds):
    return list(ds._image_files), [int(l) for l in ds._labels]

def load_all() -> Tuple[List[str], List[int], List[str]]:
    paths, labels, class_names = [], [], None
    for split in ("train", "val", "test"):
        ds = _load_split(split)
        p, y = _paths_labels(ds)
        paths.extend(p); labels.extend(y); class_names = ds.classes
    return paths, labels, class_names

def load_classification_splits():
    tr = _load_split("train"); te = _load_split("test")
    tr_p, tr_y = _paths_labels(tr); te_p, te_y = _paths_labels(te)
    return (tr_p, tr_y), (te_p, te_y), tr.classes
