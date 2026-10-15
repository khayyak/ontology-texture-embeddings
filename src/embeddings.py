"""Frozen backbones -> cached vectors. Generic over open_clip models + DINOv2."""
from functools import lru_cache
import numpy as np
from PIL import Image
import torch
import config

_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OPENCLIP = {"clip": ("ViT-B-32", "openai"), "siglip": ("ViT-B-16-SigLIP", "webli")}

def _load_dinov2():
    model = torch.hub.load("facebookresearch/dinov2", "dinov2_vitb14").eval().to(_DEVICE)
    import torchvision.transforms as T
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.ToTensor(),
                    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
    @torch.no_grad()
    def encode(imgs):
        x = torch.stack([tf(im) for im in imgs]).to(_DEVICE)
        return model(x).cpu().numpy()
    return encode

def _load_openclip_image(key):
    import open_clip
    name, pretrained = OPENCLIP[key]
    model, _, preprocess = open_clip.create_model_and_transforms(name, pretrained=pretrained)
    model.eval().to(_DEVICE)
    @torch.no_grad()
    def encode(imgs):
        x = torch.stack([preprocess(im) for im in imgs]).to(_DEVICE)
        return model.encode_image(x).cpu().numpy()
    return encode

def _image_encoder(loader_key):
    return _load_dinov2() if loader_key == "dinov2" else _load_openclip_image(loader_key)

def get_embeddings(name, loader_key, paths, tag):
    cache = config.CACHE_DIR / f"{name}__{tag}.npy"
    if cache.exists():
        emb = np.load(cache)
        if emb.shape[0] == len(paths):
            print(f"[cache] {name}/{tag}: {emb.shape}"); return emb
    encode = _image_encoder(loader_key)
    feats, B = [], 32
    for i in range(0, len(paths), B):
        batch = [Image.open(p).convert("RGB") for p in paths[i:i + B]]
        feats.append(encode(batch))
        if (i // B) % 10 == 0:
            print(f"  {name}/{tag}: {i}/{len(paths)}")
    emb = np.vstack(feats).astype(np.float32)
    np.save(cache, emb); print(f"[saved] {name}/{tag}: {emb.shape}")
    return emb

@lru_cache(maxsize=4)
def _load_openclip_text(key):
    import open_clip
    name, pretrained = OPENCLIP[key]
    model = open_clip.create_model(name, pretrained=pretrained).eval().to(_DEVICE)
    return model, open_clip.get_tokenizer(name)

def encode_texts(vlm_key, prompts):
    model, tokenizer = _load_openclip_text(vlm_key)
    with torch.no_grad():
        toks = tokenizer(prompts).to(_DEVICE)
        return model.encode_text(toks).cpu().numpy()
