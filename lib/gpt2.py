"""GPT-2 embeddings + per-layer RSMs (Notebook 2).

Ported from the GPT-2 RSA notebook: each word is encoded as ' '+word (leading
space for BPE context); multi-token words are mean-pooled; all 13 layers
(embedding + 12 transformer blocks) are kept. Embeddings cache to .npz.
"""

import os

import numpy as np


def load_model(name="gpt2"):
    """Load a pretrained GPT-2 with hidden states enabled. Returns (model, tokenizer, device)."""
    import torch
    from transformers import GPT2Model, GPT2Tokenizer

    home = os.path.expanduser("~")
    cache = os.path.join(home, "hf_cache", "hub")
    os.makedirs(cache, exist_ok=True)
    os.environ.setdefault("HF_HOME", os.path.join(home, "hf_cache"))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = GPT2Tokenizer.from_pretrained(name, cache_dir=cache)
    model = GPT2Model.from_pretrained(name, output_hidden_states=True, cache_dir=cache).to(device).eval()
    return model, tokenizer, device


def embed_words(words, model, tokenizer, device, cache=None):
    """Return dict layer 0..12 -> ndarray (n_words, 768). Cached to `cache` (.npz)."""
    import torch

    if cache and os.path.exists(str(cache)):
        z = np.load(str(cache))
        return {int(k.split("_")[1]): z[k] for k in z.files}

    n_layers = model.config.n_layer + 1
    layers = {l: [] for l in range(n_layers)}
    for word in words:
        inputs = tokenizer(" " + str(word).lower(), return_tensors="pt").to(device)
        with torch.no_grad():
            out = model(**inputs)
        for li, hs in enumerate(out.hidden_states):
            layers[li].append(hs[0].mean(dim=0).cpu().numpy())
    emb = {l: np.array(v) for l, v in layers.items()}

    if cache:
        os.makedirs(os.path.dirname(str(cache)), exist_ok=True)
        np.savez(str(cache), **{f"layer_{l}": emb[l] for l in emb})
    return emb


def layer_rsms(emb):
    """Per-layer word RSM (pairwise Pearson), same metric as the neural RSM."""
    return {l: np.corrcoef(emb[l]) for l in emb}
