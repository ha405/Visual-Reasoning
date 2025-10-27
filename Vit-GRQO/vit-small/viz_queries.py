import os
import math
from typing import List, Tuple, Optional, Dict
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

def token_grid_from_N(N: int) -> Tuple[int,int]:
    s = int(math.sqrt(N))
    if s * s == N:
        return s, s
    for d in range(s, 0, -1):
        if N % d == 0:
            return d, N // d
    return 1, N

def upsample_attention_to_image(attn_patch: np.ndarray, image_size: Tuple[int,int]):
    Himg, Wimg = image_size
    im = Image.fromarray((attn_patch * 255.0).astype(np.uint8)).resize((Wimg, Himg), resample=Image.BILINEAR)
    arr = np.array(im).astype(np.float32) / 255.0
    if arr.max() > arr.min():
        arr = (arr - arr.min()) / (arr.max() - arr.min())
    return arr

def overlay_heatmap(img_pil: Image.Image, heatmap: np.ndarray, alpha=0.45):
    img = np.array(img_pil).astype(np.float32) / 255.0
    heat3 = np.stack([heatmap]*3, axis=-1)
    out = (1.0 - alpha) * img + alpha * heat3
    out = (out * 255.0).astype(np.uint8)
    return Image.fromarray(out)

def register_decoder_attn_hooks(model, store: Dict):
    store['layer_attn'] = []
    hooks = []

    def make_hook(name):
        def hook(module, inputs, outputs):
            if isinstance(outputs, tuple) and len(outputs) >= 2:
                attn = outputs[1].detach().cpu()
                store['layer_attn'].append((name, attn))
        return hook

    for n, m in model.named_modules():
        if m.__class__.__name__ == "DecoderAttn":
            h = m.register_forward_hook(make_hook(n))
            hooks.append(h)
    store['_hooks'] = hooks
    return hooks

def remove_hooks(store: Dict):
    for h in store.get('_hooks', []):
        try:
            h.remove()
        except Exception:
            pass
    store['_hooks'] = []

def collect_visual_bank(model: torch.nn.Module,
                        dataloader,
                        device: torch.device,
                        save_path: str,
                        max_images: Optional[int] = 500,
                        patch_image_size: int = 16,
                        capture_layer_attn: bool = True,
                        model_forward_adapter: Optional[callable] = None):
    model.eval()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    store = {"records": []}
    hook_store = {}
    if capture_layer_attn:
        register_decoder_attn_hooks(model, hook_store)
    seen = 0
    pbar = tqdm(dataloader, desc="Collect visual bank")
    for batch in pbar:
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            images, labels = batch[0], batch[1]
            extra = batch[2:] if len(batch) > 2 else ()
        else:
            images = batch['images']
            labels = batch.get('labels', None)
            extra = ()
        images = images.to(device)
        labels = labels.to(device) if labels is not None else None
        with torch.set_grad_enabled(True):
            if model_forward_adapter is not None:
                out = model_forward_adapter(model, images, labels)
            else:
                raw = model(images, labels)
                if isinstance(raw, dict):
                    out = raw
                else:
                    raise RuntimeError("Model returned non-dict. Provide model_forward_adapter that returns dict with required keys.")
        prob_keys = ['prob_scores', 'selection_probs', 'w']
        perq_logits_keys = ['per_query_logits', 'query_logits']
        decoder_keys = ['decoder_out', 'queries_feat']
        attn_keys = ['attn', 'attention_map']
        def pick_key(d, keys):
            for k in keys:
                if k in d:
                    return k
            return None
        k_prob = pick_key(out, prob_keys)
        k_plog = pick_key(out, perq_logits_keys)
        k_dec = pick_key(out, decoder_keys)
        k_attn = pick_key(out, attn_keys)
        if k_prob is None or k_plog is None or k_dec is None or k_attn is None:
            raise RuntimeError(f"Missing required keys in model output. Found: {list(out.keys())}")
        prob = out[k_prob].detach().cpu()
        perq_logits = out[k_plog].detach().cpu()
        dec_out = out[k_dec].detach().cpu()
        attn = out[k_attn].detach().cpu()
        B = prob.shape[0]
        layer_attn_per_batch = []
        if capture_layer_attn:
            collected = hook_store.get('layer_attn', [])
            layer_attn_per_batch = [(n, t.cpu()) for (n, t) in collected]
            hook_store['layer_attn'] = []
        for i in range(B):
            rec = {
                "prob_scores": prob[i].numpy(),
                "perq_logits": perq_logits[i].numpy(),
                "decoder_out": dec_out[i].numpy(),
                "attn": attn[i].numpy(),
                "label": int(labels[i].item()) if labels is not None else -1,
            }
            if capture_layer_attn:
                rec['layer_attn'] = [(n, t[i].numpy()) for (n, t) in layer_attn_per_batch]
            if extra:
                e0 = extra[0]
                try:
                    candidate = e0[i] if isinstance(e0, (list, tuple)) else e0[i]
                    if isinstance(candidate, str):
                        rec['img_path'] = candidate
                    elif isinstance(candidate, Image.Image):
                        img_small = candidate.copy().resize((224,224))
                        rec['img_preview'] = np.array(img_small)
                except Exception:
                    pass
            store["records"].append(rec)
            seen += 1
            if max_images is not None and seen >= max_images:
                break
        if max_images is not None and seen >= max_images:
            break
    remove_hooks(hook_store)
    torch.save(store, save_path)
    print(f"Saved visual bank with {len(store['records'])} images to {save_path}")
    return store


def plot_attention_overlay_from_record(rec, query_id: int, image_size=(224,224), patch_grid=None, show=True):
    attn = rec['attn']
    M, N = attn.shape
    if patch_grid is None:
        patch_grid = token_grid_from_N(N)
    Hp, Wp = patch_grid
    q_attn = attn[query_id].reshape(Hp, Wp)
    heat = upsample_attention_to_image(q_attn, image_size)
    if 'img_preview' in rec:
        img = Image.fromarray(rec['img_preview'])
    elif 'img_path' in rec:
        img = Image.open(rec['img_path']).convert('RGB').resize(image_size)
    else:
        img = Image.new('RGB', image_size, (200,200,200))
    overlay = overlay_heatmap(img, heat)
    if show:
        plt.figure(figsize=(4,4))
        plt.axis('off')
        plt.imshow(overlay)
        plt.title(f"Query {query_id} overlay | label={rec.get('label', '?')}")
    return overlay

def topk_patches_for_query(visual_bank: Dict, query_id: int, k:int=9, patch_grid:Tuple[int,int]=None, patch_size:int=16):
    records = visual_bank['records']
    items = []
    for idx, rec in enumerate(records):
        attn = rec['attn']
        M, N = attn.shape
        if patch_grid is None:
            Hp, Wp = token_grid_from_N(N)
        else:
            Hp, Wp = patch_grid
        scores = attn[query_id]
        top_idx = int(scores.argmax())
        score = float(scores[top_idx])
        r, c = divmod(top_idx, Wp)
        img = None
        if 'img_preview' in rec:
            img = Image.fromarray(rec['img_preview'])
        elif 'img_path' in rec:
            try:
                img = Image.open(rec['img_path']).convert('RGB')
            except Exception:
                img = None
        items.append((score, idx, (r,c), img))
    items = sorted(items, key=lambda x: -x[0])[:k]
    cols = int(math.ceil(math.sqrt(len(items))))
    rows = int(math.ceil(len(items)/cols))
    fig, axs = plt.subplots(rows, cols, figsize=(cols*2, rows*2))
    axs = np.array(axs).reshape(-1)
    for ax in axs:
        ax.axis('off')
    for i, it in enumerate(items):
        score, idx, (r,c), img = it
        if img is None:
            axs[i].imshow(np.zeros((patch_size,patch_size,3), dtype=np.uint8) + 200)
        else:
            Himg, Wimg = img.size[1], img.size[0]
            if patch_grid is None:
                rec = visual_bank['records'][idx]
                _, N = rec['attn'].shape
                Hp, Wp = token_grid_from_N(N)
            ph = Himg // Hp
            pw = Wimg // Wp
            cy = int((r + 0.5) * ph)
            cx = int((c + 0.5) * pw)
            y0 = max(0, cy - patch_size//2)
            x0 = max(0, cx - patch_size//2)
            crop = img.crop((x0, y0, min(Wimg, x0+patch_size), min(Himg, y0+patch_size)))
            axs[i].imshow(crop)
        axs[i].set_title(f"{score:.2f}", fontsize=8)
    plt.suptitle(f"Top-{k} patches for query {query_id}")
    plt.tight_layout()
    return fig

def tsne_query_embeddings(visual_bank, sample_limit=2000, per_query=False):
    records = visual_bank['records']
    rows = []
    meta = []
    for i, rec in enumerate(records):
        dec = rec['decoder_out']
        dec = np.atleast_2d(dec)  # ensures 2D [M,D] even if M=1
        M, D = dec.shape
        for q in range(M):
            rows.append(dec[q])
            meta.append({
                "record_idx": i,
                "query_id": q,
                "label": rec.get('label', -1)
            })
    X = np.stack(rows, axis=0)
    if sample_limit is not None and X.shape[0] > sample_limit:
        idx = np.random.choice(X.shape[0], sample_limit, replace=False)
        X = X[idx]
        meta = [meta[i] for i in idx]

    from sklearn.manifold import TSNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    Z = tsne.fit_transform(X)
    return Z, meta


def ablate_query_and_eval(visual_bank: Dict, model, dataloader, device, query_id:int, adapter=None):
    model.eval()
    correct = 0
    total = 0
    for batch in dataloader:
        images, labels = batch[0].to(device), batch[1].to(device)
        with torch.no_grad():
            out = adapter(model, images) if adapter is not None else model(images)
        perq = out['per_query_logits']
        probs = out['prob_scores']
        img_logits = torch.einsum("bq,bqc->bc", probs, perq)
        probs_ab = probs.clone()
        probs_ab[:, query_id] = 0.0
        probs_ab = probs_ab / (probs_ab.sum(dim=1, keepdim=True) + 1e-12)
        img_logits_ab = torch.einsum("bq,bqc->bc", probs_ab, perq)
        preds = img_logits_ab.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    acc = correct / total
    print(f"Ablated query {query_id} accuracy = {acc:.4f}")
    return acc
