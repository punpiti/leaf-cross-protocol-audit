"""Image-level interventions and frozen ResNet-18 features."""
import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

CROP_MODES = {"crop", "crop_swap", "crop_occ", "crop_leafout", "silhouette"}


def leaf_mask_small(a8):
    """Leaf mask for a small image, relative to a background estimated from the border."""
    lab = cv2.cvtColor(a8, cv2.COLOR_RGB2LAB).astype(np.float32)
    A = cv2.GaussianBlur(lab[..., 1], (5, 5), 0)
    h, w = A.shape
    b = max(4, min(h, w) // 25)
    border = np.concatenate([A[:b].ravel(), A[-b:].ravel(),
                             A[:, :b].ravel(), A[:, -b:].ravel()])
    bg = float(np.median(border))
    spread = float(np.median(np.abs(border - bg))) + 1e-6
    m = ((bg - A) > max(4.0, 5.0 * spread)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab_im, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    big = st[1:, cv2.CC_STAT_AREA].max()
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > max(0.08 * big, 20):
            keep[lab_im == i] = 1
    return keep


def bg_patch(path):
    """Background of a donor crop, with the donor's own leaf replaced by its median background."""
    im = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    dm = load_mask(path)
    if dm is not None and dm.mean() < 0.95:
        bgmed = np.median(a[dm == 0].reshape(-1, 3), axis=0).astype(np.float32)
        a = np.where(dm[..., None] > 0, bgmed, a)
        a = cv2.GaussianBlur(a, (15, 15), 0)
    return a


def load_mask(path):
    """Mask written next to a crop by segment.crop_leaf."""
    mp = path[:-4] + "_m.png"
    if not os.path.exists(mp):
        return None
    m = np.asarray(Image.open(mp).resize((224, 224), Image.NEAREST))
    return (m > 127).astype(np.uint8)


def to_tensor(path, mode, donor=None, seed=0):
    """One image under one input condition, normalised for ResNet-18."""
    im = Image.open(path).convert("RGB").resize((224, 224), Image.BILINEAR)
    a = np.asarray(im, dtype=np.float32) / 255.0
    rng = np.random.default_rng(seed)

    if mode == "gray":
        g = a @ np.array([0.299, 0.587, 0.114], dtype=np.float32)
        a = np.repeat(g[..., None], 3, axis=2)

    elif mode == "colornorm":
        for c in range(3):
            ch = a[..., c]
            a[..., c] = (ch - ch.mean()) / (ch.std() + 1e-6) * 0.2 + 0.5
        a = np.clip(a, 0, 1)

    elif mode == "silhouette":
        m = load_mask(path)
        a = (np.repeat((m if m is not None else
                        np.zeros(a.shape[:2], np.uint8))[..., None], 3, 2)
             .astype(np.float32))

    elif mode == "crop_leafout":
        m = load_mask(path)
        if m is not None:
            a = np.where(m[..., None] > 0, np.float32(0.5), a)

    elif mode == "crop_swap":
        m = load_mask(path)
        if m is not None:
            fill = bg_patch(donor) if donor is not None else np.float32(0.5)
            a = np.where(m[..., None] > 0, a, fill)

    elif mode == "crop_occ":
        m = load_mask(path)
        ys, xs = np.nonzero(m) if m is not None else (np.array([]),) * 2
        if ys.size:
            for _ in range(3):
                ch, cw = int(rng.integers(24, 64)), int(rng.integers(24, 64))
                k = int(rng.integers(ys.size))
                y0 = max(0, min(224 - ch, int(ys[k]) - ch // 2))
                x0 = max(0, min(224 - cw, int(xs[k]) - cw // 2))
                a[y0:y0 + ch, x0:x0 + cw] = 0.5

    a = (a - np.array(MEAN, dtype=np.float32)) / np.array(STD, dtype=np.float32)
    return torch.from_numpy(a.transpose(2, 0, 1))


def embed(paths, mode, donors=None, batch=48):
    """512-d ImageNet ResNet-18 features, one row per path, under one condition."""
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net.eval()
    torch.set_num_threads(8)
    out = []
    with torch.no_grad():
        for i in range(0, len(paths), batch):
            xb = torch.stack([
                to_tensor(p, mode,
                          donors[i + k] if donors else None, seed=i + k)
                for k, p in enumerate(paths[i:i + batch])])
            out.append(net(xb).numpy())
    return np.concatenate(out)
