"""The corner condition: classify from image corners that contain neither leaf nor coin."""
import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models

from .interventions import MEAN, STD, leaf_mask_small

PATCH_FRAC = 0.15
DETECT_LONG = 512
DRAFT_DIV = 4
MAX_OBJ_FRAC = 0.02


def corner_boxes(w, h):
    s = max(8, int(round(PATCH_FRAC * min(w, h))))
    return [("tl", (0, 0, s, s)), ("tr", (w - s, 0, w, s)),
            ("bl", (0, h - s, s, h)), ("br", (w - s, h - s, w, h))]


def accepted_corners(src):
    """Return (decoded image, accepted corner names, reasons for rejection)."""
    im = Image.open(src)
    im.draft("RGB", (im.width // DRAFT_DIV, im.height // DRAFT_DIV))
    im = im.convert("RGB")

    det = im.copy()
    det.thumbnail((DETECT_LONG, DETECT_LONG), Image.BILINEAR)
    a8 = np.asarray(det, dtype=np.uint8)
    leaf = leaf_mask_small(a8)
    if leaf is None:
        leaf = np.zeros(a8.shape[:2], np.uint8)

    g = a8.astype(np.float32) @ np.array([0.299, 0.587, 0.114], np.float32)
    hh, ww = g.shape
    b = max(2, min(hh, ww) // 25)
    ring = np.concatenate([g[:b].ravel(), g[-b:].ravel(),
                           g[:, :b].ravel(), g[:, -b:].ravel()])
    bg = float(np.median(ring))
    mad = float(np.median(np.abs(ring - bg))) + 1e-6
    tol = max(25.0, 6.0 * mad)

    ok, why = [], []
    for name, (x0, y0, x1, y1) in corner_boxes(ww, hh):
        if leaf[y0:y1, x0:x1].any():
            why.append(f"{name}:leaf")
            continue
        patch = g[y0:y1, x0:x1]
        if float((np.abs(patch - bg) > tol).mean()) > MAX_OBJ_FRAC:
            why.append(f"{name}:object")
            continue
        ok.append(name)
    return im, ok, why


def patches(src):
    """224 x 224 patches of every accepted corner."""
    im, ok, why = accepted_corners(src)
    if not ok:
        return [], why
    boxes = dict(corner_boxes(im.width, im.height))
    out = [np.asarray(im.crop(boxes[n]).resize((224, 224), Image.BILINEAR),
                      dtype=np.float32) / 255.0 for n in ok]
    return out, why


def embed_arrays(groups, batch=48):
    """Embed all patches and average them back to one vector per image."""
    net = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    net.fc = nn.Identity()
    net.eval()
    torch.set_num_threads(8)

    flat, owner = [], []
    for i, ps in enumerate(groups):
        for p in ps:
            flat.append(p)
            owner.append(i)
    mean = np.array(MEAN, np.float32)
    std = np.array(STD, np.float32)
    feats = []
    with torch.no_grad():
        for i in range(0, len(flat), batch):
            xb = torch.from_numpy(
                np.stack([((p - mean) / std).transpose(2, 0, 1)
                          for p in flat[i:i + batch]]))
            feats.append(net(xb).numpy())
    F = np.concatenate(feats)
    X = np.zeros((len(groups), F.shape[1]), np.float32)
    for i in range(len(groups)):
        sel = [k for k, o in enumerate(owner) if o == i]
        X[i] = F[sel].mean(0)
    return X
