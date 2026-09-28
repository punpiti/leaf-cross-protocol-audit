"""Leaf segmentation on the a* axis of CIELAB, and the square crop around the leaf."""
import os

import cv2
import numpy as np
from PIL import Image

SIZE = 256
MARGIN = 0.15


def mask_of(rgb):
    """Leaf mask relative to a background estimated from the frame border."""
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB).astype(np.float32)
    A = cv2.GaussianBlur(lab[..., 1], (9, 9), 0)
    h, w = A.shape
    b = max(8, min(h, w) // 25)
    border = np.concatenate([A[:b].ravel(), A[-b:].ravel(),
                             A[:, :b].ravel(), A[:, -b:].ravel()])
    bg = float(np.median(border))
    spread = float(np.median(np.abs(border - bg))) + 1e-6
    m = ((bg - A) > max(4.0, 5.0 * spread)).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    n, lab_im, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None
    big = st[1:, cv2.CC_STAT_AREA].max()
    keep = np.zeros_like(m)
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > max(0.08 * big, 30):
            keep[lab_im == i] = 1
    return keep


def crop_leaf(src, dst, mdst):
    """Write a SIZE x SIZE crop around the leaf and its mask; return the leaf fraction of the crop."""
    im = Image.open(src)
    im.draft("RGB", (1024, 1024))
    im = im.convert("RGB")
    W, H = im.size
    small = im.copy()
    small.thumbnail((640, 640), Image.BILINEAR)
    m = mask_of(np.asarray(small))
    if m is None:
        return 0.0
    ys, xs = np.nonzero(m)
    sx, sy = W / small.width, H / small.height
    x0, x1 = xs.min() * sx, (xs.max() + 1) * sx
    y0, y1 = ys.min() * sy, (ys.max() + 1) * sy
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    side = max(x1 - x0, y1 - y0) * (1 + 2 * MARGIN)
    side = max(side, 32.0)
    L, T = cx - side / 2, cy - side / 2
    crop = im.crop((int(round(L)), int(round(T)),
                    int(round(L + side)), int(round(T + side))))
    crop = crop.resize((SIZE, SIZE), Image.LANCZOS)
    cm = mask_of(np.asarray(crop))
    if cm is None:
        return 0.0
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    crop.save(dst, quality=95)
    Image.fromarray(cm * 255).save(mdst)
    return float(cm.mean())
