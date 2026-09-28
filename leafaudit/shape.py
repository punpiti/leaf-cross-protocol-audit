"""Size, shape and colour of the leaf, used by the feature baseline."""
import cv2
import numpy as np
from PIL import Image

MIN_PART = 0.08


def leaf_mask(rgb):
    """Leaf mask on the green-red axis, relative to the frame's own border."""
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
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((21, 21), np.uint8))
    n, lab_im, st, _ = cv2.connectedComponentsWithStats(m, 8)
    if n <= 1:
        return None, 0
    big = st[1:, cv2.CC_STAT_AREA].max()
    keep = np.zeros_like(m)
    parts = 0
    for i in range(1, n):
        if st[i, cv2.CC_STAT_AREA] > max(MIN_PART * big, 150):
            keep[lab_im == i] = 1
            parts += 1
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    return keep, parts


def measure(path):
    """Area fraction, aspect, solidity, part count and colour of the leaf in one image."""
    im = Image.open(path)
    im.draft("RGB", (1000, 750))
    rgb = np.asarray(im.convert("RGB"))
    mask, parts = leaf_mask(rgb)
    if mask is None or not mask.any():
        return dict(ok=0)
    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return dict(ok=0)
    pts = np.vstack(cnts)
    area = float(sum(cv2.contourArea(c) for c in cnts))
    if area < 200:
        return dict(ok=0)
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
    length, width = max(rw, rh), min(rw, rh)
    hull = float(cv2.contourArea(cv2.convexHull(pts)))
    blob = mask > 0
    cie = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    return dict(
        ok=1,
        area_frac=round(area / mask.size, 5),
        aspect=round(length / max(width, 1e-6), 3),
        solidity=round(area / max(hull, 1e-6), 3),
        n_parts=parts,
        lightness=round(float(cie[..., 0][blob].mean()), 2),
        greenness=round(float(cie[..., 1][blob].mean()) - 128, 2),
        saturation=round(float(hsv[..., 1][blob].mean()), 2),
        lightness_sd=round(float(cie[..., 0][blob].std()), 2),
    )
