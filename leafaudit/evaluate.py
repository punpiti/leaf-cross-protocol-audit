"""Grouped cross-validation, the empirical null and the paired group bootstrap.

Each row is a dict with keys cls, group, cam, fold and image_id; X holds one
feature row per image in the same order.
"""
from collections import defaultdict

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 20260918


def evaluate(rows, X, train_cam, test_cam, folds, agg, labels=None):
    """Mean and SD over folds of macro-F1, predictions aggregated per image, side or group."""
    scores = []
    for fold in range(folds):
        tri = [i for i, r in enumerate(rows) if r["fold"] != fold
               and (train_cam == "both" or r["cam"] == train_cam)]
        tei = [i for i, r in enumerate(rows) if r["fold"] == fold
               and (test_cam == "both" or r["cam"] == test_cam)]
        if not tri or not tei:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000,
                                               class_weight="balanced"))
        clf.fit(X[tri], [rows[i]["cls"] for i in tri])
        P = clf.predict_proba(X[tei])
        order = list(clf.classes_)
        acc, truth = defaultdict(lambda: np.zeros(len(order))), {}
        for j, i in enumerate(tei):
            r = rows[i]
            k = ((r["image_id"],) if agg == "image"
                 else (r["group"], r["cam"]) if agg == "sides"
                 else (r["group"],))
            acc[k] += P[j]
            truth[k] = r["cls"]
        ks = list(acc)
        scores.append(f1_score([truth[k] for k in ks],
                               [order[int(np.argmax(acc[k]))] for k in ks],
                               labels=labels, average="macro",
                               zero_division=0))
    return (float(np.mean(scores)), float(np.std(scores))) if scores else (0, 0)


def oof(rows, Xtr, Xte, folds=5):
    """Out-of-fold predictions, one per test unit (group x camera), for all camera pairs."""
    out = {}
    for tr_cam, te_cam in (("canon", "phone"), ("phone", "canon"),
                           ("canon", "canon"), ("phone", "phone")):
        pred, truth, grp = {}, {}, {}
        for fold in range(folds):
            tri = [i for i, r in enumerate(rows)
                   if r["fold"] != fold and r["cam"] == tr_cam]
            tei = [i for i, r in enumerate(rows)
                   if r["fold"] == fold and r["cam"] == te_cam]
            if not tri or not tei:
                continue
            clf = make_pipeline(StandardScaler(),
                                LogisticRegression(max_iter=3000,
                                                   class_weight="balanced"))
            clf.fit(Xtr[tri], [rows[i]["cls"] for i in tri])
            P = clf.predict_proba(Xte[tei])
            order = list(clf.classes_)
            acc = defaultdict(lambda: np.zeros(len(order)))
            for j, i in enumerate(tei):
                k = (rows[i]["group"], rows[i]["cam"])
                acc[k] += P[j]
                truth[k] = rows[i]["cls"]
                grp[k] = rows[i]["group"]
            for k, v in acc.items():
                pred[k] = order[int(np.argmax(v))]
        out[f"{tr_cam}->{te_cam}"] = (pred, truth, grp)
    return out


def macro(pred, truth, keys, labels=None):
    """Macro-F1 over the given units, with the label set fixed by the caller."""
    return f1_score([truth[k] for k in keys], [pred[k] for k in keys],
                    labels=labels, average="macro", zero_division=0)


def boot_diff(a, b, n=2000, rng=None, labels=None):
    """Paired bootstrap over groups of the macro-F1 difference between two conditions."""
    rng = rng or np.random.default_rng(SEED)
    pa, ta, ga = a
    pb, tb, _ = b
    keys = [k for k in pa if k in pb]
    if not keys:
        return None
    bygroup = defaultdict(list)
    for k in keys:
        bygroup[ga[k]].append(k)
    groups = list(bygroup)
    d0 = macro(pa, ta, keys, labels) - macro(pb, tb, keys, labels)
    ds = np.empty(n)
    for i in range(n):
        gs = rng.choice(len(groups), len(groups), replace=True)
        ks = [k for j in gs for k in bygroup[groups[j]]]
        ds[i] = macro(pa, ta, ks, labels) - macro(pb, tb, ks, labels)
    lo, hi = np.percentile(ds, [2.5, 97.5])
    return dict(diff=float(d0), lo=float(lo), hi=float(hi),
                n_groups=len(groups), n_units=len(keys))


def null_of(rows, X, train_cam, test_cam, folds, perm, labels):
    """Pooled macro-F1 and its empirical null from permuting true against predicted labels."""
    pairs = []
    for fold in range(folds):
        tri = [i for i, r in enumerate(rows)
               if r["fold"] != fold and r["cam"] == train_cam]
        tei = [i for i, r in enumerate(rows)
               if r["fold"] == fold and r["cam"] == test_cam]
        if not tri or not tei:
            continue
        clf = make_pipeline(StandardScaler(),
                            LogisticRegression(max_iter=3000,
                                               class_weight="balanced"))
        clf.fit(X[tri], [rows[i]["cls"] for i in tri])
        P = clf.predict_proba(X[tei])
        order = list(clf.classes_)
        acc, truth = defaultdict(lambda: np.zeros(len(order))), {}
        for j, i in enumerate(tei):
            k = (rows[i]["group"], rows[i]["cam"])
            acc[k] += P[j]
            truth[k] = rows[i]["cls"]
        pairs += [(truth[k], order[int(np.argmax(v))]) for k, v in acc.items()]

    def pooled(ps):
        y, q = zip(*ps)
        return float(f1_score(y, q, labels=labels, average="macro",
                              zero_division=0))

    yt = np.array([t for t, _ in pairs])
    yp = np.array([q for _, q in pairs])
    obs = pooled(pairs)
    rng = np.random.default_rng(SEED)
    null = np.array([pooled(list(zip(rng.permutation(yt), yp)))
                     for _ in range(perm)])
    _, cnt = np.unique(yp, return_counts=True)
    return dict(pooled_macro_f1=obs, null_mean=float(null.mean()),
                null_hi=float(np.percentile(null, 97.5)),
                null_p=float((null >= obs).mean()),
                n_pred_classes=int(len(cnt)),
                pred_max_share=float(cnt.max() / cnt.sum()),
                n_units=len(pairs))
