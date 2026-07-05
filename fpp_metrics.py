"""FPP V4 Core Metrics — self-contained, zero external FPP dependencies."""
import numpy as np
from scipy import stats
from sklearn.feature_selection import mutual_info_regression

def pearson_r(x, y):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 3: return 0.0
    r, _ = stats.pearsonr(x[mask], y[mask])
    return float(r) if np.isfinite(r) else 0.0

def _entropy_estimate(x, n_bins=20):
    x = np.asarray(x).ravel(); x = x[np.isfinite(x)]
    if len(x) < 2: return 0.0
    counts, _ = np.histogram(x, bins=n_bins)
    probs = counts / counts.sum(); probs = probs[probs > 0]
    return -np.sum(probs * np.log2(probs))

def mutual_info(x, y, n_neighbors=3):
    x, y = np.asarray(x).ravel(), np.asarray(y).ravel()
    mask = np.isfinite(x) & np.isfinite(y)
    if mask.sum() < 10: return 0.0
    xc, yc = x[mask].reshape(-1,1), y[mask].reshape(-1,1)
    if len(np.unique(xc)) < 5: xc = xc + np.random.randn(*xc.shape)*1e-10
    if len(np.unique(yc)) < 5: yc = yc + np.random.randn(*yc.shape)*1e-10
    try:
        mi = mutual_info_regression(xc, yc.ravel(), n_neighbors=min(n_neighbors, len(xc)//5))
        hx = _entropy_estimate(xc.ravel()); hy = _entropy_estimate(yc.ravel())
        norm = np.sqrt(max(hx*hy, 1e-10))
        return float(np.clip(mi[0]/norm, 0, 1)) if norm > 1e-10 else 0.0
    except Exception: return 0.0

def deception_index(pearson_val, mi_val):
    return abs(pearson_val - mi_val)

def run_fpp(model, tokenizer, layers, text, n=10):
    """Run FPP measurement on any model with given layers."""
    gv, miv, phv, dcv, ipv = [], [], [], [], []
    for _ in range(n):
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=64).input_ids
        outs = {}; hooks = []
        def hk(i):
            def h(_, __, o): outs[i] = o[0].detach() if isinstance(o, tuple) else o.detach(); return None
            return h
        for i, _ in enumerate(layers): hooks.append(layers[i].register_forward_hook(hk(i)))
        with torch.no_grad():
            import torch
            _ = model(ids); fwd = dict(outs)
            _ = model(torch.flip(ids, [1])); rev = dict(outs)
        for h in hooks: h.remove()
        fv = [fwd[i].float() for i in sorted(fwd.keys())]
        rv = [rev[i].float() for i in sorted(rev.keys())]
        gs = np.mean([pearson_r(fv[i].cpu().numpy().flatten(), rv[len(rv)-1-i].cpu().numpy().flatten()) for i in range(len(fv))])
        mi = np.mean([mutual_info(fv[i].cpu().numpy().flatten()[:500], fv[-1].cpu().numpy().flatten()[:500]) for i in range(len(fv))])
        pv = pearson_r(fv[0].cpu().numpy().flatten(), fv[-1].cpu().numpy().flatten())
        dc = deception_index(pv, mi)
        nl = len(fv); corr = np.array([[pearson_r(fv[i].cpu().numpy().flatten(), fv[j].cpu().numpy().flatten()) for j in range(nl)] for i in range(nl)])
        ph = np.std(corr); ipr = 1.0 / max(np.sum(np.mean(np.abs(corr), axis=0) ** 2), 1e-8)
        gv.append(gs); miv.append(mi); phv.append(ph); dcv.append(dc); ipv.append(ipr)
    return {"gs": float(np.mean(gv)), "gs_std": float(np.std(gv)), "mi": float(np.mean(miv)), "mi_std": float(np.std(miv)),
            "ph": float(np.mean(phv)), "ph_std": float(np.std(phv)), "dc": float(np.mean(dcv)), "ipr": float(np.mean(ipv))}
