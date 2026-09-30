"""LOSS / logit baselines and LiRA (offline + online) computed purely from saved logits.

  python src/attacks.py --out $OUT                      # main comparison table + ROC plot
  python src/attacks.py --out $OUT --n-shadows 16       # fewer shadows

Everything is evaluated on ALL rows; "member" = the target's training split.
"""
import argparse, glob, json, os, warnings
import numpy as np
from scipy.special import logsumexp
from sklearn.metrics import roc_curve, roc_auc_score

warnings.filterwarnings("ignore", category=RuntimeWarning)   # nanmean over empty slices is handled explicitly below
FPRS = (0.001, 0.01, 0.05, 0.1)

# ---------------------------------------------------------------- signals (higher = more member-like)
def true_logit(logits, y):
    return logits[np.arange(len(y)), y]

def neg_loss(logits, y):
    return true_logit(logits, y) - logsumexp(logits, axis=1)          # log p(y|x) = -CE

def logit_scaled(logits, y):
    """LiRA's phi = log(p/(1-p)) = z_y - logsumexp(z_other), stable for p close to 1."""
    z = logits.copy(); zy = z[np.arange(len(y)), y]; z[np.arange(len(y)), y] = -np.inf
    return zy - logsumexp(z, axis=1)

SIGNALS = {"neg_loss": neg_loss, "true_logit": true_logit, "logit_scaled": logit_scaled}

# ---------------------------------------------------------------- metrics
def metrics(score, member):
    fpr, tpr, _ = roc_curve(member, score)
    out = {"auc": float(roc_auc_score(member, score)), "bal_acc": float(np.max(0.5 * (tpr + 1 - fpr)))}
    for f in FPRS: out[f"tpr@{f:g}"] = float(tpr[fpr <= f].max())
    return out, (fpr, tpr)

# ---------------------------------------------------------------- LiRA
def _stats(S, mask):
    """Per-example mean/variance over the shadows selected by mask. Examples with < 2 samples get the pooled variance."""
    x = np.where(mask, S, np.nan)
    n = mask.sum(0)
    mu = np.nanmean(x, 0)
    var = np.nanvar(x, 0, ddof=1)
    pooled = np.nanmean(var)
    return mu, np.where(n >= 2, var, pooled), pooled

def lira_offline(obs, S, M, fixed_var=False):
    """obs: (N,) target signal. S: (K,N) shadow signals, M: (K,N) shadow membership. Only OUT shadows are used."""
    mu, var, pooled = _stats(S, ~M)
    mu = np.where(np.isnan(mu), np.nanmean(mu), mu)
    sd = np.sqrt(pooled) if fixed_var else np.sqrt(var)
    return (obs - mu) / (sd + 1e-8)                                       # monotone in 1 - P[Z > obs], Z ~ N(mu_out, sd^2)

def lira_online(obs, S, M, fixed_var=False):
    """Likelihood ratio of N(mu_in, sd_in^2) vs N(mu_out, sd_out^2) at the observed signal."""
    mi, vi, pi = _stats(S, M)
    mo, vo, po = _stats(S, ~M)
    shift = np.nanmean(mi - mo)                                           # fill examples with no in/out shadow at all
    mi = np.where(np.isnan(mi), mo + shift, mi); mo = np.where(np.isnan(mo), mi - shift, mo)
    mi = np.where(np.isnan(mi), np.nanmean(mi), mi); mo = np.where(np.isnan(mo), np.nanmean(mo), mo)
    if fixed_var: vi = np.full_like(vi, pi); vo = np.full_like(vo, po)
    vi, vo = vi + 1e-8, vo + 1e-8
    ll = lambda m, v: -0.5 * np.log(2 * np.pi * v) - 0.5 * (obs - m) ** 2 / v
    return ll(mi, vi) - ll(mo, vo)

# ---------------------------------------------------------------- IO
def load_pool(folder, n, sig, rng=None):
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    if rng is not None: files = list(rng.permutation(files))
    files = files[:n] if n else files
    Z = [np.load(f) for f in files]
    return np.stack([sig(z["logits"], z["labels"]) for z in Z]), np.stack([z["member"] for z in Z]), len(files)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--target", default="target/target_seed0.npz")
    ap.add_argument("--n-shadows", type=int, default=0, help="0 = use all available")
    ap.add_argument("--save", default="results/attack_main.json")
    a = ap.parse_args()

    t = np.load(os.path.join(a.out, a.target))
    y, member, tl = t["labels"], t["member"], t["logits"]
    rows, curves = {}, {}
    def add(name, score):
        m, c = metrics(score, member); rows[name] = m; curves[name] = c

    add("LOSS attack (neg_loss)", neg_loss(tl, y))
    add("Logit attack (true_logit)", true_logit(tl, y))
    obs = logit_scaled(tl, y)
    for pool in ("offline", "online"):
        folder = os.path.join(a.out, f"shadow_{pool}")
        if not glob.glob(os.path.join(folder, "*.npz")): continue
        S, M, k = load_pool(folder, a.n_shadows, logit_scaled)
        f = lira_offline if pool == "offline" else lira_online
        add(f"LiRA {pool} (per-example var, K={k})", f(obs, S, M))
        add(f"LiRA {pool} (fixed var, K={k})", f(obs, S, M, fixed_var=True))

    cols = ["auc", "bal_acc"] + [f"tpr@{x:g}" for x in FPRS]
    print(f"{'attack':42s}" + "".join(f"{c:>10s}" for c in cols))
    for n, m in rows.items(): print(f"{n:42s}" + "".join(f"{m[c]:10.4f}" for c in cols))
    os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
    json.dump(rows, open(a.save, "w"), indent=1)
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5, 5))
        for n, (fpr, tpr) in curves.items(): ax.plot(fpr, tpr, label=n)
        ax.plot([1e-4, 1], [1e-4, 1], "k:", lw=0.8); ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(1e-3, 1); ax.set_ylim(1e-3, 1)
        ax.set_xlabel("FPR"); ax.set_ylabel("TPR"); ax.legend(fontsize=6); fig.tight_layout()
        fig.savefig(a.save.replace(".json", "_roc.png"), dpi=150)
    except ImportError:
        print("(matplotlib not installed: skipped ROC plot)")

if __name__ == "__main__":
    main()
