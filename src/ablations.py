"""Ablations computed from saved logits.

  python src/ablations.py --out $OUT                    # everything that has data
  python src/ablations.py --out $OUT --which sweep,signals

sweep    number of shadow models K, fixed vs per-example variance (mean/std over random K-subsets)
signals  LiRA with different per-example signals (logit-scaled, true logit, negative loss)
overfit  targets trained for different numbers of epochs, each attacked with shadows of the SAME recipe
seeds    main recipe, several target seeds: mean/std of each attack
"""
import argparse, glob, json, os
import numpy as np
from attacks import SIGNALS, logit_scaled, neg_loss, lira_offline, lira_online, metrics

KEYS = ["auc", "bal_acc", "tpr@0.001", "tpr@0.01", "tpr@0.05", "tpr@0.1"]
SHOW = ["auc", "tpr@0.001", "tpr@0.01", "tpr@0.1"]

def lira(pool, obs, S, M, fixed):
    return (lira_offline if pool == "offline" else lira_online)(obs, S, M, fixed_var=fixed)

def load_pool(folder, sig, limit=0):
    files = sorted(glob.glob(os.path.join(folder, "*.npz")))
    files = files[:limit] if limit else files
    Z = [np.load(f) for f in files]
    return np.stack([sig(z["logits"], z["labels"]) for z in Z]), np.stack([z["member"] for z in Z])

def target(out, name="target/target_seed0.npz"):
    t = np.load(os.path.join(out, name)); return t["logits"], t["labels"], t["member"]

def row(m, keys=SHOW): return "".join(f"{m[k]:>11.4f}" for k in keys)
def header(first, keys=SHOW): return f"{first:38s}" + "".join(f"{k:>11s}" for k in keys)

def sweep(out, res, R=5, ks=(2, 4, 8, 16, 32, 64, 128, 256)):
    tl, y, member = target(out); obs = logit_scaled(tl, y)
    print("\n== SWEEP: number of shadow models (mean +- std over random subsets)")
    for pool in ("offline", "online"):
        folder = os.path.join(out, f"shadow_{pool}")
        if not glob.glob(folder + "/*.npz"): continue
        S, M = load_pool(folder, logit_scaled); rng = np.random.default_rng(0)
        for fixed in (False, True):
            print(header(f"{pool} / {'fixed' if fixed else 'per-example'} var"))
            for k in ks:
                if k > len(S): continue
                ms = [metrics(lira(pool, obs, S[i], M[i], fixed), member)[0]
                      for i in (rng.choice(len(S), k, replace=False) for _ in range(R if k < len(S) else 1))]
                agg = {c: (float(np.mean([m[c] for m in ms])), float(np.std([m[c] for m in ms]))) for c in KEYS}
                res[f"sweep|{pool}|{'fixed' if fixed else 'per_example'}|{k}"] = agg
                print(f"  K={k:<34d}" + "".join(f"{agg[c][0]:>7.3f}+-{agg[c][1]:.3f}" for c in SHOW))

def signals(out, res):
    tl, y, member = target(out)
    print("\n== SIGNALS: LiRA with different signals (all shadows)")
    print(header("signal / pool / variance"))
    for name, sig in SIGNALS.items():
        obs = sig(tl, y)
        for pool in ("offline", "online"):
            folder = os.path.join(out, f"shadow_{pool}")
            if not glob.glob(folder + "/*.npz"): continue
            S, M = load_pool(folder, sig)
            for fixed in (False, True):
                m = metrics(lira(pool, obs, S, M, fixed), member)[0]
                res[f"signals|{name}|{pool}|{'fixed' if fixed else 'per_example'}"] = m
                print(f"{name + ' / ' + pool + ' / ' + ('fixed' if fixed else 'per-ex'):38s}" + row(m))

def overfit(out, res, k=16):
    tags = sorted(os.path.basename(f)[len("target_seed0_"):-4] for f in glob.glob(os.path.join(out, "target/target_seed0_*.npz")))
    print("\n== OVERFIT: epochs of target and matching shadows (offline LiRA, K<=%d)" % k)
    print(header("recipe", ["train_acc", "test_acc", "loss_gap"] + SHOW))
    for tag in ["main"] + tags:
        tname = "target/target_seed0.npz" if tag == "main" else f"target/target_seed0_{tag}.npz"
        folder = os.path.join(out, "shadow_offline" if tag == "main" else f"shadow_offline_{tag}")
        if not os.path.exists(os.path.join(out, tname)) or not glob.glob(folder + "/*.npz"): continue
        tl, y, member = target(out, tname)
        acc = tl.argmax(1) == y; loss = -neg_loss(tl, y)
        S, M = load_pool(folder, logit_scaled, k); obs = logit_scaled(tl, y)
        ms = {"LOSS attack": metrics(neg_loss(tl, y), member)[0],
              f"LiRA offline fixed (K={len(S)})": metrics(lira("offline", obs, S, M, True), member)[0]}
        for n, m in ms.items():
            res[f"overfit|{tag}|{n}"] = {**m, "train_acc": float(acc[member].mean()), "test_acc": float(acc[~member].mean()),
                                        "loss_gap": float(loss[~member].mean() - loss[member].mean())}
            print(f"{tag + ' / ' + n:38s}{acc[member].mean():>11.3f}{acc[~member].mean():>11.3f}{loss[~member].mean() - loss[member].mean():>11.4f}" + row(m))

def seeds(out, res):
    files = sorted(f for f in glob.glob(os.path.join(out, "target/target_seed[0-9]*.npz")) if os.path.basename(f)[11:-4].isdigit())
    if len(files) < 2: return
    print(f"\n== SEEDS: {len(files)} target seeds, main recipe (mean +- std)")
    pools = {p: load_pool(os.path.join(out, f"shadow_{p}"), logit_scaled) for p in ("offline", "online")
             if glob.glob(os.path.join(out, f"shadow_{p}/*.npz"))}
    acc = {}
    for f in files:
        tl, y, member = np.load(f)["logits"], np.load(f)["labels"], np.load(f)["member"]; obs = logit_scaled(tl, y)
        acc.setdefault("LOSS attack", []).append(metrics(neg_loss(tl, y), member)[0])
        for p, (S, M) in pools.items():
            for fixed in (False, True):
                acc.setdefault(f"LiRA {p} ({'fixed' if fixed else 'per-example'} var)", []).append(metrics(lira(p, obs, S, M, fixed), member)[0])
    print(header("attack"))
    for n, ms in acc.items():
        agg = {c: (float(np.mean([m[c] for m in ms])), float(np.std([m[c] for m in ms]))) for c in KEYS}
        res[f"seeds|{n}"] = agg
        print(f"{n:38s}" + "".join(f"{agg[c][0]:>6.3f}+-{agg[c][1]:.3f}" for c in SHOW))

def plot(res, path):
    try:
        import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    except ImportError: return
    fig, axs = plt.subplots(1, 2, figsize=(9, 3.5))
    for ax, key, title in zip(axs, ("auc", "tpr@0.01"), ("AUC", "TPR @ 1% FPR")):
        for pool in ("offline", "online"):
            for var in ("fixed", "per_example"):
                pts = sorted((int(k.split("|")[-1]), v[key]) for k, v in res.items() if k.startswith(f"sweep|{pool}|{var}|"))
                if pts: ax.errorbar([p[0] for p in pts], [p[1][0] for p in pts], [p[1][1] for p in pts], marker="o", ms=3, capsize=2, label=f"{pool}/{var}")
        ax.set_xscale("log", base=2); ax.set_xlabel("number of shadow models"); ax.set_title(title)
    axs[0].legend(fontsize=7); fig.tight_layout(); fig.savefig(path, dpi=150)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--which", default="sweep,signals,overfit,seeds")
    ap.add_argument("--save", default="results/ablations.json")
    a = ap.parse_args()
    res = {}
    for w in a.which.split(","): {"sweep": sweep, "signals": signals, "overfit": overfit, "seeds": seeds}[w](a.out, res)
    os.makedirs(os.path.dirname(a.save) or ".", exist_ok=True)
    json.dump(res, open(a.save, "w"), indent=1)
    plot(res, a.save.replace(".json", "_sweep.png"))

if __name__ == "__main__":
    main()
