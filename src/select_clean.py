"""Build the clean real-seed pool (~9k): drop noisy/conflicting rows, then prune hub intents of the confusion graph."""
import sys
from collections import Counter
import numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

TARGET = int(sys.argv[1]) if len(sys.argv) > 1 else 9000

def main():
    d = pd.read_parquet("data/seeds_audited.parquet")
    d["votes"] = d.suspect.astype(int) + d.lr_bad.astype(int) + d.knn_bad.astype(int)
    p = np.load("data/seed_cvprob_bge.npy")
    d["cv_pred"] = p.argmax(1)

    # row-level: any detector flag, or in a near-duplicate pair with a different label
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(d.text.str.lower())
    dist, idx = NearestNeighbors(n_neighbors=4, metric="cosine").fit(X).kneighbors(X)
    conflict = np.zeros(len(d), bool)
    for i in range(len(d)):
        for dj, j in zip(dist[i, 1:], idx[i, 1:]):
            if 1 - dj >= 0.85 and d.label.values[i] != d.label.values[j]: conflict[i] = conflict[j] = True
    d["conflict_nd"] = conflict
    keep = d[(d.votes == 0) & ~d.conflict_nd].copy()
    print(f"row-level: {len(d)} -> {len(keep)} (flagged {int((d.votes>0).sum())}, conflicting near-dup {int(conflict.sum())})")

    # intent-level: confusion graph from CV predictions on the rows we kept; drop hub intents greedily
    kk = keep[keep.cv_pred != keep.label]
    mass = Counter()
    edges = Counter()
    for a, b in zip(kk.intent, kk.cv_pred.map(dict(zip(d.label, d.intent)))):
        edges[a] += 1; edges[b] += 1; mass[tuple(sorted((a, b)))] += 1
    dropped = []
    while len(keep) > TARGET:
        deg = Counter()
        for (a, b), n in mass.items(): deg[a] += n; deg[b] += n
        if not deg: break
        worst = deg.most_common(1)[0][0]
        dropped.append((worst, int(deg[worst]), int((keep.intent == worst).sum())))
        keep = keep[keep.intent != worst]
        mass = Counter({k: v for k, v in mass.items() if worst not in k})
    print(f"intent-level: dropped {len(dropped)} intents -> {len(keep)} rows, {keep.intent.nunique()} intents")
    for n, m, r in dropped: print(f"  drop {n:45s} confusion={m:3d} rows={r}")
    keep.drop(columns=["cv_pred"]).to_parquet("data/clean_pool.parquet")
    print("per-intent rows: min", keep.intent.value_counts().min(), "max", keep.intent.value_counts().max())

if __name__ == "__main__":
    main()
