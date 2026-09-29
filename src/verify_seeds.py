"""Independent audit of data/seeds.parquet: different source, normalization, and detectors than prep_seeds.py."""
import re, unicodedata, itertools
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sentence_transformers import SentenceTransformer

def strict(s):  # NFKC + casefold + alnum only (no spaces at all)
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", s).casefold())

def main():
    # ---- independent source: mteb/banking77 jsonl
    raw = pd.concat([pd.read_json(hf_hub_download("mteb/banking77", f"{s}.jsonl", repo_type="dataset"), lines=True).assign(orig_split=s)
                     for s in ("train", "test")], ignore_index=True)
    print("mteb columns:", list(raw.columns), "rows:", len(raw))
    raw = raw.rename(columns={c: "intent" for c in raw.columns if c in ("label_text", "intent")})
    saved = pd.read_parquet("data/seeds.parquet")

    print("\n== 1. SOURCE CROSS-CHECK")
    a, b = set(map(strict, raw["text"])), set(map(strict, saved["text"]))
    print(f"unique(strict) mteb={len(a)} saved={len(b)}  only-in-mteb={len(a-b)} only-in-saved={len(b-a)}")

    print("\n== 2. EXACT DUPLICATES (strict norm, space-free)")
    print("saved rows:", len(saved), " unique strict:", saved["text"].map(strict).nunique())
    print("saved unique bag-of-words (order-insensitive):", saved["text"].map(lambda s: " ".join(sorted(re.findall(r"\w+", s.lower())))).nunique())
    d = raw[raw["text"].map(strict).duplicated(keep=False)]
    print(f"raw dup rows: {len(d)}; spanning train/test: {d.groupby(d['text'].map(strict)).orig_split.nunique().gt(1).sum()} groups; label-conflicting:",
          d.groupby(d['text'].map(strict)).intent.nunique().gt(1).sum())

    print("\n== 3. NEAR DUPLICATES (char 3-5gram TF-IDF cosine) in saved set")
    X = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit_transform(saved["text"].str.lower())
    nn = NearestNeighbors(n_neighbors=4, metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    pairs = {}
    for i in range(len(saved)):
        for dj, j in zip(dist[i, 1:], idx[i, 1:]):
            sim = 1 - dj
            if sim >= 0.85 and i != j: pairs[tuple(sorted((i, j)))] = sim
    P = pd.DataFrame([(i, j, s) for (i, j), s in pairs.items()], columns=["i", "j", "sim"])
    P["same_label"] = saved.label.values[P.i] == saved.label.values[P.j]
    P["cross_split"] = saved.orig_split.values[P.i] != saved.orig_split.values[P.j]
    for t in (0.85, 0.9, 0.95):
        q = P[P.sim >= t]
        print(f"sim>={t}: {len(q)} pairs | cross orig train/test: {q.cross_split.sum()} | DIFFERENT labels: {(~q.same_label).sum()}")
    diff = P[~P.same_label].sort_values("sim", ascending=False)
    print("\n-- near-identical pairs with DIFFERENT labels (model-free evidence of ambiguity):")
    for _, r in diff.head(15).iterrows():
        print(f"  {r.sim:.2f} | {saved.text[r.i]!r} [{saved.intent[r.i]}]  <->  {saved.text[r.j]!r} [{saved.intent[r.j]}]")
    print(f"total conflicting near-dup pairs @0.85: {len(diff)}; distinct rows involved: {len(set(diff.i)|set(diff.j))}")

    print("\n== 4. INDEPENDENT LABEL AUDIT")
    emb = SentenceTransformer("BAAI/bge-small-en-v1.5").encode(saved["text"].tolist(), batch_size=256, normalize_embeddings=True)
    y = saved.label.values
    cv = StratifiedKFold(5, shuffle=True, random_state=123)
    p_lr = cross_val_predict(LogisticRegression(max_iter=3000, C=20), emb, y, cv=cv, method="predict_proba")
    # kNN vote (k=10, leave-self-out) as a second, non-parametric detector
    knn = NearestNeighbors(n_neighbors=11, metric="cosine").fit(emb)
    _, kidx = knn.kneighbors(emb)
    vote_true = np.array([(y[kidx[i, 1:]] == y[i]).mean() for i in range(len(y))])
    lr_bad = p_lr[np.arange(len(y)), y] < 0.1
    knn_bad = vote_true == 0
    both = lr_bad & knn_bad
    prev = saved.suspect.values
    print(f"LR(bge) true-label prob<0.1: {lr_bad.sum()} | kNN 0/10 neighbours agree: {knn_bad.sum()} | both: {both.sum()}")
    print(f"previous flags (cleanlab+MiniLM): {prev.sum()} | overlap with 'both': {(prev & both).sum()} | overlap with either: {(prev & (lr_bad | knn_bad)).sum()}")
    # original protocol check: train on original train, predict original test
    tr, te = (saved.orig_split == "train").values, (saved.orig_split == "test").values
    clf = LogisticRegression(max_iter=3000, C=20).fit(emb[tr], y[tr])
    print(f"train->test accuracy (bge+LR): {(clf.predict(emb[te]) == y[te]).mean():.3f}  (a very low value would signal noisy test labels)")

    print("\n-- 25 random rows flagged by BOTH new detectors (read them yourself):")
    names = saved.intent.values
    cls = dict(zip(saved.label, saved.intent))
    for i in np.random.default_rng(0).permutation(np.where(both)[0])[:25]:
        print(f"  {saved.text[i]!r}  labeled={names[i]}  model says={cls[p_lr[i].argmax()]}  prev_flag={prev[i]}")
    print("\n-- 10 random rows flagged by previous pass but NOT by new detectors:")
    for i in np.random.default_rng(1).permutation(np.where(prev & ~(lr_bad | knn_bad))[0])[:10]:
        print(f"  {saved.text[i]!r}  labeled={names[i]}  model says={cls[p_lr[i].argmax()]}")
    out = saved.assign(lr_bad=lr_bad, knn_bad=knn_bad, both_bad=both)
    out["votes"] = out.suspect.astype(int) + out.lr_bad.astype(int) + out.knn_bad.astype(int)
    out["suspect_consensus"] = out.votes >= 2                                # flagged by >= 2 of 3 detectors
    out.to_parquet("data/seeds_audited.parquet")
    # artifacts used by select_clean.py (confusion graph over the same embeddings)
    np.save("data/seed_emb_bge.npy", emb)
    np.save("data/seed_cvprob_bge.npy", cross_val_predict(LogisticRegression(max_iter=3000, C=20), emb, y,
                                                          cv=StratifiedKFold(5, shuffle=True, random_state=7), method="predict_proba"))

if __name__ == "__main__":
    main()
