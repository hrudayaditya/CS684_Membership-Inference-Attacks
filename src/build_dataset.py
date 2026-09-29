"""Verified synthetic rows -> deduped, balanced, PII-filled dataset with a stratified 1/3 train / 2/3 test split
in which near-duplicate clusters never cross the split."""
import argparse, datetime, re, random, unicodedata
import numpy as np, pandas as pd
from faker import Faker
from sentence_transformers import SentenceTransformer

END = datetime.date(2026, 1, 1)   # fixed so fake dates do not depend on the clock
TYPES = {"NAME": "name", "AMOUNT": "amount", "LAST4": "card_digits", "MERCHANT": "merchant", "CITY": "location", "DATE": "date", "REF": "reference"}
MERCHANTS = ["Amazon", "Starbucks", "Tesco", "Uber", "Netflix", "Spotify", "Walmart", "Target", "IKEA", "Zara", "Airbnb", "Apple Store",
             "Best Buy", "Lidl", "Deliveroo", "Ryanair", "Booking.com", "Nike", "eBay", "Shell"]

def strict(s):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", s).casefold())

def fill(t, fk, rng):
    """Fresh fake values per placeholder occurrence (Faker + RNG only; nothing real)."""
    gen = {"NAME": lambda: fk.first_name() if rng.random() < 0.6 else fk.name(),
           "AMOUNT": lambda: rng.choice(["£", "£", "£", "$", "€", ""]) + str(rng.choice([rng.randint(5, 99), rng.randint(100, 999), round(rng.uniform(5, 900), 2)])),
           "LAST4": lambda: f"{rng.randint(0, 9999):04d}", "MERCHANT": lambda: rng.choice(MERCHANTS), "CITY": fk.city,
           "DATE": lambda: rng.choice([fk.date_object(END).strftime("%d/%m"), fk.date_object(END).strftime("%-d %B"), f"the {rng.randint(1, 28)}th"]),
           "REF": lambda: rng.choice([f"TXN-{rng.randint(100000, 999999)}", f"{rng.randint(10**7, 10**8 - 1)}", f"REF{rng.randint(1000, 9999)}X"])}
    return re.sub(r"\{(\w+)\}", lambda m: str(gen[m.group(1)]()), t)

def components(E, thr):
    """Connected components of the cosine >= thr graph (union-find)."""
    par = list(range(len(E)))
    def f(x):
        while par[x] != x: par[x] = par[par[x]]; x = par[x]
        return x
    S = E @ E.T
    for i, j in zip(*np.where(np.triu(S >= thr, 1))): par[f(i)] = f(j)
    return np.array([f(i) for i in range(len(E))])

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/synth_verified.jsonl")
    ap.add_argument("--out", default="data/dataset.parquet")
    ap.add_argument("--per-intent", type=int, default=160)
    ap.add_argument("--noisy-quota", type=int, default=0, help="per intent, take up to this many llm_noisy rows first")
    ap.add_argument("--sim", type=float, default=0.96, help="within-synthetic near-duplicate cosine threshold (bge)")
    ap.add_argument("--real-sim", type=float, default=0.97, help="drop synthetic rows this close to any real seed")
    ap.add_argument("--split-sim", type=float, default=0.95, help="rows this similar are kept on the same side of the split")
    ap.add_argument("--max-words", type=int, default=100, help="drop degenerate generations (e.g. several messages concatenated)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng, fk = random.Random(a.seed), Faker("en_US"); Faker.seed(a.seed)
    slots = set(TYPES)

    df = pd.read_json(a.inp, lines=True)
    if "noise_type" not in df: df["noise_type"] = "clean"
    n0 = len(df)
    df = df[df.verified]; n_ver = len(df)
    df = df[df.text.str.split().str.len() <= a.max_words]
    df = df[df.text.map(lambda t: set(re.findall(r"\{(\w+)\}", t)) <= slots)]; n_slot = len(df)
    real = pd.read_parquet("data/seeds_audited.parquet")                 # ALL real seeds, incl. dropped ones
    df = df[~df.text.map(strict).isin(set(real.text.map(strict)))]
    df = df.assign(k=df.text.map(strict)).drop_duplicates("k").drop(columns="k").sample(frac=1, random_state=a.seed).reset_index(drop=True); n_exact = len(df)

    m = SentenceTransformer("BAAI/bge-small-en-v1.5")
    E = m.encode(df.text.tolist(), batch_size=256, normalize_embeddings=True)
    R = m.encode(real.text.tolist(), batch_size=256, normalize_embeddings=True)
    keep = np.ones(len(df), bool)
    keep &= (E @ R.T).max(1) < a.real_sim                                  # too close to a real seed
    for i, g in df.groupby("intent").indices.items():                      # greedy near-dup removal within intent
        kept = []
        for j in g:
            if not keep[j]: continue
            if kept and (E[kept] @ E[j]).max() >= a.sim: keep[j] = False
            else: kept.append(j)
    df = df[keep].assign(ei=np.where(keep)[0]); n_near = len(df)

    parts = []
    for _, g in df.groupby("intent"):
        nz = g[g.noise_type == "llm_noisy"].head(a.noisy_quota)
        parts.append(pd.concat([nz, g.drop(nz.index)]).head(a.per_intent))
    df = pd.concat(parts).reset_index(drop=True)

    df["text_template"] = df.text
    df["pii_types"] = df.text.map(lambda t: sorted({TYPES[s] for s in re.findall(r"\{(\w+)\}", t)}))
    df["has_pii"] = df.pii_types.map(len) > 0
    df["text"] = [fill(t, fk, rng) for t in df.text]
    df["label"] = df.intent.astype("category").cat.codes

    # stratified 1/3 train / 2/3 test; near-dup clusters (cos >= split-sim) move together
    df["split"] = "test"
    for _, g in df.groupby("intent"):
        comp = components(E[g.ei.values], a.split_sim)
        cids = list(dict.fromkeys(comp)); rng.shuffle(cids)
        want, got = round(len(g) / 3), 0
        for c in cids:
            n = int((comp == c).sum())
            if got >= want: break
            if got + n > want + 2: continue                                # don't let a big cluster overshoot the quota
            df.loc[g.index[comp == c], "split"] = "train"; got += n
    cross = 0                                                              # audit: any cross-split pair above split-sim?
    for _, g in df.groupby("intent"):
        S = E[g.ei.values] @ E[g.ei.values].T; sp = (g.split == "train").values
        cross += int(((S >= a.split_sim) & (sp[:, None] != sp[None, :])).sum() // 2)
    df[["text", "text_template", "intent", "label", "has_pii", "pii_types", "noise_type", "split"]].to_parquet(a.out)
    print(f"generated {n0} -> verified {n_ver} -> valid slots {n_slot} -> exact-dedup/not-in-real {n_exact} -> near-dedup {n_near} -> balanced {len(df)}")
    print(f"intents={df.intent.nunique()} per-intent min/max={df.intent.value_counts().min()}/{df.intent.value_counts().max()} "
          f"has_pii={df.has_pii.mean():.1%} train={int((df.split=='train').sum())} test={int((df.split=='test').sum())} "
          f"cross-split pairs >= {a.split_sim}: {cross}")
    print("noise_type:", df.noise_type.value_counts().to_dict(), "| pii_types:", df.pii_types.explode().value_counts().to_dict())

if __name__ == "__main__":
    main()
