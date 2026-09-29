"""v2 candidates: clean v1 rows (25% perturbed) + messy LLM rows, then PII injected up to a target rate. Labels are never touched."""
import argparse, json, random, re
import pandas as pd

SLANG = [("please", "pls"), ("thanks", "thx"), ("because", "bc"), ("going to", "gonna"), ("want to", "wanna"), ("do not", "dont"),
         ("cannot", "cant"), ("i am", "im"), ("you are", "ur"), ("what is", "whats"), ("i have", "ive"), ("my account", "my acct")]
PREFIX = ["Hi, I'm {NAME}. ", "Hello, this is {NAME}. ", "{NAME} here. ", "hi its {NAME}, ", "Hi, {NAME} speaking. "]
SUFFIX = [" My ref is {REF}.", " ref {REF}", " (account holder {NAME})", " I'm in {CITY} at the moment.", " This was on {DATE}.", " Ref: {REF}"]
TYPES = {"NAME": "name", "AMOUNT": "amount", "LAST4": "card_digits", "MERCHANT": "merchant", "CITY": "location", "DATE": "date", "REF": "reference"}

def perturb(text, rng):
    """Light input noise: slang, typos, dropped word/punctuation, casing. Placeholders are never touched."""
    for f, s in SLANG:
        if rng.random() < 0.35: text = re.sub(rf"\b{f}\b", s, text, flags=re.I)
    w = text.split()
    for _ in range(rng.choice([1, 1, 2, 3])):
        i = rng.randrange(len(w))
        t = w[i]
        if "{" in t or len(t) < 4: continue
        j = rng.randrange(1, len(t) - 1)
        op = rng.choice(["swap", "drop", "dup"])
        w[i] = t[:j] + t[j + 1] + t[j] + t[j + 2:] if op == "swap" else t[:j] + t[j + 1:] if op == "drop" else t[:j] + t[j] + t[j:]
    k = rng.randrange(1, len(w) - 1) if len(w) > 5 else 0
    if k and rng.random() < 0.3 and "{" not in w[k]: w.pop(k)
    text = " ".join(w)
    if rng.random() < 0.4: text = text.rstrip("?.!")
    if rng.random() < 0.35 and "{" not in text: text = text.lower()
    return text

def inject(text, rng):
    if rng.random() < 0.5: return rng.choice(PREFIX) + (text if rng.random() < 0.5 else text[0].lower() + text[1:])
    return text.rstrip() + rng.choice(SUFFIX)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perturb-frac", type=float, default=0.25)
    ap.add_argument("--pii-target", type=float, default=0.32)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="data/cand_v2.jsonl")
    a = ap.parse_args()
    rng = random.Random(a.seed)

    clean = pd.read_json("data/synth_verified.jsonl", lines=True)
    clean = clean[clean.verified][["intent", "text"]].assign(noise_type="clean")
    noisy = pd.read_json("data/synth_raw_v2.jsonl", lines=True).assign(noise_type="llm_noisy")
    pert = clean.sample(frac=a.perturb_frac, random_state=a.seed)
    clean = clean.drop(pert.index)
    pert = pert.assign(text=[perturb(t, rng) for t in pert.text], noise_type="perturbed")
    df = pd.concat([clean, pert, noisy], ignore_index=True)

    has = df.text.str.contains(r"\{\w+\}")
    need = int(a.pii_target * len(df)) - int(has.sum())
    cand = df.index[~has].tolist()
    rng.shuffle(cand)
    for i in cand[:max(0, need)]: df.at[i, "text"] = inject(df.at[i, "text"], rng)
    after = df.text.str.contains(r"\{\w+\}").mean()
    print(f"rows={len(df)} noise_type={df.noise_type.value_counts().to_dict()} pii_rate {has.mean():.1%} -> {after:.1%}")
    df.to_json(a.out, orient="records", lines=True)

if __name__ == "__main__":
    main()
