"""Independent label check: deepseek-v4-pro classifies each generated query among the intents; keep only agreeing rows.
Also calibrates the verifier on held-out REAL rows so we know its ceiling."""
import argparse, json
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
from llm import chat_json, usage_line

def build_system(intents, examples):
    listing = "\n".join(f"{k}: {n} | e.g. " + " / ".join(examples[n]) for k, n in enumerate(intents))
    return ("You classify messages sent to a retail bank's support chat into exactly one intent. Intents (id: name | examples):\n"
            f"{listing}\n\nFor each numbered message choose the single best-fitting intent id. Placeholders like {{NAME}} are fake personal details; ignore them. "
            'Return JSON {"a": [id, id, ...]} with exactly one id per message, in order.')

def classify(system, texts, model):
    user = "\n".join(f"{i+1}. {t}" for i, t in enumerate(texts))
    a = chat_json(model, system, user, max_tokens=400, temperature=0.0).get("a", [])
    if len(a) == len(texts): return a
    if len(texts) == 1: return [-1]
    h = len(texts) // 2
    return classify(system, texts[:h], model) + classify(system, texts[h:], model)

def run(system, df, model, batch, workers):
    chunks = [df.text.tolist()[i:i + batch] for i in range(0, len(df), batch)]
    with ThreadPoolExecutor(workers) as ex: res = list(ex.map(lambda c: classify(system, c, model), chunks))
    return [x for r in res for x in r]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inp", default="data/synth_raw.jsonl")
    ap.add_argument("--out", default="data/synth_verified.jsonl")
    ap.add_argument("--model", default="deepseek-v4-pro")
    ap.add_argument("--batch", type=int, default=40)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--calibrate", type=int, default=0, help="only score N held-out real rows, then exit")
    a = ap.parse_args()

    pool = pd.read_parquet("data/clean_pool.parquet")
    intents = sorted(pool.intent.unique())
    ex_rows = pool.groupby("intent").sample(2, random_state=0)          # in-prompt examples
    examples = {i: g.text.tolist() for i, g in ex_rows.groupby("intent")}
    system = build_system(intents, examples)
    idx = {n: k for k, n in enumerate(intents)}

    if a.calibrate:
        held = pool.drop(ex_rows.index).sample(a.calibrate, random_state=1)
        pred = run(system, held, a.model, a.batch, a.workers)
        acc = (pd.Series(pred).values == held.intent.map(idx).values).mean()
        print(f"verifier accuracy on {len(held)} held-out REAL rows: {acc:.3f}\n{usage_line()}")
        return

    df = pd.read_json(a.inp, lines=True)
    df["pred"] = run(system, df, a.model, a.batch, a.workers)
    df["verified"] = df.pred == df.intent.map(idx)
    df.to_json(a.out, orient="records", lines=True)
    print(f"verified {df.verified.mean():.1%} of {len(df)}\n{df.groupby('intent').verified.mean().sort_values().head(8).round(2)}\n{usage_line()}")

if __name__ == "__main__":
    main()
