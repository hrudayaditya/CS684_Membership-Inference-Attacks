"""Generate synthetic banking queries per intent with DeepSeek, seeded with real clean-pool examples."""
import argparse, json, os, random, threading
from concurrent.futures import ThreadPoolExecutor
import numpy as np, pandas as pd
from llm import chat_json, usage_line

SYSTEM = """You write realistic customer messages sent to a retail bank's support chat. Given a target intent, real examples of it, and confusable OTHER intents, write NEW messages that clearly express ONLY the target intent.
Rules:
- Every message must be unmistakably about the target intent and must NOT be answerable as any of the confusable other intents.
- Do not copy or lightly paraphrase the examples; vary wording, structure, length, tone, typos and punctuation as instructed by the style hint.
- About 1 in 3 messages should include personal details, written ONLY as these placeholders (never real-looking values): {NAME} {AMOUNT} {LAST4} {MERCHANT} {CITY}. The rest must have no personal details and no placeholders.
- One message per string, no numbering, no quotes inside unless natural.
Return JSON: {"q": ["...", ...]}"""

STYLES = ["very short and terse (3-8 words)", "one casual sentence, lowercase, a typo or two", "polite and formal, 1-2 sentences",
          "frustrated or worried, 2 sentences", "a question phrased indirectly", "detailed, gives the backstory, 2-3 sentences",
          "non-native English speaker phrasing", "mix of short and medium messages"]

SYSTEM_NOISY = """You write realistic, MESSY customer messages sent to a retail bank's support chat, typed quickly on a phone. Given a target intent, real examples of it, and confusable OTHER intents, write NEW messages that clearly express ONLY the target intent.
Rules:
- Every message must still be unmistakably about the target intent and must NOT be answerable as any confusable other intent.
- Do NOT write polished, grammatically perfect sentences. Do not copy the examples.
- Follow the mess hint for the whole batch, but mix in variety: about 30% with typos or misspellings, about 20% very short (under 8 words), about 15% rambling run-ons, some with slang ("pls", "thx", "idk", "gonna", "bc"), some with no punctuation or "!!!" / "???", occasional mixed register.
- About 1 in 3 messages should include personal details, written ONLY as these placeholders (never real-looking values): {NAME} {AMOUNT} {LAST4} {MERCHANT} {CITY} {DATE} {REF}. Use a placeholder only where it fits the message naturally. The rest must have no placeholders.
- One message per string, no numbering.
Return JSON: {"q": ["...", ...]}"""

MESS = ["heavy typos and misspellings", "very short fragments, under 8 words", "rambling run-on sentences with backstory",
        "texting slang and abbreviations, no punctuation", "frustrated, caps and !!! and ???", "mixed formal and rude register",
        "odd or unusual word choices, non-native phrasing", "incomplete sentences and trailing off"]

def neighbors(intents):
    seeds = pd.read_parquet("data/seeds_audited.parquet")
    emb = np.load("data/seed_emb_bge.npy")
    cent = {i: emb[(seeds.intent == i).values].mean(0) for i in seeds.intent.unique()}
    for k in cent: cent[k] /= np.linalg.norm(cent[k])
    out = {}
    for i in intents:
        sims = sorted(((float(cent[i] @ cent[j]), j) for j in intents if j != i), reverse=True)
        out[i] = [j for _, j in sims[:2]]
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-intent", type=int, default=195)
    ap.add_argument("--batch", type=int, default=15)
    ap.add_argument("--intents", type=int, default=0, help="pilot: only first N intents")
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--out", default="data/synth_raw.jsonl")
    ap.add_argument("--noisy", action="store_true", help="messy-style prompt with extra PII slots")
    a = ap.parse_args()

    pool = pd.read_parquet("data/clean_pool.parquet")
    intents = sorted(pool.intent.unique())
    nb = neighbors(intents)
    by = {i: pool[pool.intent == i].text.tolist() for i in intents}
    todo_intents = intents[: a.intents] if a.intents else intents
    have = {i: 0 for i in intents}
    if os.path.exists(a.out):
        for l in open(a.out): have[json.loads(l)["intent"]] += 1
    jobs = []
    for i in todo_intents:
        for _ in range(max(0, -(-(a.per_intent - have[i]) // a.batch))): jobs.append(i)
    print(f"{len(jobs)} calls planned", flush=True)
    lock, f = threading.Lock(), open(a.out, "a")

    def run(intent):
        rng = random.Random()
        ex = rng.sample(by[intent], min(6, len(by[intent])))
        user = (f"Target intent: {intent}\nReal examples:\n" + "\n".join(f"- {s}" for s in ex) +
                "\nConfusable OTHER intents (do NOT write these):\n" +
                "\n".join(f"- {n}, e.g. \"{rng.choice(by[n])}\"" for n in nb[intent]) +
                f"\n\n{'Mess hint' if a.noisy else 'Style hint'}: {rng.choice(MESS if a.noisy else STYLES)}\nWrite {a.batch} new messages.")
        qs = chat_json("deepseek-flash", SYSTEM_NOISY if a.noisy else SYSTEM, user, max_tokens=1500).get("q", [])
        with lock:
            for q in qs:
                if isinstance(q, str) and q.strip(): f.write(json.dumps({"intent": intent, "text": q.strip()}) + "\n")
            f.flush()

    with ThreadPoolExecutor(a.workers) as ex: list(ex.map(run, jobs))
    print(usage_line())

if __name__ == "__main__":
    main()
