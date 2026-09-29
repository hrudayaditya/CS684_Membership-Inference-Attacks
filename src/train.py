"""Fine-tune ModernBERT-base on FinGuard-Privacy-Benchmark and save per-example logits (all MIA needs).

  python src/train.py target --out $OUT                       # the victim: trains on split == 'train'
  python src/train.py shadow --pool offline --start 0 --end 128 --out $OUT
  python src/train.py shadow --pool online  --start 0 --end 256 --out $OUT

Every model is trained with the SAME recipe (--lr/--epochs/--bs). Each run writes one .npz with
logits over ALL rows plus the boolean membership mask, so attacks and ablations never retrain.
"""
import argparse, os, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

MODEL = "answerdotai/ModernBERT-base"

def load(path, limit, tok, max_len, dev):
    df = pd.read_parquet(path)
    n_cls = int(df.label.max()) + 1
    if limit: df = pd.concat([df[df.split == s].sample(limit, random_state=0) for s in ("train", "test")])   # smoke tests only
    df = df.reset_index(drop=True)
    enc = tok(df.text.tolist(), padding="max_length", truncation=True, max_length=max_len, return_tensors="pt")
    return df, n_cls, enc["input_ids"].to(dev), enc["attention_mask"].to(dev), torch.tensor(df.label.values, dtype=torch.long).to(dev)

def train_model(ids, mask, y, members, n_cls, cfg, seed, dev):
    torch.manual_seed(seed)
    g = torch.Generator().manual_seed(seed)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL, num_labels=n_cls, attn_implementation="sdpa").to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.wd)
    steps = cfg.epochs * -(-len(members) // cfg.bs)
    sched = get_cosine_schedule_with_warmup(opt, int(cfg.warmup * steps), steps)
    members = torch.as_tensor(members, device=dev)
    model.train()
    for _ in range(cfg.epochs):
        perm = members[torch.randperm(len(members), generator=g).to(dev)]
        for b in range(0, len(perm), cfg.bs):
            i = perm[b:b + cfg.bs]
            L = int(mask[i].sum(1).max())                                   # trim padding to the longest in batch
            with torch.autocast(dev.type, dtype=torch.bfloat16):
                logits = model(input_ids=ids[i, :L], attention_mask=mask[i, :L]).logits
            F.cross_entropy(logits.float(), y[i]).backward()                # no label smoothing
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
    return model

@torch.no_grad()
def all_logits(model, ids, mask, dev, bs=256):
    model.eval()
    out = []
    for b in range(0, len(ids), bs):
        L = int(mask[b:b + bs].sum(1).max())
        with torch.autocast(dev.type, dtype=torch.bfloat16):
            out.append(model(input_ids=ids[b:b + bs, :L], attention_mask=mask[b:b + bs, :L]).logits.float().cpu())
    return torch.cat(out).numpy()

def report(logits, y, member):
    """Accuracy and a quick LOSS-attack AUC so every run shows the memorisation gap."""
    yt = torch.tensor(y)
    loss = F.cross_entropy(torch.tensor(logits), yt, reduction="none").numpy()
    acc = (logits.argmax(1) == y)
    return (f"acc in={acc[member].mean():.3f} out={acc[~member].mean():.3f} | mean loss in={loss[member].mean():.4f} "
            f"out={loss[~member].mean():.4f} | LOSS-attack AUC={roc_auc_score(member, -loss):.3f}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["target", "shadow"])
    ap.add_argument("--out", required=True, help="folder for logits (big files live here, not in git)")
    ap.add_argument("--data", default="data/dataset_v2.parquet")
    ap.add_argument("--pool", choices=["offline", "online"], default="offline")
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--end", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0, help="target seed (target mode)")
    ap.add_argument("--tag", default="", help="name for ablation targets, e.g. ep10")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--warmup", type=float, default=0.1)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--limit", type=int, default=0, help="smoke test: rows per split")
    ap.add_argument("--save-model", action="store_true", help="target only: also save weights to --out")
    a = ap.parse_args()

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(MODEL)
    df, n_cls, ids, mask, y = load(a.data, a.limit, tok, a.max_len, dev)
    yn = df.label.values.astype(np.int64)
    train_idx, test_idx = np.where(df.split == "train")[0], np.where(df.split == "test")[0]
    print(f"device={dev} rows={len(df)} train={len(train_idx)} test={len(test_idx)} classes={n_cls} tokens<= {int(mask.sum(1).max())}", flush=True)

    jobs = []   # (name, member indices, seed)
    if a.mode == "target":
        jobs.append((f"target/target_seed{a.seed}{'_' + a.tag if a.tag else ''}", train_idx, a.seed))
    else:
        for k in range(a.start, a.end):
            rng = np.random.default_rng(1_000_000 * (a.pool == "online") + k)
            if a.pool == "offline": mem = rng.choice(test_idx, len(test_idx) // 2, replace=False)      # half of the test split
            else: mem = rng.choice(len(df), len(train_idx), replace=False)                              # target-sized subset of ALL rows
            jobs.append((f"shadow_{a.pool}/shadow_{k:03d}", np.sort(mem), 100_000 * (a.pool == "online") + 1000 + k))

    for name, members, seed in jobs:
        path = os.path.join(a.out, name + ".npz")
        if os.path.exists(path): print("skip", name); continue
        os.makedirs(os.path.dirname(path), exist_ok=True)
        t = time.time()
        model = train_model(ids, mask, y, members, n_cls, a, seed, dev)
        logits = all_logits(model, ids, mask, dev)
        member = np.zeros(len(df), bool); member[members] = True
        np.savez_compressed(path, logits=logits.astype(np.float32), member=member, labels=yn, seed=seed, lr=a.lr, epochs=a.epochs, bs=a.bs)
        if a.mode == "target" and a.save_model: model.save_pretrained(path[:-4] + "_model")
        print(f"{name}: {time.time() - t:.1f}s | {report(logits, yn, member)}", flush=True)
        del model
        if dev.type == "cuda": torch.cuda.empty_cache()

if __name__ == "__main__":
    main()
