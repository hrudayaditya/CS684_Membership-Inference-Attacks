"""Pool Banking77, dedup, and flag likely-mislabeled seeds via cross-validated confident learning."""
import re
import unicodedata
import numpy as np
import pandas as pd
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from cleanlab.filter import find_label_issues

def norm(s):
    """NFKC + casefold + keep alphanumerics only (no spaces), so 'top-up' == 'top up' == 'topup'."""
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", s).casefold())

def main():
    ds = load_dataset("legacy-datasets/banking77")
    names = ds["train"].features["label"].names
    df = pd.concat([ds["train"].to_pandas().assign(orig_split="train"),
                    ds["test"].to_pandas().assign(orig_split="test")], ignore_index=True)
    df["intent"] = df["label"].map(dict(enumerate(names)))
    df["norm"] = df["text"].map(norm)
    print(f"pooled: {len(df)}")

    # exact duplicates after normalization; report label conflicts (same text, different intent)
    grp = df.groupby("norm")["label"].nunique()
    print(f"norm-duplicate groups: {(df.duplicated('norm', keep=False)).sum()} rows, "
          f"{(grp > 1).sum()} texts with conflicting labels")
    df = df[~df["norm"].isin(grp[grp > 1].index)]          # ambiguous text: drop entirely
    df = df.drop_duplicates("norm", keep="first").reset_index(drop=True)
    print(f"after exact dedup: {len(df)}")

    emb = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2").encode(
        df["text"].tolist(), batch_size=256, normalize_embeddings=True, show_progress_bar=False)
    probs = cross_val_predict(LogisticRegression(max_iter=2000, C=10), emb, df["label"],
                              cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")
    issues = find_label_issues(df["label"].values, probs, return_indices_ranked_by="self_confidence")
    df["suspect"] = False
    df.loc[issues, "suspect"] = True
    df["cv_pred"] = probs.argmax(1)
    df["cv_conf_true"] = probs[np.arange(len(df)), df["label"]]
    print(f"suspect seeds: {df.suspect.sum()} ({df.suspect.mean():.1%})")
    print("top intents by suspect rate:\n", df.groupby("intent").suspect.mean().sort_values(ascending=False).head(8).round(2))
    df.drop(columns=["label"]).assign(label=df["label"]).to_parquet("data/seeds.parquet")
    np.save("data/seed_emb.npy", emb)


if __name__ == "__main__":
    main()
