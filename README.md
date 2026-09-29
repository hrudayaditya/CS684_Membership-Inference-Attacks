# Membership Inference Attacks on a Banking Intent Classifier

CS684 Assignment 1. We build a privacy-flavoured dataset, train a target model on it, and test how well LiRA can tell which examples the model was trained on.

**Status:** Task A (dataset) is done. Model training and the LOSS / LiRA attacks are next.

## The dataset: FinGuard-Privacy-Benchmark

9,120 short bank-support messages, 57 intents, 160 per intent. Labels are clean, and about 35% of the messages contain fake personal details (names, cities, amounts, dates, reference numbers). Everything is generated, nothing is real customer data.

- Hugging Face: [onlyaady/FinGuard-Privacy-Benchmark](https://huggingface.co/datasets/onlyaady/FinGuard-Privacy-Benchmark) (the dataset card explains how it was built)
- Local copy: `data/dataset_v2.parquet`. The `split` column marks the 1/3 train (3,023) and 2/3 test (6,097) split.

**Why synthetic?** Banking77 is public and it might have been in ModernBERT's pretraining data, and the we intend to avoid data that's might have been included in the the pretraining. So we used Banking77 only as grounding: we cleaned it, had an LLM (DeepSeek) write new messages in the same style, and had a second model check that each message really matches its intent. The original Banking77 text is not in the dataset.

## Setup

```bash
conda create -n mia python=3.11 -y
conda activate mia
pip install datasets pandas scikit-learn sentence-transformers cleanlab torch faker requests huggingface_hub
```

Only the generation steps need a key: `export DEEPSEEK_API_KEY=...` (and `HF_TOKEN` if the Hugging Face download asks for one).

## Reproducing the dataset

Run everything from the repo root.

```bash
export PYTHONPATH=src

# 1. Clean the real Banking77 seeds
python src/prep_seeds.py           # pool, dedup, first label-noise check
python src/verify_seeds.py         # second, independent audit
python src/select_clean.py 9000    # clean pool: 8,971 rows, 57 intents

# 2. Generate and verify synthetic messages (cost API credit)
python src/gen_synthetic.py
python src/verify_synthetic.py
python src/gen_synthetic.py --noisy --per-intent 90 --out data/synth_raw_v2.jsonl
python src/make_candidates.py
python src/verify_synthetic.py --inp data/cand_v2.jsonl --out data/synth_verified_v2.jsonl

# 3. Build the final dataset
python src/build_dataset.py --inp data/synth_verified_v2.jsonl --out data/dataset_v2.parquet --noisy-quota 64
```

The LLM outputs are kept in `data/`, so you can skip step 2 and only run step 3. It rebuilds `dataset_v2.parquet` exactly (fixed seed) and takes about a minute. LLM generation itself is not deterministic, so re-running step 2 gives a similar dataset, not an identical one.

## What's in the repo

| Path | What it is |
|---|---|
| `src/` | all the code (seed cleaning, generation, verification, dataset build) |
| `data/dataset_v2.parquet` | the final dataset |
| `data/clean_pool.parquet`, `data/seeds_audited.parquet` | the cleaned real seeds and the audit flags |
| `data/synth_*.jsonl` | generated and verified messages (intermediate) |

Banking77 is by Casanueva et al. (2020), CC BY 4.0.
