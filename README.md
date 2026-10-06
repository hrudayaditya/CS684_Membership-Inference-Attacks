# Membership Inference Attacks on a Banking Intent Classifier

CS684 Assignment 1. We build a privacy-flavoured dataset, train a target model on it, and test how well LiRA can tell which examples the model was trained on.

**Status:** Tasks A and B (dataset, target and shadow models, LOSS and LiRA attacks, ablations) are done. The Task C discussion and the report are drafted in `report/` and still need a team review before we export the PDF.

## The dataset: FinGuard-Privacy-Benchmark

9,120 short bank-support messages, 57 intents, 160 per intent. Labels are clean, and about 35% of the messages contain fake personal details (names, cities, amounts, dates, reference numbers). Everything is generated, nothing is real customer data.

- Hugging Face: [onlyaady/FinGuard-Privacy-Benchmark](https://huggingface.co/datasets/onlyaady/FinGuard-Privacy-Benchmark) (the dataset card explains how it was built)
- Local copy: `data/dataset_v2.parquet`. The `split` column marks the 1/3 train (3,023) and 2/3 test (6,097) split.

**Why synthetic?** Banking77 is public and it might have been in ModernBERT's pretraining data, and the we intend to avoid data that's might have been included in the the pretraining. So we used Banking77 only as grounding: we cleaned it, had an LLM (DeepSeek) write new messages in the same style, and had a second model check that each message really matches its intent. The original Banking77 text is not in the dataset.

## Setup

```bash
conda create -n mia python=3.11 -y
conda activate mia
pip install datasets pandas pyarrow scikit-learn scipy matplotlib sentence-transformers cleanlab torch transformers faker requests huggingface_hub
```

Only the generation steps need a key: `export DEEPSEEK_API_KEY=...` (and `HF_TOKEN` if the Hugging Face download asks for one). Training needs a GPU (we used an A100); the dataset steps run fine on a laptop.

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

## Training the models and running the attacks

**Victim (target) model:** ModernBERT-base, fully fine-tuned on the 3,023 training messages. 4 epochs, learning rate 5e-5, batch size 32, bf16, no label smoothing. It reaches 100% accuracy on its training messages and 96% on the rest. The shadow models use exactly the same recipe, and only the training subset and the random seed change.

- **Offline shadows (128):** each one trains on a random half of the test split, so none has ever seen the victim's training messages.
- **Online shadows (256, extra credit):** each one trains on a random subset of *all* messages, the same size as the victim's training set.

Training stores only each model's logits on all 9,120 messages, not the weights. Big files go to a folder you choose (`$OUT`), not into git. One model takes about 20 seconds on an A100, so we ran a few processes side by side.

```bash
export OUT=/path/with/space          # where logits are written
python src/train.py target --out $OUT
python src/train.py shadow --pool offline --start 0 --end 128 --out $OUT
python src/train.py shadow --pool online  --start 0 --end 256 --out $OUT
python src/attacks.py --out $OUT     # LOSS, logit and LiRA attacks -> results/attack_main.*
```

Each `train.py` run skips models that already exist, so you can split `--start/--end` across several processes and safely rerun after a crash.

**Ablations** need a few more models (four other epoch counts, each with its own 16 matching shadows, plus four more target seeds):

```bash
for e in 1 2 10 20; do
  python src/train.py target --epochs $e --tag ep$e --out $OUT
  python src/train.py shadow --pool offline --start 0 --end 16 --epochs $e --tag ep$e --out $OUT
done
for s in 1 2 3 4; do python src/train.py target --seed $s --out $OUT; done
python src/ablations.py --out $OUT   # -> results/ablations.json and the sweep plot
```

## Results

Mean ± std over 5 victim seeds (same training set, different initialisation and batch order). "TPR @ x% FPR" is the share of true members caught when only x% of non-members are wrongly accused.

| Attack | AUC | TPR @ 0.1% FPR | TPR @ 1% FPR | TPR @ 10% FPR |
|---|---|---|---|---|
| LOSS (baseline) | 0.684 ± 0.003 | 0.2% ± 0.1 | 2.1% ± 0.3 | 17.9% ± 0.4 |
| LiRA offline, 128 shadows | 0.736 ± 0.008 | 6.2% ± 0.5 | 13.5% ± 0.4 | 34.8% ± 0.8 |
| LiRA online, 256 shadows | 0.785 ± 0.005 | 9.0% ± 0.8 | 16.6% ± 0.5 | 40.1% ± 0.9 |

(LiRA numbers use one shared variance; per-example variance gives close to the same at this many shadows. `results/attack_main.json` has the single-run table with balanced accuracy, and `results/ablations.json` has the sweep, signal, training-length and seed numbers.)

What the ablations show:
- **Number of shadows:** most of the gain arrives by 16 to 32 shadows. With very few shadows, one shared variance beats per-example variance. Online LiRA needs more shadows than offline.
- **Signal:** the logit-scaled confidence LiRA uses beats the true-class logit, and both beat plain negative loss.
- **Training length:** a barely trained model (1 epoch) leaks much less (AUC 0.60). Beyond about 4 epochs the leak stops growing, because this dataset is easy and the model already fits its training messages.

Online LiRA assumes the attacker can put the candidate messages into shadow training. That is stronger than the assignment's grey-box attacker, so read it as an upper bound.

## What's in the repo

| Path | What it is |
|---|---|
| `src/` | all the code: dataset building, `train.py`, `attacks.py`, `ablations.py` |
| `data/dataset_v2.parquet` | the final dataset |
| `data/clean_pool.parquet`, `data/seeds_audited.parquet` | the cleaned real seeds and the audit flags |
| `data/synth_*.jsonl` | generated and verified messages (intermediate) |
| `results/` | attack tables (`.json`) and plots (`.png`) |
| `report/` | draft report with our answers (Word, answers in blue) and its figures |

Banking77 is by Casanueva et al. (2020), CC BY 4.0.
