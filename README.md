# BT4012: Fraudulent Bitcoin Transaction Detection

NUS student number: **e1122358**  
Competition: [BT4012 In-class Kaggle Competition 2026](https://www.kaggle.com/competitions/bt-4012-competition-2026)  
Repository: [Miennnn/BT4012](https://github.com/Miennnn/BT4012)

The task is to rank Bitcoin transactions as licit (0) or illicit (1) from anonymised numerical features. The goal is to help prioritise transactions for investigation; a model score is not proof of wrongdoing. The competition uses ROC AUC, with 30% of the test set on the public leaderboard and 70% on the private leaderboard.

## Final model

A `CatBoostClassifier` with depth 8, learning rate 0.03, L2 leaf regularisation 5 and seed 4012, trained on all labelled transactions using the 165 supplied numerical `feat_*` columns. The number of trees is the median best iteration across four expanding chronological validation windows. The final submission scored **0.94881 public** and **0.94061 private** ROC AUC.

## Repository contents

| Path | Purpose |
| --- | --- |
| `experiments/catboost_pipeline.py` | Chronological validation, training and submission generation |
| `requirements-catboost.txt` | Python dependencies for the CatBoost experiment |

The competition data and generated CSV files are not committed. The final model does not use `txs_edgelist.csv`.

## Setup and execution

Use Python 3.9 or newer.

```bash
python3 -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-catboost.txt
```

Download `train.csv`, `test.csv` and `sample_submission.csv` from the competition page into a `data/` folder in the repository root. Then run:

```bash
python -m experiments.catboost_pipeline --data-dir data --output outputs/submission.csv
```

The script writes `outputs/submission.csv` and `outputs/submission.metrics.json`. It can also be run with the default output path `outputs/catboost_submission.csv`.

CatBoost uses all CPU threads and a fixed seed. Reruns should be similar, though small differences may occur across machines or library versions.

## Method summary

- **Preprocessing.** Only labelled rows enter supervised training. The pipeline checks for duplicate or overlapping transaction IDs, non-binary known labels, non-finite features and mismatches between test indices and `sample_submission.csv`. Features are read as `float32`; no imputation or scaling is needed.
- **Features.** The 165 supplied `feat_*` columns are used as provided. `txId`, the submission index and `time_step` are excluded from model inputs. No graph features are engineered and the edge list is not read.
- **Class imbalance.** About 11.7% of labelled transactions are illicit. No resampling, SMOTE, pseudo-labelling or class weights are used.
- **Validation.** Four expanding chronological windows fit on periods 1–15, 1–21, 1–25 and 1–29, validating on 16–29, 22–35, 26–35 and 30–35. ROC AUC guides early stopping (150 rounds, up to 1,500 iterations); average precision is recorded as a diagnostic. The windows overlap and were reused during model selection, so they are development evidence rather than independent cross-validation estimates.
- **Final fit.** The model is refit on all labelled rows using the median best iteration across the four windows.
- **Model choice.** CatBoost depth 8 has the highest recorded public and private scores. The earlier submissions are not all reproducible from retained code, so their leaderboard scores are not a controlled model comparison.

## Submission history

| # | Model | Public ROC AUC | Private ROC AUC |
| - | --- | ---: | ---: |
| 1 | Early baseline (code not retained) | 0.94622 | 0.93331 |
| 2 | Early baseline (code not retained) | 0.92141 | 0.91583 |
| 3 | Early baseline (code not retained) | 0.94748 | 0.93469 |
| 4 | Early baseline (code not retained) | 0.92196 | 0.91586 |
| 5 | Early baseline (code not retained) | 0.94770 | 0.93461 |
| 6 | Early baseline (code not retained) | 0.94734 | 0.93442 |
| 7 | CatBoost, first run | 0.94778 | 0.93634 |
| 8 | CatBoost, second run | 0.94788 | 0.93621 |
| 9 | **CatBoost, depth 8 (final)** | **0.94881** | **0.94061** |

## Limitations

The model does not use the transaction graph or unlabelled rows. There was one run per CatBoost setting, with no repeated seeds, and no decision threshold was studied.