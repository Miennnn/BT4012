# BT4012: Fraudulent Bitcoin Transaction Detection

NUS student number: **e1122358**  
Competition: [BT4012 In-class Kaggle Competition 2026](https://www.kaggle.com/competitions/bt-4012-competition-2026)  
Repository: [Miennnn/BT4012](https://github.com/Miennnn/BT4012)

Classify Bitcoin transactions as licit (0) or illicit (1), using numerical transaction features and chronological validation. The official metric is **ROC AUC**, evaluated on a public 30% / private 70% split.

**Best public score reported for this project: 0.94881 (CatBoostClassifier, depth 8).** This supersedes the earlier 0.94770 score from the 75% reference + 25% raw-feature HGB blend. The CatBoost model uses the 165 original numerical features. The private score and final leaderboard standing are not available.

## Reproducibility scope

`experiments/catboost_pipeline.py` trains the current best-reported model using known labels and the original numerical features, with chronological validation and a reproducible submission-generation command. The reported 0.94881 result is a public leaderboard score; no private score is available. The earlier HGB/reference blend used an external reference-prediction file, and its original training code was not supplied, so that historical blend is not reproducible from raw data alone.

## Setup and execution

Use **Python 3.12**. Core numerical package versions match the verified run.

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download the competition data through your own authorized Kaggle account and extract into `data/`:

- `train.csv`
- `test.csv`
- `sample_submission.csv`
- `txs_edgelist.csv` (needed only for the historical graph experiments)

Open the required notebook and run all cells:

```bash
jupyter lab e1122358.ipynb
```

The notebook creates `outputs/submission.csv`, `outputs/standalone_hgb.csv`, validation metrics and a reproducibility manifest. It checks the best submission against its recorded SHA-256. You may set `BT4012_DATA_DIR` to use a different data directory.

The same final training and blending pipeline is available without Jupyter:

```bash
python -m src.pipeline --data-dir data --reference artifacts/reference_submission.csv --reference-weight 0.75 --output outputs/submission.csv --validate
```

Omit `--reference` for the standalone HGB model. To regenerate the later 50/50 experiment, set `--reference-weight 0.50` and use a different output filename. This trial scored 0.94734 and was not the best public submission.

## Data and preprocessing

| Item | Count / range |
| --- | ---: |
| Training transactions | 141,772 |
| Labelled transactions | 31,235 |
| Licit / illicit labelled transactions | 27,591 / 3,644 |
| Unlabelled transactions | 110,537 |
| Test transactions | 15,329 |
| Training / test periods | 1–35 / 36–49 |
| Original numerical features | 165 |
| Directed graph edges | 234,355 |

Only rows with known labels enter supervised fitting. Unlabelled transactions are not recoded as negative examples. Features are cast to float32; IDs, submission indices and explicit time steps are excluded from the model. No missing or non-finite feature values were found, so no imputation is applied. Duplicate transaction IDs and train/test ID overlap are rejected. No scaling, oversampling, SMOTE, pseudo-labelling or class weighting is used in the selected component.

Tree boosting captures nonlinear numerical relationships without requiring feature scaling. ROC AUC is the selection metric; average precision and threshold-specific precision/recall/F1 provide additional checks for class imbalance. The threshold 0.5 is diagnostic, not tuned for deployment costs.

## Model development

Initial experiments compared random forests, ExtraTrees and histogram gradient boosting on original and graph-augmented features. Graph features include in/out degree, neighbour degree summaries, reciprocity, connected-component measures and relative PageRank. They use graph structure only. Graph experiments include supplied unlabelled nodes structurally; the selected raw-feature component does not explicitly exploit unlabelled rows. The supplied anonymized features may themselves contain aggregates whose construction is outside this code.

An initial 50/50 graph HGB–ExtraTrees ensemble achieved local ROC AUC 0.99761 on periods 31–35 after fitting periods 1–30, but only 0.92196 publicly. This discrepancy motivated longer-horizon chronological development checks. It does not by itself prove a particular cause or data leakage.

Three expanding-window checks were then used:

| Training periods | Validation periods |
| --- | --- |
| 1–15 | 16–29 |
| 1–21 | 22–35 |
| 1–25 | 26–35 |

These validation windows overlap, and periods previously held out were reused during development. Their means are descriptive model-selection evidence, **not** independent cross-validation estimates or a new untouched holdout. The reference model has no out-of-fold predictions, so these local scores evaluate the trainable components, not the final reference blend.

| Model | Mean development ROC AUC | Worst-window ROC AUC | Mean average precision |
| --- | ---: | ---: | ---: |
| Original graph ensemble | 0.91193 | 0.82840 | 0.87050 |
| Raw-feature HGB, 350 iterations | **0.92973** | **0.89595** | 0.87397 |
| Raw-feature HGB, 700 iterations | 0.92673 | 0.89160 | 0.87460 |

The 350-iteration raw model had the highest mean and worst-window ROC AUC among the compared variants. It did not win every metric: the original ensemble had lower mean log loss. Full results, including graph ablations, stronger regularization and recency weighting, are in `results/rolling_summary.csv`.

Selected HGB settings: learning rate 0.06, 350 iterations, at most 31 leaves, minimum 25 samples per leaf, L2 regularization 10, early stopping disabled, seed 4012. The final fit uses all 31,235 known labels. The 75/25 blend weight was a conservative heuristic, subsequently compared with 50/50 using public feedback; it was not optimized using hidden labels. AUC is computed on probability rankings, so probabilities are not converted to hard labels for submission.

## Submission history

The first six submissions are recorded below. The CatBoost result was reported later and is listed separately as the current best public score.

| Attempt | Method / evidence available | Public ROC AUC |
| --- | --- | ---: |
| 1 | Uploaded predictions; training code unavailable | 0.94622 |
| 2 | Uploaded predictions; training code unavailable | 0.92141 |
| 3 | Supplied reference; training code unavailable | 0.94748 |
| 4 | 50% graph HGB + 50% graph ExtraTrees | 0.92196 |
| 5 | **75% reference + 25% raw HGB** | **0.94770** |
| 6 | 50% reference + 50% raw HGB | 0.94734 |
| Latest reported | CatBoostClassifier, depth 8 | **0.94881** |

The depth-8 CatBoost result is 0.00111 ROC AUC above the previous best HGB/reference blend (0.94770). The private score and official baseline score have not been supplied. Public results cannot establish the final private ranking or guarantee improvement over the official baseline. No test accuracy, precision, recall or F1 is claimed without test labels.

## Project files

| Path | Purpose |
| --- | --- |
| `e1122358.ipynb` | Required final-model notebook |
| `src/pipeline.py` | Data checks, chronological validation, training, blending and CSV generation |
| `requirements.txt` | Core dependencies plus notebook tooling |
| `experiments/original_pipeline.py` | Original graph feature construction and four-model comparison |
| `experiments/compare_models.py` | Extended chronological comparisons and ablations |
| `results/` | Aggregate historical validation and submission scores |
| `artifacts/reference_submission.csv` | Required external prediction artifact for the final blend |
| `artifacts/attempts/` | Exact supplied submission files, preserved for provenance |
| `docs/report_notes.md` | Evidence and limitations to use when writing the report |
| `tests/test_submission.py` | Checks for label handling, identifier alignment and invalid predictions |

Raw competition data and generated model outputs are excluded by `.gitignore`. The included submission artifacts are part of this project; keep repository access consistent with the competition's individual-work and sharing rules.

## Verification performed

All eight notebook code cells were executed sequentially in a fresh Python process against the original competition data. The three validation-window ROC AUC values matched the historical results, and the generated best submission matched uploaded attempt 5 byte for byte (SHA-256 `7337e125e1d1666f6f5dba9ffcf7c732e391992839a5ef1a5a339fead59227d8`). All six regression checks passed. Notebook outputs are from that real execution. A Jupyter kernel was unavailable in the verification environment; the notebook interface itself was not launched. See `results/reproduction_check.json`.

## Re-running historical experiments

From the repository root:

```bash
python -m experiments.original_pipeline --data-dir data --output-dir outputs/original
python -m experiments.compare_models
python -m unittest discover -s tests -v
```

The full historical comparisons take substantially longer than the final notebook. Historical JSON files preserve the information available at their original run, including an old note that the official metric was then unknown; the supplied competition description now confirms ROC AUC.

## Assignment handoff

The code notebook is named `e1122358.ipynb`. The separate report should be named `e1122358.pdf`, be at most four pages, include the repository URL, and cover introduction, preprocessing/feature engineering, development/validation, exploration/final strategy, results/discussion and reflection. The supplied brief sets the report deadline to **25 October 2026, 23:59 Singapore time** and the competition deadline to **24 October 2026, 23:55 Singapore time**. This repository does not submit to Kaggle or select final leaderboard entries automatically.


## CatBoost experiment

A standalone CatBoost trial is available in `experiments/catboost_pipeline.py`. It fits only rows with known labels, uses the 165 `feat_*` columns, and keeps transaction IDs and time steps out of the model inputs. Time steps define four chronological validation windows: periods 1–15 to 16–29, 1–21 to 22–35, 1–25 to 26–35, and 1–29 to 30–35. The final model is refit on all known labels using the median best iteration across those windows. The latest reported public ROC AUC is 0.94881 for depth 8, making CatBoost the current best-scoring model in this project; the private score is unavailable.

Install the experiment dependencies and run it from the repository root:

```bash
python -m pip install -r requirements-catboost.txt
python -m experiments.catboost_pipeline --data-dir data --output outputs/catboost_submission.csv
```

The command writes class-1 probabilities in the sample-submission index order and saves validation details beside the CSV as `catboost_submission.metrics.json`. Use `--skip-validation` to train directly with the configured iteration count. The reported 0.94881 score is public leaderboard feedback; the repository does not contain a private score.
