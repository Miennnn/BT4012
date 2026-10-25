# BT4012: Fraudulent Bitcoin Transaction Detection

NUS student number: **e1122358**  
Competition: [BT4012 In-class Kaggle Competition 2026](https://www.kaggle.com/competitions/bt-4012-competition-2026)  
Repository: [Miennnn/BT4012](https://github.com/Miennnn/BT4012)

The task is to rank Bitcoin transactions as licit (0) or illicit (1) from anonymised numerical features. A useful ranking can help investigators prioritise transactions for review; it is not proof that a transaction is criminal. ROC AUC is the competition metric. The public leaderboard uses 30% of the test set and the private leaderboard uses the remaining 70%.

## Final result

The final model is a `CatBoostClassifier` with depth 8, learning rate 0.03, L2 leaf regularisation 5 and seed 4012. It was trained on all labelled transactions using the 165 supplied `feat_*` columns. The final submission, number 9, scored **0.94881 public** and **0.94061 private** ROC AUC, the highest of my nine submissions on both leaderboards.

If selecting two entries for the final private ranking, choose **submission 9** and **submission 7**. Their private ROC AUC scores were 0.94061 and 0.93634, the two highest recorded.

## Reproducibility

- Submission 9 can be reproduced from the competition data with `e1122358.ipynb` or `experiments/catboost_pipeline.py`.
- Submissions 7 and 8 came from earlier CatBoost runs. The first committed pipeline used depth 7 and three validation windows; the final run uses depth 8 and four windows.
- Submissions 1-6 are included as historical leaderboard results. Submissions 5 and 6 blended raw-feature HGB predictions with the predictions from submission 3. These results provide context, but the early models were not evaluated on the same features and validation windows as CatBoost.

## Repository contents

| Path | Purpose |
| --- | --- |
| `e1122358.ipynb` | Final model: data checks, validation, training and submission output |
| `experiments/catboost_pipeline.py` | Command-line version of the training pipeline |
| `requirements.txt` | Dependencies for the notebook and script |
| `requirements-catboost.txt` | Minimal dependencies for the standalone script |

The competition data and generated CSV files are not committed.

## Setup and execution

Use Python 3.9 or newer. From the repository root:

```bash
python -m venv .venv
# macOS/Linux:
source .venv/bin/activate
# Windows PowerShell:
.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Download `train.csv`, `test.csv` and `sample_submission.csv` from the competition page into a `data/` folder. The final model does not use `txs_edgelist.csv`.

Run all cells in the notebook:

```bash
jupyter lab e1122358.ipynb
```

Or run the pipeline from the repository root:

```bash
python -m experiments.catboost_pipeline --data-dir data --output outputs/submission.csv
```

Both write `outputs/submission.csv` and `outputs/submission.metrics.json`. The notebook accepts `BT4012_DATA_DIR`, `BT4012_OUTPUT_DIR` and `BT4012_MAX_ITERATIONS`; set the last value to something small, such as 60, for a quick check. CatBoost uses all CPU threads and seed 4012, so predictions should be close across runs, though libraries and hardware can cause small differences.

## Data and preprocessing

| Item | Count or range |
| --- | ---: |
| Training transactions | 141,772 |
| Labelled transactions | 31,235 |
| Licit / illicit labels | 27,591 / 3,644 |
| Unlabelled transactions | 110,537 |
| Test transactions | 15,329 |
| Labelled / test periods | 1-35 / 36-49 |
| Numerical features | 165 |
| Directed graph edges | 234,355 |

Only rows with known labels enter supervised training; unlabelled rows are not treated as licit. The pipeline checks transaction IDs, label values, finite features and test-index alignment with the sample submission. No missing or non-finite feature values were found, so no imputation was needed. Features are read as `float32`. The model excludes `txId`, the submission index and `time_step`; it uses the 165 numerical columns as supplied, with no scaling or feature selection. No SMOTE, resampling, pseudo-labelling or class weights are used.

About 11.7% of labelled transactions are illicit. ROC AUC is the selection metric; average precision is recorded as an additional check on the rare class. The hidden test labels are not used to calculate accuracy, precision, recall or F1.

## Model development

I began with Logistic Regression and Random Forest baselines, then tried graph features, ExtraTrees and histogram gradient boosting (HGB). The graph experiments used in-degree, out-degree, neighbour-degree summaries, reciprocity, connected components and relative PageRank. They used the edge list as structural data, including unlabelled nodes, without using labels to build the features. The final CatBoost pipeline does not use the edge list. Some graph information may already be present in the supplied features.

One early 50/50 graph HGB–ExtraTrees blend scored 0.99761 ROC AUC on a short local window but 0.92196 on the public leaderboard. That gap does not prove leakage, but it showed that a single short window was not a reliable guide. The HGB comparisons therefore used three expanding chronological windows:

| Fit periods | Validation periods |
| --- | --- |
| 1-15 | 16-29 |
| 1-21 | 22-35 |
| 1-25 | 26-35 |

The windows overlap and were reused during development, so they are model-selection evidence rather than independent cross-validation estimates.

| Historical model | Mean ROC AUC | Worst-window ROC AUC | Mean average precision |
| --- | ---: | ---: | ---: |
| Graph ensemble | 0.91193 | 0.82840 | 0.87050 |
| Raw-feature HGB, 350 iterations | **0.92973** | **0.89595** | 0.87397 |
| Raw-feature HGB, 700 iterations | 0.92673 | 0.89160 | **0.87460** |

Raw-feature HGB with 350 iterations had the best mean and worst-window ROC AUC. It did not lead on every measure: the graph ensemble had lower mean log loss. These are historical results, not CatBoost validation scores.

The CatBoost pipeline uses four expanding windows: fit on periods 1-15, 1-21, 1-25 and 1-29; validate on 16-29, 22-35, 26-35 and 30-35. It fits with Logloss and uses validation ROC AUC for early stopping, with a 150-round patience and at most 1,500 iterations. The final fit uses the median best iteration across the four windows.

CatBoost may have helped because its trees can represent nonlinear thresholds and combinations among the numerical features. The low learning rate, L2 regularisation and early stopping limit model complexity. These are plausible reasons, not a causal finding: the earlier models were not all rerun on the same folds, and there was one run per CatBoost setting.

## Submission history

| # | Model | Public ROC AUC | Private ROC AUC |
| ---: | --- | ---: | ---: |
| 1 | Early baseline | 0.94622 | 0.93331 |
| 2 | Early baseline | 0.92141 | 0.91583 |
| 3 | Early baseline | 0.94748 | 0.93469 |
| 4 | Early baseline | 0.92196 | 0.91586 |
| 5 | Early baseline | 0.94770 | 0.93461 |
| 6 | Early baseline | 0.94734 | 0.93442 |
| 7 | CatBoost, first run | 0.94778 | **0.93634** |
| 8 | CatBoost, second run | **0.94788** | 0.93621 |
| 9 | **CatBoost, depth 8 (final)** | **0.94881** | **0.94061** |

The best earlier public result was 0.94788 from submission 8; the best earlier private result was 0.93634 from submission 7. The depth-8 run improved on both. The private score was not available during model selection, so it serves as a check on the final choice.

## Limitations

The final model ignores the transaction graph and unlabelled rows. There was one run per CatBoost setting, and no decision threshold or investigator review budget was studied. The early submission results came from different experimental setups, so they provide context rather than a controlled comparison with CatBoost.