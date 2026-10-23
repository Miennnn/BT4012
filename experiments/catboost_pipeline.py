"""Train a chronological CatBoost baseline and write a Kaggle submission."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score


SEED = 4012
VALIDATION_WINDOWS = (
    (15, 16, 29),
    (21, 22, 35),
    (25, 26, 35),
    (29, 30, 35),
)


def feature_columns(columns: list[str]) -> list[str]:
    """Return the numbered numerical feature columns in their natural order."""
    features = [name for name in columns if name.startswith("feat_")]
    return sorted(features, key=lambda name: int(name.removeprefix("feat_")))


def load_data(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, list[str]]:
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    sample_path = data_dir / "sample_submission.csv"
    for path in (train_path, test_path, sample_path):
        if not path.is_file():
            raise FileNotFoundError(f"Required competition file not found: {path}")

    header = pd.read_csv(train_path, nrows=0).columns.tolist()
    features = feature_columns(header)
    required_train = {"txId", "time_step", "label"}
    if not features or not required_train.issubset(header):
        raise ValueError("train.csv must contain txId, time_step, label, and feat_* columns")

    train_dtypes: dict[str, Any] = {
        "txId": "int64",
        "time_step": "int16",
        "label": "float32",
        **{name: "float32" for name in features},
    }
    test_dtypes: dict[str, Any] = {
        "index": "int64",
        "txId": "int64",
        "time_step": "int16",
        **{name: "float32" for name in features},
    }
    train = pd.read_csv(
        train_path,
        usecols=list(train_dtypes),
        dtype=train_dtypes,
    )
    test = pd.read_csv(
        test_path,
        usecols=list(test_dtypes),
        dtype=test_dtypes,
    )
    sample = pd.read_csv(sample_path)

    if "index" not in sample or "target" not in sample:
        raise ValueError("sample_submission.csv must contain index and target columns")
    if not train["txId"].is_unique or not test["txId"].is_unique:
        raise ValueError("Duplicate transaction IDs found")
    if set(train["txId"]).intersection(test["txId"]):
        raise ValueError("Transaction IDs overlap between train.csv and test.csv")
    if not test["index"].is_unique or not sample["index"].is_unique:
        raise ValueError("Duplicate submission indices found")
    for name, frame in (("train.csv", train), ("test.csv", test)):
        if not np.isfinite(frame[features].to_numpy(copy=False)).all():
            raise ValueError(f"Non-finite feature value found in {name}")

    known = train["label"].dropna()
    if not set(known.unique()).issubset({0.0, 1.0}):
        raise ValueError("Known labels must be binary values 0 and 1")
    if not set(sample["index"]) == set(test["index"]):
        raise ValueError("sample_submission indices do not match test.csv")

    return train, test, sample, features


def new_model(iterations: int, verbose: bool):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise RuntimeError(
            "CatBoost is not installed. Install the project dependencies with "
            "python -m pip install -r requirements-catboost.txt."
        ) from exc

    return CatBoostClassifier(
        iterations=iterations,
        depth=8,
        learning_rate=0.03,
        loss_function="Logloss",
        eval_metric="AUC",
        l2_leaf_reg=5.0,
        random_seed=SEED,
        thread_count=-1,
        allow_writing_files=False,
        verbose=200 if verbose else False,
    )


def run_validation(
    train: pd.DataFrame,
    features: list[str],
    max_iterations: int,
    early_stopping_rounds: int,
    verbose: bool,
) -> list[dict[str, Any]]:
    labeled = train.loc[train["label"].notna()]
    periods = labeled["time_step"]
    results: list[dict[str, Any]] = []

    for train_through, valid_from, valid_through in VALIDATION_WINDOWS:
        fit_mask = periods <= train_through
        valid_mask = periods.between(valid_from, valid_through)
        fit = labeled.loc[fit_mask]
        valid = labeled.loc[valid_mask]
        if fit.empty or valid.empty or valid["label"].nunique() != 2:
            raise ValueError(
                "A chronological validation window is empty or has fewer than two classes: "
                f"train<=period {train_through}, validation {valid_from}-{valid_through}"
            )

        model = new_model(max_iterations, verbose)
        model.fit(
            fit[features],
            fit["label"].astype("int8"),
            eval_set=(valid[features], valid["label"].astype("int8")),
            early_stopping_rounds=early_stopping_rounds,
            use_best_model=True,
            verbose=200 if verbose else False,
        )
        probabilities = model.predict_proba(valid[features])[:, 1]
        best_iteration = model.get_best_iteration()
        if best_iteration < 0:
            best_iteration = model.tree_count_ - 1
        result = {
            "train_periods": f"1-{train_through}",
            "validation_periods": f"{valid_from}-{valid_through}",
            "train_rows": int(len(fit)),
            "validation_rows": int(len(valid)),
            "roc_auc": float(roc_auc_score(valid["label"], probabilities)),
            "average_precision": float(average_precision_score(valid["label"], probabilities)),
            "best_iterations": int(best_iteration + 1),
        }
        results.append(result)
        print(json.dumps(result, sort_keys=True))

    return results


def make_submission(
    sample: pd.DataFrame,
    test_indices: pd.Series,
    probabilities: np.ndarray,
) -> pd.DataFrame:
    prediction_by_index = pd.Series(probabilities, index=test_indices.to_numpy())
    submission = sample.copy()
    submission["target"] = submission["index"].map(prediction_by_index)
    if submission["target"].isna().any():
        raise ValueError("Could not align predictions to every sample_submission index")
    if not np.isfinite(submission["target"].to_numpy()).all():
        raise ValueError("Predictions contain non-finite values")
    if not submission["target"].between(0.0, 1.0).all():
        raise ValueError("Predicted probabilities are outside [0, 1]")
    return submission


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, default=Path("outputs/catboost_submission.csv"))
    parser.add_argument("--iterations", type=int, default=1500)
    parser.add_argument("--early-stopping-rounds", type=int, default=150)
    parser.add_argument("--skip-validation", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()
    if args.iterations < 1 or args.early_stopping_rounds < 1:
        parser.error("iterations and early-stopping-rounds must be positive")

    train, test, sample, features = load_data(args.data_dir)
    labeled = train.loc[train["label"].notna()]
    validation: list[dict[str, Any]] = []
    if not args.skip_validation:
        validation = run_validation(
            train,
            features,
            args.iterations,
            args.early_stopping_rounds,
            args.verbose,
        )
        final_iterations = max(
            1,
            int(statistics.median(item["best_iterations"] for item in validation)),
        )
    else:
        final_iterations = args.iterations

    final_model = new_model(final_iterations, args.verbose)
    final_model.fit(labeled[features], labeled["label"].astype("int8"))
    probabilities = final_model.predict_proba(test[features])[:, 1]
    submission = make_submission(sample, test["index"], probabilities)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(args.output, index=False)
    metrics_path = args.output.with_suffix(".metrics.json")
    metrics = {
        "model": "CatBoostClassifier",
        "seed": SEED,
        "feature_count": len(features),
        "labeled_training_rows": int(len(labeled)),
        "unlabeled_training_rows_excluded": int(train["label"].isna().sum()),
        "test_rows": int(len(test)),
        "final_iterations": final_iterations,
        "validation": validation,
        "submission": str(args.output),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(submission)} predictions to {args.output}")
    print(f"Wrote validation and run details to {metrics_path}")


if __name__ == "__main__":
    main()
