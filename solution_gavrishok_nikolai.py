from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import time
from pathlib import Path

os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(max(1, (os.cpu_count() or 2) // 2)))

import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, cohen_kappa_score, confusion_matrix, f1_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import OneHotEncoder

from kairos import features
from kairos.data import load_followers, load_labels, load_posts
from kairos.label import author_level

BASE_NUM = features.CONTENT_NUM + features.HISTORY_NUM + features.ARTEFACT_NUM
EXTRA_NUM = ["recent_level_5", "recent_level_15", "recent_trend", "past_std",
             "prev_reactions_delta", "mean_word_length", "exclamations", "digit_share"]


def make_features(posts: pd.DataFrame, followers: pd.Series) -> pd.DataFrame:
    levelled = author_level(posts)
    out = features.build(levelled, followers)
    out["age_days"] = (out["crawled_at"] - out["posted_at"]).dt.total_seconds() / 86400
    past = out.groupby("author", sort=False)["log_reactions"]
    for window in (5, 15):
        out[f"recent_level_{window}"] = past.transform(
            lambda s: s.shift(1).rolling(window, min_periods=3).median())
    out["recent_trend"] = out["recent_level_5"] - out["level"]
    out["past_std"] = past.transform(lambda s: s.shift(1).expanding(5).std())
    out["prev_reactions_delta"] = past.shift(1) - out["level"]
    out["mean_word_length"] = out["n_chars"] / out["n_words"].clip(lower=1)
    out["exclamations"] = out["text"].str.count("!")
    out["digit_share"] = out["text"].str.count(r"\d") / out["n_chars"].clip(lower=1)
    out["content_type"] = out["content_type"].fillna("unknown").astype(str)
    out[BASE_NUM + EXTRA_NUM] = out[BASE_NUM + EXTRA_NUM].replace([np.inf, -np.inf], np.nan)
    return out

def clean_text(text: str) -> str:
    text = re.sub(r"⟨([A-Z]+)_\d+⟩", r" \1 ", str(text))
    text = re.sub(r"https?://\S+", " URL ", text)
    return text.casefold() if text.strip() else "emptypost"

def numeric_model(extra: bool, seed: int, quick: bool) -> Pipeline:
    columns = BASE_NUM + (EXTRA_NUM if extra else [])
    return Pipeline([
        ("columns", ColumnTransformer([
            ("numbers", "passthrough", columns),
            ("type", OneHotEncoder(handle_unknown="ignore", min_frequency=20,
                                   sparse_output=False), features.CONTENT_CAT),
        ], sparse_threshold=0)),
        ("model", HistGradientBoostingClassifier(
            max_iter=50 if quick else 300, learning_rate=0.08,
            class_weight="balanced", early_stopping=False, random_state=seed)),
    ])

def text_model(seed: int, quick: bool) -> Pipeline:
    return Pipeline([
        ("vectorize", FeatureUnion([
            ("words", TfidfVectorizer(preprocessor=clean_text, lowercase=False,
                strip_accents=None, ngram_range=(1, 2), min_df=3,
                max_df=1.0, sublinear_tf=True, dtype=np.float32,
                max_features=10000 if quick else 80000)),
            ("chars", TfidfVectorizer(preprocessor=clean_text, lowercase=False,
                analyzer="char_wb", ngram_range=(3, 5), min_df=4,
                sublinear_tf=True, dtype=np.float32,
                max_features=5000 if quick else 50000)),
        ], transformer_weights={"words": 1.0, "chars": 0.6})),
        ("model", LogisticRegression(C=3.0, class_weight="balanced", solver="lbfgs",
            max_iter=400, random_state=seed)),
    ])

def metrics(truth, pred) -> dict:
    return {
        "macro_f1": float(f1_score(truth, pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(truth, pred, labels=[0, 1, 2], weights="quadratic")),
        "accuracy": float(accuracy_score(truth, pred)),
        "f1_by_class": f1_score(truth, pred, labels=[0, 1, 2], average=None, zero_division=0).tolist(),
        "confusion_matrix": confusion_matrix(truth, pred, labels=[0, 1, 2]).tolist(),
    }

def fit_prob(model, train, valid, text=False):
    x = train["text"] if text else train
    v = valid["text"] if text else valid
    model.fit(x, train["label"])
    if list(model.classes_) != [0, 1, 2]:
        raise ValueError("В обучающей части должны присутствовать все три класса.")
    return model.predict_proba(v)

NEW_COLUMNS = ["relative_chars", "chars_delta", "relative_words", "posts_last_7d",
               "posts_last_30d", "relative_pause"]


def add_relative_features(table):
    out = table.copy()
    for column in ("n_chars", "n_words", "days_since_prev"):
        history = out.groupby("author", sort=False)[column].transform(
            lambda s: s.shift(1).expanding(min_periods=3).median())
        if column == "n_chars":
            out["relative_chars"] = out[column] / history.clip(lower=1)
            out["chars_delta"] = out[column] - history
        elif column == "n_words":
            out["relative_words"] = out[column] / history.clip(lower=1)
        else:
            out["relative_pause"] = out[column] / history.clip(lower=1 / 24)
    for days in (7, 30):
        counts = np.zeros(len(out), dtype=np.int32)
        for indices in out.groupby("author", sort=False).indices.values():
            dates = out.iloc[indices]["posted_at"].astype("int64").to_numpy()
            counts[indices] = np.searchsorted(dates, dates, side="left") - np.searchsorted(
                dates, dates - days * 86400 * 10**9, side="left")
        out[f"posts_last_{days}d"] = counts
    return out


def improved_numeric(seed=42):
    model = numeric_model(True, seed, False)
    model.named_steps["columns"].transformers[0][2].extend(NEW_COLUMNS)
    return model


def choose(numeric, text, config):
    mixed = (1 - config["text_weight"]) * numeric + config["text_weight"] * text
    return (mixed * config["class_weights"]).argmax(axis=1)


def fast_f1(y, pred):
    matrix = np.bincount(y * 3 + pred, minlength=9).reshape(3, 3)
    denominator = matrix.sum(axis=0) + matrix.sum(axis=1)
    return float(np.mean(np.divide(2 * matrix.diagonal(), denominator,
                                   out=np.zeros(3), where=denominator != 0)))


def tune(numeric, text, y):
    best = {"text_weight": 0.2, "class_weights": [0.8, 1.0, 0.8]}
    value = fast_f1(y, choose(numeric, text, best))
    for weight in (0.1, 0.15, 0.2, 0.25, 0.3):
        for low in (0.7, 0.8, 0.9, 1.0):
            for high in (0.7, 0.8, 0.9, 1.0):
                config = {"text_weight": weight, "class_weights": [low, 1.0, high]}
                score = fast_f1(y, choose(numeric, text, config))
                if score > value:
                    best, value = config, score
    return best


def main(argv=None):
    ap = argparse.ArgumentParser(description='Лабораторная работа 1')
    ap.add_argument("--data", type=Path, default=Path("data"))
    ap.add_argument("--out", type=Path, default=Path("results/gavrishok-nikolai"))
    ap.add_argument("--final", action="store_true")
    args = ap.parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    started = time.time()
    print("Читаем данные и считаем относительные признаки...", flush=True)
    features = add_relative_features(make_features(load_posts(args.data), load_followers(args.data)))
    labels = load_labels(args.data)
    table = labels.merge(features, on=["author", "post"], how="left", validate="one_to_one")
    if table["level"].isna().any():
        raise ValueError("Неполная история постов")
    train = table.loc[table["split"] == "train"].reset_index(drop=True)
    test = table.loc[table["split"] == "test"].reset_index(drop=True)
    fit_i, val_i = next(GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42).split(
        train, groups=train["author"]))
    fit, val = train.iloc[fit_i], train.iloc[val_i]
    tune_i, confirm_i = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=1234).split(
        val, groups=val["author"]))
    old_config = {"text_weight": 0.2, "class_weights": [0.8, 1.0, 0.8]}
    print("Обучаем прежнюю числовую модель...", flush=True)
    old_p = fit_prob(numeric_model(True, 42, False), fit, val)
    print("Обучаем числовую модель с новыми признаками...", flush=True)
    new_p = fit_prob(improved_numeric(), fit, val)
    print("Обучаем одну общую текстовую модель...", flush=True)
    text_p = fit_prob(text_model(42, False), fit, val, text=True)
    probabilities = val[["author", "post", "label"]].copy()
    for name, matrix in (("old", old_p), ("relative", new_p), ("text", text_p)):
        for cls in range(3):
            probabilities[f"{name}_{cls}"] = matrix[:, cls]
    probabilities.to_csv(args.out / "validation_probabilities.csv", index=False)
    candidates = {
        "current": (old_p, old_config),
        "finer_weights": (old_p, tune(old_p[tune_i], text_p[tune_i], val["label"].to_numpy()[tune_i])),
        "relative_features": (new_p, old_config),
    }
    results = {}
    for name, (numeric, config) in candidates.items():
        predicted = choose(numeric, text_p, config)
        results[name] = {
            "config": config,
            "all_validation": metrics(val["label"], predicted),
            "tuning_authors": metrics(val.iloc[tune_i]["label"], predicted[tune_i]),
            "confirmation_authors": metrics(val.iloc[confirm_i]["label"], predicted[confirm_i]),
        }
        print(f"{name}: общая проверка {results[name]['all_validation']['macro_f1']:.5f}; "
              f"подтверждение {results[name]['confirmation_authors']['macro_f1']:.5f}", flush=True)
    reference = results["current"]
    accepted = [name for name in ("finer_weights", "relative_features")
                if results[name]["confirmation_authors"]["macro_f1"] >
                   reference["confirmation_authors"]["macro_f1"] + 0.001
                and results[name]["all_validation"]["macro_f1"] >
                   reference["all_validation"]["macro_f1"]]
    winner = max(accepted, key=lambda name: results[name]["confirmation_authors"]["macro_f1"]) if accepted else "current"
    report = {
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "pandas": pd.__version__, "sklearn": sklearn.__version__},
        "configuration": {"seed": 42, "valid_size": 0.2, "confirmation_seed": 1234},
        "labels_sha256": hashlib.sha256((args.data / "labels.csv").read_bytes()).hexdigest(),
        "n_train": len(train), "n_test": len(test),
        "experiments": results, "selected": winner,
        "new_features": NEW_COLUMNS,
        "tuning_authors": sorted(val.iloc[tune_i]["author"].unique().tolist()),
        "confirmation_authors": sorted(val.iloc[confirm_i]["author"].unique().tolist()),
        "note": "Параметры finer_weights выбраны на половине внутренней проверки. "
                "Выбор из двух гипотез подтверждён на другой половине. Это ограниченная проверка, не новая внешняя выборка.",
    }
    report_path = args.out / "report.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Выбран вариант: {winner}", flush=True)
    if args.final and winner != "current":
        config = results[winner]["config"]
        print("Обучаем выбранную числовую модель на всех обучающих авторах...", flush=True)
        numeric = improved_numeric() if winner == "relative_features" else numeric_model(True, 42, False)
        final_p = (1 - config["text_weight"]) * fit_prob(numeric, train, test)
        print("Обучаем итоговую текстовую модель...", flush=True)
        final_p += config["text_weight"] * fit_prob(text_model(42, False), train, test, text=True)
        pred = (final_p * config["class_weights"]).argmax(axis=1)
        test[["author", "post"]].assign(label=pred).to_csv(args.out / "predictions.csv", index=False)
        report["final_test"] = metrics(test["label"], pred)
        print(f"Итоговая оценка: {report['final_test']['macro_f1']:.5f}", flush=True)
    report["elapsed_seconds"] = round(time.time() - started, 1)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
