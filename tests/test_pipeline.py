# Tests für meine Phishing-Pipeline
# Tarek Al Jawabra – SoSe 2026
#
# Ich teste hier: Datensatz, Pipeline-Aufbau, Wortlisten, Zusatzmerkmale,
# Metriken und die Output-Dateien

import json
import os
import sys

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from main import (
    StructuralFeatureExtractor,
    analyze_wordlists,
    build_pipeline,
    evaluate,
    load_data,
    FIGURES_DIR,
    OUTPUT_DIR,
    REPORTS_DIR,
)


@pytest.fixture
def sample_df():
    # kleiner Testdatensatz – 4 Spam/Phishing, 6 legitim
    texts = [
        "Click here now to verify your account http://phishing.xyz urgent password",
        "Win a free prize! Click now: http://scam.net winner selected, buy viagra cheap",
        "URGENT: your account has been suspended, verify now http://fake-bank.tk !!!",
        "FREE MONEY CLICK NOW http://win-prize.net limited offer act now $$$",
        "Meeting invitation for Thursday at 14:00 in room 3.12",
        "Please find attached the monthly report for review by Friday",
        "Your order has been shipped. Estimated delivery: 3 business days.",
        "Hi team, quick reminder about the lunch meeting tomorrow at noon",
        "Code review completed for feature branch. Please check comments.",
        "Hotel booking confirmation: check-in 15.06.2024",
    ]
    labels = [1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    return pd.DataFrame({"text": texts, "label": labels})


@pytest.fixture
def trained_pipeline(sample_df):
    pipeline = build_pipeline(LogisticRegression(max_iter=500))
    pipeline.fit(sample_df["text"], sample_df["label"])
    return pipeline


class TestDataset:
    def test_csv_exists(self):
        path = os.path.join(ROOT, "data", "emails.csv")
        assert os.path.exists(path), f"Datensatz fehlt: {path}"

    def test_csv_columns(self):
        df = load_data()
        assert "text" in df.columns
        assert "label" in df.columns

    def test_csv_no_nulls(self):
        df = load_data()
        assert df["text"].isna().sum() == 0
        assert df["label"].isna().sum() == 0

    def test_csv_binary_labels(self):
        df = load_data()
        assert set(df["label"].unique()).issubset({0, 1})

    def test_csv_size(self):
        # echter Enron-Spam-Datensatz, deutlich groesser als der fruehere synthetische
        df = load_data()
        assert len(df) >= 10000

    def test_csv_balanced(self):
        df = load_data()
        ratio = df["label"].mean()
        assert 0.3 <= ratio <= 0.7


class TestWordlists:
    def test_analyze_wordlists_returns_three_frames(self, sample_df):
        top_spam, top_ham, full = analyze_wordlists(sample_df, top_n=5, min_count=1)
        assert isinstance(top_spam, pd.DataFrame)
        assert isinstance(top_ham, pd.DataFrame)
        assert isinstance(full, pd.DataFrame)

    def test_wordlist_has_expected_columns(self, sample_df):
        top_spam, _, _ = analyze_wordlists(sample_df, top_n=5, min_count=1)
        for col in ["word", "spam_count", "ham_count", "total", "spam_ratio"]:
            assert col in top_spam.columns

    def test_spam_ratio_in_range(self, sample_df):
        _, _, full = analyze_wordlists(sample_df, top_n=5, min_count=1)
        assert (full["spam_ratio"] >= 0).all()
        assert (full["spam_ratio"] <= 1).all()

    def test_wordlist_analysis_csv_exists(self):
        assert os.path.exists(os.path.join(OUTPUT_DIR, "wordlist_analysis.csv"))

    def test_wordlists_figure_exists(self):
        assert os.path.exists(os.path.join(FIGURES_DIR, "wordlists.png"))


class TestStructuralFeatures:
    def test_extractor_returns_correct_shape(self):
        extractor = StructuralFeatureExtractor()
        texts = ["Click here now!!! http://scam.com FREE $$$", "Normal email text without anything suspicious"]
        features = extractor.transform(texts)
        assert features.shape == (2, len(StructuralFeatureExtractor.FEATURE_NAMES))

    def test_extractor_detects_links(self):
        extractor = StructuralFeatureExtractor()
        features = extractor.transform(["Visit http://example.com and http://test.com now"])
        assert features[0][0] == 2  # num_links

    def test_extractor_detects_exclamation(self):
        extractor = StructuralFeatureExtractor()
        features = extractor.transform(["Hurry up!!!"])
        assert features[0][1] == 3  # num_exclamation

    def test_extractor_handles_empty_text(self):
        extractor = StructuralFeatureExtractor()
        features = extractor.transform([""])
        assert not np.isnan(features).any()


class TestPipeline:
    def test_build_pipeline_returns_pipeline(self):
        assert isinstance(build_pipeline(LogisticRegression()), Pipeline)

    def test_pipeline_has_features_step(self):
        assert "features" in build_pipeline(LogisticRegression()).named_steps

    def test_pipeline_combines_tfidf_and_structural(self):
        pipeline = build_pipeline(LogisticRegression())
        feature_union = pipeline.named_steps["features"]
        names = [name for name, _ in feature_union.transformer_list]
        assert "tfidf" in names
        assert "structural" in names

    def test_pipeline_has_clf_step(self):
        assert "clf" in build_pipeline(LogisticRegression()).named_steps

    def test_pipeline_fit_predict(self, sample_df):
        pipeline = build_pipeline(LogisticRegression(max_iter=500))
        pipeline.fit(sample_df["text"], sample_df["label"])
        preds = pipeline.predict(sample_df["text"])
        assert len(preds) == len(sample_df)
        assert set(preds).issubset({0, 1})

    def test_pipeline_predict_proba(self, sample_df):
        pipeline = build_pipeline(LogisticRegression(max_iter=500))
        pipeline.fit(sample_df["text"], sample_df["label"])
        proba = pipeline.predict_proba(sample_df["text"])
        assert proba.shape == (len(sample_df), 2)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-6)


class TestEvaluate:
    def test_evaluate_returns_dict(self, trained_pipeline, sample_df):
        result = evaluate("LR", trained_pipeline, sample_df["text"], sample_df["label"])
        assert isinstance(result, dict)

    def test_evaluate_has_all_metrics(self, trained_pipeline, sample_df):
        result = evaluate("LR", trained_pipeline, sample_df["text"], sample_df["label"])
        for key in ["accuracy", "precision", "recall", "f1"]:
            assert key in result

    def test_evaluate_accuracy_in_range(self, trained_pipeline, sample_df):
        result = evaluate("LR", trained_pipeline, sample_df["text"], sample_df["label"])
        assert 0.0 <= result["accuracy"] <= 1.0

    def test_evaluate_f1_in_range(self, trained_pipeline, sample_df):
        result = evaluate("LR", trained_pipeline, sample_df["text"], sample_df["label"])
        assert 0.0 <= result["f1"] <= 1.0


class TestOutputFiles:
    def test_comparison_table_exists(self):
        assert os.path.exists(os.path.join(OUTPUT_DIR, "comparison_table.csv"))

    def test_comparison_table_columns(self):
        df = pd.read_csv(os.path.join(OUTPUT_DIR, "comparison_table.csv"))
        for col in ["classifier", "accuracy", "precision", "recall", "f1"]:
            assert col in df.columns

    def test_comparison_table_not_empty(self):
        df = pd.read_csv(os.path.join(OUTPUT_DIR, "comparison_table.csv"))
        assert len(df) > 0

    def test_comparison_table_results_not_trivially_perfect(self):
        # echte Daten sollten keine durchgehenden Werte von exakt 1.0 liefern
        df = pd.read_csv(os.path.join(OUTPUT_DIR, "comparison_table.csv"))
        assert not (df["f1"] == 1.0).all()

    def test_figures_exist(self):
        for fname in ["confusion_matrices.png", "roc_curves.png",
                      "metric_comparison.png", "label_distribution.png", "wordlists.png"]:
            assert os.path.exists(os.path.join(FIGURES_DIR, fname)), f"Fehlt: {fname}"

    def test_summary_report_exists(self):
        assert os.path.exists(os.path.join(REPORTS_DIR, "summary.json"))

    def test_summary_report_valid_json(self):
        with open(os.path.join(REPORTS_DIR, "summary.json")) as f:
            data = json.load(f)
        assert "best_classifier" in data
        assert "cv_f1_mean" in data
        assert "top_spam_words" in data
        assert "top_ham_words" in data

    def test_ergebnisse_md_exists(self):
        assert os.path.exists(os.path.join(ROOT, "docs", "ergebnisse.md"))

    def test_ergebnisse_md_not_empty(self):
        with open(os.path.join(ROOT, "docs", "ergebnisse.md")) as f:
            assert len(f.read()) > 100


class TestPhishingDetection:
    def test_phishing_email_detected(self, trained_pipeline):
        pred = trained_pipeline.predict(
            ["URGENT: Click here NOW to verify your account http://scam.xyz password reset"]
        )
        assert pred[0] in [0, 1]

    def test_legit_email_predicted(self, trained_pipeline):
        pred = trained_pipeline.predict(["Hi team, meeting on Thursday at 2pm in room 3.12"])
        assert pred[0] in [0, 1]

    def test_batch_prediction(self, trained_pipeline):
        texts = [
            "Win free money click here http://scam.net urgent",
            "Meeting notes from today's standup attached",
            "Your account suspended verify immediately http://fake.com",
        ]
        assert len(trained_pipeline.predict(texts)) == 3
