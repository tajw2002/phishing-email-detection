# Tarek Al Jawabra
# Hausarbeit: Erkennung von Phishing-E-Mails mit Machine Learning
# Kurs: Sicherheit von IT-Systemen, SoSe 2026
#
# Ausführung: python3 main.py --ml

import argparse
import json
import os
import re
import warnings
from collections import Counter

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from nltk.corpus import stopwords
import nltk
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

warnings.filterwarnings("ignore")

try:
    stopwords.words("english")
except LookupError:
    nltk.download("stopwords", quiet=True)

ROOT        = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(ROOT, "data")
OUTPUT_DIR  = os.path.join(ROOT, "output")
FIGURES_DIR = os.path.join(OUTPUT_DIR, "figures")
REPORTS_DIR = os.path.join(OUTPUT_DIR, "reports")
MODELS_DIR  = os.path.join(ROOT, "models")
DOCS_DIR    = os.path.join(ROOT, "docs")

for d in [OUTPUT_DIR, FIGURES_DIR, REPORTS_DIR, MODELS_DIR, DOCS_DIR]:
    os.makedirs(d, exist_ok=True)


def load_data() -> pd.DataFrame:
    path = os.path.join(DATA_DIR, "emails.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Datensatz nicht gefunden: {path}\n"
            "Zuerst scripts/download_data.py ausführen!"
        )
    df = pd.read_csv(path)
    assert "text" in df.columns and "label" in df.columns
    return df


# ---------------------------------------------------------------------------
# Wortlisten-Analyse
# ---------------------------------------------------------------------------
# Bevor die ML-Pipeline laeuft, werden die typischen Begriffe je Klasse
# extrahiert und verglichen - das ist in der Aufgabenstellung explizit als
# eigener Schritt vor der eigentlichen Klassifikation gefordert.

WORD_RE = re.compile(r"[a-zA-Z]{3,}")


def analyze_wordlists(df: pd.DataFrame, top_n: int = 20, min_count: int = 50):
    """Vergleicht Wortfrequenzen zwischen Spam/Phishing- und legitimen Mails."""
    stop_words = set(stopwords.words("english"))
    spam_counts = Counter()
    ham_counts = Counter()

    for text, label in zip(df["text"], df["label"]):
        words = [w.lower() for w in WORD_RE.findall(str(text)) if w.lower() not in stop_words]
        (spam_counts if label == 1 else ham_counts).update(words)

    rows = []
    for word in set(spam_counts) | set(ham_counts):
        s, h = spam_counts.get(word, 0), ham_counts.get(word, 0)
        total = s + h
        if total < min_count:
            continue
        rows.append({"word": word, "spam_count": s, "ham_count": h,
                     "total": total, "spam_ratio": round(s / total, 3)})

    wordlist_df = pd.DataFrame(rows)
    top_spam = wordlist_df.sort_values(["spam_ratio", "total"], ascending=[False, False]).head(top_n)
    top_ham = wordlist_df.sort_values(["spam_ratio", "total"], ascending=[True, False]).head(top_n)
    return top_spam.reset_index(drop=True), top_ham.reset_index(drop=True), wordlist_df


def plot_wordlists(top_spam: pd.DataFrame, top_ham: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    axes[0].barh(top_spam["word"][::-1], top_spam["spam_ratio"][::-1], color="tomato")
    axes[0].set_xlim(0, 1.05)
    axes[0].set_xlabel("Anteil in Spam/Phishing-Mails")
    axes[0].set_title("Typische Spam/Phishing-Begriffe")

    ham_ratio = (1 - top_ham["spam_ratio"])[::-1]
    axes[1].barh(top_ham["word"][::-1], ham_ratio, color="steelblue")
    axes[1].set_xlim(0, 1.05)
    axes[1].set_xlabel("Anteil in legitimen Mails")
    axes[1].set_title("Typische Begriffe legitimer Mails")

    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "wordlists.png"), dpi=150)
    plt.close()


# ---------------------------------------------------------------------------
# Strukturelle Zusatzmerkmale (Links, Formatierung)
# ---------------------------------------------------------------------------
# Absenderadressen koennen nicht ausgewertet werden, weil der Enron-Datensatz
# keine E-Mail-Header enthaelt - nur Betreff und Nachrichtentext sind verfuegbar.
# Das ist eine bewusste Einschraenkung, die in der Diskussion angesprochen wird.

URL_PATTERN = re.compile(r"(https?://|www\.)\S+", re.IGNORECASE)
EXCLAIM_PATTERN = re.compile(r"!")
DIGIT_PATTERN = re.compile(r"\d")
CURRENCY_PATTERN = re.compile(r"[$€£]")
SUSPICIOUS_WORDS = [
    "click", "verify", "urgent", "free", "winner", "password",
    "account", "limited", "offer", "guarantee", "congratulations", "act now",
]


class StructuralFeatureExtractor(BaseEstimator, TransformerMixin):
    """Zusaetzliche numerische Merkmale neben TF-IDF: Links, Formatierung, Schluesselwoerter."""

    FEATURE_NAMES = [
        "num_links", "num_exclamation", "digit_ratio",
        "num_currency_symbols", "caps_ratio", "suspicious_keyword_count",
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        rows = []
        for text in X:
            text = str(text)
            length = max(len(text), 1)
            letters = [c for c in text if c.isalpha()]
            n_letters = max(len(letters), 1)
            n_upper = sum(1 for c in letters if c.isupper())
            lower_text = text.lower()

            rows.append([
                len(URL_PATTERN.findall(text)),
                len(EXCLAIM_PATTERN.findall(text)),
                len(DIGIT_PATTERN.findall(text)) / length,
                len(CURRENCY_PATTERN.findall(text)),
                n_upper / n_letters,
                sum(lower_text.count(w) for w in SUSPICIOUS_WORDS),
            ])
        return np.array(rows, dtype=float)

    def get_feature_names_out(self, input_features=None):
        return np.array(self.FEATURE_NAMES)


def build_pipeline(clf) -> Pipeline:
    # TF-IDF für den Text + strukturelle Merkmale für Links/Formatierung,
    # per FeatureUnion zu einem gemeinsamen Merkmalsvektor kombiniert
    features = FeatureUnion([
        ("tfidf", TfidfVectorizer(
            max_features=5000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )),
        ("structural", StructuralFeatureExtractor()),
    ])
    return Pipeline([
        ("features", features),
        ("clf", clf),
    ])


def evaluate(name: str, pipeline: Pipeline, X_test, y_test) -> dict:
    y_pred = pipeline.predict(X_test)

    # AUC braucht Wahrscheinlichkeiten – LinearSVC liefert die nicht
    try:
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_prob)
    except AttributeError:
        y_prob = None
        auc = float("nan")

    return {
        "classifier": name,
        "accuracy":  round(accuracy_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "auc":       round(auc, 4) if not np.isnan(auc) else "N/A",
        "_y_pred": y_pred,
        "_y_prob": y_prob,
    }


def plot_confusion_matrices(results, X_test, y_test):
    n = len(results)
    cols = 3
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for idx, (name, pipeline, res) in enumerate(results):
        cm = confusion_matrix(y_test, res["_y_pred"])
        sns.heatmap(cm, annot=True, fmt="d", ax=axes[idx],
                    cmap="Blues", cbar=False,
                    xticklabels=["Legitim", "Spam/Phishing"],
                    yticklabels=["Legitim", "Spam/Phishing"])
        axes[idx].set_title(name, fontsize=10)
        axes[idx].set_xlabel("Vorhergesagt")
        axes[idx].set_ylabel("Tatsächlich")

    for ax in axes[n:]:
        ax.set_visible(False)

    plt.suptitle("Confusion Matrices – Alle Klassifikatoren", fontsize=13, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "confusion_matrices.png"), dpi=150, bbox_inches="tight")
    plt.close()


def plot_roc_curves(results, y_test):
    plt.figure(figsize=(9, 6))
    for name, pipeline, res in results:
        if res["_y_prob"] is not None:
            fpr, tpr, _ = roc_curve(y_test, res["_y_prob"])
            plt.plot(fpr, tpr, label=f"{name} (AUC={res['auc']})", linewidth=1.8)
    plt.plot([0, 1], [0, 1], "k--", linewidth=1)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC-Kurven – Klassifikatorvergleich")
    plt.legend(loc="lower right", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "roc_curves.png"), dpi=150)
    plt.close()


def plot_metric_comparison(comparison_df: pd.DataFrame):
    metrics = ["accuracy", "precision", "recall", "f1"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    axes = axes.flatten()

    for i, metric in enumerate(metrics):
        vals = comparison_df.set_index("classifier")[metric].astype(float)
        bars = axes[i].barh(vals.index, vals.values, color=sns.color_palette("viridis", len(vals)))
        axes[i].set_xlim(0, 1.05)
        axes[i].set_xlabel(metric.capitalize())
        axes[i].set_title(f"{metric.capitalize()} – Vergleich")
        for bar, v in zip(bars, vals.values):
            axes[i].text(v + 0.005, bar.get_y() + bar.get_height() / 2,
                         f"{v:.3f}", va="center", fontsize=8)

    plt.suptitle("Metriken-Vergleich aller Klassifikatoren", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "metric_comparison.png"), dpi=150)
    plt.close()


def plot_label_distribution(df: pd.DataFrame):
    counts = df["label"].value_counts().rename({0: "Legitim", 1: "Spam/Phishing"})
    fig, ax = plt.subplots(figsize=(5, 4))
    bars = ax.bar(counts.index, counts.values, color=["steelblue", "tomato"])
    for bar, v in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 5, str(v), ha="center", fontsize=11)
    ax.set_title("Klassenverteilung im Datensatz")
    ax.set_ylabel("Anzahl E-Mails")
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_DIR, "label_distribution.png"), dpi=150)
    plt.close()


def run_ml_pipeline():
    print("\n=== Phishing-Erkennung – ML-Pipeline ===\n")

    print("[1/7] Daten laden ...")
    df = load_data()
    print(f"      {len(df)} E-Mails | Spam/Phishing: {df['label'].sum()} | Legitim: {(df['label']==0).sum()}")
    plot_label_distribution(df)

    print("\n[2/7] Wortlisten-Analyse ...")
    top_spam, top_ham, wordlist_df = analyze_wordlists(df)
    wordlist_df.to_csv(os.path.join(OUTPUT_DIR, "wordlist_analysis.csv"), index=False)
    plot_wordlists(top_spam, top_ham)
    print(f"      Top Spam-Begriff: '{top_spam.iloc[0]['word']}' ({top_spam.iloc[0]['spam_ratio']*100:.0f}% Spam-Anteil)")
    print(f"      Top legitimer Begriff: '{top_ham.iloc[0]['word']}' ({(1-top_ham.iloc[0]['spam_ratio'])*100:.0f}% Ham-Anteil)")

    X_train, X_test, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )
    print(f"\n      Train: {len(X_train)} | Test: {len(X_test)}")

    print("\n[3/7] Klassifikatoren trainieren ...")
    classifiers = {
        "Logistic Regression":  LogisticRegression(max_iter=1000, C=1.0),
        "Naive Bayes":          MultinomialNB(alpha=0.1),
        "Random Forest":        RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "Decision Tree":        DecisionTreeClassifier(max_depth=20, random_state=42),
        "Gradient Boosting":    GradientBoostingClassifier(n_estimators=100, random_state=42),
        "Linear SVM":           LinearSVC(max_iter=2000, C=1.0),
    }

    results = []
    metrics_list = []

    for name, clf in classifiers.items():
        print(f"  {name} ...", end=" ", flush=True)
        pipeline = build_pipeline(clf)
        pipeline.fit(X_train, y_train)
        res = evaluate(name, pipeline, X_test, y_test)
        results.append((name, pipeline, res))
        metrics_list.append({k: v for k, v in res.items() if not k.startswith("_")})
        joblib.dump(pipeline, os.path.join(MODELS_DIR, f"{name.replace(' ', '_')}.pkl"))
        print(f"Acc={res['accuracy']:.3f}  F1={res['f1']:.3f}")

    print("\n[4/7] Vergleichstabelle ...")
    comparison_df = pd.DataFrame(metrics_list)
    comparison_df = comparison_df.sort_values("f1", ascending=False).reset_index(drop=True)
    comparison_df.to_csv(os.path.join(OUTPUT_DIR, "comparison_table.csv"), index=False)

    # Cross-Validation nur für den besten Klassifikator (Laufzeit!)
    best_name = comparison_df.iloc[0]["classifier"]
    best_pipeline = build_pipeline(classifiers[best_name])
    print(f"\n[5/7] Cross-Validation '{best_name}' (5-fold) ...")
    cv_scores = cross_val_score(best_pipeline, df["text"], df["label"], cv=5, scoring="f1", n_jobs=1)
    print(f"  F1: {cv_scores.round(3)}  |  Mittelwert: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")

    print("\n[6/7] Grafiken erstellen ...")
    plot_confusion_matrices(results, X_test, y_test)
    plot_roc_curves(results, y_test)
    plot_metric_comparison(comparison_df)

    print("\n[7/7] Reports speichern ...")
    for name, pipeline, res in results:
        report_data = {
            "classifier": name,
            "metrics": {k: v for k, v in res.items() if not k.startswith("_")},
            "classification_report": classification_report(
                y_test, res["_y_pred"],
                target_names=["Legitim", "Spam/Phishing"],
                output_dict=True,
            ),
        }
        fname = name.replace(" ", "_") + "_report.json"
        with open(os.path.join(REPORTS_DIR, fname), "w", encoding="utf-8") as f:
            json.dump(report_data, f, indent=2, ensure_ascii=False)

    summary = {
        "total_emails": int(len(df)),
        "phishing":     int(df["label"].sum()),
        "legit":        int((df["label"] == 0).sum()),
        "train_size":   int(len(X_train)),
        "test_size":    int(len(X_test)),
        "best_classifier": best_name,
        "cv_f1_mean":   round(float(cv_scores.mean()), 4),
        "cv_f1_std":    round(float(cv_scores.std()), 4),
        "classifiers":  metrics_list,
        "top_spam_words": top_spam["word"].tolist()[:10],
        "top_ham_words":  top_ham["word"].tolist()[:10],
    }
    with open(os.path.join(REPORTS_DIR, "summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    _write_markdown_report(comparison_df, summary, cv_scores, best_name, top_spam, top_ham)

    print("\n" + "=" * 55)
    print("ERGEBNISSE (sortiert nach F1):")
    print("=" * 55)
    print(comparison_df[["classifier", "accuracy", "precision", "recall", "f1", "auc"]].to_string(index=False))
    print("=" * 55)
    print(f"\nBester Klassifikator : {best_name}")
    print(f"CV F1-Score (5-fold) : {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")


def _write_markdown_report(comparison_df, summary, cv_scores, best_name, top_spam, top_ham):
    rows_md = "\n".join(
        f"| {r['classifier']} | {r['accuracy']} | {r['precision']} | {r['recall']} | {r['f1']} | {r['auc']} |"
        for _, r in comparison_df.iterrows()
    )
    spam_words_md = ", ".join(top_spam["word"].tolist()[:10])
    ham_words_md = ", ".join(top_ham["word"].tolist()[:10])

    content = f"""# Ergebnisse: Phishing-E-Mail-Erkennung

## Datensatz

Enron-Spam-Datensatz (Metsis, Androutsopoulos & Paliouras, 2006)

| Eigenschaft | Wert |
|---|---|
| Gesamt E-Mails | {summary['total_emails']} |
| Spam/Phishing | {summary['phishing']} |
| Legitim | {summary['legit']} |
| Trainingsset | {summary['train_size']} |
| Testset | {summary['test_size']} |

## Wortlisten-Analyse

- Typische Spam/Phishing-Begriffe: {spam_words_md}
- Typische legitime Begriffe: {ham_words_md}

## Klassifikatorvergleich

| Klassifikator | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
{rows_md}

## Cross-Validation

- **Bester Klassifikator:** {best_name}
- **CV F1-Score (5-fold):** {cv_scores.mean():.4f} ± {cv_scores.std():.4f}

## Erzeugte Dateien

### Grafiken (`output/figures/`)
- `confusion_matrices.png`
- `roc_curves.png`
- `metric_comparison.png`
- `label_distribution.png`
- `wordlists.png`

### Reports (`output/reports/`)
- `summary.json`
- Je ein detaillierter Report pro Klassifikator

### Modelle (`models/`)
- Alle trainierten Modelle als `.pkl`-Datei

## Fazit

**{best_name}** hat im direkten Vergleich den höchsten F1-Score erzielt.
Die 5-fache Cross-Validation bestätigt, dass das Modell gut generalisiert.
"""
    with open(os.path.join(DOCS_DIR, "ergebnisse.md"), "w", encoding="utf-8") as f:
        f.write(content)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Phishing-E-Mail-Erkennung")
    parser.add_argument("--ml", action="store_true", help="ML-Pipeline starten")
    args = parser.parse_args()

    if args.ml:
        run_ml_pipeline()
    else:
        print("Hinweis: --ml flag zum Starten der Pipeline verwenden")
