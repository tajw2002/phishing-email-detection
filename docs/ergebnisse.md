# Ergebnisse: Phishing-E-Mail-Erkennung

## Datensatz

Enron-Spam-Datensatz (Metsis, Androutsopoulos & Paliouras, 2006)

| Eigenschaft | Wert |
|---|---|
| Gesamt E-Mails | 33665 |
| Spam/Phishing | 17120 |
| Legitim | 16545 |
| Trainingsset | 26932 |
| Testset | 6733 |

## Wortlisten-Analyse

- Typische Spam/Phishing-Begriffe: pills, viagra, cialis, computron, voip, ooking, photoshop, nbsp, wysak, wiil
- Typische legitime Begriffe: enron, ect, dynegy, kaminski, ees, ena, mmbtu, dbcaps, hourahead, fastow

## Klassifikatorvergleich

| Klassifikator | Accuracy | Precision | Recall | F1-Score | AUC |
|---|---|---|---|---|---|
| Logistic Regression | 0.9898 | 0.9855 | 0.9945 | 0.99 | 0.9991 |
| Linear SVM | 0.9896 | 0.9881 | 0.9915 | 0.9898 | N/A |
| Random Forest | 0.9869 | 0.9804 | 0.9942 | 0.9872 | 0.9989 |
| Gradient Boosting | 0.9663 | 0.9471 | 0.9889 | 0.9676 | 0.9945 |
| Naive Bayes | 0.966 | 0.9675 | 0.9655 | 0.9665 | 0.9874 |
| Decision Tree | 0.9513 | 0.9398 | 0.9661 | 0.9528 | 0.9634 |

## Cross-Validation

- **Bester Klassifikator:** Logistic Regression
- **CV F1-Score (5-fold):** 0.9800 ± 0.0066

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

**Logistic Regression** hat im direkten Vergleich den höchsten F1-Score erzielt.
Die 5-fache Cross-Validation bestätigt, dass das Modell gut generalisiert.
