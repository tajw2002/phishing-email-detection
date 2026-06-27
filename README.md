# Erkennung von Phishing-E-Mails durch datenbasierte Analyse

Praktischer Teil der Hausarbeit von Tarek Al Jawabra (Matrikelnummer 888997).

Python-Pipeline, die Spam-/Phishing-E-Mails mit sechs Machine-Learning-Klassifikatoren
auf Basis des Enron-Spam-Korpus erkennt und vergleicht.

## Voraussetzungen

- Python 3.9 oder neuer
- Internetzugang beim ersten Lauf (zum automatischen Download des Datensatzes)

## Installation

```bash
pip3 install -r requirements.txt
```

## Ausführung

```bash
# 1. Datensatz herunterladen und vorbereiten (Enron-Spam-Korpus)
python3 scripts/download_data.py

# 2. Komplette ML-Pipeline ausführen
python3 main.py --ml

# 3. Tests ausführen
python3 -m pytest tests/ -v
```

## Was die Pipeline erzeugt

| Pfad | Inhalt |
|------|--------|
| `data/emails.csv` | bereinigter Datensatz (wird automatisch erzeugt) |
| `output/comparison_table.csv` | Vergleichstabelle aller Klassifikatoren |
| `output/wordlist_analysis.csv` | Wortlisten-Analyse (Spam- vs. legitime Begriffe) |
| `output/figures/*.png` | Konfusionsmatrizen, ROC-Kurven, Metrikvergleich, Wortlisten |
| `output/reports/*.json` | detaillierte Metriken je Klassifikator + Zusammenfassung |
| `models/*.pkl` | trainierte Modelle |
| `docs/ergebnisse.md` | automatisch erzeugter Ergebnisbericht |
| `docs/projektbeschreibung.txt` | Schritt-für-Schritt-Beschreibung des Vorgehens |

## Projektstruktur

```
PhishingEmail/
├── main.py                  # Hauptpipeline (Feature-Extraktion, Training, Evaluation)
├── scripts/download_data.py # Download + Bereinigung des Enron-Spam-Korpus
├── tests/test_pipeline.py   # 37 automatisierte Tests (pytest)
├── requirements.txt
├── docs/                    # Ergebnisbericht und Vorgehensdokumentation
└── output/                  # erzeugte Tabellen, Grafiken und Reports
```

## Hinweis zu den Daten

Datensatz (`data/`) und trainierte Modelle (`models/`) sind hier im Repository
enthalten, werden aber auch automatisch durch die beiden Skripte oben neu erzeugt –
ein Löschen und Neuausführen liefert dieselben Ergebnisse. Der verwendete
Enron-Spam-Korpus stammt aus:

> Metsis, V., Androutsopoulos, I., & Paliouras, G. (2006). Spam filtering with naive
> Bayes – Which naive Bayes? In Proceedings of the 3rd Conference on Email and
> Anti-Spam (CEAS).
