# Tarek Al Jawabra - SoSe 2026
# Echten Datensatz fuer die Phishing/Spam-Erkennung vorbereiten
#
# Verwendet den Enron-Spam-Datensatz (Metsis, Androutsopoulos & Paliouras, 2006).
# Datengrundlage laut Aufgabenstellung: Enron Spam Dataset ODER SpamAssassin Corpus.
# Ich habe mich fuer Enron entschieden.

import os
import urllib.request
import zipfile

import pandas as pd

RAW_URL = "https://raw.githubusercontent.com/MWiechmann/enron_spam_data/master/enron_spam_data.zip"

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "..", "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
RAW_ZIP = os.path.join(RAW_DIR, "enron_spam_data.zip")
RAW_CSV = os.path.join(RAW_DIR, "enron_spam_data.csv")

os.makedirs(RAW_DIR, exist_ok=True)


def ensure_raw_data():
    """Laedt den Enron-Spam-Datensatz herunter, falls er nicht schon lokal liegt."""
    if os.path.exists(RAW_CSV):
        print(f"Rohdaten bereits vorhanden: {RAW_CSV}")
        return

    print(f"Lade Enron-Spam-Datensatz herunter von {RAW_URL} ...")
    urllib.request.urlretrieve(RAW_URL, RAW_ZIP)
    with zipfile.ZipFile(RAW_ZIP) as z:
        z.extractall(RAW_DIR)
    os.remove(RAW_ZIP)
    print("Download abgeschlossen.")


def clean_dataset() -> pd.DataFrame:
    """Liest die Rohdaten ein und bereitet sie fuer die Pipeline vor."""
    df = pd.read_csv(RAW_CSV)

    # Zeilen ohne jeglichen Inhalt (Subject UND Message leer) verwerfen
    df = df.dropna(subset=["Subject", "Message"], how="all")

    # Einzelne fehlende Subject/Message-Werte mit Leerstring auffuellen
    df["Subject"] = df["Subject"].fillna("")
    df["Message"] = df["Message"].fillna("")

    # Betreff + Nachrichtentext zu einem Volltext zusammenfuehren
    df["text"] = (df["Subject"] + " " + df["Message"]).str.strip()
    df = df[df["text"].str.len() > 0]

    # Spam/Ham -> binaeres Label (1 = Spam/Phishing-Proxy, 0 = legitim)
    df["label"] = (df["Spam/Ham"] == "spam").astype(int)

    return df[["text", "label"]].reset_index(drop=True)


if __name__ == "__main__":
    ensure_raw_data()
    print("Datensatz wird bereinigt ...")
    df = clean_dataset()

    out_path = os.path.join(DATA_DIR, "emails.csv")
    df.to_csv(out_path, index=False)
    print(f"Gespeichert: {out_path}")
    print(f"Gesamt: {len(df)} | Spam/Phishing: {df['label'].sum()} | Legitim: {(df['label']==0).sum()}")
