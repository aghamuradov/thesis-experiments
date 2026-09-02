# Dieses Skript wird NICHT im Cloud-Sandbox ausgefuehrt (dort ist der Zugriff auf
# huggingface.co durch die Organisations-Firewall blockiert), sondern vom Nutzer in
# seinem eigenen Terminal - genau wie die LLM-Skripte. Es benoetigt KEINEN API-Key
# und verursacht KEINE Kosten (rein lokales, quelloffenes Modell).
#
# Es laedt das vortrainierte Sentence-Transformer-Modell 'all-MiniLM-L6-v2'
# (ca. 90 MB, einmaliger Download) und berechnet semantische Satz-Einbettungen fuer
# alle TF4-Texte (thread_category + title). Die Einbettungen werden als .npy-Dateien
# gespeichert und danach in die Cloud-Umgebung zurueckgeholt, wo der eigentliche
# Klassifikator (ohne Internetzugriff) trainiert wird.

import subprocess
import sys

try:
    import sentence_transformers  # noqa
except ImportError:
    print("sentence-transformers wird installiert...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "sentence-transformers"])

import pandas as pd
import numpy as np
import time
import os

from sentence_transformers import SentenceTransformer

os.makedirs("results", exist_ok=True)

print("Lade Modell 'sentence-transformers/all-MiniLM-L6-v2' (einmaliger Download, ca. 90 MB)...")
t0 = time.time()
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
print(f"Modell geladen in {time.time()-t0:.1f}s.")

eval_df = pd.read_csv("benchmark/tf4_semantic_eval.csv")
app_df = pd.read_csv("benchmark/tf4_application_set.csv")

eval_df["text"] = eval_df["thread_category"].fillna("") + " || " + eval_df["title"].fillna("")
app_df["text"] = app_df["thread_category"].fillna("") + " || " + app_df["title"].fillna("")

print(f"Berechne Einbettungen fuer {len(eval_df)} Eval-Zeilen und {len(app_df)} Anwendungs-Zeilen...")
t0 = time.time()
eval_embeddings = model.encode(eval_df["text"].tolist(), show_progress_bar=True, batch_size=64)
app_embeddings = model.encode(app_df["text"].tolist(), show_progress_bar=True, batch_size=64)
encode_time = time.time() - t0
print(f"Fertig in {encode_time:.1f}s.")

np.save("results/tf4_sentence_embeddings_eval.npy", eval_embeddings)
np.save("results/tf4_sentence_embeddings_app.npy", app_embeddings)
eval_df[["row_id", "split", "true_parent_category"]].to_csv("results/tf4_sentence_embeddings_eval_meta.csv", index=False)
app_df[["row_id"]].to_csv("results/tf4_sentence_embeddings_app_meta.csv", index=False)

import json
with open("results/tf4_sentence_embeddings_info.json", "w") as f:
    json.dump({
        "model_name": "sentence-transformers/all-MiniLM-L6-v2",
        "embedding_dim": int(eval_embeddings.shape[1]),
        "n_eval": len(eval_df),
        "n_application": len(app_df),
        "encode_time_sec": encode_time,
    }, f, indent=2)

print("Gespeichert: results/tf4_sentence_embeddings_{eval,app}.npy (+ _meta.csv, + info.json)")
print("EMBEDDINGS_ERSTELLT_OK")
