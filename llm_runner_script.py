import pandas as pd
import numpy as np
import json
import time
import os

from anthropic import Anthropic, RateLimitError, APIError
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix, classification_report

os.makedirs("results", exist_ok=True)

MODELL_NAME = "claude-sonnet-5"

# Preisliste Stand 31.08.2026 (https://platform.claude.com/docs/en/about-claude/pricing):
# Claude Sonnet 5, Standard-API-Preise (nicht Batch-API).
PREIS_PRO_MTOK_INPUT = 2.00
PREIS_PRO_MTOK_OUTPUT = 10.00

client = Anthropic()  # liest ANTHROPIC_API_KEY automatisch aus der Umgebung
print(f"Anthropic-Client initialisiert. Modell: {MODELL_NAME}")


pool = pd.read_csv("benchmark/tf1_duplicate_pool.csv").set_index("row_id")
pairs = pd.read_csv("benchmark/tf1_duplicate_pairs.csv")
test_pairs = pairs[pairs["split"] == "test"].reset_index(drop=True)

print(f"Test-Paare (identisch zum XGBoost-Notebook): {len(test_pairs)}")
print(test_pairs["label"].value_counts())


SYSTEM_PROMPT = """Du bist ein Experte für Datenqualität, spezialisiert auf die Erkennung von \
Duplikaten (erneut geposteten Angeboten) in einem Online-Deal-Forum.

Dir werden zwei Forenbeitrag-Einträge (A und B) im JSON-Format gezeigt. Beurteile anhand \
aller verfügbaren Informationen (Titel, Autor, Preis, Ersparnis, Datum, Quelle, Kategorie, URL), \
ob beide Einträge dieselbe reale Anzeige/denselben realen Deal beschreiben (z.B. ein erneut \
gepostetes oder von einem anderen Nutzer geteiltes Angebot), auch wenn Formulierung, \
Zeitstempel oder Schreibweise leicht abweichen.

Antworte AUSSCHLIESSLICH mit einem JSON-Objekt in folgendem Format, ohne zusätzlichen Text:
{"is_duplicate": true oder false, "reasoning": "kurze Begründung in 1-2 Sätzen"}
"""

def erstelle_user_prompt(row_a, row_b):
    felder = ["title", "author", "price", "saving", "creation_date", "source",
              "parent_category", "thread_category", "url"]
    a = {f: (None if pd.isna(row_a[f]) else row_a[f]) for f in felder}
    b = {f: (None if pd.isna(row_b[f]) else row_b[f]) for f in felder}
    return f"Eintrag A:\n{json.dumps(a, ensure_ascii=False)}\n\nEintrag B:\n{json.dumps(b, ensure_ascii=False)}"

print(erstelle_user_prompt(pool.loc[test_pairs.iloc[0]['row_id_a']], pool.loc[test_pairs.iloc[0]['row_id_b']]))


def rufe_claude_api_mit_backoff(system_prompt, user_prompt, model, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model,
                max_tokens=300,
                system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                thinking={"type": "disabled"},
            )
            raw_text = response.content[0].text.strip()
            usage = {"input_tokens": response.usage.input_tokens,
                     "output_tokens": response.usage.output_tokens}

            start, end = raw_text.find("{"), raw_text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise json.JSONDecodeError("JSON nicht extrahierbar.", raw_text, 0)
            parsed = json.loads(raw_text[start:end + 1])
            return parsed, usage

        except RateLimitError:
            delay = 2 ** attempt
            print(f"  Ratenlimit erreicht, warte {delay}s (Versuch {attempt + 1}/{max_retries})...")
            time.sleep(delay)
        except APIError as e:
            print(f"  API-Fehler bei Versuch {attempt + 1}: {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return None, None
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"  Antwort nicht verwertbar: {e}")
            return None, None
    return None, None


CHECKPOINT_PATH = "results/tf1_llm_checkpoint.csv"
TIMING_STATE_PATH = "results/tf1_llm_timing_state.json"
TIME_BUDGET_SEC = float(os.environ.get("LLM_TIME_BUDGET_SEC", "1e9"))

if os.path.exists(CHECKPOINT_PATH):
    ergebnisse = pd.read_csv(CHECKPOINT_PATH).to_dict("records")
    done_keys = {(r["row_id_a"], r["row_id_b"]) for r in ergebnisse}
    print(f"Checkpoint gefunden: {len(ergebnisse)} von {len(test_pairs)} Paaren bereits verarbeitet.")
else:
    ergebnisse, done_keys = [], set()
    print("Kein Checkpoint gefunden, starte von vorne.")

if os.path.exists(TIMING_STATE_PATH):
    timing_state = json.load(open(TIMING_STATE_PATH))
else:
    timing_state = {"cumulative_wall_time_sec": 0.0}

fehler = [r for r in ergebnisse if "FEHLER" in str(r.get("reasoning", ""))]
total_input_tok = int(sum(r.get("input_tokens", 0) for r in ergebnisse))
total_output_tok = int(sum(r.get("output_tokens", 0) for r in ergebnisse))

start_zeit = time.time()
verarbeitet_in_dieser_sitzung = 0

for i, p in test_pairs.iterrows():
    key = (p["row_id_a"], p["row_id_b"])
    if key in done_keys:
        continue
    if time.time() - start_zeit > TIME_BUDGET_SEC:
        print(f"Zeitbudget ({TIME_BUDGET_SEC:.0f}s) fuer diese Sitzung erreicht, breche kontrolliert ab.")
        break

    row_a, row_b = pool.loc[p["row_id_a"]], pool.loc[p["row_id_b"]]
    user_prompt = erstelle_user_prompt(row_a, row_b)
    result, usage = rufe_claude_api_mit_backoff(SYSTEM_PROMPT, user_prompt, MODELL_NAME)

    if result is not None:
        total_input_tok += usage["input_tokens"]
        total_output_tok += usage["output_tokens"]
        eintrag = {
            "row_id_a": p["row_id_a"], "row_id_b": p["row_id_b"], "label": p["label"],
            "prediction": int(bool(result.get("is_duplicate", False))),
            "reasoning": result.get("reasoning", ""),
            "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"],
        }
    else:
        eintrag = {
            "row_id_a": p["row_id_a"], "row_id_b": p["row_id_b"], "label": p["label"],
            "prediction": 0, "reasoning": "[FEHLER/FALLBACK - kein gültiges LLM-Ergebnis]",
            "input_tokens": 0, "output_tokens": 0,
        }
        fehler.append(eintrag)

    ergebnisse.append(eintrag)
    done_keys.add(key)
    verarbeitet_in_dieser_sitzung += 1

    # Sofortiges Speichern nach jedem einzelnen Paar (Checkpoint) - kein Datenverlust bei Abbruch.
    pd.DataFrame(ergebnisse).to_csv(CHECKPOINT_PATH, index=False)

    if verarbeitet_in_dieser_sitzung % 10 == 0:
        print(f"  ...{len(ergebnisse)} von {len(test_pairs)} Paaren insgesamt "
              f"({verarbeitet_in_dieser_sitzung} in dieser Sitzung).")

wall_time_sitzung = time.time() - start_zeit
timing_state["cumulative_wall_time_sec"] += wall_time_sitzung
json.dump(timing_state, open(TIMING_STATE_PATH, "w"))
wall_time = timing_state["cumulative_wall_time_sec"]

print("-" * 70)
print(f"Diese Sitzung: {verarbeitet_in_dieser_sitzung} Paare in {wall_time_sitzung:.1f}s verarbeitet.")
print(f"Insgesamt: {len(ergebnisse)} von {len(test_pairs)} Paaren, kumulierte Laufzeit {wall_time:.1f}s.")
print(f"Fehlgeschlagene Paare (insgesamt): {len(fehler)}")
print(f"Tokens insgesamt: {total_input_tok} Input, {total_output_tok} Output")

if len(ergebnisse) < len(test_pairs):
    raise RuntimeError(
        f"Nur {len(ergebnisse)} von {len(test_pairs)} Paaren verarbeitet (Zeitbudget erreicht). "
        "Zelle/Skript erneut ausfuehren, um mit dem Checkpoint fortzufahren."
    )


results_df = pd.DataFrame(ergebnisse)
y_true, y_pred = results_df["label"], results_df["prediction"]

precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print(f"Precision: {precision:.3f}  Recall: {recall:.3f}  F1: {f1:.3f}")
print()
print(classification_report(y_true, y_pred, target_names=["kein Duplikat", "Duplikat"]))
print("Konfusionsmatrix:")
print(confusion_matrix(y_true, y_pred))


results_df.to_csv("results/tf1_llm_predictions.csv", index=False)

estimated_cost = (total_input_tok / 1_000_000 * PREIS_PRO_MTOK_INPUT +
                   total_output_tok / 1_000_000 * PREIS_PRO_MTOK_OUTPUT)

metrics = {
    "experiment": "TF1_Duplikate", "method": "Claude_LLM", "model": MODELL_NAME,
    "n_test": len(test_pairs), "n_fehler": len(fehler),
    "precision": precision, "recall": recall, "f1": f1,
    "wall_time_sec": wall_time,
    "input_tokens": total_input_tok, "output_tokens": total_output_tok,
    "estimated_cost_usd": estimated_cost,
}
with open("results/tf1_llm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

log_row = pd.DataFrame([{
    "experiment": "TF1_Duplikate", "method": "Claude_LLM", "n_items": len(test_pairs),
    "wall_time_sec": wall_time, "input_tokens": total_input_tok, "output_tokens": total_output_tok,
    "estimated_cost_usd": estimated_cost, "model_name": MODELL_NAME,
}])
log_path = "results/laufzeit_kosten_log.csv"
if os.path.exists(log_path):
    _old_log = pd.read_csv(log_path)
    _new_keys = set(zip(log_row["experiment"], log_row["method"]))
    _old_log = _old_log[~_old_log.apply(lambda r: (r["experiment"], r["method"]) in _new_keys, axis=1)]
    _combined_log = pd.concat([_old_log, log_row], ignore_index=True)
else:
    _combined_log = log_row
_combined_log.to_csv(log_path, index=False)

print(f"Geschätzte API-Kosten: ${estimated_cost:.4f}")
print("Gespeichert: results/tf1_llm_predictions.csv, results/tf1_llm_metrics.json")
print("Laufzeit-Log aktualisiert: results/laufzeit_kosten_log.csv")
