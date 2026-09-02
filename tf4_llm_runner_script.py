import pandas as pd
import numpy as np
import json
import time
import os

from anthropic import Anthropic, RateLimitError, APIError
from sklearn.metrics import accuracy_score, f1_score

os.makedirs("results", exist_ok=True)

MODELL_NAME = "claude-sonnet-5"
# Preise geprueft am 31.08.2026 gegen https://platform.claude.com/docs/en/about-claude/pricing
PREIS_PRO_MTOK_INPUT = 2.00
PREIS_PRO_MTOK_OUTPUT = 10.00

TIME_BUDGET_SEC = float(os.environ.get("LLM_TIME_BUDGET_SEC", "1e9"))

client = Anthropic()
eval_df = pd.read_csv("benchmark/tf4_semantic_eval.csv")
app_df = pd.read_csv("benchmark/tf4_application_set.csv")
test = eval_df[eval_df["split"] == "test"].reset_index(drop=True)

VALID_CATEGORIES = sorted(eval_df["true_parent_category"].unique().tolist())
print(f"Anthropic-Client initialisiert. Modell: {MODELL_NAME}. Test: {len(test)}, Anwendung: {len(app_df)}")


SYSTEM_PROMPT = f"""Du bist ein Experte für Produktkategorisierung in einem Online-Deal-Forum. \
Ordne die gezeigte Zeile (Unterkategorie thread_category und Titel) GENAU EINER der folgenden \
zulässigen Hauptkategorien zu:

{VALID_CATEGORIES}

Antworte AUSSCHLIESSLICH mit einem JSON-Objekt:
{{"parent_category": "<eine der zulässigen Kategorien>", "reasoning": "kurze Begründung"}}
"""

def build_prompt(row):
    return f'thread_category: "{row["thread_category"]}"\ntitle: "{row["title"]}"'

def rufe_claude_api_mit_backoff(system_prompt, user_prompt, model, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model, max_tokens=150, system=system_prompt,
                messages=[{"role": "user", "content": user_prompt}],
                thinking={"type": "disabled"},
            )
            raw_text = response.content[0].text.strip()
            usage = {"input_tokens": response.usage.input_tokens, "output_tokens": response.usage.output_tokens}
            start, end = raw_text.find("{"), raw_text.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise json.JSONDecodeError("JSON nicht extrahierbar.", raw_text, 0)
            return json.loads(raw_text[start:end + 1]), usage
        except RateLimitError:
            delay = 2 ** attempt
            print(f"  Ratenlimit, warte {delay}s (Versuch {attempt + 1}/{max_retries})...")
            time.sleep(delay)
        except APIError as e:
            print(f"  API-Fehler (Versuch {attempt + 1}): {e}")
            if attempt < max_retries - 1:
                time.sleep(5)
            else:
                return None, None
        except (json.JSONDecodeError, IndexError, KeyError) as e:
            print(f"  Antwort nicht verwertbar: {e}")
            return None, None
    return None, None


def process_loop_mit_checkpoint(items_df, id_col, checkpoint_path, timing_state, timing_key, label):
    if os.path.exists(checkpoint_path):
        done = pd.read_csv(checkpoint_path)
        done_ids = set(done[id_col].tolist())
        print(f"  Checkpoint gefunden: {len(done_ids)} von {len(items_df)} {label}-Zeilen bereits verarbeitet.")
    else:
        done = pd.DataFrame(columns=[id_col])
        done_ids = set()

    t0 = time.time()
    budget_exceeded = False
    for i, (_, row) in enumerate(items_df.iterrows()):
        if row[id_col] in done_ids:
            continue
        if time.time() - t0 > TIME_BUDGET_SEC:
            budget_exceeded = True
            print(f"  Zeitbudget ({TIME_BUDGET_SEC:.0f}s) erreicht, breche {label} kontrolliert ab.")
            break
        result, usage = rufe_claude_api_mit_backoff(SYSTEM_PROMPT, build_prompt(row), MODELL_NAME)
        if result is not None:
            pred = result.get("parent_category")
            if pred not in VALID_CATEGORIES:
                pred = None
            neue_zeile = {id_col: row[id_col], "parent_category_pred_llm": pred, "reasoning": result.get("reasoning", ""),
                          "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]}
        else:
            neue_zeile = {id_col: row[id_col], "parent_category_pred_llm": None, "reasoning": "[FEHLER/FALLBACK - kein gueltiges LLM-Ergebnis]",
                          "input_tokens": 0, "output_tokens": 0}
        done = pd.concat([done, pd.DataFrame([neue_zeile])], ignore_index=True)
        done.to_csv(checkpoint_path, index=False)
        if len(done) % 50 == 0:
            print(f"  {label}: {len(done)}/{len(items_df)}")

    timing_state[timing_key] = timing_state.get(timing_key, 0.0) + (time.time() - t0)

    if budget_exceeded or len(done) < len(items_df):
        raise RuntimeError(f"{label}-Aufgabe unvollstaendig ({len(done)}/{len(items_df)}). Notebook erneut ausfuehren.")

    total_in = int(done["input_tokens"].sum())
    total_out = int(done["output_tokens"].sum())
    return done, total_in, total_out, timing_state[timing_key]


TIMING_STATE_PATH = "results/tf4_llm_timing_state.json"
if os.path.exists(TIMING_STATE_PATH):
    with open(TIMING_STATE_PATH) as f:
        timing_state = json.load(f)
else:
    timing_state = {}


results_df, total_in, total_out, wall_time = process_loop_mit_checkpoint(
    test, "row_id", "results/tf4_llm_test_checkpoint.csv", timing_state, "test_sec", "test")

with open(TIMING_STATE_PATH, "w") as f:
    json.dump(timing_state, f, indent=2)

merged = results_df.merge(test[["row_id", "true_parent_category"]], on="row_id")
valid = merged.dropna(subset=["parent_category_pred_llm"])

accuracy = accuracy_score(valid["true_parent_category"], valid["parent_category_pred_llm"])
macro_f1 = f1_score(valid["true_parent_category"], valid["parent_category_pred_llm"], average="macro")
print(f"Accuracy: {accuracy:.3f}  Macro-F1: {macro_f1:.3f}  (n_valid={len(valid)}/{len(merged)}, kumulierte Laufzeit {wall_time:.1f}s)")

merged.to_csv("results/tf4_llm_predictions.csv", index=False)


app_results_df, total_in_app, total_out_app, wall_time_app = process_loop_mit_checkpoint(
    app_df, "row_id", "results/tf4_llm_application_checkpoint.csv", timing_state, "application_sec", "Anwendung")

with open(TIMING_STATE_PATH, "w") as f:
    json.dump(timing_state, f, indent=2)

app_out = app_df[["row_id", "thread_category", "title"]].merge(
    app_results_df[["row_id", "parent_category_pred_llm"]], on="row_id")
app_out.to_csv("results/tf4_llm_application_predictions.csv", index=False)
print(f"Anwendungsmenge verarbeitet: {len(app_out)} Zeilen (kumulierte Laufzeit {wall_time_app:.1f}s)")


cost_test = total_in/1_000_000*PREIS_PRO_MTOK_INPUT + total_out/1_000_000*PREIS_PRO_MTOK_OUTPUT
cost_app = total_in_app/1_000_000*PREIS_PRO_MTOK_INPUT + total_out_app/1_000_000*PREIS_PRO_MTOK_OUTPUT

metrics = {
    "experiment": "TF4_Semantik", "method": "Claude_LLM", "model": MODELL_NAME,
    "accuracy": accuracy, "macro_f1": macro_f1, "n_valid": len(valid),
    "wall_time_sec": wall_time, "estimated_cost_usd": cost_test + cost_app,
    "n_application_rows": len(app_df),
}
with open("results/tf4_llm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

log_rows = pd.DataFrame([{
    "experiment": "TF4_Semantik", "method": "Claude_LLM", "n_items": len(test),
    "wall_time_sec": wall_time, "input_tokens": total_in, "output_tokens": total_out,
    "estimated_cost_usd": cost_test, "model_name": MODELL_NAME,
}])
log_path = "results/laufzeit_kosten_log.csv"
if os.path.exists(log_path):
    _old_log = pd.read_csv(log_path)
    _new_keys = set(zip(log_rows["experiment"], log_rows["method"]))
    _old_log = _old_log[~_old_log.apply(lambda r: (r["experiment"], r["method"]) in _new_keys, axis=1)]
    _combined_log = pd.concat([_old_log, log_rows], ignore_index=True)
else:
    _combined_log = log_rows
_combined_log.to_csv(log_path, index=False)

print(f"Geschaetzte Gesamtkosten TF4 (LLM, Test+Anwendung): ${cost_test + cost_app:.4f}")
print("Gespeichert: results/tf4_llm_predictions.csv, results/tf4_llm_application_predictions.csv, results/tf4_llm_metrics.json")
print("NOTEBOOK_EXECUTED_OK")
