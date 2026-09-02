import pandas as pd
import numpy as np
import json
import time
import re
import os

from anthropic import Anthropic, RateLimitError, APIError
from sklearn.metrics import mean_absolute_error, mean_squared_error, accuracy_score, f1_score

os.makedirs("results", exist_ok=True)

MODELL_NAME = "claude-sonnet-5"
# Preise gepr\u00fcft am 31.08.2026 gegen https://platform.claude.com/docs/en/about-claude/pricing
# (die urspruenglich fuer 01.09.2026 angekuendigte Erhoehung auf $3/$15 wurde storniert;
#  $2/$10 ist der geltende Standardpreis fuer Claude Sonnet 5).
PREIS_PRO_MTOK_INPUT = 2.00
PREIS_PRO_MTOK_OUTPUT = 10.00

TIME_BUDGET_SEC = float(os.environ.get("LLM_TIME_BUDGET_SEC", "1e9"))

client = Anthropic()
print(f"Anthropic-Client initialisiert. Modell: {MODELL_NAME}")


def rufe_claude_api_mit_backoff(system_prompt, user_prompt, model, max_retries=5):
    for attempt in range(max_retries):
        try:
            response = client.messages.create(
                model=model, max_tokens=200, system=system_prompt,
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


price_df = pd.read_csv("benchmark/tf2_missing_price.csv")
price_truth = pd.read_csv("benchmark/tf2_missing_price_groundtruth.csv").set_index("row_id")["price_true"]
price_test = price_df[price_df["split"] == "test"].reset_index(drop=True)

SYSTEM_PROMPT_PRICE = """Du bist ein Experte für Datenqualität. Dir wird eine Zeile aus einem \
Online-Deal-Forum gezeigt, bei der der Preis (price, in USD) fehlt. Schätze auf Basis der \
übrigen Angaben (Kategorie, Quelle, Interaktionszahlen) einen plausiblen Preis.

Antworte AUSSCHLIESSLICH mit einem JSON-Objekt:
{"estimated_price": <Zahl>, "reasoning": "kurze Begründung"}
"""

def prompt_price(row):
    felder = ["views", "votes", "replies", "parent_category", "thread_category", "source"]
    ctx = {f: (None if pd.isna(row[f]) else row[f]) for f in felder}
    return f"Zeile mit fehlendem Preis:\n{json.dumps(ctx, ensure_ascii=False)}"

CHECKPOINT_PRICE = "results/tf2_llm_price_checkpoint.csv"
TIMING_STATE_PATH = "results/tf2_llm_timing_state.json"

if os.path.exists(CHECKPOINT_PRICE):
    done_price = pd.read_csv(CHECKPOINT_PRICE)
    done_ids_price = set(done_price["row_id"].tolist())
    print(f"Checkpoint gefunden: {len(done_ids_price)} von {len(price_test)} price-Zeilen bereits verarbeitet.")
else:
    done_price = pd.DataFrame(columns=["row_id", "price_pred_llm", "reasoning", "input_tokens", "output_tokens"])
    done_ids_price = set()

if os.path.exists(TIMING_STATE_PATH):
    with open(TIMING_STATE_PATH) as f:
        timing_state = json.load(f)
else:
    timing_state = {"price_wall_time_sec": 0.0, "parent_category_wall_time_sec": 0.0}

t0 = time.time()
budget_exceeded = False
for i, row in price_test.iterrows():
    if row["row_id"] in done_ids_price:
        continue
    if time.time() - t0 > TIME_BUDGET_SEC:
        budget_exceeded = True
        print(f"Zeitbudget ({TIME_BUDGET_SEC:.0f}s) erreicht, breche kontrolliert ab.")
        break
    result, usage = rufe_claude_api_mit_backoff(SYSTEM_PROMPT_PRICE, prompt_price(row), MODELL_NAME)
    if result is not None:
        try:
            est = float(result.get("estimated_price"))
        except (TypeError, ValueError):
            est = np.nan
        neue_zeile = {"row_id": row["row_id"], "price_pred_llm": est, "reasoning": result.get("reasoning", ""),
                      "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]}
    else:
        neue_zeile = {"row_id": row["row_id"], "price_pred_llm": np.nan, "reasoning": "[FEHLER/FALLBACK - kein gueltiges LLM-Ergebnis]",
                      "input_tokens": 0, "output_tokens": 0}
    done_price = pd.concat([done_price, pd.DataFrame([neue_zeile])], ignore_index=True)
    done_price.to_csv(CHECKPOINT_PRICE, index=False)  # sofortiger Checkpoint - kein Datenverlust bei Abbruch
    if len(done_price) % 25 == 0:
        print(f"  price: {len(done_price)}/{len(price_test)}")

timing_state["price_wall_time_sec"] += time.time() - t0
with open(TIMING_STATE_PATH, "w") as f:
    json.dump(timing_state, f, indent=2)

if budget_exceeded or len(done_price) < len(price_test):
    raise RuntimeError(
        f"price-Aufgabe unvollstaendig ({len(done_price)}/{len(price_test)}). "
        "Notebook erneut ausfuehren, um am Checkpoint fortzusetzen."
    )

price_llm_df = done_price.set_index("row_id")
price_llm_df["price_true"] = price_truth.loc[price_llm_df.index]
valid = price_llm_df.dropna(subset=["price_pred_llm"])
total_in_p = int(done_price["input_tokens"].sum())
total_out_p = int(done_price["output_tokens"].sum())

mae_llm = mean_absolute_error(valid["price_true"], valid["price_pred_llm"])
rmse_llm = np.sqrt(mean_squared_error(valid["price_true"], valid["price_pred_llm"]))
wall_time_price = timing_state["price_wall_time_sec"]
print(f"price (LLM) - MAE: {mae_llm:.3f}  RMSE: {rmse_llm:.3f}  (n_valid={len(valid)}/{len(price_llm_df)}, kumulierte Laufzeit {wall_time_price:.1f}s)")

price_llm_df.reset_index().to_csv("results/tf2_llm_price_predictions.csv", index=False)


cat_df = pd.read_csv("benchmark/tf2_missing_parent_category.csv")
cat_truth = pd.read_csv("benchmark/tf2_missing_parent_category_groundtruth.csv").set_index("row_id")["parent_category_true"]
cat_test = cat_df[cat_df["split"] == "test"].reset_index(drop=True)

VALID_CATEGORIES = sorted(cat_truth.unique().tolist())

SYSTEM_PROMPT_CAT = f"""Du bist ein Experte für Produktkategorisierung in einem Online-Deal-Forum. \
Dir wird eine Zeile gezeigt, bei der die Hauptkategorie (parent_category) fehlt. Wähle GENAU EINE \
der folgenden zulässigen Kategorien basierend auf Titel, Unterkategorie (thread_category) und den \
übrigen Angaben:

{VALID_CATEGORIES}

Antworte AUSSCHLIESSLICH mit einem JSON-Objekt:
{{"parent_category": "<eine der zulässigen Kategorien>", "reasoning": "kurze Begründung"}}
"""

def prompt_cat(row):
    felder = ["title", "thread_category", "source", "price", "saving", "views", "votes", "replies"]
    ctx = {f: (None if pd.isna(row[f]) else row[f]) for f in felder}
    return f"Zeile mit fehlender parent_category:\n{json.dumps(ctx, ensure_ascii=False)}"

CHECKPOINT_CAT = "results/tf2_llm_parent_category_checkpoint.csv"

if os.path.exists(CHECKPOINT_CAT):
    done_cat = pd.read_csv(CHECKPOINT_CAT)
    done_ids_cat = set(done_cat["row_id"].tolist())
    print(f"Checkpoint gefunden: {len(done_ids_cat)} von {len(cat_test)} parent_category-Zeilen bereits verarbeitet.")
else:
    done_cat = pd.DataFrame(columns=["row_id", "parent_category_pred_llm", "reasoning", "input_tokens", "output_tokens"])
    done_ids_cat = set()

t0 = time.time()
budget_exceeded_cat = False
for i, row in cat_test.iterrows():
    if row["row_id"] in done_ids_cat:
        continue
    if time.time() - t0 > TIME_BUDGET_SEC:
        budget_exceeded_cat = True
        print(f"Zeitbudget ({TIME_BUDGET_SEC:.0f}s) erreicht, breche kontrolliert ab.")
        break
    result, usage = rufe_claude_api_mit_backoff(SYSTEM_PROMPT_CAT, prompt_cat(row), MODELL_NAME)
    if result is not None:
        pred = result.get("parent_category", None)
        if pred not in VALID_CATEGORIES:
            pred = None
        neue_zeile = {"row_id": row["row_id"], "parent_category_pred_llm": pred, "reasoning": result.get("reasoning", ""),
                      "input_tokens": usage["input_tokens"], "output_tokens": usage["output_tokens"]}
    else:
        neue_zeile = {"row_id": row["row_id"], "parent_category_pred_llm": None, "reasoning": "[FEHLER/FALLBACK - kein gueltiges LLM-Ergebnis]",
                      "input_tokens": 0, "output_tokens": 0}
    done_cat = pd.concat([done_cat, pd.DataFrame([neue_zeile])], ignore_index=True)
    done_cat.to_csv(CHECKPOINT_CAT, index=False)
    if len(done_cat) % 25 == 0:
        print(f"  parent_category: {len(done_cat)}/{len(cat_test)}")

timing_state["parent_category_wall_time_sec"] += time.time() - t0
with open(TIMING_STATE_PATH, "w") as f:
    json.dump(timing_state, f, indent=2)

if budget_exceeded_cat or len(done_cat) < len(cat_test):
    raise RuntimeError(
        f"parent_category-Aufgabe unvollstaendig ({len(done_cat)}/{len(cat_test)}). "
        "Notebook erneut ausfuehren, um am Checkpoint fortzusetzen."
    )

cat_llm_df = done_cat.set_index("row_id")
cat_llm_df["parent_category_true"] = cat_truth.loc[cat_llm_df.index]
valid_cat = cat_llm_df.dropna(subset=["parent_category_pred_llm"])
total_in_c = int(done_cat["input_tokens"].sum())
total_out_c = int(done_cat["output_tokens"].sum())

acc_llm = accuracy_score(valid_cat["parent_category_true"], valid_cat["parent_category_pred_llm"])
macro_f1_llm = f1_score(valid_cat["parent_category_true"], valid_cat["parent_category_pred_llm"], average="macro")
wall_time_cat = timing_state["parent_category_wall_time_sec"]
print(f"parent_category (LLM) - Accuracy: {acc_llm:.3f}  Macro-F1: {macro_f1_llm:.3f}  (n_valid={len(valid_cat)}/{len(cat_llm_df)}, kumulierte Laufzeit {wall_time_cat:.1f}s)")

cat_llm_df.reset_index().to_csv("results/tf2_llm_parent_category_predictions.csv", index=False)


cost_price = total_in_p/1_000_000*PREIS_PRO_MTOK_INPUT + total_out_p/1_000_000*PREIS_PRO_MTOK_OUTPUT
cost_cat = total_in_c/1_000_000*PREIS_PRO_MTOK_INPUT + total_out_c/1_000_000*PREIS_PRO_MTOK_OUTPUT

metrics = {
    "experiment": "TF2_FehlendeWerte", "method": "Claude_LLM", "model": MODELL_NAME,
    "price_mae": mae_llm, "price_rmse": rmse_llm, "price_n_valid": len(valid),
    "parent_category_accuracy": acc_llm, "parent_category_macro_f1": macro_f1_llm, "parent_category_n_valid": len(valid_cat),
    "wall_time_price_sec": wall_time_price, "wall_time_parent_category_sec": wall_time_cat,
    "input_tokens_price": total_in_p, "output_tokens_price": total_out_p,
    "input_tokens_parent_category": total_in_c, "output_tokens_parent_category": total_out_c,
    "estimated_cost_usd": cost_price + cost_cat,
}
with open("results/tf2_llm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

log_rows = pd.DataFrame([
    {"experiment": "TF2_FehlendeWerte_price", "method": "Claude_LLM", "n_items": len(price_test),
     "wall_time_sec": wall_time_price, "input_tokens": total_in_p, "output_tokens": total_out_p,
     "estimated_cost_usd": cost_price, "model_name": MODELL_NAME},
    {"experiment": "TF2_FehlendeWerte_parent_category", "method": "Claude_LLM", "n_items": len(cat_test),
     "wall_time_sec": wall_time_cat, "input_tokens": total_in_c, "output_tokens": total_out_c,
     "estimated_cost_usd": cost_cat, "model_name": MODELL_NAME},
])
log_path = "results/laufzeit_kosten_log.csv"
if os.path.exists(log_path):
    _old_log = pd.read_csv(log_path)
    _new_keys = set(zip(log_rows["experiment"], log_rows["method"]))
    _old_log = _old_log[~_old_log.apply(lambda r: (r["experiment"], r["method"]) in _new_keys, axis=1)]
    _combined_log = pd.concat([_old_log, log_rows], ignore_index=True)
else:
    _combined_log = log_rows
_combined_log.to_csv(log_path, index=False)

print(f"Geschaetzte Gesamtkosten TF2 (LLM): ${cost_price + cost_cat:.4f}")
print("Gespeichert: results/tf2_llm_price_predictions.csv, results/tf2_llm_parent_category_predictions.csv, results/tf2_llm_metrics.json")
print("NOTEBOOK_EXECUTED_OK")
