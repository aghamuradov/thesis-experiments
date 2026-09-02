import pandas as pd
import numpy as np
import json
import time
import re
import os

from anthropic import Anthropic, RateLimitError, APIError

os.makedirs("results", exist_ok=True)

MODELL_NAME = "claude-sonnet-5"
# Preise geprueft am 31.08.2026 gegen https://platform.claude.com/docs/en/about-claude/pricing
PREIS_PRO_MTOK_INPUT = 2.00
PREIS_PRO_MTOK_OUTPUT = 10.00

TIME_BUDGET_SEC = float(os.environ.get("LLM_TIME_BUDGET_SEC", "1e9"))

client = Anthropic()
df_raw = pd.read_csv("data/rfd_main.csv").drop(columns=["Unnamed: 0"]).reset_index(drop=True)
df_raw["row_id"] = df_raw.index
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


def process_loop_mit_checkpoint(items_df, id_col, checkpoint_path, timing_state, timing_key,
                                 system_prompt, prompt_fn, parse_fn, empty_row_fn, label):
    """Verarbeitet eine Test-DataFrame zeilenweise mit Checkpoint/Resume:
    - liest ggf. vorhandenen Checkpoint ein und ueberspringt bereits erledigte IDs,
    - schreibt nach jeder einzelnen Zeile sofort den Checkpoint,
    - bricht kontrolliert ab, wenn TIME_BUDGET_SEC ueberschritten wird (Exception, kein Datenverlust),
    - kumuliert die Laufzeit ueber mehrere Notebook-Ausfuehrungen hinweg in timing_state.
    """
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
        result, usage = rufe_claude_api_mit_backoff(system_prompt, prompt_fn(row), MODELL_NAME)
        if result is not None:
            neue_zeile = parse_fn(row, result)
            neue_zeile["input_tokens"] = usage["input_tokens"]
            neue_zeile["output_tokens"] = usage["output_tokens"]
        else:
            neue_zeile = empty_row_fn(row)
            neue_zeile["input_tokens"] = 0
            neue_zeile["output_tokens"] = 0
        done = pd.concat([done, pd.DataFrame([neue_zeile])], ignore_index=True)
        done.to_csv(checkpoint_path, index=False)
        if len(done) % 50 == 0:
            print(f"  {label}: {len(done)}/{len(items_df)}")

    timing_state[timing_key] = timing_state.get(timing_key, 0.0) + (time.time() - t0)

    if budget_exceeded or len(done) < len(items_df):
        raise RuntimeError(
            f"{label}-Aufgabe unvollstaendig ({len(done)}/{len(items_df)}). "
            "Notebook erneut ausfuehren, um am Checkpoint fortzusetzen."
        )

    total_in = int(done["input_tokens"].sum())
    total_out = int(done["output_tokens"].sum())
    return done, total_in, total_out, timing_state[timing_key]


TIMING_STATE_PATH = "results/tf3_llm_timing_state.json"
if os.path.exists(TIMING_STATE_PATH):
    with open(TIMING_STATE_PATH) as f:
        timing_state = json.load(f)
else:
    timing_state = {}


price_bench = pd.read_csv("benchmark/tf3_format_price.csv")
price_test = price_bench[price_bench["split"] == "test"].reset_index(drop=True)

SYSTEM_PRICE = """Du bist ein Experte für Datenbereinigung. Dir wird ein unsauberer, roher \
Preis-Rohwert aus einem Deal-Forum gezeigt (z.B. "$39.99", "Free", "19.99-29.99"). \
Extrahiere den bereinigten numerischen Preis in USD als Dezimalzahl (ohne Währungssymbol). \
"Free" bedeutet 0. Bei einer Preisspanne verwende den Mittelwert. Ist der Wert nicht \
sinnvoll interpretierbar (z.B. "varies"), gib null zurück.

Antworte AUSSCHLIESSLICH mit einem JSON-Objekt: {"clean_price": <Zahl oder null>}
"""

def price_prompt(row):
    return f'Roh-Preis: "{row["raw_value"]}"'

def price_parse(row, result):
    val = result.get("clean_price")
    pred = float(val) if isinstance(val, (int, float)) else None
    return {"row_id_raw": row["row_id_raw"], "raw_value": row["raw_value"],
            "true_clean_value": row["true_clean_value"], "price_pred_llm": pred}

def price_empty(row):
    return {"row_id_raw": row["row_id_raw"], "raw_value": row["raw_value"],
            "true_clean_value": row["true_clean_value"], "price_pred_llm": None}

price_llm_df, in_p, out_p, time_p = process_loop_mit_checkpoint(
    price_test, "row_id_raw", "results/tf3_llm_price_checkpoint.csv", timing_state, "price_sec",
    SYSTEM_PRICE, price_prompt, price_parse, price_empty, "price")

with open(TIMING_STATE_PATH, "w") as f:
    json.dump(timing_state, f, indent=2)

exact_p = (price_llm_df["price_pred_llm"].sub(price_llm_df["true_clean_value"]).abs() < 0.01).mean()
valid_p = price_llm_df["price_pred_llm"].notna().mean()
print(f"price (LLM) - Exact-Match-Rate: {exact_p:.3f}  Valid-Format-Rate: {valid_p:.3f}  (kumulierte Laufzeit {time_p:.1f}s)")
price_llm_df.to_csv("results/tf3_llm_price_predictions.csv", index=False)


saving_bench = pd.read_csv("benchmark/tf3_format_saving.csv")
saving_test = saving_bench[saving_bench["split"] == "test"].reset_index(drop=True)
raw_price_lookup = df_raw.set_index("row_id")["price"]
saving_test = saving_test.copy()
saving_test["raw_price"] = raw_price_lookup.loc[saving_test["row_id_raw"]].values

SYSTEM_SAVING = """Du bist ein Experte für Datenbereinigung. Dir werden der rohe Ersparnis-Wert \
(saving) und der rohe Preis-Wert (price) derselben Forenanzeige gezeigt. Berechne das \
bereinigte Ersparnis-Verhältnis (saving_ratio) wie folgt:

1. Bereinige zunächst den Preis zu einer Zahl (Regeln wie üblich: "$" entfernen, "Free"=0, ...).
2. Ist die Ersparnis bereits ein Prozentsatz (z.B. "50%"), gilt: saving_ratio = Prozent / 100.
3. Ist die Ersparnis ein Dollarbetrag oder eine reine Zahl (z.B. "$90", "100 off", "72"), gilt:
   saving_ratio = ersparnis_betrag / (bereinigter_preis + ersparnis_betrag)

Antworte AUSSCHLIESSLICH mit einem JSON-Objekt: {"saving_ratio": <Zahl zwischen 0 und 1, oder null>}
"""

def saving_prompt(row):
    return f'Roher Ersparnis-Wert: "{row["raw_value"]}"\nRoher Preis-Wert: "{row["raw_price"]}"'

def saving_parse(row, result):
    val = result.get("saving_ratio")
    pred = float(val) if isinstance(val, (int, float)) else None
    return {"row_id_raw": row["row_id_raw"], "raw_value": row["raw_value"],
            "true_clean_value": row["true_clean_value"], "saving_pred_llm": pred}

def saving_empty(row):
    return {"row_id_raw": row["row_id_raw"], "raw_value": row["raw_value"],
            "true_clean_value": row["true_clean_value"], "saving_pred_llm": None}

saving_llm_df, in_s, out_s, time_s = process_loop_mit_checkpoint(
    saving_test, "row_id_raw", "results/tf3_llm_saving_checkpoint.csv", timing_state, "saving_sec",
    SYSTEM_SAVING, saving_prompt, saving_parse, saving_empty, "saving")

with open(TIMING_STATE_PATH, "w") as f:
    json.dump(timing_state, f, indent=2)

exact_s = (saving_llm_df["saving_pred_llm"].sub(saving_llm_df["true_clean_value"]).abs() < 0.01).mean()
valid_s = saving_llm_df["saving_pred_llm"].notna().mean()
print(f"saving (LLM) - Exact-Match-Rate: {exact_s:.3f}  Valid-Format-Rate: {valid_s:.3f}  (kumulierte Laufzeit {time_s:.1f}s)")
saving_llm_df.to_csv("results/tf3_llm_saving_predictions.csv", index=False)


date_bench = pd.read_csv("benchmark/tf3_format_date.csv")
date_test = date_bench[date_bench["split"] == "test"].reset_index(drop=True)

SYSTEM_DATE = """Du bist ein Experte für Datenbereinigung. Dir wird ein roher Datums-/Zeitwert \
aus einem Online-Forum gezeigt (z.B. "Jul 12th, 2020 8:09 pm" oder "July 29, 2020"). \
Normalisiere ihn ins ISO-Format "YYYY-MM-DD HH:MM:SS". Enthält der Rohwert keine Uhrzeit, \
verwende "00:00:00".

Antworte AUSSCHLIESSLICH mit einem JSON-Objekt: {"clean_datetime": "<ISO-String>"}
"""

def date_prompt(row):
    return f'Roher Datumswert ({row["source_column"]}): "{row["raw_value"]}"'

def date_parse(row, result):
    return {"row_id_raw": row["row_id_raw"], "source_column": row["source_column"],
            "raw_value": row["raw_value"], "true_clean_value": row["true_clean_value"],
            "date_pred_llm": result.get("clean_datetime")}

def date_empty(row):
    return {"row_id_raw": row["row_id_raw"], "source_column": row["source_column"],
            "raw_value": row["raw_value"], "true_clean_value": row["true_clean_value"],
            "date_pred_llm": None}

date_llm_df, in_d, out_d, time_d = process_loop_mit_checkpoint(
    date_test, "row_id_raw", "results/tf3_llm_date_checkpoint.csv", timing_state, "date_sec",
    SYSTEM_DATE, date_prompt, date_parse, date_empty, "date")

with open(TIMING_STATE_PATH, "w") as f:
    json.dump(timing_state, f, indent=2)

date_llm_df["parsed"] = pd.to_datetime(date_llm_df["date_pred_llm"], errors="coerce")
date_llm_df["true_parsed"] = pd.to_datetime(date_llm_df["true_clean_value"], errors="coerce")

def dates_match(row):
    if pd.isna(row["parsed"]) or pd.isna(row["true_parsed"]):
        return False
    if row["source_column"] == "expiry":
        return row["parsed"].date() == row["true_parsed"].date()
    return row["parsed"].floor("min") == row["true_parsed"].floor("min")

date_llm_df["match"] = date_llm_df.apply(dates_match, axis=1)
exact_d = date_llm_df["match"].mean()
valid_d = date_llm_df["parsed"].notna().mean()
print(f"date (LLM) - Exact-Match-Rate: {exact_d:.3f}  Valid-Format-Rate: {valid_d:.3f}  (kumulierte Laufzeit {time_d:.1f}s)")
print(date_llm_df.groupby("source_column")["match"].mean())
date_llm_df.drop(columns=["parsed", "true_parsed"]).to_csv("results/tf3_llm_date_predictions.csv", index=False)


cost_p = in_p/1_000_000*PREIS_PRO_MTOK_INPUT + out_p/1_000_000*PREIS_PRO_MTOK_OUTPUT
cost_s = in_s/1_000_000*PREIS_PRO_MTOK_INPUT + out_s/1_000_000*PREIS_PRO_MTOK_OUTPUT
cost_d = in_d/1_000_000*PREIS_PRO_MTOK_INPUT + out_d/1_000_000*PREIS_PRO_MTOK_OUTPUT

metrics = {
    "experiment": "TF3_Formatierung", "method": "Claude_LLM", "model": MODELL_NAME,
    "price_exact_match": exact_p, "price_valid_rate": valid_p,
    "saving_exact_match": exact_s, "saving_valid_rate": valid_s,
    "date_exact_match": exact_d, "date_valid_rate": valid_d,
    "estimated_cost_usd": cost_p + cost_s + cost_d,
}
with open("results/tf3_llm_metrics.json", "w") as f:
    json.dump(metrics, f, indent=2)

log_rows = pd.DataFrame([
    {"experiment": "TF3_Formatierung_price", "method": "Claude_LLM", "n_items": len(price_test),
     "wall_time_sec": time_p, "input_tokens": in_p, "output_tokens": out_p, "estimated_cost_usd": cost_p, "model_name": MODELL_NAME},
    {"experiment": "TF3_Formatierung_saving", "method": "Claude_LLM", "n_items": len(saving_test),
     "wall_time_sec": time_s, "input_tokens": in_s, "output_tokens": out_s, "estimated_cost_usd": cost_s, "model_name": MODELL_NAME},
    {"experiment": "TF3_Formatierung_date", "method": "Claude_LLM", "n_items": len(date_test),
     "wall_time_sec": time_d, "input_tokens": in_d, "output_tokens": out_d, "estimated_cost_usd": cost_d, "model_name": MODELL_NAME},
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

print(f"Geschaetzte Gesamtkosten TF3 (LLM): ${cost_p + cost_s + cost_d:.4f}")
print("Gespeichert: results/tf3_llm_{price,saving,date}_predictions.csv, results/tf3_llm_metrics.json")
print("NOTEBOOK_EXECUTED_OK")
