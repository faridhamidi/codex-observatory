# model_audit_mini_app

A local web dashboard for auditing your Codex model usage — see which models were used, how many tokens were consumed, and whether the numbers add up. Runs entirely on your machine, no accounts or internet required.

## Requirements

- Python 3.10+
- [Codex CLI](https://github.com/openai/codex) (session logs must exist at `~/.codex/sessions`)

## Usage

```bash
./run_model_audit_dashboard.sh
```

Then open http://127.0.0.1:8765 in your browser.

That's it. The dashboard loads your Codex session history automatically.

### Options

```
./run_model_audit_dashboard.sh \
  -p 8765                        # port (default: 8765)
  -h 127.0.0.1                   # host (default: 127.0.0.1)
  -s ~/.codex/sessions           # Codex sessions directory
  -r .                           # reports output directory
  --max-retention-days 31        # drop events older than N days (default: 31)
  --max-csv-rows 0               # hard row cap, 0 = disabled (default: 0)
```

### Refreshing Data

Hit the **Refresh** button in the dashboard to re-parse your latest session logs without restarting the server.

---

## What It Shows

| Section | What you get |
|---|---|
| Routing | Which model and sandbox policy was used per session turn |
| Queries | Your messages and the queries sent to the model |
| Token Usage | Input/output/cached token counts per query |
| Tool Calls | Which tools were invoked and how long they took |

### Audit Checks

The dashboard runs 5 automatic checks on your token data and flags any anomalies:

| Check | What it verifies |
|---|---|
| A1 | `total_tokens` == `input_tokens + output_tokens` |
| A2 | `cached_input_tokens` <= `input_tokens` |
| A3 | `reasoning_output_tokens` <= `output_tokens` |
| A4 | All token values are non-negative |
| A5 | Average tokens per query is internally consistent |

---

## For Developers

### Project Structure

```
model_audit_mini_app/
├── model_audit_local_app.py     # HTTP server, JSONL parser, CSV writer
├── metrics_audit.py             # Audit checks and reconciliation logic
├── model_audit_dashboard.html   # Frontend dashboard (served at /)
├── run_model_audit_dashboard.sh # Shell launcher
├── test_model_audit_local_app.py
└── data/                        # Generated at runtime, not committed
    ├── catalog.json
    ├── routing_context.csv
    ├── query_messages.csv
    ├── usage_tokens.csv
    └── tool_calls.csv
```

### Running Tests

```bash
# Using the local venv
.venv/bin/pytest

# Or directly
python3 -m pytest
```

### Extending

The app is a single-file server with no framework dependencies — keep it that way.

To add a new event type:
1. Add a new `*_CSV_HEADERS` list
2. Add an entry to `DATASET_SPECS`
3. Add parsing logic in `extract_event_rows()`
4. Add a corresponding test fixture in `test_model_audit_local_app.py`
