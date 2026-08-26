# Ping-Pong-Predictor

# Ping Pong Advanced Research Predictor

A read-only pipeline that collects completed Ping Pong rounds, stores them in SQLite, trains a leakage-safe ensemble, exposes results through FastAPI, and presents them in a live Streamlit research dashboard.

> The project does not place bets, choose stakes, click BET/CASH OUT, control accounts, or guarantee results. Predictions count as evidence only when they beat a baseline on chronologically later held-out data.

## System overview

```text
Visible completed-round stream
          |
          v
Playwright collector  -->  data/pingpong.sqlite3
                                  |
                   +--------------+--------------+
                   |                             |
                   v                             v
            live_trainer.py                  FastAPI :8010
                   |                             |
                   v                             v
     models/pingpong_ensemble.joblib     Streamlit :8501
```

The model estimates threshold probabilities such as `P(next >= 2x)`. It does not produce a guaranteed exact multiplier.

## Quick start on the current VM

Use four terminals and keep each process running.

### Terminal 1 — real-data collector

```bash
cd /home/azureuser/Mubarak/TTS/pingpong_predictor
source .venv/bin/activate
python collector.py
```

Expected output:

```text
Observing completed-round WebSocket: wss://...
Captured completed round 3147000: 2.31x
```

### Terminal 2 — automatic trainer

```bash
cd /home/azureuser/Mubarak/TTS/pingpong_predictor
source .venv/bin/activate
python live_trainer.py
```

The trainer waits silently until enough data exists. It trains automatically and then retrains every 500 new rounds.

### Terminal 3 — API

```bash
cd /home/azureuser/Mubarak/TTS/pingpong_predictor
source .venv/bin/activate
python -m uvicorn api:app --host 127.0.0.1 --port 8010
```

Verify it:

```bash
curl http://127.0.0.1:8010/health
```

### Terminal 4 — dashboard

```bash
cd /home/azureuser/Mubarak/TTS/pingpong_predictor
source .venv/bin/activate
streamlit run dashboard/app.py --server.address 0.0.0.0 --server.port 8501
```

For a remote IDE, open its **Ports** panel, forward port `8501`, keep visibility private, and click **Open in Browser**. A remote VM's `127.0.0.1` is not your laptop's `127.0.0.1`.

## One-command service stack

Instead of Terminals 2–4:

```bash
cd /home/azureuser/Mubarak/TTS/pingpong_predictor
source .venv/bin/activate
python scripts/start_stack.py
```

This starts the trainer, API on port 8010, and dashboard on port 8501. Run `python collector.py` separately.

## First-time installation

Python 3.11+ is recommended.

```bash
cd /home/azureuser/Mubarak/TTS/pingpong_predictor
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
python -m playwright install chromium
```

Confirm the environment:

```bash
which python
python --version
which streamlit
```

## Real-data collection

`config.yaml` is configured for the completed-round WebSocket stream:

```yaml
collector:
  mode: websocket
  headless: true
  websocket_url_contains: /ws/ng/games/ping-pong/
  websocket_message_type: ROUND_COEFFICIENT
  json_items_path: data
  round_id_field: id
  multiplier_field: houseCoefficient
  timestamp_field: createdAt
```

Only completed `ROUND_COEFFICIENT` messages are stored. Ongoing multipliers are ignored, preventing target leakage. Real round IDs provide deduplication.

Production data:

```text
data/pingpong.sqlite3
```

Check progress:

```bash
python -c "from src.pingpong.config import load_config; from src.pingpong.storage import connect,count_rounds,latest_round; c=load_config('config.yaml'); d=connect(c['storage']['sqlite_path']); print('rounds:',count_rounds(d)); print('latest:',latest_round(d))"
```

If the visible upstream stream changes, use `discover_feed.py` to investigate. The project does not bypass authentication, anti-bot measures, or access controls.

## Training readiness

Production uses a 250-round maximum rolling window and requires 3,000 usable rows:

```text
250 warm-up rounds + 3,000 usable rows = about 3,250 total rounds
```

Before that, the dashboard correctly shows `COLLECTING` or `NOT TRAINED`. This is not an error.

Manual training after enough real data exists:

```bash
python train.py
```

Manual prediction after training:

```bash
python predict.py
```

Artifact:

```text
models/pingpong_ensemble.joblib
```

## Model and evidence gates

The ensemble combines:

1. logistic regression;
2. calibrated histogram gradient boosting;
3. calibrated Extra Trees.

Features include 80 log-multiplier lags, rolling windows from 5 to 250 rounds, distribution statistics, threshold frequencies, streaks, and rounds-since-exceedance values. Splits are strictly chronological.

A threshold reports signal only when all configured checks pass:

```text
test samples >= 300
relative held-out log-loss improvement >= 0.5%
held-out Brier improvement >= 0.001
held-out ROC-AUC >= 0.515
```

Statuses:

- `MODEL_HAS_HELD_OUT_SIGNAL`: at least one threshold passed every gate;
- `NO_PREDICTIVE_EDGE`: the model did not beat the baseline strongly enough.

Neither status guarantees a future result. `NO_PREDICTIVE_EDGE` means probabilities should be treated as descriptive estimates.

## Dashboard

The dashboard has five workspaces:

- **Overview** — collector state, latest result, training progress, and observed rates;
- **Next Prediction** — next-round threshold curve and coarse p50/p75/p90 summaries;
- **Live Rounds** — outcome timeline, histogram, and recent-round table;
- **Model Evidence** — held-out AUC, log loss, Brier improvement, and gate result;
- **Diagnostics** — autocorrelation, mutual information, runs test, Ljung–Box, and KS drift.

The prediction workspace unlocks automatically after training finishes.

## API routes

Base URL: `http://127.0.0.1:8010`

```text
GET /health
GET /history?limit=300
GET /prediction
GET /diagnostics
GET /model/metrics
GET /transformer/prediction
GET /transformer/metrics
WS  /ws
```

Examples:

```bash
curl http://127.0.0.1:8010/health
curl 'http://127.0.0.1:8010/history?limit=10'
curl http://127.0.0.1:8010/diagnostics
curl http://127.0.0.1:8010/prediction
```

`/prediction` returns HTTP 503 until a model exists. That is expected.

## Optional Temporal Transformer

The Transformer requires 20,000 completed rounds by default and uses 128 previous multipliers per sequence.

```bash
python -m pip install -r requirements-transformer.txt
python train_transformer.py
python predict_transformer.py
```

Artifact:

```text
models/pingpong_transformer.pt
```

The dashboard detects it automatically. A Transformer can overfit random data, so it uses the same future-only evidence gates.

## Import historical real data

CSV format:

```csv
round_id,timestamp,multiplier
3140001,2026-08-26T10:00:00Z,1.32
3140002,2026-08-26T10:00:10Z,4.28
```

```bash
python import_history.py history.csv
```

Only import genuine completed rounds in chronological order. Never import synthetic smoke rows into the production database.

## Manual diagnostics

```bash
python diagnostics.py
```

Diagnostics detect dependence and distribution changes. They do not independently prove next-round predictability.

## Optional isolated smoke test

Smoke testing uses separate database and model paths from `config.smoke.yaml`:

```bash
python synthetic_demo.py --config config.smoke.yaml --n 1200 --seed 42
python train.py --config config.smoke.yaml
python diagnostics.py --config config.smoke.yaml
python predict.py --config config.smoke.yaml
```

Independent synthetic data should usually produce `NO_PREDICTIVE_EDGE`.

## Troubleshooting

### Dashboard says API unavailable

Confirm:

```bash
rg -n 'api_base_url' config.yaml
```

Expected:

```yaml
api_base_url: http://127.0.0.1:8010
```

Start and verify the API:

```bash
python -m uvicorn api:app --host 127.0.0.1 --port 8010
curl http://127.0.0.1:8010/health
```

### Port 8501 is unavailable

A dashboard already owns it:

```bash
ss -ltnp | rg ':8501'
```

Use the existing forwarded port, or stop its terminal with `Ctrl+C` before restarting.

### Port 8010 is unavailable

```bash
ss -ltnp | rg ':8010'
curl http://127.0.0.1:8010/health
```

Do not start a duplicate API if the existing one is healthy.

### Dashboard works on the VM but not in Opera/Chrome

Forward port 8501 through the IDE, or create an SSH tunnel from your computer:

```bash
ssh -L 8501:127.0.0.1:8501 YOUR_USER@YOUR_VM_HOST
```

Then open `http://127.0.0.1:8501` on your computer.

### Dashboard says model not trained

The first production run needs about 3,250 total rounds. Do not lower the threshold simply to force early predictions; small samples can create convincing false positives.

### Collector captures nothing

```bash
python -m playwright install chromium
python collector.py
```

Look for `Observing completed-round WebSocket`. If absent, the upstream page or stream may have changed.

## Stop everything

Press `Ctrl+C` in each foreground terminal. When using `scripts/start_stack.py`, one `Ctrl+C` stops its trainer, API, and dashboard children; stop the collector separately.

## Recommended research scale

```text
3,250 rounds       first configured ensemble run
20,000 rounds      early research / Transformer minimum
50,000 rounds      stronger temporal validation
100,000+ rounds    regime and drift studies
```

Repeat evaluation across multiple future windows rather than trusting one split. The goal is to measure whether signal exists, not manufacture certainty from random outcomes.
