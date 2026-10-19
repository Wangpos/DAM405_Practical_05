# Practical 5 — Logging and Monitoring for an ML Model

## Files
| File | Task |
|---|---|
| `train_model.py` | Trains the model (`model.pkl`) and saves training feature means (`train_stats.json`) |
| `logger_config.py` | **Task 1** – logger writing to console + rotating `service.log` |
| `app.py` | **Task 2** – Flask `/predict` with request, prediction, latency and error logging; `/stats`, `/drift` |
| `log_levels_demo.py` | **Task 3** – all five log levels |
| `monitoring.py` | **Task 4** counters + summary, **Task 5** drift check |
| `client.py` | Sends normal, bad and drifted traffic to produce the expected results |

> If you have your own model from Practical 3, replace `model.pkl` / `train_model.py` with yours and update `FEATURE_NAMES` in `app.py`.

## How to run
```bash
pip install flask scikit-learn numpy joblib

python train_model.py                       # 1. train + save model

LOG_LEVEL=DEBUG python log_levels_demo.py   # 2. Task 3 (shows all 5 levels)
LOG_LEVEL=INFO  python log_levels_demo.py   #    DEBUG line disappears

python app.py                               # 3. start service (terminal 1)
python client.py                            # 4. send traffic (terminal 2)

cat service.log                             # 5. inspect the log file
```
On Windows PowerShell use `$env:LOG_LEVEL="DEBUG"; python log_levels_demo.py`.

Manual test with curl:
```bash
curl -X POST http://127.0.0.1:5000/predict -H "Content-Type: application/json" -d '{"features":[5.1,3.5,1.4,0.2]}'
curl http://127.0.0.1:5000/stats
curl http://127.0.0.1:5000/drift
```

## Screenshots to take for the report
1. `log_levels_demo.py` output at DEBUG vs INFO
2. `service.log` showing `prediction=… latency_ms=…` lines
3. ERROR lines from the bad requests
4. A `SUMMARY requests=… errors=… avg_latency_ms=…` line
5. The `possible drift on petal_length …` WARNING
6. The project folder showing `service.log` (and `service.log.1` etc. if it rotates)

## Reflection answers (starting points — write them in your own words)

**1. Why logging instead of print()?**
`print()` only writes plain text to stdout. The `logging` module adds timestamps and severity levels, can send the same message to several destinations (console, file, log aggregators) at once, supports rotation so files don't fill the disk, and lets you change verbosity through configuration without editing code. Levels also make logs filterable — you can search for every ERROR in a day.

**2. Which level in production?**
INFO (or WARNING for very high-traffic services). DEBUG logs the raw request features on every call, which produces huge volumes of logs, adds I/O overhead that increases latency, costs storage, and can leak sensitive or personal input data. DEBUG is switched on temporarily when investigating a specific problem.

**3. What does a mean-only drift check miss?**
- Changes in **spread/shape** — variance can double while the mean stays the same.
- **Multi-modal shifts** — two groups moving in opposite directions cancel out in the mean.
- **Relationships between features** (correlation changes) and **concept drift** (the input→label relationship changes while inputs look the same).
- It is sensitive to outliers and breaks when the training mean is near zero (relative change explodes).
- Categorical features and missing-value rates are not covered.

Improvements: use statistical tests on whole distributions (Kolmogorov–Smirnov test, Population Stability Index, Wasserstein distance), check variance and percentiles, monitor the prediction distribution and — once labels arrive — actual accuracy, and use a tool such as **Evidently** to produce a full data-drift report comparing reference and current data.
