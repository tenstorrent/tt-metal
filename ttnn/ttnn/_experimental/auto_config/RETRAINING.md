# Retraining & Update Guide

## When to Retrain

Re-calibrate the auto-config scorer when:
- Matmul kernel implementations change (new dataflow, new tile sizes)
- New hardware is added (Blackhole, etc.)
- New program config types are added to TTNN

## Step 1: Collect Benchmark Data

```bash
cd tt-metal
PYTHONPATH=ttnn python -m ttnn._experimental.auto_config.benchmark matmul_benchmark.csv
```

This sweeps representative shapes and writes per-shape timing to CSV.

## Step 2a: Update Heuristic Weights

Edit `scorer/heuristic.py` — adjust the weight constants
(`_W_UTILIZATION`, `_W_BLOCK`, etc.) based on benchmark results.
The heuristic is the default scorer and requires no training.

## Step 2b: Train DNN Scorer (Optional)

```python
from ttnn._experimental.auto_config.scorer.dnn_scorer import DNNConfigGenerator
DNNConfigGenerator.train_from_csv("matmul_benchmark.csv", "dnn_config_model.pt", epochs=200)
```

Place the trained model at `ttnn/_experimental/auto_config/data/dnn_config_model.pt`.
The DNN scorer will be used automatically when available, falling back to
heuristic when no model file exists.

## Step 3: Validate

```bash
PYTHONPATH=ttnn pytest tests/ttnn/unit_tests/operations/test_matmul_auto/test_mutation.py -v
PYTHONPATH=ttnn pytest tests/ttnn/unit_tests/operations/test_matmul_auto/test_falcon7b_perf.py -v
```

Verify that `test_auto_beats_default` and `test_geomean_speedup` still pass.

## Architecture

```
matmul_auto(a, b)
    │
    ├── feature_extraction.extract_matmul_features()
    ├── candidate_generator.generate_matmul_candidates()
    ├── constraint_validator.validate_candidate()
    ├── scorer.heuristic.HeuristicScorer  ← calibrate weights here
    │   └── scorer.dnn_scorer.DNNConfigGenerator  ← optional, trained model
    ├── config_cache.ConfigCache  ← local JSON cache
    └── execute with selected config OR fallback to default ttnn.matmul
```

The modular design means you can swap the scorer without touching the
pipeline. The heuristic scorer works offline with no training data.
The DNN scorer requires benchmark data but can learn hardware-specific
patterns that the heuristic cannot.
