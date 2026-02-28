# How to Retrain When Matmul Internals Change

This document describes how to update the auto-config infrastructure when
underlying matmul implementations change. This is an explicit bounty
deliverable for retrainability.

## When to Retrain

Retrain whenever:
- The C++ matmul kernel implementations change
- New MatmulProgramConfig types are added
- Tile sizes or L1 budgets change
- New hardware architectures are introduced
- The heuristic scorer weights need adjustment based on real data

## Step-by-Step Retraining Workflow

### Step 1: Invalidate Stale Cache

Bump `SELECTOR_VERSION` in `config_cache.py`. This automatically invalidates
all cached entries from previous versions:

```python
# config_cache.py
SELECTOR_VERSION = "1.1.0"  # was "1.0.0"
```

### Step 2: Run Benchmark Sweep

Collect fresh telemetry data across representative shapes:

```bash
python -m ttnn.operations.auto_config.benchmark --sweep --output sweep_results.json -v
```

This runs 21+ representative shapes (small, medium, large, tall, wide, LLM,
attention, non-power-of-2) and measures device latency for all valid
configuration candidates.

### Step 3: Review Results

Check the benchmark output to see which config families actually win
for each shape category. Use this to tune heuristic weights:

```bash
cat sweep_results.json | python -m json.tool
```

### Step 4: (Optional) Retrain DNN Scorer

If you want to use the DNN-based scorer instead of/alongside the heuristic:

```bash
python -m ttnn.operations.auto_config.benchmark --sweep --train-dnn
```

This trains the MLP scorer from collected telemetry and saves the weights
to `~/.ttnn/auto_config_cache/dnn_scorer_weights.json`.

### Step 5: Update Heuristic Weights

Based on benchmark results, adjust the weights in `scorer/heuristic.py`:

```python
class HeuristicScorer:
    WEIGHT_UTILIZATION = 0.35   # How important is core utilization?
    WEIGHT_BLOCK_EFFICIENCY = 0.25  # How important is block size?
    WEIGHT_LAYOUT_ALIGNMENT = 0.20  # How important is mcast direction?
    WEIGHT_SUBBLOCK = 0.10     # How important is subblock efficiency?
    WEIGHT_BACKEND = 0.10      # How important is backend preference?
```

### Step 6: Verify

Run the test suite to confirm correctness is maintained:

```bash
pytest tests/ttnn/unit_tests/operations/test_matmul_auto/test_correctness.py -v
pytest tests/ttnn/unit_tests/operations/test_matmul_auto/test_performance.py -v
```

## Adding New Config Families

When a new MatmulProgramConfig type is added to tt-metal:

1. Add a new generator function in `candidate_generator.py`:
   ```python
   def _generate_new_family_candidates(features):
       # Generate candidates for the new family
       ...
   ```

2. Register it in `generate_matmul_candidates()`:
   ```python
   candidates.extend(_generate_new_family_candidates(features))
   ```

3. Add validation rules in `constraint_validator.py` if needed.

4. Add scoring logic in `scorer/heuristic.py` if the family has
   unique performance characteristics.

5. Bump `SELECTOR_VERSION` to invalidate stale cache entries.

## Adding New Hardware Architectures

The cache key includes the silicon architecture (e.g., "wormhole_b0"),
so cache entries are automatically separated per architecture. When a
new architecture is added:

1. Verify that `feature_extraction.py` correctly detects the new
   architecture and grid size.

2. Run a full benchmark sweep on the new hardware.

3. Adjust heuristic weights or train a new DNN scorer if needed.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `TTNN_AUTO_CONFIG_CACHE_DIR` | `~/.ttnn/auto_config_cache` | Override cache directory (useful for CI) |
