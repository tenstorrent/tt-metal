# Matmul config model (opt-in)

An experimental, default-off way for `ttnn.matmul` / `ttnn.linear` / `ttnn.addmm` to choose their program config
when the caller passes none (#57884). A learned performance model scores a list of candidate configs and the
lowest predicted time is used. Anything the model doesn't cover keeps today's default selection unchanged.

## Turning it on

```
TTNN_CONFIG_OVERRIDES='{"matmul_model_config": true, "matmul_model_path": "/path/to/model.blob"}'
```

or from Python, before the first matmul:

```python
ttnn.CONFIG.matmul_model_path = "/path/to/model.blob"
ttnn.CONFIG.matmul_model_config = True
```

When the flag is off nothing here runs and nothing is loaded. When it is on, the blob is loaded once, lazily, on
the first matmul the model is eligible for. If it can't be loaded, a warning is logged once and matmul behaves as
with the flag off. `ttnn.matmul_model_last_pick()` returns, for this thread's last matmul,
`"model: <program config>"` or `"default: <reason>"` (for tests and benchmarks).

## When the model chooses

The model chooses only when all of the following hold. Otherwise today's default selection runs.

- The arch is Blackhole, which is the only device the model was trained on.
- No `program_config` and no `core_grid` were given.
- A, B and the output are interleaved (DRAM or L1), with 32x32 tiles in TILE layout.
- The dtypes are bf16, bfp8_b, bfp4_b or fp32.
- There is no global CB, sub-device, prefetcher pipe, `untilize_out` or `in0_column_offset`.
- A fused bias, if any, is a single row.
- `sparse_matmul` is never affected.

It decides once per matmul, in `bound_matmul`, on the inputs, bias and attributes the device op receives:

- **Transposes.** The legacy manual-transpose decision has already been taken.
- **Bias.** Fused or post-processed exactly as with the flag off.
- **Activation.** It runs as the separate op it is without a core grid.
- **Compute kernel config.** It is the one resolved with no program config. The math fidelity stays at the default
  path's HiFi2 rather than the LoFi an explicit program config would get.

`linear` is treated the same as `matmul`. On a mesh, the config is chosen from the per-device shapes, the device's
`compute_with_storage_grid_size` and its free L1, the same view the default selection uses.

## How it chooses

1. **Candidates.** `matmul_candidates.{hpp,cpp}` ports the `EnumeratingSource` of Ryan Miller's #57884 selector
   (tt-metal `4b385bbeace`) for interleaved tensors.
   - It covers 2D, 1D (in0 and in1 multicast) and Reuse layouts on the device grid, with several K depths, output
     blocks and subblocks per layout.
   - Each candidate fits the L1 left for circular buffers. That budget is `get_max_l1_space`, less an interleaved L1
     output and 16 KiB.
2. **Scores.** `matmul_model_features.{hpp,cpp}` computes an analytic base latency and 57 features per candidate.
   `gbdt_model.{hpp,cpp}` evaluates the LightGBM residual. The prediction is `base_ns * exp(gbdt(features))`.
3. **Pick.** Candidates are tried in order of prediction; ties keep candidate order. The first one that passes the
   device op's own validation (`validate_on_program_cache_miss`) and the live L1 budget is used. If none passes, or
   none can be scored, the default selection runs.
4. **Cache.** Picks are cached process-wide, keyed on the call's full signature and the model's sha256. On every
   hit the cached pick is re-checked against the current L1 budget.

A candidate that the device op rejects costs a caught exception and the critical-level log line that every
`TT_FATAL` writes. Candidates are checked best first, so this happens only when the best-ranked ones are invalid.

## The model blob

The blob is not checked in. It is produced from the model repository `tt-matmul-predictor`:

```
mmodel export ...                                                    # the repository's export entry point
python -m bench.export.export_gbdt <lightgbm_model.txt> model.blob --format blob   # what it runs today
```

The blob carries the GBDT trees, the feature names (checked against `kFeatureNames` at load) and the sha256 of the
LightGBM model it was exported from. The analytic part (`matmul_model_features.cpp`,
`matmul_model_device_params.cpp`) is code. Its constants must match the model the blob was trained with.

## Tests

- **`unit_tests_ttnn`, `GbdtModel.*` and `MatmulModel*.*`.** These need no device. Set `TT_GBDT_PARITY_DIR` and
  `TT_MATMUL_MODEL_BLOB` to also run the checks against the real model.
- **`tests/ttnn/unit_tests/operations/matmul/test_matmul_model_config.py`.** This needs a Blackhole device and
  `TT_MATMUL_MODEL_BLOB`.
