# Matmul config model (opt-in)

An experimental, default-off way for `ttnn.matmul` / `ttnn.linear` / `ttnn.addmm` to choose their program config
when the caller passes none (#57884). A learned performance model scores a list of candidate configs and the
lowest predicted time is used. Anything the model doesn't cover keeps the default selection unchanged.

## Turning it on

```
TTNN_CONFIG_OVERRIDES='{"matmul_model_config": true}'
```

or from Python, before the first matmul:

```python
ttnn.CONFIG.matmul_model_config = True
```

The model is the blob that `ttnn.CONFIG.matmul_model_path` names. The build fetches it (see "The model blob" below)
to `runtime/matmul_model/matmul_model.blob` under the runtime root, and `import ttnn` sets `matmul_model_path` to it
unless the path is already set. A C++ program that does not import ttnn's Python package must set `matmul_model_path`
itself: while it is unset, a matmul with the flag on logs that the model can't be loaded and uses the default configs.

When the flag is off nothing here runs and nothing is loaded. When it is on, the blob at `matmul_model_path` is
loaded on the first matmul the model is eligible for, and loaded again on the first eligible matmul after
`matmul_model_path` changes; picks are kept per model, so a new model makes its own. If the blob can't be loaded, a
warning is logged and matmul behaves as with the flag off until a model loads: the load is tried again when the path
changes or the file at it appears or changes. Like the rest of `ttnn.CONFIG`, `matmul_model_path` is not synchronized:
set it while no other thread runs a matmul.
`ttnn.operations.matmul.matmul_model_last_pick()` returns, for this thread's last matmul, `"model: <program config>"`
or `"default: <reason>"` (for tests and benchmarks).

## When the model chooses

The model chooses only when all of the following hold. Otherwise the default selection runs.

- The arch is Blackhole, which is the only device the model was trained on.
- No `program_config` and no `core_grid` were given.
- A, B and the output are interleaved (DRAM or L1), with 32x32 tiles in TILE layout.
- The dtypes are bf16, bfp8_b, bfp4_b or fp32.
- There is no global CB, sub-device, prefetcher pipe, `untilize_out` or `in0_column_offset`.
- A fused bias, if any, is a single row.
- If B is batched, A has the same batch count. A broadcast A over a batched B keeps the default: the model describes
  a single batch count.
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

1. **Candidates.** `matmul_candidates.{hpp,cpp}` ports the candidate enumeration of the default-config selector
   proposed in #58143, for interleaved tensors.
   - It covers 2D, 1D (in0 and in1 multicast) and Reuse layouts on the device grid and on the grid one column
     narrower, with several K depths, output blocks and subblocks per layout. The narrower grid helps when a
     dimension splits into awkward per-core tile counts on the full grid (#62076: N = 448 tiles over 11 columns is 41
     per core, a prime).
   - Each candidate fits the L1 left for circular buffers. That budget is `get_max_l1_space`, less an interleaved L1
     output and 16 KiB.
2. **Scores.** `matmul_model_features.{hpp,cpp}` computes an analytic base latency and 57 features per candidate.
   `gbdt_model.{hpp,cpp}` evaluates the LightGBM residual. The prediction is `base_ns * exp(gbdt(features))`.
3. **Precision.** The model changes only the program config: output dtype, math fidelity and accumulation stay what
   the caller asked for. Its K blocking (`in0_block_w`) can differ from the default's, and the K blocking changes how
   partial sums are rounded: between K blocks they leave DST and are summed in L1 (packer L1 accumulation) or kept in
   the output format, in 16 bits or fewer without fp32 accumulation. So a result with the flag on can differ slightly
   from the default's, in either direction. On 303 matmuls from current models (Blackhole Galaxy, random inputs, PCC
   against an fp32 reference) the change was within ±1e-5 on 149, better on 23, worse by up to 1e-4 on 101, by 1e-4
   to 1e-3 on 29 and by 1.3e-3 on one; with fp32 accumulation it was never worse. One case is bounded: with a bfp8_b
   or bfp4_b output and neither packer L1 nor fp32 accumulation, partial sums are re-quantized to the output format at
   every K block, so only candidates with no more K blocks than the default's config are kept. The default's K
   blocking depends on live L1, so a cached pick is checked against it again when that changes.
4. **Pick.** The default selection's own config is scored with the candidates when it is one of the modeled layouts
   (2D without transpose_mcast, 1D without gather_in0, Reuse). If the model predicts it fastest, or predicts no
   candidate at least 2% faster than it, the default runs; picks predicted within 2% of the default are within the
   model's error. When the default config can't be scored (MultiCore, or a layout the model doesn't cover), no margin
   applies. Otherwise the best-predicted candidate that fits the live L1 budget is used; ties keep candidate order. It
   is checked once with the device op's own validation (`validate_on_program_cache_miss`) before it is used. If no
   candidate fits, none can be scored, or validation rejects the pick, the default selection runs.
5. **Cache.** Picks are cached process-wide for the loaded model, keyed on the call's full signature. On every hit the cached pick is re-checked against the current L1 budget (and the precision bound), and
   a pick that found no candidate for lack of L1 is made again once more L1 is free.

The pick can depend on live L1, like the default selection's. With trace capture, warm up each matmul shape once, in
the same allocation state as the capture, before capturing: the first call of a shape also does the selection's host
work (model load, candidate enumeration, validation).

Validation is a backstop, not a filter. The enumerator emits only configs that pass the device op's checks (the
`MatmulModelCandidates.AllPassTheDeviceOpRules` gtest), so a rejection is a selector bug: it is logged as a warning and
the default runs for that call signature. Validation fails through `TT_FATAL`, which writes a critical log line and a
backtrace, and calls `abort()` when `TT_ASSERT_ABORT` is set, so a pick runs it once and never moves on to the next
candidate.

## The model blob

`matmul_model.blob` (12 MB: 3000 GBDT trees of 255 leaves) is not in the repository. It is a one-layer OCI artifact,
`ghcr.io/tenstorrent/tt-metal/matmul-model`, and CMake fetches it at configure time by its sha256 (an OCI layer's digest
is the sha256 of its content, so the download is pinned and verified) into `runtime/matmul_model/`, like the SFPI
toolchain; it is installed and packaged in the wheel at the same path under the runtime root. A failed fetch fails
configure; `-DTT_MATMUL_MODEL_FETCH=OFF` builds without the blob, and then turning the model on logs a warning and keeps
the default configs. A file at the target path with a different sha256 is deleted before fetching. To use a local blob
instead, set `matmul_model_path` (or `TT_MATMUL_MODEL_BLOB` for the tests).

Replace it only with a retrained model; the analytic constants below must match the model it was trained with. The blob
is a LightGBM model exported to the format `gbdt_model.cpp` reads. It carries the GBDT trees, the feature names (checked against `kFeatureNames` at load, so a blob trained on
other features is refused) and the sha256 of the LightGBM model it was exported from. The current blob: file sha256
`7dc37af22b6ae0d2ecb4f415a14cdc7e9b6e0208ee2ac2c2aacd049576ee9ae2`, model sha256
`46dc43100ba9d1bdd6d30e408d23c7012a5e535cefab4d32ddb04910f1ba73a5` (logged at load), trained on device-time
measurements of matmul program configs on Blackhole p150 (11x10 grid). The analytic part (`matmul_model_features.cpp`,
`matmul_model_device_params.cpp`) is code. Its constants must match the model the blob was trained with.

To publish a new blob, push it as a new tag and update the sha256 in `../../../CMakeLists.txt` (the matmul op's):

```
oras push ghcr.io/tenstorrent/tt-metal/matmul-model:<tag> --artifact-type application/vnd.tenstorrent.matmul-model.v1 \
    matmul_model.blob:application/octet-stream
```

## Tests

- **`unit_tests_ttnn`, `GbdtModel.*` and `MatmulModel*.*`.** These need no device and score candidates with the
  fetched model (or `TT_MATMUL_MODEL_BLOB`). Set `TT_GBDT_PARITY_DIR` to also check the evaluator against LightGBM.
- **`tests/ttnn/unit_tests/operations/matmul/test_matmul_model_config.py`.** This needs a Blackhole device; it uses
  the fetched model (or `TT_MATMUL_MODEL_BLOB`).
