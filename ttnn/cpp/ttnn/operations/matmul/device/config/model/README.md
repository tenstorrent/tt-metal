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

The model is the blob that `ttnn.CONFIG.matmul_model_path` names. Unless that path is already set, `import ttnn`
sets it, once, to the blob a source build fetched (`runtime/matmul_model/matmul_model.blob` under the runtime root, see
"The model blob" below), or, when there is none (a wheel, or a build configured with `-DTT_MATMUL_MODEL_FETCH=OFF`), to
`<ttnn.CONFIG.cache_path>/matmul_model/<sha256>.blob`; changing `cache_path` later does not move it. ttnn downloads the
blob to that cache path when the flag is on at `import ttnn` or when a device is opened (`ttnn.open_device`,
`ttnn.open_mesh_device`) and the file isn't there yet. It tries once per process: if the download fails, a warning is
logged and matmul uses the default configs. Turned on after the device is open, or to download ahead of time (say,
into a container image), run `python -m ttnn.matmul_model` or call `ttnn.matmul_model.download()`. Nothing is
downloaded automatically while the flag is off, or to a `matmul_model_path` set to another file. The code is
`ttnn/ttnn/matmul_model/`. A C++ program that does not import ttnn's Python package must set `matmul_model_path`
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
- A, B and the output are interleaved (DRAM or L1), with 32x32 tiles in TILE layout; A's and B's tiles are not stored
  transposed.
- The dtypes are bf16, bfp8_b, bfp4_b or fp32.
- There is no global CB, `sub_device_id`, prefetcher pipe, `untilize_out` or `in0_column_offset`, and no sub-device
  manager is loaded: the model's configs use the whole grid.
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
2. **Scores.** `matmul_model_features.{hpp,cpp}` computes an analytic base latency and the model's features per
   candidate.
   `gbdt_model.{hpp,cpp}` evaluates the LightGBM residual. The prediction is `base_ns * exp(gbdt(features))`.
3. **Precision.** The model changes only the program config: output dtype, math fidelity and accumulation stay what
   the caller asked for. Its K blocking (`in0_block_w`) can differ from the default's, and the K blocking changes how
   partial sums are rounded: between K blocks they leave DST and are summed in L1 (packer L1 accumulation) or kept in
   the output format, in 16 bits or fewer without fp32 accumulation. So a result with the flag on can differ slightly
   from the default's, in either direction; with fp32 accumulation the difference is smallest. One case is bounded:
   with a bfp8_b
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
work (model load, candidate enumeration, validation). Have the blob in place before warming up too: a model that loads
mid-run (its file appeared at `matmul_model_path`) changes the picks from then on.

Validation is a backstop, not a filter. The enumerator emits only configs that pass the device op's checks (the
`MatmulModelCandidates.AllPassTheDeviceOpRules` and `MatmulModelMockDevice.*` gtests), so a rejection is a selector bug:
it is logged as a warning and the default runs for that call signature. Validation fails through `TT_FATAL`, which
writes a critical log line and a backtrace, and calls `abort()` when `TT_ASSERT_ABORT` is set, so a pick runs it once
and never moves on to the next candidate.

## The model blob

`matmul_model.blob` is not in the repository. It is a one-layer OCI artifact, `ghcr.io/tenstorrent/tt-metal/matmul-model`,
pinned by its sha256 in `matmul_model.json` (an OCI layer's digest is the sha256 of its content, so the pin also
verifies the download). CMake fetches it at configure time into `runtime/matmul_model/`, like the SFPI toolchain, and
installs it at the same path under the runtime root. A failed fetch fails configure; `-DTT_MATMUL_MODEL_FETCH=OFF`
builds without the blob. A file at the target path with a different sha256 is deleted before fetching. The wheel leaves
the blob out, to stay within PyPI's file size limit, and carries `matmul_model.json`: `ttnn/ttnn/matmul_model/`
downloads the same artifact into ttnn's cache ("Turning it on" above). To use a local blob instead, set
`matmul_model_path` (or `TT_MATMUL_MODEL_BLOB` for the tests).

Replace it only with a retrained model. The blob is a LightGBM model exported to the format `gbdt_model.cpp` reads. It
carries the GBDT trees, the feature names (checked against `kFeatureNames` at load, so a blob trained on other features
is refused) and the sha256 of the LightGBM model it was exported from, which is logged at load. It is trained on
device-time measurements of matmul program configs on Blackhole. The analytic part (`matmul_model_features.cpp`,
`matmul_model_device_params.cpp`) is code, and its constants must match the model the blob was trained with.

To publish a new blob, push it as a new tag, update the sha256 in `matmul_model.json` (the build and ttnn's Python
package both read it), and update the predictions the `MatmulModelSelector.GoldenPredictions` gtest pins:

```
oras push ghcr.io/tenstorrent/tt-metal/matmul-model:<tag> --artifact-type application/vnd.tenstorrent.matmul-model.v1 \
    matmul_model.blob:application/octet-stream
```

## Tests

The tests that need the model blob are skipped when there is none (a build configured with
`-DTT_MATMUL_MODEL_FETCH=OFF`).

- **`unit_tests_ttnn`, `GbdtModel.*` and `MatmulModel*.*`.** These need no device and score candidates with the
  fetched model (or `TT_MATMUL_MODEL_BLOB`). The `Golden*` tests pin the enumerated candidates, the features and the
  prediction for a few matmuls, so a change to the enumerator, the equation or the device constants fails them (and
  then needs a retrained model). Set `TT_GBDT_PARITY_DIR` to also check the evaluator against LightGBM.
- **`unit_tests_ttnn_mock_allocator`, `MatmulModelMockDevice.*`.** On a mock Blackhole P150, no hardware: enumerated
  candidates for a set of matmuls pass the device op's own validation, and the L1 buffers their programs allocate fit
  the selector's estimate (`circular_buffer_bytes`), so a factory change the selector doesn't follow fails here.
- **`tests/ttnn/unit_tests/operations/matmul/test_matmul_model_config.py`.** This needs a Blackhole device; it uses
  the fetched model (or `TT_MATMUL_MODEL_BLOB`).
- **`tests/ttnn/unit_tests/operations/matmul/test_matmul_model_download.py`.** The download into ttnn's cache,
  against a fake registry: no device, no network.
