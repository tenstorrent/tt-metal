# Verification Report: rms_norm

Device: Blackhole, 120-Tensix-core compute grid, AICLK 1350 MHz (from profiler FW cycles / ns).
Golden results dir: `/tmp/rms_norm_verify_r1/` (full `verifier_report.json` is 25 MB, so it is not committed;
its summary and loud buckets are in `verifier_report_summary.json` next to this file).

## Code Review

### Fixed

1. **γ row-0 expansion mechanism (reader kernel, perf).** `rms_norm_reader.cpp` rebuilt the γ tiles every
   (block, chunk) with RISC-V word-by-word `volatile` L1 copies (16 loads + 16 stores per tile for bf16).
   Measured on the reader (NCRISC): 47.9 µs of a 52.6 µs decode 32×7168. Replaced with the mechanism
   `op_design.md` names: local L1→L1 NoC reads (2 per tile, 16 B-aligned 32/64-byte segments, one barrier
   per chunk). **1.15–1.28× on every γ-bearing decode / 640-row shape; no-γ and prefill shapes flat (±1 %).**
   The remaining γ serialization on the reader is a design-anticipated lamp (L4). It needs a γ-path
   rewrite, so it is carried by Refinement 1 (see Design Conformance).
2. **DRY: intermediate tile bytes had two sources.** `footprint()` (the L1 solve) priced
   `cb_sumsq_partial` / `cb_inv_rms` / `cb_x_gamma` at a hard-coded Float32 tile (`T32`). The CB
   declarations derived their format from `fp32_dest_acc_en`. Both now read `Geometry.T_interm` /
   `Geometry.interm_format`, computed once in `intermediate_format()`. The `fp32_dest_acc_en=False` refinement
   would otherwise have silently over-reserved L1 (a smaller `block_rows` than necessary).
3. **DRY: reduce-scaler format.** `bfloat16` was restated in the CB declaration and in `_fixed_bytes()`; it
   is now one constant, `SCALER_FORMAT`.
4. **Ledger currency** (`l1_ledger.md`): `T32` → `T_interm` in the capacity expressions, symbol table and
   footprint total. Added the missing sharing justification for the disjoint-lifetime pair
   `cb_sumsq_partial` ↔ `cb_x_gamma` / `cb_y_tiles`. Added verifier notes (measurements below).

### Checked, no change needed

- **Kernel hygiene**: `void kernel_main()`; `api/dataflow/dataflow_api.h` includes; `TensorAccessor`
  throughout (no `InterleavedAddrGen`).
- **CB push/pop balance**:
  - `cb_x_tiles` pushes `Wp` per tile-row and is popped `R_b·Wp` per block;
  - `cb_inv_rms` gets `R_b` push/pop;
  - `cb_gamma_tiles` / `cb_output` / `cb_x_rm_sticks` move C-tile quanta;
  - `cb_reduce_scaler` is popped once at kernel end.
- **Helper usage**: every compute phase is a kernel-lib helper (`tilize`, `eltwise_chain` ×3, `reduce` +
  post-op, `untilize`). RM full chunks use `read_sticks_for_tilize` / `write_sticks_after_untilize`. The raw
  dataflow paths (TILE page reads/writes, RM tail chunk, W-pad zeroing) are the ones the design shows no
  helper can express. No inter-core communication, so no multicast helper applies.
- **Broadcasts**:
  - `scale_by_gamma` uses `BroadcastDim::Row` against row-0-only γ tiles (no full-tile fill);
  - `scale_by_inv_rms` uses `BroadcastDim::Col` against the column-0-valid REDUCE_ROW output;
  - the scaler is one constant tile.
- **Padding**:
  - W-pad columns of the last TILE tile are zeroed in L1 by overwrite, not multiply, so NaN/Inf-safe;
  - pad tiles `[Wt, Wp)` are zeroed once at fixed ring positions;
  - RM tail-chunk pad bytes are zeroed per chunk;
  - the divisor is the logical `1/W` (fp32 bit pattern).
  - The poison-padding loose cases are xfail at Phase 0 (they pin TILE γ). The acceptance test covers
    poisoned TILE inputs, and they pass.
- **`math_approx_mode` honoured**: `rsqrt_tile` uses `APPROX`. The caller's config is passed verbatim as the
  compute kernel `config=`.

### Recommendations (not refinements)

- CB slot indices are duplicated as literals in the three kernels and in the descriptor (`CB_X_TILES = 1`
  …). These are not block knobs and are consistent today. Passing them as CT args would make a slot
  renumbering a one-place change.
- `op_design.md` still states `W_CHUNK_TILES = 8`. The implementation uses a cap of 32, falling back to 16 / 8 /
  … under the L1 solve, a measured implementer change recorded in `l1_ledger.md`. The ledger is current;
  the design text is stale on this one knob value.
- RM tail-chunk reader: the pad-head element stores are issued before the stick reads' barrier. They
  target bytes disjoint from the reads, so this is safe. Keep it that way if the tail path is touched.

## Prompt Conformance (`eval/prompts/rms_norm.txt`)

The prompt has no `## Rules` section; its requirements were checked directly:
- signature + keyword-only args (plus an extra `memory_config` kwarg the golden runner passes for
  sharded cells);
- exported `default_compute_kernel_config()` used by both the entry point and `axes.py`;
- no host-side `to_layout` / `tilize` / `pad` / `slice`;
- output `Layout` == input `Layout`;
- native non-aligned H/W in both `Layout`s;
- `float32 + fp32_dest_acc_en=False` refused (EXCLUSIONS);
- `ValueError` for rank < 2 and for a γ last-dim mismatch;
- exactly one `ttnn.generic_op` dispatch per call.

**Accepted deviation**: the prompt says `tag_gamma_dtype` / `tag_gamma_layout` MUST be INPUT_TAGGERS.
The implementer classifies them in `_classify()` with the `"none"` sentinel instead. Registering them would
remove those axes from the golden cartesian (`eval/feature_matrix.py`), so every real γ cell would collapse
to `"none"`. The golden counts confirm real γ dtype/`Layout` cells are generated and pass. `"none"` is in
SUPPORTED for both axes and never refused.

## Registry Conformance

- `INPUT_TAGGERS` = `{alignment, rank}`, both `(inputs, axes)` signatures. `SUPPORTED` covers every gated
  axis: dtype, fp32_dest_acc_en, layout, alignment, rank, gamma_mode, gamma_dtype, gamma_layout,
  memory_layout. `EXCLUSIONS = [{dtype: float32, fp32_dest_acc_en: False}]`.
- `validate()` order:
  1. hard `ValueError`s;
  2. output `memory_layout` gate;
  3. SUPPORTED per axis (`UnsupportedAxisValue`);
  4. EXCLUSIONS (`ExcludedCell`).

  `rms_norm()` calls `validate()` first. The op file does **not** declare `INVALID`.
- No SUPPORTED auto-fixes were needed (0 `xpass_drift`).

### INVALID audit (`eval/golden_tests/rms_norm/feature_spec.py`, not edited — report to the user)

- Canonical `{dtype: bfloat8_b, layout: ROW_MAJOR}` is present, plus the γ analogue. The no-γ
  canonicalization via the `"none"` sentinel is complete (7 entries, both directions). These pass the three
  sanity rules.
- **Flag — not structural, and cross-tensor:** `{layout: ROW_MAJOR, memory_layout: *_SHARDED,
  gamma_layout: TILE}` (3 entries). These couple the **activation** `layout` / `memory_layout` with the
  **γ** `gamma_layout`. The entries call themselves "author-scoped exclusions … COULD eventually be
  supported". That violates both the single-tensor rule and the universe-must-change rule. Recommend
  removing them, letting the op refuse via `EXCLUSIONS` if the sharded refinements cannot serve them.
- **Flag — not structural:** `{dtype: bfloat8_b, alignment: w_non_aligned | h_non_aligned}` (2 entries).
  These are single-tensor but "out of scope for now", not impossible. Recommend moving them to op-side
  `EXCLUSIONS` when Refinement 2 adds `bfloat8_b`. Note that the translated tests already run 2 such
  cells (`invalid_unexpected = 2`, both correctly refused today because bf8b ∉ SUPPORTED). Refinement 2 is
  told to add the EXCLUSION so they stay refused.

## Design Conformance

- **Algorithm / pipeline / work split: conforms to `op_design.md` regime R1.**
  - Σx² is one DEST-accumulated FPU multiply per tile-row;
  - `×1/W`, `+ε` and `rsqrt` are the reduce post-op;
  - γ is applied first, then `inv_rms`;
  - tile-rows are split over the full grid with `row_wise=True` (120 Tensix cores busy on many-row shapes);
  - x crosses DRAM once.
- **Blocking-model fidelity: knobs are parameters with one source.**
  - `W_CHUNK_TILES`, `STREAM_DEPTH`, `MIN_BLOCKS_PER_CORE` and `READ_BATCH_CHUNKS` are host constants.
  - `block_rows`, `w_chunk_tiles`, `num_w_chunks` and `x_depth` are CT args in a shared prefix.
  - Every CB capacity and loop trip derives from them.
  - The only Wt-proportional CB (`cb_x_tiles`) is predicate-guarded by the regime-R1 residency predicate,
    with an explicit refusal. That is the sanctioned resident fast-path, not a collapsed knob.
- **Expression** (block-scoped schedule):
  - compute runs each helper once per block (tilize, square_accumulate, finalize), or per (block, chunk)
    for the scalings, at `R_b` rows;
  - the TILE reader reserves/reads/barriers/pushes a whole tile-row (Wp tiles) per handshake, and the block
    factor sizes the live set (`x_depth·R·Wp`);
  - the writer drains C-tile quanta with a flush (not a full barrier) each.
  - No per-tile handshake on a block axis.
- **Finding (performance conformance) → Refinement 1.** The design promises `x_depth = 2` buys "reader
  fetches block k+1 while compute runs the scaling pass of block k". With γ present, the reader runs the γ
  expansion after the x load of block k and blocks on `cb_gamma_tiles` until compute drains those chunks.
  The prefetch of block k+1 therefore overlaps only the last ≤ 2 chunks. The cheap part (copy mechanism)
  is fixed above. Moving γ supply off the reader (lamp L4) is a γ-path rewrite, so it is bundled into
  Refinement 1, which rewrites that path for TILE γ anyway.
- **Occupancy (known, design-acknowledged).** One-tile-row decode shapes (32×W) run on 1 Tensix core, and
  640-row shapes on 20. The design defers the W-split (regime R3) with a positive reason. The occupancy
  side is now queued in Refinement 4 together with the residency failure it also fixes.

## L1 Ledger Audit

- **Currency**: every declared CB has a row, and every row is a live CB. Size expressions match the code:
  - `cb_x_tiles = x_depth·R·Wp`;
  - `cb_sumsq_partial = cb_inv_rms = R`;
  - `cb_x_gamma = cb_y_tiles = R·C`;
  - `cb_output = cb_x_rm_sticks = cb_gamma_tiles = STREAM_DEPTH·C`;
  - `cb_gamma_row = Wt` pages of `32·es_g` bytes;
  - `cb_reduce_scaler = 1`.

  Fixed: the `T32` → `T_interm` currency drift (the code now has one source).
- **Capacity vs live set**: no over-sized buffer without a reason. The double-buffered streaming CBs state
  their double-buffering reason. `cb_x_gamma` / `cb_y_tiles` hold `R_b·C` because two sequential helpers on
  the same TRISCs hand off a whole sub-block (in-place rejected with a stated deadlock reason). Under: the
  axis accounting is consistent: `cb_x_tiles` spans `tile_rows` (→R) and `width_reduce` (→Wp), and its
  capacity scales with both. The streaming CBs span only `width_stream` (→C).
- **Page format vs DEST width**:
  - Float32 intermediates under `fp32_dest_acc_en=True`: correct.
  - `cb_y_tiles` carries the output dtype with a reasoned justification (final rounding point; untilize is
    lossless).
  - `cb_reduce_scaler` is a dataflow-only bf16 payload (1.0, exact).
  - The 16-bit DEST path already derives 16-bit pages (`intermediate_format()`).
- **Disjoint lifetimes**:
  - `cb_sumsq_partial` ↔ `cb_inv_rms` and `cb_x_gamma` ↔ `cb_y_tiles` were justified (concurrent in one
    helper).
  - `cb_sumsq_partial` ↔ `cb_x_gamma` / `cb_y_tiles` was **unjustified** → fixed in the ledger: disjoint,
    not aliased; saving ≤ `R·T_interm` (< 1 % of the footprint) vs `alias_with` lockstep across three
    helpers.
- **Bounds / closed form**: every symbol is bounded with its predicate. The total is closed-form in
  `R, x_depth, C, Wp, Wt` and the dtype tile sizes.
- **Data-movement budget**: present and consistent with the code. x and out each cross DRAM once; γ is
  read `active_cores ×` one row (≤ 1/32 of x bytes); there is no cross-core traffic. The
  cheapest-traffic split (regime R1) is the one implemented. Regime R3 is a `deferred` row with a
  positive reason (no Phase-0-SUPPORTED cell exceeds residency).
- **Block-size defaults**: full grid first, then the coarsest block that fits, **departed from** by
  `MIN_BLOCKS_PER_CORE = 2` (≥ 2 blocks per core so `x_depth = 2` actually prefetches). The implementer's
  justification was confounded with a chunk-width change, so I re-measured it in isolation (TILE, bf16,
  HiFi2, fp32 DEST). Against `MIN_BLOCKS_PER_CORE = 1`: **1.02–1.05×** on 7168×1024, 7168×2048 and
  8192×1024 with γ; ±1 % elsewhere. The departure stands on measurement.
- **Per-core footprint**:

  ```
  F(R, x_depth) = x_depth·R·Wp·T_in + R·(2·T_interm + has_γ·C·T_interm + is_rm·C·T_out)
                  + 2048 + STREAM_DEPTH·C·(T_out + is_rm·T_in + has_γ·T_g) + has_γ·Wt·32·es_g
  ```

  Only `cb_x_tiles` (and `cb_gamma_row`, at 1/32 the rate) scales with W. The worst Phase 0 cell, fp32 RM
  W=8192 with fp32 γ, is 1,353,728 bytes at C = 8 tiles and `x_depth = 1`; I re-derived this by hand.

## Precision Baseline

`tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py`: γ present (same dtype
as input), HiFi4, `fp32_dest_acc_en=True`. ULP is in the output dtype. The ratio is `actual / expected`
over elements with `|expected| > 5 %` of max.

| Shape | dtype | Layout | PCC | Max Abs Err | Mean Abs Err | Rel RMS Err | ULP max / mean | ratio median [p5, p95] |
|-------|-------|--------|-----|-------------|--------------|-------------|----------------|------------------------|
| (1,1,32,64) | bf16 | TILE | 0.999999 | 0.0148 | 0.00112 | 0.00196 | 1.0 / 0.15 | 0.99915 [0.9964, 1.0018] |
| (1,1,32,64) | bf16 | RM | 0.999999 | 0.0148 | 0.00112 | 0.00196 | 1.0 / 0.15 | 0.99915 [0.9964, 1.0018] |
| (1,1,32,64) | fp32 | TILE | 1.000000 | 0.0170 | 0.00099 | 0.00160 | 5.2e4 / 1.7e4 | 0.99861 [0.9974, 0.9997] |
| (1,1,32,64) | fp32 | RM | 1.000000 | 0.0184 | 0.00123 | 0.00194 | 5.7e4 / 2.1e4 | 0.99827 [0.9970, 0.9994] |
| (2,4,128,512) | bf16 | TILE | 0.999998 | 0.0422 | 0.00103 | 0.00192 | 1.0 / 0.15 | 0.99921 [0.9964, 1.0019] |
| (2,4,128,512) | bf16 | RM | 0.999998 | 0.0422 | 0.00103 | 0.00192 | 1.0 / 0.15 | 0.99921 [0.9964, 1.0019] |
| (2,4,128,512) | fp32 | TILE | 1.000000 | 0.0286 | 0.00093 | 0.00160 | 6.3e4 / 1.7e4 | 0.99862 [0.9974, 0.9997] |
| (2,4,128,512) | fp32 | RM | 1.000000 | 0.0323 | 0.00115 | 0.00193 | 7.0e4 / 2.1e4 | 0.99828 [0.9970, 0.9994] |
| (1,1,128,4096) | bf16 | TILE | 0.999998 | 0.0478 | 0.00099 | 0.00191 | 1.0 / 0.14 | 0.99924 [0.9965, 1.0019] |
| (1,1,128,4096) | bf16 | RM | 0.999998 | 0.0478 | 0.00099 | 0.00191 | 1.0 / 0.14 | 0.99924 [0.9965, 1.0019] |
| (1,1,128,4096) | fp32 | TILE | 1.000000 | 0.0301 | 0.00092 | 0.00162 | 6.1e4 / 1.7e4 | 0.99858 [0.9974, 0.9997] |
| (1,1,128,4096) | fp32 | RM | 1.000000 | 0.0341 | 0.00112 | 0.00193 | 6.8e4 / 2.1e4 | 0.99827 [0.9970, 0.9994] |
| (4,8,47,100) | bf16 | TILE | 0.999998 | 0.0252 | 0.00110 | 0.00194 | 1.0 / 0.15 | 0.99917 [0.9963, 1.0019] |
| (4,8,47,100) | bf16 | RM | 0.999998 | 0.0252 | 0.00110 | 0.00194 | 1.0 / 0.15 | 0.99917 [0.9963, 1.0019] |
| (4,8,47,100) | fp32 | TILE | 1.000000 | 0.0174 | 0.00099 | 0.00156 | 6.4e4 / 1.7e4 | 0.99860 [0.9974, 0.9997] |
| (4,8,47,100) | fp32 | RM | 1.000000 | 0.0204 | 0.00122 | 0.00189 | 7.0e4 / 2.1e4 | 0.99826 [0.9970, 0.9994] |

**Assessment**: bf16 is at the format floor: max 1 ULP and mean 0.15 ULP in every case, and both `Layout`s
are bit-identical. fp32 is well inside tolerance (rel RMS ≈ 0.0016–0.0019 vs the 0.02 gate) but is
**tf32-limited**, not fp32-accurate: ~2·10⁴ fp32 ULP mean. It also has a small systematic **negative bias**
(ratio median 0.9986 TILE / 0.9983 RM).

**Scale-vs-precision triage**: rel RMS is ~0.002, far below the "≳ 0.1" structural-scale signature. A
host-side simulation that truncates each FPU operand to tf32 (10-bit mantissa, round-toward-zero at unpack)
reproduces the bias (median 0.9990, p5–p95 0.9982–0.9997). The bias is three truncations: x·x, x·γ, and
(x·γ)·inv_rms. RM is a further ~0.03 % lower because the fp32 `tilize` runs in `Fast` (tf32) mode. This
is precision, not a bug. The only lever is moving fp32 multiplies to the SFPU, which the catalog measures
as 0.58× (`compute_fusion`). No refinement is filed for it.

**Recommended tolerances** (Phase 0 corner): bf16 PCC ≥ 0.9999, rel RMS ≤ 0.005, ≤ 2 ULP; fp32 PCC ≥
0.99999, rel RMS ≤ 0.005, atol 0.05. The golden gates (0.995/0.04 bf16, 0.999/0.02 fp32) hold with wide
margin.

## Performance Baseline (proxy config, for headroom only)

`test_rms_norm_perf_baseline.py`, profiler DEVICE KERNEL DURATION. Config: bf16 `Layout::TILE` input,
bf16 `Layout::ROW_MAJOR` γ, HiFi2, `fp32_dest_acc_en=True`, `TensorMemoryLayout::INTERLEAVED`. The interleaved
`_perf_case` loose cases pin `fp32_dest_acc_en=False` and a TILE γ (outside Phase 0 SUPPORTED), so these
numbers are a headroom indicator, **not** the perf-refinement acceptance baseline. Same 1350 MHz clock
as the references, so no scaling is needed.

| rows × W | Tensix cores | with γ, before fix (ns) | with γ, after fix (ns) | no γ (ns) | `achievable_ns` (interleaved) |
|----------|------|------------|-----------|----------|------|
| 32 × 1024 | 1 | 11,803 | 9,193 | 6,827 | 9,149 |
| 32 × 2304 | 1 | 19,167 | 15,789 | 11,047 | 17,003 |
| 32 × 5120 | 1 | 35,412 | 30,763 | 20,703 | 75,825 |
| 32 × 7168 | 1 | 52,579 | 41,467 | 27,584 | 14,894 |
| 640 × 6144 | 20 | 59,670 | 54,457 | 47,929 | 101,968 |
| 640 × 7168 | 20 | 74,121 | 63,227 | 54,535 | 119,307 |
| 7168 × 1024 | 120 | 75,892 | 75,286 | 73,853 | 89,392 |
| 7168 × 2048 | 120 | 144,221 | 144,098 | 139,809 | 172,892 |
| 8192 × 1024 | 120 | 86,506 | 84,478 | 84,024 | 96,744 |
| 8192 × 2304 | 120 | 183,273 | 184,879 | 181,966 | 211,345 |
| 8192 × 5120 | 120 | 404,932 | 401,279 | 397,877 | 738,307 |
| 8192 × 7168 | 120 | 559,035 | 557,753 | 540,234 | 1,032,281 |

Reading:
- **Many-row prefill** is already under `achievable_ns` and DRAM-bound at ~390–420 GB/s of read+write
  traffic (≈ 80 % of Blackhole peak), so modest data-movement headroom remains → Refinement 3.
- **Decode** runs on one Tensix core and is compute-bound on TRISC1. 32×7168 is 2.8× over its reference
  because the reference W-splits → Refinement 4 (occupancy) + Refinement 6 (tuning).
- The γ overhead on decode is now mostly the reader serialization → Refinement 1.

## Verifier CLI Summary (after fixes; `/tmp/rms_norm_verify_r1/verifier_report.json`)

- supported_pass: 436
- xfail_expected: 6481
- invalid_skipped: 33900
- supported_fail: 0
- xpass_drift: 0
- xfail_wrong_mode: 0
- other buckets:
  - `invalid_unexpected` = 2 (translated bf8b non-aligned sharded tests, correctly refused; see INVALID audit);
  - `infeasible_skipped` = 2 (`99991×64` WIDTH_SHARDED: the input shard itself did not fit L1 — harness-infeasible, not op-charged);
  - `no_axes_found` = 15 (`test_regression.py` numerics tests, all passed).

Golden runner: PASSED=451 FAILED=0 HANGS=0. The acceptance test is 150/150 in `--dev` and production
modes.

**xfail_expected by missing axis value** (a cell can miss several):
- fp32_dest_acc_en=False: 3615
- gamma_layout=TILE: 2839
- memory_layout: BLOCK_SHARDED 1623, WIDTH_SHARDED 1622, HEIGHT_SHARDED 1593
- dtype=bfloat8_b: 962
- gamma_dtype=bfloat8_b: 860

Every one maps to a refinement: TILE γ → R1; 16-bit DEST, bf8b and bf8b γ → R2; WIDTH/BLOCK → R4;
HEIGHT → R5. The 481 `{float32, fp32_dest_acc_en=False}` cells are the native EXCLUSION and stay xfail
by design.

## Recommendations

- **Queue order** (`op_requirements.md`):
  - R1 (TILE γ) and R2 (16-bit DEST / bf8b) land the full config of the interleaved `_perf_case` loose
    cases, which perf R3 optimizes.
  - R4 builds the cross-core combine (WIDTH/BLOCK sharded + interleaved W-split; the hardest work).
  - R5 is the local HEIGHT placement.
  - R6 tunes decode + sharded perf.
- **Expected `supported_fail` window**: after R1, `_WIDE (1,1,32,32768)` becomes in-SUPPORTED and is
  regime-refused (row too wide for L1 residency) until R4. This is a resource boundary like OOM. Do not
  EXCLUDE it.
- **16-bit DEST precision (R2)**: Σx² over up to 224 tiles in a 16-bit DEST accumulator (lamp L8) is the
  likely pressure point for the perf cases' 0.9995 soft PCC gate.
- **feature_spec.py INVALID cleanup** (user action): the 5 author-scoped, non-structural entries flagged
  above.
