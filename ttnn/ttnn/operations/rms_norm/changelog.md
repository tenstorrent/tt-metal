# Changelog: rms_norm

## Phase 0 — Core Implementation

- **Date**: 2026-07-28
- **Device**: blackhole_p150b, 11 × 10 = 110-core compute grid, AICLK 1350 MHz

- **What was done**: Initial implementation via the incremental pipeline (planner → implementer → verifier).
  Generic-op (`ProgramDescriptor`) RMSNorm over the last dimension. The independent tile-row axis
  is split across the whole compute grid (`ttnn.split_work_to_cores(..., row_wise=True)`); each core
  owns whole rows and reduces the dependent `W` axis sequentially in-core over `NW` coarse chunks,
  holding `x` (and `gamma`) resident in L1 whenever a size predicate says they fit, with a bounded
  streaming fallback. TILE and ROW_MAJOR are both native (in-kernel `tilize`/`untilize`), and
  non-tile-aligned `H` and `W` are handled natively by a masked reduce plus a valid-stick writer —
  no host-side `to_layout` / `pad` / `slice` anywhere.

- **SUPPORTED at Phase 0**:
  - `dtype = [float32, bfloat16]`
  - `fp32_dest_acc_en = [True]`
  - `layout = [TILE, ROW_MAJOR]`
  - `alignment = [tile_aligned, w_non_aligned, h_non_aligned]`
  - `rank = [2, 3, 4]`
  - `gamma_mode = [gamma, no_gamma]`
  - `gamma_dtype = [float32, bfloat16, "none"]`
  - `gamma_layout = [TILE, ROW_MAJOR, "none"]`
  - `memory_layout = [INTERLEAVED]`
  - `EXCLUSIONS = [{dtype: float32, fp32_dest_acc_en: False}]` (permanent op-side refusal)

- **Accuracy achieved** (measured on 16 cells = 4 shapes × {bf16, fp32} × {TILE, ROW_MAJOR} via
  `test_rms_norm_precision_baseline.py`; gamma present, `epsilon=1e-6`, HiFi4 + `fp32_dest_acc_en=True`):
  - `bfloat16`: PCC ≥ 0.995 on every cell, max_abs_err ≤ 4.8e-02, mean_abs_err ≈ 1.2e-03,
    relative RMS err 2.34e-03 … 2.46e-03 (gate 4.0e-02 → ~17× headroom)
  - `float32`: PCC ≥ 0.999 on every cell, max_abs_err ≤ 2.2e-02, mean_abs_err 4.1e-04 … 1.0e-03,
    relative RMS err 8.28e-04 … 1.84e-03 (gate 2.0e-02 → ~13× headroom)
  - got/true ratio spread: median 0.9984 … 1.0003, p5/p95 within ±0.5 %. The mild fp32 low bias
    was traced (`probes/probe_001.py`) to FPU SrcA/SrcB 19-bit truncation on the two multiplies,
    **not** a scale/structural bug — the reduce itself is unbiased and the offset is smaller than
    the random spread.
  - Error does not grow with `W`: the widest and only chunked-reduce cell `(1,1,32,4096)` (`NW = 5`)
    is the most accurate fp32 row.

- **Measured device performance** (bf16 / HiFi4 / `fp32_dest_acc_en=True`, DEVICE KERNEL DURATION ns,
  from `test_rms_norm_perf.py`; "first commit" = the correctness-only baseline at 56cbc4f8):

  | shape | cores | first commit | Phase 0 | speedup |
  |---|---:|---:|---:|---:|
  | (1,1,32,1024) | 1 | 13 559 | 12 172 | 1.11× |
  | (1,1,32,2304) | 1 | 25 404 | 21 382 | 1.19× |
  | (1,1,32,5120) | 1 | 51 434 | 42 439 | 1.21× |
  | (1,1,32,7168) | 1 | 63 313 | 57 474 | 1.10× |
  | (1,1,8192,1024) | 110 | 103 238 | 102 114 | 1.01× |
  | (1,1,8192,2304) | 110 | 214 633 | 215 181 | 1.00× |
  | (1,1,8192,5120) | 110 | 482 655 | 483 074 | 1.00× |
  | (1,1,8192,7168) | 110 | 835 407 | 831 973 | 1.00× |

  Prefill is reader-bound (NCRISC 90–99 % of kernel time) — at the interleaved-DRAM read floor.
  Decode is latency-bound on **1 of 110 cores** (`ht_total == 1`), which is the standing gap and the
  subject of Refinements 2 + 3.

- **Golden suite at Phase 0**: **755 / 755** supported cells passing
  (`supported_pass` 755, `xfail_expected` 5 768, `invalid_skipped` 33 900, `no_axes_found` 15 uncharged;
  `supported_fail` = `xpass_drift` = `xfail_wrong_mode` = `supported_marked_xfail` = `invalid_unexpected` = **0**).
  Runner line: `770/40438 passed (0 failed, 0 errors, 33900 skipped, 0 hangs)`.
  Per-core CB totals 408–920 KB against `L1_CB_BUDGET_BYTES = 1 100 000` — no OOM anywhere in the suite.

- **Issues encountered**:
  - *Found and fixed by the implementer before verification*: `cb_scaler`'s data format was a fixed
    `Float16_b` per `op_design.md` R4, but `reduce_accumulate_via_add`'s `fold_partial_last` reads the
    partial-W mask through srcB **without** reconfiguring srcB (reduce entry already programmed it at
    the input format), so an fp32 input reinterpreted the bf16 mask as fp32 — 82 golden failures, all
    `float32 × w_non_aligned`. Fixed by deriving `cb_scaler`'s format from the input dtype; pinned by
    `test_rms_norm_debug.py`'s hand-calculable all-ones cases.
  - *Found and fixed during this verification pass* (three single-source / DRY repairs in
    `rms_norm_program_descriptor.py`, no math change):
    1. The reader's and writer's 18-entry compile-time-arg blocks were two separate literal
       expressions building the identical list, while both kernels hard-code the same CT indices and
       `TensorAccessorArgs<18>`. Now built once as `dataflow_ct_args`, with an assert pinning the
       block length so host and kernels cannot desync.
    2. `X_RESIDENT_DEPTH` was raised to 2 on the ROW_MAJOR path, where `cb_input_tiles` is produced
       *and* consumed by the compute RISC — a second strip can never be filled ahead, so it only
       spent L1 (and could evict `GAMMA_RESIDENT`). Now gated on `not is_rm`.
    3. The compute grid was derived twice (once to size the blocking, once inside
       `_core_assignment`), giving two sources for the quantity `grid_full` keys off. `_core_assignment`
       now takes the caller's grid.
  - *Reported, not edited* (`eval/golden_tests/rms_norm/feature_spec.py` is the golden-test author's):
    three INVALID entries couple the activation's `layout`/`memory_layout` with the **gamma tensor's**
    `gamma_layout` (cross-tensor coupling — 1 260 cells skipped that should be xfailing), and two
    `{bfloat8_b, *_non_aligned}` entries encode "out of scope for now" rather than impossibility
    (720 cells; they belong in the op's `EXCLUSIONS`). Both shrink the exercised universe, so nothing
    is over-claimed. See `verification_report.md` → "INVALID audit".
  - *Design-doc drift noted, not blocking*: `op_design.md` R4 (`cb_scaler` format) and §6
    (`cb_output_rm` page size) no longer match the shipped, helper-mandated behaviour; the
    implementation is correct in both cases.

- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm.py` (89 — immutable acceptance spec)
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_debug.py` (31 — deterministic bug repros)
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_perf.py` (9 — on-device perf harness + blocking report)
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_baseline.py` (16 — **new this pass**:
    PCC + max/mean abs error + relative RMS + got/true ratio-spread scale-bug tripwire)
  - `tests/ttnn/unit_tests/operations/rms_norm/probes/probe_001.py` (**new this pass** — the fp32
    bias attribution probe referenced in the precision baseline)
  - Total: **145 passed**, `scripts/run_safe_pytest.sh --dev --run-all tests/ttnn/unit_tests/operations/rms_norm/`

- **Artifacts**: `verification_report.md`, `verifier_report.json` (trimmed: summary + loud-category
  node lists + xfail axis histogram), `op_requirements.md` (5 refinements: 3 generality + 2 measured
  perf, at the 2:1 cadence), this changelog.

## Refinement 1 — Numerical configurability expansion (unlocks every perf target)

- **Date**: 2026-07-28
- **Device**: blackhole_p150b, 11 × 10 = 110-core compute grid, AICLK 1350 MHz

- **What was done**: opened the full float surface `{bfloat16, float32, bfloat8_b} ×
  {fp32_dest_acc_en True, False}`. **Zero kernel changes** — the compute/reader/writer triple was
  already helper-based and dtype-agnostic (runtime `get_tile_size`, host-computed byte counts), so
  the skill's pass condition held and every edit is in the op file + program descriptor.

  - `SUPPORTED`: `dtype += bfloat8_b`, `gamma_dtype += bfloat8_b`, `fp32_dest_acc_en = [True, False]`.
  - `default_compute_kernel_config()` **unchanged** (`fp32_dest_acc_en=True`), so
    `axes.py:40-43` and `test_rms_norm_default_config_matches_factory` still hold.
  - The existing `{float32, fp32_dest_acc_en=False}` `EXCLUSIONS` entry became *reachable* — kept,
    as the queue requires; it is now exercised (64 skips in the precision matrix).
  - `_row_elem_bytes()`: `Tensor.element_size()` **raises** for block-float dtypes (no per-datum
    width). It is consumed only on the ROW_MAJOR stick path, which a block-quantized tensor can
    never take, so it returns a documented structurally-unused placeholder rather than letting the
    exception escape a branch that discards it.
  - `x_squared_dtype`: the reduce programs srcA/srcB from its **input CB** (`cb_x_squared`), so that
    — not the input tensor's dtype — is the format `fold_partial_last` reads the partial-W mask at.
    Phase 0 conflated the two because they were always equal. Block-float inputs now square into a
    **bfloat16** `cb_x_squared`; `cb_scaler` derives from it, single-source.

- **Accuracy achieved** (`precision_matrix_results.md`, 320 cells = 8 shapes × 3 dtypes ×
  4 math_fidelity × 2 fp32_dest_acc_en × 2 distributions; gate PCC ≥ 0.99):
  - **320 / 320 pass**; worst PCC anywhere is **0.999325** (bfloat8_b / LoFi / bf16 DEST), ~75×
    inside the gate. 64 cells skipped via the op's own `EXCLUSIONS`.
  - `bfloat16`: PCC ≥ 0.999980 / rel-RMS ≤ 6.2e-03 at HiFi4 with `fp32_dest_acc_en=False`
    (vs 0.999992 / 2.5e-03 with it on) — the new DEST mode costs ~2.5× rel-RMS at HiFi4 and
    **nothing** at HiFi2/LoFi, where fidelity already dominates.
  - `bfloat8_b`: PCC ≥ 0.99979 / rel-RMS ≤ 2.2e-02 at HiFi4 (both DEST modes), flat across
    fidelity — its error is input/output block-float quantization, not the compute pipeline.
  - `float32`: unchanged from Phase 0 (PCC ≥ 0.999999 / rel-RMS 1.6e-03 at HiFi4).
  - **Accuracy watch item resolved**: the 8 interleaved perf loose cases (bf16 / HiFi2 /
    `fp32_dest_acc_en=False`, W up to 7168, `NW = 7`) pass their tighter `pcc_threshold = 0.9995`
    soft gate. The pairwise-add `AccumulateViaAdd` datapath was the right one; no SFPU finalize
    needed, no gate widened.

- **Golden test progress**: the refinement's gating criterion is met — **all 8 interleaved
  perf-flagged loose cases now run as `supported_pass`** (they were 8/8 xfail before, because every
  one pins `fp32_dest_acc_en=False`). Nothing downstream is blocked on measurement any more. The
  remaining 8 loose xfails are the sharded schemes (Refinements 2 / 4). Targeted cartesian slice
  over `1x1x64x128` (aligned) + `1x1x64x17` (W non-aligned) + `1x1x17x64` (H non-aligned):
  **120 passed, 0 failed, 0 XPASS-drift**, with bfloat8_b cells passing at every `gamma_dtype`
  including bf8 gamma. Unit suite: **456 passed** (136 prior + 320 new matrix cells) + 25 element-count
  cells, no regression.

- **Issues encountered**:
  1. **A silently-wrong bfloat8_b cell that PCC could not see.** With a `Bfp8_b` reduce datapath the
     `Float16_b` partial-W mask decoded as all-zeros (a bf8 tile's leading bytes are the
     shared-exponent header), so the final reduce-dim tile contributed **nothing**: an all-ones
     `W=49` row summed to **32**, not 49. Random-data PCC was **0.9998** and the golden gate
     (0.99 / 0.10) passed anyway — because dropping elements only *rescales* each row and PCC is
     scale-invariant. Caught by a deterministic all-ones probe (`probes/probe_005.py`), fixed by the
     `x_squared_dtype` rule above, and pinned by
     `test_partial_w_reduce_counts_every_element`, which inverts the kernel's own output back into
     the element count it actually summed. Post-fix the recovered sum is exact at
     `W = 33 / 49 / 63 / 100 / 4097`. **No `EXCLUSIONS` entry was needed for bf8 non-aligned** —
     both buckets are genuinely correct now (they remain `feature_spec.INVALID`-skipped, so this is
     honesty rather than new golden coverage).
  2. **`UnpackToDestFp32` is unavailable on the generic-op path** — investigated and reverted, see
     "Measured null" below.

- **Measured null — the `UnpackToDestFp32` upside the queue flagged**: the verifier suggested
  tagging the fp32 *inputs* to the two FPU multiplies. That is forbidden outright by the tag's
  exclusivity rule (a tagged CB can never be an FPU operand), so the only legal candidates here are
  `cb_partials` (the `AccumulateViaAdd` running sum, reloaded via `copy_tile`) and `cb_rms_sum`
  (read only by `CopyTile` → SFPU). Those looked genuinely promising: the reduce path's accuracy
  degrades exactly with the reload count — **12.02 effective mantissa bits at `NW=2`, 11.11 at
  `NW=8`, 9.89 at `NW=16`** (`probes/probe_007/008.py`) — the signature of per-reload TF32
  truncation compounding. The tag was implemented and correctly reached the descriptor (CBs 25/26
  tagged, vector length 64, verified by readback). It produced **bitwise-identical output**.
  Falsification test (`probes/probe_010.py`, plus three *separate processes* on a fresh shape to
  rule out the program cache): tagging `cb_rms_recip` — an **FPU srcB operand**, which by the
  exclusivity rule *must* corrupt the result if the tag were honoured — is also bitwise identical
  (`md5=a7c265c57d4719bb` for off / legal / illegal alike). Conclusion:
  `ComputeConfigDescriptor.unpack_to_dest_mode` is **inert on the `ttnn.generic_op` ProgramDescriptor
  path**. Reverted rather than left in as dead code, because the wrapper had to rebuild the caller's
  config field-by-field and would silently drop any future `ComputeConfigDescriptor` field — a real
  latent hazard bought for a measured-zero benefit. fp32 retains ≥13× tolerance headroom regardless.

- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_precision_matrix.py` (**new** — 320
    matrix cells + 25 partial-W element-count cells; imports the op's `EXCLUSIONS` rather than
    copying it, so the skip list cannot drift)
  - `tests/ttnn/unit_tests/operations/rms_norm/precision_matrix_results.md` (**new** — the §10
    results file)
  - `probes/probe_003..013.py` (bf8 bring-up, the deterministic mask probe that found the bug, the
    fp32 error attribution, and the `UnpackToDestFp32` falsification)
