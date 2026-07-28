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
