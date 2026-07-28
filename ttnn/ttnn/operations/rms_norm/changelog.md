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

---

## Refinement 2 — Cross-core `W`-split: partial-sum combine + `1/rms` multicast

- **Date**: 2026-07-28

- **What was done**: built op_design.md's Lamp L1 / §4.2 — the **dependent** `W` axis is now split
  across cores. Each core reduces its own `W`-slice to a raw `Σx²`, its **writer** unicasts that tile
  into slot `slot` of the group root's `cb_group_partials` and bumps the root's gather counter; the
  root's **reader** waits for all `CW` of them, hands them to compute, which folds them into one
  `mean(x²)` per tile-row with `n_reduced = W` (the grand total); the reader then **multicasts** that
  tile back over the group via `kernel_lib/mcast_pipe.hpp` (`SenderPipe`/`ReceiverPipe` +
  `ttnn.Mcast2D` on the host), and every core finalizes `rsqrt(mean+ε)` and scales its own slice.
  `SUPPORTED["memory_layout"]` gains `WIDTH_SHARDED` and `BLOCK_SHARDED`, consumed **natively**:
  `ttnn.cb_descriptor_from_sharded_tensor` backs `cb_input_tiles`/`cb_output_tiles` on the core's own
  L1 shard, so the reader issues **no input read at all** and the writer **no output write**.

  The delta is deliberately small. `_Blocking` is untouched in shape — the per-core `W` extent is
  simply the axis every knob now derives against, so `WT_CHUNK` / `NW` / `HT_BLOCK` / the residency
  predicates / every CB page count keep their existing single source. The new structure is one class
  (`_Placement`: row-split | W-split | shard-pinned), four fp32 CBs, and a combine block that slots
  between compute phases 3 and 4 exactly as the design promised. Compute phases 1/2/5/6/7 are
  byte-identical; only the *producer* of `cb_rms_sum` changes (reader multicast instead of phase 3).

- **Accuracy achieved**: PCC ≥ 0.9999 on every shape measured. Golden `test_op`: **3258 passed,
  0 failed, 0 XPASS, 3181 xfail** (was 3240 pass before the axis values; the sharded cells are the
  new pass mass). All 18 supported loose cases pass, including all 5 sharded perf-flagged geometries
  and the three interleaved `_WIDE` cases. Worst PCC seen across the sharded probe matrix: 0.999850
  (bfloat8_b BLOCK_SHARDED).

- **Perf** (blackhole_p150b, 110-core grid, bf16 / HiFi4 / fp32-on, DEVICE KERNEL DURATION,
  `RMS_NORM_W_SPLIT=0` vs default, one fresh-cache run each):

      shape             row-split  cores | W-split  cores | speedup
      (1,1,32,1024)        12_271      1 |   6_999     32 |  1.75x
      (1,1,32,2304)        21_429      1 |   7_663     36 |  2.80x
      (1,1,32,5120)        42_484      1 |   9_611     40 |  4.42x
      (1,1,32,7168)        57_608      1 |  11_279     56 |  5.11x
      (1,1,8192,*)                   110 |             110 |  0.99-1.01x (noise band)

  The decode column was 1 busy core of 110; it is now 32-56. Prefill is not engaged (its row axis
  already fills the grid) and is unchanged.

- **Golden test progress**: 3258/3258 supported cross-product cells + 18/18 supported loose cases.
  `WIDTH_SHARDED` and `BLOCK_SHARDED` are in SUPPORTED; `HEIGHT_SHARDED` remains the only
  `memory_layout` xfail (Refinement 4).

- **Issues encountered** — three real bugs, each invisible to at least one obvious check:

  1. **The combine must gather the RAW accumulator, not the reduced tile.** `AccumulateViaAdd`'s
     finalize writes the row sum into column 0 and leaves the surviving `x²` lanes in columns 1..31,
     so a second `REDUCE_ROW` over *finalized* partials double-counts them. Measured with DEVICE_PRINT:
     an all-ones `W=64` produced `mean(x²) = 8.75` instead of `1.0` — and **PCC scored 0.9999**,
     because rescaling every row by one factor is invisible to a scale-invariant metric. Fixed by
     running every chunk with `Accumulate::at` (never `at_last`) so `cb_partials` keeps the raw
     elementwise accumulator, and letting the root's `reduce_mean` do the one within-tile fold. The
     combine is then *literally* the local chunk-accumulate, done across cores instead of across
     chunks. Pinned by `test_cross_core_combine_counts_every_element` (absolute, not correlational).

  2. **A multicast rectangle is a VIRTUAL rectangle, and the logical grid is not virtually
     contiguous.** On this box logical x 0..6 → virtual 1..7 but logical x 7..10 → virtual 10..13, so
     virtual columns 8-9 are **not worker cores**. A group whose bounding box straddles that seam
     multicasts into non-worker endpoints — a hard device hang on the WIDTH_SHARDED cells whose shard
     grid is wider than one run. Fixed by splitting each group's broadcast into **one multicast family
     per virtually-contiguous run** (`_virtual_x_runs` / `_split_rect_by_runs`, two `Mcast2D` per
     group, two `McastArgs` in the reader) and by confining the *interleaved* split's group
     rectangles to a single run. Pinned by
     `test_every_combine_group_is_one_virtual_rectangle_per_family`.

  3. **`noc_semaphore_inc` is a non-posted atomic and needs draining.** The gather leg's counter bump
     left an outstanding atomic at kernel exit; on the sharded path there is no trailing output-write
     barrier to absorb it, so the core's NoC transaction counters never balanced and dispatch
     completion stalled. Fixed with a `noc_async_atomic_barrier()` at the end of the reader and the
     writer under `W_SPLIT`. **Only `--dev` caught this** — plain mode passed the same cells.

  Two geometry cases `auto_shard_config` really emits also needed handling: a **ragged** shard core
  grid (full rows + a partial last row), padded up to its bounding box with zero-work *filler* cores
  so the broadcast rectangle stays legal and dense-ack accounting stays right; and a shard grid that
  **over-covers W** (86 cores × 3 tiles vs 256 real tiles), whose trailing padding tiles are
  uninitialized L1 — the reader zeroes them so they contribute nothing to `Σx²`.

  Deferred to EXCLUSIONS with a note: `{ROW_MAJOR, WIDTH_SHARDED}` and `{ROW_MAJOR, BLOCK_SHARDED}`.
  `eval.sharding`'s ROW_MAJOR granule is `(1 row, L1_align/elem_bytes columns)`, so an RM shard is
  e.g. `[1, 128]` or `[64, 8]` — a single stick, or 8 of a tile's 32 columns. The kernels tilize a
  32-stick × 32-column block in place, which a core physically does not hold under those shards;
  forming one needs a cross-core *stick* gather, a different scheme from this refinement's W-split.

- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_wsplit.py` (**new**, 29 cases) — the
    split engages on wide/few-row shapes and does *not* on prefill or narrow W; every group's
    broadcast sub-rectangle is virtually contiguous; the absolute element-count assertion for the
    combine; the pinned + auto sharded geometries incl. ragged / padding-tile / degenerate-1-core;
    and `test_sharded_input_is_consumed_in_place`, which asserts the input and output CBs are
    **aliased onto the tensor buffers** — an accessor read of a core's own shard passes every
    numerical gate, so this is checked on the descriptor, not on the test colour.
  - `test_rms_norm_perf.py::test_report_blocking` now reports `CW` and the real core count, and the
    module docstring carries the row-split vs W-split A/B table.

---

## Refinement 3 — Speed up the perf-flagged **decode** column

- **Date**: 2026-07-28
- **Device**: blackhole_p150b, 11 × 10 = 110-core compute grid, measured AICLK
  **1349.98 MHz** — the references' `reference_aiclk_mhz`, so `scaled_ns == achievable_ns`
  and no clock scaling applies to any number below.

- **What was done**: a perf-only refinement (no SUPPORTED change) on the four pinned decode
  profiles `(1,1,32,W)`, `W ∈ {1024, 2304, 5120, 7168}`, measured at the config
  `feature_spec` actually pins for them — bf16 / TILE / `fp32_dest_acc_en=False` /
  `math_fidelity=HiFi2` / bf16 TILE gamma — never at the op's default HiFi4 / fp32-on corner.

  **Measure first.** A new `test_rms_norm_perf_decode_pinned` re-baselined the column at that
  config, and a new opt-in ablation (`test_rms_norm_ablate`, `RMS_NORM_ABLATE=combine[,gamma]`)
  attributed the time. The ablation holds the cross-core placement byte-for-byte — same core
  count, same per-core W slice, same DRAM reads and writes, same per-core square/reduce/scale —
  and removes only the named stage, so the delta is that stage's real cost on the critical path:

      shape           cores    full   no-combine   combine   share
      (1,1,32,1024)      32   6_938        3_524     3_414     49%
      (1,1,32,2304)      36   7_555        3_827     3_728     49%
      (1,1,32,5120)      40   9_309        5_152     4_157     45%
      (1,1,32,7168)      56  10_929        5_759     5_170     47%

  The **combine was the bottleneck**, not the data movement (NCRISC 1.0–2.2 µs) and not gamma
  (117–468 ns total, i.e. noise). It also grew ~73 ns per contributor — the signature of one
  core absorbing `CW × 4 KB` of NoC writes and then running `CW` fp32 tile-adds back to back.

  **Lever 1 — the two-stage combine** (`examples/tensix_all_reduce`'s measured
  `two_stage_grid_reduce`, the topology R2 deliberately deferred to a tuning phase). Row members
  gather to their grid-row leader, the row leaders gather to the group root: the serial fan-in
  becomes `cx + cy` instead of `cx · cy`. Deliberately small delta — one CB
  (`cb_group_partials2`), one semaphore, and a `CW1 × CW2` factorization of the existing `CW`.
  The leader's fold is *literally the worker's own chunk-accumulate* (`Accumulate::at`, never
  `at_last`, so the raw elementwise accumulator survives and the root's single finalize cannot
  double-count the surviving x² lanes), and its row sum rides the **same** `cb_partial_out` the
  slice partial did — one producer, one consumer, two sequential pushes, so no CB was added for it.
  Fan-in 32/36/40/56 → **12/12/13/15**; per-core gather L1 174/204/248/312 KB → **94/108/140/148 KB**.

  **Lever 2 — delete the republishing `copy` pass.** When a gather source comes from a *single*
  accumulate call (`NW == 1` workers, and every leader's stage-1 fold) the accumulator is written
  once and never reloaded, so the reduce packs straight into `cb_partial_out` instead of into
  `cb_partials` + a `copy`. That is a whole compute pass (~320 ns fixed, `examples/compute_block_size`)
  off the combine's serial path, twice over on a leader. `NW > 1` keeps the copy — there the
  accumulator genuinely is a compute→compute read-modify-write.

  Both levers are single-source knobs, not inlined constants: `COMBINE_MAX_FLAT_FANIN` selects the
  topology at fixed `CW` (raise it past any reachable group area to get R2's flat root back), and
  the existing `L1_GATHER_BUDGET_BYTES` still caps `CW` — now interpreted through `_gather_tiles`,
  which charges `cx + cy` for a staged rectangle and `cx · cy` for a flat one, so the same budget
  reaches a much wider group once the gather is staged.

- **Measured result** (pinned perf config, DEVICE KERNEL DURATION ns, one fresh-cache run per
  variant; a repeat of the winning configuration reproduced to 0.4–2.3 %, consistent with the
  op's established 1–2 % noise band):

      shape          R2 flat   +two-stage   +no-copy   speedup   ceiling   margin
      (1,1,32,1024)    6_938        5_987      5_709     1.22x     9_149    1.60x inside
      (1,1,32,2304)    7_555        6_513      6_219     1.21x    17_003    2.73x inside
      (1,1,32,5120)    9_309        8_320      7_933     1.17x    75_825    9.56x inside
      (1,1,32,7168)   10_929        9_200      8_917     1.23x    14_894    1.67x inside

  The headline `(1,1,32,7168)` ceiling is `104_259 / minimum_expected_speedup 7.0`; the op now
  delivers **104 259 / 8 917 = 11.7×** against that 7.0× requirement.

  At the op's *default* config the same shapes go 6_999 → 5_762 · 7_663 → 6_318 · 9_611 → 8_273 ·
  11_279 → 9_273 (1.16–1.22×), and **prefill is unmoved** — 101_152 / 219_255 / 479_801 / 831_397 ns
  against R2's 101_922 / 215_814 / 487_581 / 834_799, all inside the noise band. The W-split is
  not engaged at prefill (its row axis already fills the grid), so `CW == 1` and none of this code
  runs there.

- **Accuracy achieved**: `pcc_threshold = 0.9995` holds on all four pinned decode cases (asserted
  by `test_rms_norm_perf_decode_pinned`). Staging changes the *association order* of an exact
  elementwise fp32 add tree, nothing else — worst PCC across the golden slices is unchanged, and
  the all-ones absolute check recovers `mean(x²) = 1.0` exactly on both topologies.

- **Golden test progress**: unchanged by design (perf refinement, no SUPPORTED change).
  `test_op_loose` **18 passed / 1 xfail** (the xfail is HEIGHT_SHARDED — Refinement 4), including
  all five sharded perf geometries; the `(8,4)` and `(7,4)` WIDTH_SHARDED ones are dense
  rectangles and now take the staged combine too. Golden cross-product slices, 0 failed and
  0 XPASS-drift throughout: `1x1x32x8192` 108 passed (wide → staged combine, across TILE/RM ×
  gamma/no-gamma × every dtype × interleaved/WIDTH/BLOCK sharded), `2x1x64x4096` 108, `1x1x17x50`
  72 (both-non-aligned, no split), `4x8x32x47` 72 (W-non-aligned, row-split with the grid full).
  Unit suite **527 passed / 73 skipped**; the W-split suite is green under `--dev` (watcher + NoC
  sanitizer), which R2's changelog flags as non-optional for this scheme — the second gather leg's
  `noc_semaphore_inc` is drained by the existing `noc_async_atomic_barrier()`.

- **Issues encountered**: none — no hang, no numerical failure, no debug cycle. The two structural
  hazards were anticipated from R2's findings rather than hit: (a) a leader must fold with
  `Accumulate::at` and never `at_last`, or the root's finalize double-counts the surviving x² lanes
  exactly as R2 measured (8.75 instead of 1.0) with PCC scoring it 0.9999; and (b) two-stage needs a
  **dense** rectangle — every grid row needs a real core at column `x0` to lead it — so a ragged
  shard grid (auto_shard_config's full rows + partial last row, whose filler cores own no work) and
  any single-row group keep the flat topology by construction.

- **Knob defaults are measured, not assumed**:
  - `COMBINE_MAX_FLAT_FANIN = 24`. The second stage buys back ~73 ns per contributor removed from
    the root but costs one extra gather round (~1.3 µs, fitted from the matched-CW A/B), so it only
    pays above a wide fan-in. The cap is set at the widest flat gather still measured to be
    competitive rather than at the fitted break-even (~28): every group at or below it — notably the
    24-core `(1,1,64,12288)` loose case — keeps R2's flat topology byte-for-byte, and staging engages
    only in the 32–56 range where it is measured to win.
  - `CW` stays at *widest that fits*. Re-swept under the staged topology, that is now optimal
    (`(1,1,32,7168)`: 8 cores 12_016 ns, 32 cores 10_116, widest-56 **8_917**) — the reverse of the
    flat topology, where the same sweep had shown narrowing to be a 1.06–1.17× *win*. That inversion
    is the cleanest evidence the fan-in cost is what got removed. `CW` is in any case already at its
    structural maximum for these shapes: it must divide `Wt` and its rectangle must fit one
    virtually-contiguous column run (≤ 7 wide on this part).

- **Remaining headroom — a finding, not a queued task**. The combine is still the largest single
  item (~2.4–3.4 µs, 30–38 %) but its *shape* changed: it now fits ≈ **2.35 µs fixed + 73 ns ×
  (cw1+cw2)**, i.e. it is dominated by the fixed cost of two semaphore rounds plus the multicast
  handshake, not by fan-in — so a third gather stage would add another ~1.3 µs round and lose. The
  rest is a ~2.7 µs compute floor that is **per-pass-overhead bound**, not tile-work bound (1
  tile/core still costs 3.5 µs across 5–6 helper passes, ≈ 540 ns each). The next lever is therefore
  pass elimination, not topology: fold `AddUnary(eps) → Rsqrt` into the root's stage-2 reduce as a
  `post_reduce_op` so the multicast carries `1/rms` instead of `mean(x²)`, deleting one whole compute
  pass from *every* core on the serial tail after the broadcast (~5 %, and the value becomes
  bit-identical across the group instead of recomputed per core). Not done here because `reduce_mean`
  already occupies the `post_reduce_op` slot with its 1/N multiply, so it means dropping to
  `reduce<SUM>` and re-inlining that normalization — duplicating helper logic on the numerics path —
  for ~5 % on top of a result that already clears its ceiling by 1.67×. Fusing phases 5 and 6 (the
  two broadcast multiplies) into one `eltwise_chain` was investigated and is **not** expressible:
  `DestReuseBinary` combines DEST with a CB tile but carries no `BroadcastDim`, and the gamma
  multiply needs a row broadcast.

- **Tests added**:
  - `test_rms_norm_perf.py::test_rms_norm_perf_decode_pinned` (**new**, 4 cases) — the decode column
    at the pinned perf config, with the tighter `pcc_threshold = 0.9995` soft gate those loose cases
    carry. `perf_compute_kernel_config()` is the single source for that config so a knob A/B can
    never be taken on the wrong datapath.
  - `test_rms_norm_perf.py::test_rms_norm_ablate` (**new**, opt-in via `RMS_NORM_ABLATE`) — the
    stage-peeling ablation harness. Asserts nothing by design (never PCC-gate an ablated kernel) and
    is skipped unless explicitly requested, so a normal suite run never executes it.
  - `test_rms_norm_wsplit.py::test_combine_topologies_agree` (**new**, 4 cases) — an *absolute*
    all-ones check on **both** topologies plus an agreement check between them at matched `CW`.
    Staging is exactly where a premature within-tile fold could creep back one level up, and PCC is
    blind to it (one scale factor per row), so this is the R2 element-count discipline extended to
    the new fan-in tree.
  - `test_rms_norm_perf.py` knob overrides `RMS_NORM_GATHER_BUDGET_KB` / `RMS_NORM_MAX_FLAT_FANIN`,
    so `CW` and the combine topology stay A/B-measurable without editing the op.
  - `probes/probe_026.py`, `probes/probe_027.py` — clock/placement readback and the topology
    resolution check.

---

## Refinement 4 — `HEIGHT_SHARDED` placement (local shard, zero-copy)

- **Date**: 2026-07-28
- **Device**: blackhole_p150b, 11 × 10 = 110-core compute grid, AICLK 1349.98 MHz,
  per-core L1 bank **1 461 504 B** (read from the live device, not assumed)

- **What was done**: op_design Lamp L3 — the Phase-0 row split made *physical*. `SUPPORTED["memory_layout"]`
  gains `HEIGHT_SHARDED`. It really is a knob-turn: **zero kernel changes**, and the whole placement is one
  ~10-line branch in `_placement_sharded` that hands back `cw = 1`, no groups, no semaphores and no multicast,
  with the shard grid as the core assignment and the shard height as `rows_core_max`. Everything downstream
  was already generalized by Refinement 2 and is reused verbatim: `ttnn.cb_descriptor_from_sharded_tensor`
  backs `cb_input_tiles` **and** `cb_output_tiles` on the core's own L1 shards, so the reader issues no input
  read and the writer no output write, and the compute phases are byte-identical to the interleaved resident
  regime. Because each core holds WHOLE rows, the reduce stays entirely local — none of the R2/R3 combine
  machinery is built, allocated or executed.

  Two supporting changes, both single-source:

  1. **`ROW_MAJOR` shards route to the accessor path** (`tile_shard` gate, 3 lines in
     `create_program_descriptor`). "Sharded" in this op means *zero-copy*, which is only meaningful for a TILE
     shard. `eval.sharding`'s RM granule is `(1 row, L1_align/elem_bytes columns)`, so an RM height shard is
     `[1..3, W]` — a handful of sticks. That is **not** this core's block, and no amount of CB placement makes
     it one.
  2. **The L1 budget is now two budgets** (see "Issues encountered" — this is the R2 verifier note coming due).

- **Accuracy achieved**: PCC ≥ 0.999 on every cell measured; PCC = **1.000000** on the fp32 and the
  multi-tile-row shards, 0.999870 worst case (bfloat8_b). The all-ones **absolute** element-count check
  (`test_height_shard_counts_every_element`) recovers `mean(x²) = 1.0` exactly on
  `(1,1,256,512)` / `(1,1,2048,256)` / `(1,1,32,4096)` / `(1,1,17,64)` / `(1,1,32,50)` — chosen over a PCC gate
  because a wrong row→core map, a dropped W-tile or a wrong `n_reduced` only *rescales* rows, which PCC scores
  0.9999 (the R1/R2 lesson).

- **Golden test progress**: `test_op_loose` is **19 passed / 0 xfail** — the `(1,1,256,512)` HEIGHT_SHARDED case
  was the last remaining loose xfail in the suite. Cross-product slice over five shapes spanning all four
  placements (`1x1x256x512`, `1x1x17x64`, `4x8x32x256`, `1x1x32x8192`, `2x512x1024` — TILE/RM × gamma/no-gamma ×
  every dtype × INTERLEAVED/HEIGHT/WIDTH/BLOCK): **536 passed, 0 failed, 0 XPASS**, 228 xfailed.
  `memory_layout` is now complete against TARGET — no value is left queued.

- **Issues encountered** — one real bug, found exactly where Refinement 2's verifier note pointed
  ("make sure the block shrink is not what is paying for the shard"):

  **The resident shard was being charged against the CB budget.** A zero-copy sharded CB is *aliased* onto the
  tensor's own buffer, which the **buffer** allocator already reserved out of the same L1 bank — it is not part
  of the program's CB region. Charging it against `L1_CB_BUDGET_BYTES` (a budget calibrated for
  *program-allocated* CBs on the interleaved path, where tensors live in DRAM) double-counts it, and the
  halve-and-re-derive loop then pays for the shard by shrinking the block. HEIGHT shards are full-W and so the
  largest of the three schemes, which is what made this load-bearing rather than cosmetic:

  - bf16 `(1,1,32,8192)`: `WT_CHUNK` collapsed **32 → 1** (`NW = 256` on a single core) — a 256-pass compute
    loop where 16 passes fit;
  - every fp32 `W = 4096` HEIGHT cell (`(1,1,32,4096)`, `(1,1,128,4096)`, `(2,1,64,4096)`, `(1,32,4096)`,
    `(32,4096)`) was **refused outright** — `AssertionError`, i.e. hard `supported_fail` — by **10 KB**, with
    **361 KB of the 1 461 504 B bank still free**.

  Fixed by budgeting the two quantities separately in `_Blocking._fits`:

      program CBs                  <= L1_CB_BUDGET_BYTES              (unchanged)
      program CBs + resident shard <= bank size - L1_ALLOC_HEADROOM_BYTES

  The bank size is read from the live device (`ttnn.get_memory_view(...).total_bytes_per_bank`), never
  hardcoded — it is arch- and dispatch-config-specific. After the fix: fp32 `W = 4096` runs at `WT_CHUNK = 8`,
  bf16 `W = 8192` at `WT_CHUNK = 16`.

  **No regression, and it is provable rather than merely measured.** On an interleaved tensor the shard term is
  0, so the second wall can never bind and the derivation is *byte-identical* to the single-budget model
  (pinned by `test_interleaved_blocking_is_unchanged_by_the_two_budget_split`). On the sharded side exactly one
  of the geometries the suites exercise moves — surveyed across all six pinned + eight auto geometries — and it
  gets **faster** (pinned perf config, one fresh-cache run per variant via `RMS_NORM_L1_HEADROOM_KB`):

      geometry                                single-budget   two-budget
      (1,1,32,1024)   WIDTH [32,128]  (8,1)           5_007        5_016
      (1,1,32,2304)   WIDTH [32,256]  (9,1)           5_684        5_667
      (1,1,32,5120)   WIDTH [32,160]  (8,4)           5_878        5_890
      (1,1,32,7168)   WIDTH [32,256]  (7,4)           6_295        6_284
      (1,1,8192,1024) BLOCK [1024,128](8,8)          89_413       85_107   HT_BLOCK 4 -> 8, 1.05x

  **No EXCLUSIONS entry was needed**, which was not the expected outcome. `ROW_MAJOR × HEIGHT_SHARDED` looked
  like R2's `{ROW_MAJOR, WIDTH/BLOCK_SHARDED}` precedent, but the two differ: an RM height shard's tile-row is
  spread over up to 32 *different* cores, so the read is genuinely **non-local** — precisely the case
  `TensorAccessor` exists for, and the opposite of re-reading a core's own block. Routing it there costs 3
  lines and it passes at PCC ≥ 0.99999, clean under `--dev`. The same mechanism very likely unblocks R2's two
  RM exclusions; deliberately **not** attempted here, as it is outside this heading's scope.

  One case is refused and correctly so: fp32 `(1,1,32,8192)` HEIGHT_SHARDED needs 2 MB of shards on one core
  against a 1.46 MB bank. It is unreachable through the public entry point — `allocate_tensor_on_device` OOMs
  on the output shard first, which the harness classifies as an infeasible skip.

- **Perf note, a finding for Refinement 5 rather than a task it must rediscover**: the A/B above also
  re-measures the five pinned sharded geometries against their `achievable_ns` references. The four WIDTH ones
  are close (5 016–6 284 ns vs 4 110–5 481, i.e. 1.15–1.22× away), but `(1,1,8192,1024)` BLOCK_SHARDED sits at
  **85 107 ns against a 25 640 ns reference — a 3.3× gap**, by far the largest of the five and the obvious
  first target for R5's sharded column.

- **Tests added**:
  - `tests/ttnn/unit_tests/operations/rms_norm/test_rms_norm_height_sharded.py` (**new**, 47 cases) — the
    placement is pinned by the shard and the reduce stays local (core map, per-core row count, `cw == 1`,
    empty `groups`, no semaphores in the descriptor, tile-rows covered exactly once); the input **and** output
    CBs are aliased onto the shards (descriptor-level, because an accessor read of a core's own shard passes
    every numerical gate in the file); the RM shard is asserted **not** aliased, since it is the deliberate
    non-local exception; the all-ones absolute element-count check; the reference across
    dtype × gamma × 5 shapes; and the two budget-model guards described above.
  - `test_rms_norm_perf.py::test_rms_norm_perf_sharded_pinned` (**new**, 5 cases) — the five pinned sharded
    geometries at the pinned perf config, one dispatch each with the 0.9995 soft gate. Added as this
    refinement's no-regression guard for the budget change; it is also R5's measurement surface.
  - `RMS_NORM_L1_HEADROOM_KB` env override (read in the op, same style as `RMS_NORM_W_SPLIT`) so the
    single-budget vs two-budget A/B stays re-runnable without editing the op.
  - `probes/probe_028..034.py` — HEIGHT bring-up, the RM-shard granule check, the L1-tight corner, and the
    old-vs-new blocking survey across every sharded geometry the suites emit.
  - Unit suite: **582 passed, 73 skipped** (`--run-all`), and the sharded suites green under `--dev`.

---

## Refinement 5 — Speed up the perf-flagged **prefill** column and the sharded geometries

- **Date**: 2026-07-28
- **Device**: blackhole_p150b, 11 × 10 = 110-core compute grid, measured AICLK **1349.98 MHz** —
  the references' `reference_aiclk_mhz`, so `scaled_ns == achievable_ns` and nothing below is scaled.

- **What was done**: a perf-only refinement (no SUPPORTED change) on the four interleaved prefill
  profiles `(1,1,8192,W)` and the five pinned sharded geometries, all at the config `feature_spec`
  pins for them (bf16 / TILE / `fp32_dest_acc_en=False` / HiFi2 / bf16 TILE gamma).

  **Measure first, and the measurement redirected the phase.** A new `test_rms_norm_perf_prefill_pinned`
  re-baselined prefill at the pinned config (the Phase-0/R2/R4 prefill numbers were all taken at the
  op's *default* HiFi4 / fp32-on corner — the wrong datapath), and a new **sharded** twin of R3's
  ablation harness (`test_rms_norm_ablate_sharded`) attributed the sharded time. The ablation holds
  the shard placement, the core count and the per-core slice byte-for-byte and removes only the
  gather + root fold + multicast:

      geometry            full   no-combine   combine   share   BRISC   NCRISC
      WIDTH 32x1024      4_913        3_088     1_825     37%      95    1_139
      WIDTH 32x2304      5_503        3_658     1_845     34%     282    1_476
      WIDTH 32x5120      5_664        3_288     2_376     42%     220    1_871
      WIDTH 32x7168      6_079        3_562     2_517     41%     191    2_286
      BLOCK 8192x1024   74_838       55_522    19_316     26%     293    2_801

  `BLOCK_SHARDED` is **MATH-bound**: with the combine removed it is 55.5 µs of pure TRISC time
  against **0.3 µs of BRISC and 2.8 µs of NCRISC**. So the verifier's suggested placement/NoC tune
  could not have moved it — only FPU-op count and DEST-sync overhead can. (The NoC pairing was in
  any case already correct by construction: reader = NCRISC/NOC_0 (+x,+y), writer = BRISC/NOC_1
  (−x,−y), and every gather root sits at the *low* corner of its group, so the gather's −x/−y
  traffic and the multicast's +x/+y traffic are each on the NoC that routes them natively.)

  **Lever (a) — one L1 wall instead of two.** `L1_CB_BUDGET_BYTES = 1_100_000` was a Phase-0 guess
  at "worker L1 minus firmware headroom". R4 then read the real bank from the live device
  (1 461 504 B) and made `prog + shard <= bank − headroom` the second wall, but kept the guess as a
  first wall on program-allocated CBs alone. That first wall is a *proxy* for the quantity the second
  one measures, and strictly more conservative, so it could only ever cost block size. Retired:
  one condition, `program CBs + resident shards <= live bank − L1_ALLOC_HEADROOM_BYTES`.

  Blast radius verified by A/B over the whole perf set (`probes/probe_036.py`): **exactly one cell's
  derivation changes** — `(1,1,8192,7168)` gains a **resident gamma** (needs 1 195 648 B; the guessed
  wall refused it, the real 1 330 432 B one accepts). gamma is the *same bytes on every core* and is
  re-read once per row-block when not resident, so that deletes 2 of 3 gamma passes ≈ **100 MB of the
  shape's 386 MB of DRAM traffic**. Every other cell is byte-identical.

  **Lever (b) — `FUSE_SQ`, the fused square-accumulate.** Phases 2 and 3 were two FPU passes over the
  same block: square into `cb_x_squared`, then elementwise-accumulate that block back out. The second
  one's operation is an *add*, so the FPU's accumulate-into-DEST mode collapses them —
  `mul_tiles(x, x, acc_to_dest)` over a sticky D0 leaves `Σ_w x_w²` in DEST, which is **exactly** the
  raw elementwise accumulator both the local finalize and the cross-core combine already consume
  (op_design's "the combine is literally the local chunk accumulate"). `eltwise_chain`'s
  `DestAccumulation` walk expresses it directly: D0 stays acquired across an outer row's whole `Wt`
  and is packed once per row. Removes one FPU op per input tile of four, the whole `cb_x_squared` L1
  round trip, and `W−1` of every `W` packs. Two preconditions, decided on the host and
  `static_assert`ed in the kernel: `NW == 1` (a DEST accumulator dies at the next `tile_regs_acquire`,
  and the chain forbids composing DEST with L1 accumulation) and no partial-W mask (that rides the
  reduce helper's partial-scaler hook, which this path does not go through). `scaler_dtype` follows
  the reduce's input CB to Float32 — R1's rule, applied, not a new one.

  **Lever (c) — `DEST_BLOCK`.** `EltwiseShape`'s `block_size` defaults to **1**, and at 1 the chain
  runs a whole `tile_regs_acquire/commit/wait/release` round *plus a pack phase* around **every single
  tile** (`examples/compute_block_size`: ~1.6 µs per extra pass, 1.65× end to end). Every chain in the
  compute kernel was doing this. It now asks for `DEST_AUTO_LIMIT` and lets `eltwise_chain` clamp to
  its own compile-time DEST capacity (`chain_max_block_v`) — so it is "the coarsest block that fits
  DEST", re-derived per chain and per `fp32_dest_acc_en` / `dst_full_sync_en` setting, not a constant.

- **Measured result** (pinned perf config, DEVICE KERNEL DURATION ns, best of two fresh-cache runs
  that agreed to 0.02–1.6 %; the `(a)+(c)` column is `RMS_NORM_FUSE_SQ=0`, so both levers are shown
  pulling separately and together):

      case              R4 base   (a)+(c)   +FUSE_SQ  speedup      ref   vs ref
      WIDTH 32x1024       5_002     4_975      4_900   1.021x    4_110    1.19x
      WIDTH 32x2304       5_730     5_735      5_503   1.041x    4_617    1.19x
      WIDTH 32x5120       5_895     5_776      5_664   1.041x    5_267    1.08x
      WIDTH 32x7168       6_344     6_217      6_079   1.044x    5_481    1.11x
      BLOCK 8192x1024    85_245    77_561     74_813   1.139x   25_640    2.92x
      prefill 8192x1024  97_097    97_632     98_229   ~1.00    96_744    1.02x
      prefill 8192x2304 222_213   219_159    217_673   1.021x  211_345    1.03x
      prefill 8192x5120 480_766   471_674    470_143   1.023x  738_307    0.64x
      prefill 8192x7168 810_299   652_079    659_050   1.229x  1_032_281  0.64x
      decode x4                                        ~1.00            all inside

  On `BLOCK_SHARDED` the split is `DEST_BLOCK` 1.099× then `FUSE_SQ` a further 1.037×. `8192×1024`,
  `8192×5120` and the four decode shapes are flat *inside the noise band* — which this phase also
  measured properly: re-running a **byte-identical** `8192×5120` program gave 480 766 / 497 328 /
  471 787 / 470 143 / 473 456, a ±3.4 % spread, so the large prefill shapes are noisier than the
  1–2 % the decode column showed.

- **The prefill column has no data-movement headroom, and that is measured rather than argued.** The
  verifier note said to check the roofline before manufacturing a change. `/perf-ceiling-dm`'s NPE CLI
  is a test-build target not configured in this tree, so a stronger empirical form was used instead —
  a real op on the real box rather than an interpolated model. `test_prefill_dm_ceiling_reference`
  dispatches **`ttnn.clone`** on the same tensor with the same interleaved placement: every tile read
  once, written once, no reduction, no gamma. Nothing that reads and writes this tensor can beat it.

      W       clone ns   clone GB/s   rms_norm ns   rms_norm GB/s
      1024      83_100          404        98_229             415
      2304     192_568          392       219_966             417
      5120     408_984          410       470_143             434
      7168     586_752          400       659_050             433

  rms_norm moves its bytes **4–6 % more efficiently than a plain DRAM copy of the same tensor**, and
  comes in *under* `clone × 1.215` — its byte-scaled floor, since gamma is
  `110·Wt / (2·256·Wt + 110·Wt)` = a flat **17.7 %** of traffic independent of `W` — on all four
  shapes. Two of the four prefill shapes are 1.57× *ahead* of their references; the other two are at
  theirs within the noise band.

- **Accuracy achieved**: PCC ≥ 0.9995 on every pinned shape (asserted by the three `*_pinned` tests,
  which carry the loose cases' tighter soft gate). `FUSE_SQ` changes *where* the running sum lives —
  a sticky DEST register instead of an fp32 L1 accumulator reloaded per pair — so it was pinned with
  an **absolute** check and an equivalence check, never a PCC: all-ones input recovers
  `mean(x²) = 1.0` exactly on every fused shape, and fused vs pairwise agree to a median ratio of
  1.000 on random data.

- **Golden test progress**: unchanged by design (perf refinement, no SUPPORTED change).
  `test_op_loose` **19 passed / 0 xfail**. Cross-product slices, **0 failed and 0 XPASS-drift
  throughout**: `1x1x64x128` + `1x1x32x8192` + `1x1x17x50` + `4x8x32x256` = **534 passed**
  (fused and non-fused, aligned and both-non-aligned, TILE/RM × gamma/no-gamma × every dtype ×
  interleaved/WIDTH/BLOCK/HEIGHT), and `1x1x256x512` + `2x512x1024` + `1x1x32x4096` = **296 passed**
  (the sharded placements). Unit suite **605 passed / 78 skipped**; the W-split and HEIGHT suites
  green under `--dev` (watcher + NoC sanitizer), which R2's changelog flags as non-optional for this
  scheme since the compute kernel that feeds the gather changed.

- **Issues encountered** — one real bug, in the new lever, caught immediately by the pinned gates:

  **`block_size > 1` with `OutputLifecycle::Streaming` gives PCC 0.0.** `PackTile` is not a CB-reader
  element, so it does **not** constrain `block_size` (`chain_supports_block` only inspects the input
  side) — but `Streaming` is `{ReservePolicy::PerTile, PushPolicy::PerTile}`, reserving and pushing
  **one** tile per block-iteration while the pack loop writes `inner_count` of them. `Chunked` is the
  matching policy. Fixed on the three blocked pack sites (phases 2, 5, 6, both their resident and
  streaming spellings); the two chains whose operands are per-tile (phase 4's `CopyTile`, the
  republishing `copy`) are clamped to `block_size 1` regardless, so they keep `Streaming` and an
  unblocked shape rather than carrying a knob that can never turn. Catastrophic and immediate, so it
  cost one cycle — but it is exactly the class of thing that would have been invisible had the block
  happened to be 1 on the shapes under test (`(1,1,32,1024)`, whose `WT_CHUNK` is 1, was the single
  case that passed while the other 12 failed).

  Two of my *test* premises were also wrong and worth recording, since both would mislead a reader:
  `(1,1,32,8192)` is **fused** even though its whole-tensor `Wt` is 256 — the W-split hands each core
  only 8 W-tiles, so the precondition is on the *per-core* chunk count, not on `W`; and R4's
  `test_resident_shard_is_not_charged_against_the_cb_budget` A/B has to accept **outright refusal**
  as well as a shrunk block, because that is the form R4 measured on fp32 `W=4096`.

- **Remaining headroom — findings, not queued tasks.** `BLOCK_SHARDED` is still 2.9× from its
  25 640 ns reference and is math-bound, so the next lever there is **fusing phases 5 and 6**
  (`x·(1/rms)` then `·gamma`): `DestReuseBinary` carries no `BroadcastDim` (R3 already established
  this), but pre-expanding gamma to full tiles *once per core* makes it expressible — `Wt` tiles,
  cheap at the sharded `Wt = 4…8`, and it can reuse the L1 that `cb_scaled` vacates. It deletes the
  whole `cb_scaled` round trip (128 packs + 128 unpacks per core) but **no FPU ops**, so on a
  math-bound kernel I estimate only ~5 %, which is why it lost to spending the remaining budget on
  the regression net. The combine's residual 19.3 µs (26 %) is **not** topology-bound: a
  `BLOCK_SHARDED` group is one grid row (`cw_y == 1`), so R3's two-stage tree cannot apply by
  construction, and `master.md`'s `two_phase_reduce_mcast` — measured 1.69× on a 1×8 line at exactly
  this payload — is a T3-shaped restructure of which core owns which tile-row. For **prefill**, the
  one remaining byte reduction is a **grid-wide gamma multicast** (op_design's Lamp L2), worth ~1.2×
  on the column; not built because it adds a global sync to the highest-volume path,
  `examples/shared_input_reuse` measures such a broadcast at only 1.71× for an 11× read reduction,
  and the two shapes it would help are already at their references within the noise band.

- **Tests added**:
  - `test_rms_norm_fused_reduce.py` (**new**, 14 cases) — the `FUSE_SQ` datapath: it *engages* where
    expected (else every assertion below would pass on the fallback path), it *refuses* where
    structurally wrong (`NW > 1`, partial W), all-ones recovers `mean(x²) = 1.0` **absolutely**, and
    fused agrees with pairwise on random data by median ratio. The file's docstring records why a PCC
    gate is not acceptable here: R1, R2 and R4 each shipped an accumulation bug that PCC scored
    ≥ 0.9998 because miscounting elements only rescales rows.
  - `test_rms_norm_perf.py::test_rms_norm_perf_prefill_pinned` (**new**, 4 cases) — prefill at the
    pinned perf config, the datapath its `achievable_ns` references were taken at.
  - `test_rms_norm_perf.py::test_prefill_dm_ceiling_reference` (**new**, 4 cases) — the measured DM
    ceiling (`ttnn.clone` on the same tensor), so "prefill is saturated" stays a re-runnable
    measurement instead of a claim in a changelog.
  - `test_rms_norm_perf.py::test_rms_norm_ablate_sharded` (**new**, 5 cases, opt-in via
    `RMS_NORM_ABLATE`) — the sharded twin of R3's ablation harness.
  - `test_rms_norm_height_sharded.py::test_live_bank_budget_buys_gamma_residency` (**new**) — pins
    lever (a) exactly where it is load-bearing, including that the *block factor* must not move.
  - `RMS_NORM_FUSE_SQ` env override, so lever (b) stays A/B-measurable without editing the op.
  - `probes/probe_035.py`, `probes/probe_036.py` — the blocking survey across every perf geometry and
    the one-cell-changes A/B for the budget retirement.

---

## Perf 1 — fan-out perf tournament (5 ideas measured, 2 graduated)

- **Date**: 2026-07-28
- **Device**: blackhole_p150b, 11 × 10 = 110-core compute grid, measured AICLK **1349.98 MHz** —
  the references' `reference_aiclk_mhz`, so `scaled_ns == achievable_ns` and nothing below is scaled.
  Per-core L1 bank 1 461 504 B, usable budget 1 330 432 B (read from the live device).
- **`SUPPORTED` is byte-identical** to Refinement 5 (`git diff` on `rms_norm.py` is empty). A perf
  tournament moves nothing in the registry; the signal is device-ns.

### Permanent per-stage instrumentation (the thing that made the round work)

Every stage boundary of all three kernels now carries `MaybeDeviceZoneScope` from
`kernel_lib/perf_instrumentation.hpp` — free when the profiler is off, so it is a **permanent
fixture** and must never be removed:

- compute: `cmp_gamma_tilize` / `cmp_tilize_a` / `cmp_wait_x` / `cmp_square` / `cmp_rowsum` /
  `cmp_publish` / `cmp_combine` / `cmp_rsqrt` / `cmp_scale` / `cmp_gamma_mul` / `cmp_tilize_b` /
  `cmp_untilize`
- reader: `rdr_scaler` / `rdr_gamma_resident` / `rdr_shard_publish` / `rdr_read_a` /
  `rdr_gather_wait` / `rdr_mcast` / `rdr_read_b`
- writer: `wtr_gather_hop` / `wtr_write`

The compute kernel is compiled three times, so each zone reports separately on UNPACK / MATH / PACK.
That split is what distinguishes an FPU-op-count problem from an unpack/pack-throughput problem, and
it is what found this round's headline. Aggregator:
`tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/zone_report.py`.

### Focus shape

`feature_spec.LOOSE_CASES` carries no literal `attention:` note, so the perf-flagged entries are the
nine `_perf_case` rows with `extras.achievable_ns`. Of those, **`(1,1,8192,1024)` `BLOCK_SHARDED`,
shard `[1024,128]`, grid `(8,8)`** was the furthest from its goal — 76 112 ns against a 25 640 ns
reference, **2.97× over**, the single largest absolute gap in the whole set. It is the primary
target. Its full config was checked against `SUPPORTED` before any measurement: bf16 / TILE /
`fp32_dest_acc_en=False` / `math_fidelity=HiFi2` / bf16 TILE gamma / `BLOCK_SHARDED` — **all
supported, no generality gap**, so it was optimized exactly, never via a proxy.

Derived knobs (from the live device, not assumed): 64 cores, `cw=8 cw1=8 cw2=1` (flat combine), per
core 32 tile-rows × `Wt=4`, `nw=1`, `ht_block=8`, `nh_core=4` row-blocks, `fuse_sq=1`,
`x_resident=1`, `gamma_resident=1`.

### Measured breakdown, and the ranked bottleneck

Per-stage zones plus **cumulative** ablation (`RMS_NORM_ABLATE=combine`, which holds the shard
placement, core count and per-core slice byte-for-byte and removes only gather + root fold + mcast).
Ablation was decisive here and **corrected my first reading of the zones**, which is exactly why
`/perf-measure` insists on peeling cumulatively rather than trusting a single stage's number:

| stage | zone reading (full run) | truth after ablation | share |
|---|---|---|---|
| `cmp_rsqrt` | 56 133 ns MATH | **29 000 ns of real work** + ~27 000 ns of *waiting* for the multicast | **38 %** |
| combine round trip | `rdr_mcast` 56 515, `wtr_gather_hop` 59 152, `cmp_combine` 25 407 max | **19 400 ns of critical path** (76 112 → 56 732 ablated) | 26 % |
| `cmp_scale` | 17 633 ns UNPACK | **~3 700 ns of real work**; ~13 900 ns is `cb_wait_front(cb_rms_recip)` *inside the zone* | 5 % |
| `cmp_square` | 6 599 ns MATH | real | 9 % |
| `cmp_gamma_mul` | 3 949 ns MATH | real | 5 % |
| reader/writer DM | `rdr_shard_publish` 79, `wtr_write` 54 | zero-copy sharded both sides — **saturated at zero** | ~0 % |

Two zone readings were **wait, not work**, and both would have sent the round to the wrong place:

1. `cmp_rsqrt` conflates the mcast wait with real SFPU work. The combine-ablated run — no cross-core
   traffic at all, nothing to wait for — *still* spends 29.0 µs there on 32 fp32 tiles per core, i.e.
   **906 ns/tile**. That is the op's single largest real compute item and **five refinements never saw
   it**, because Refinement 5's ablation attributed 26 % to the combine and read the 55.5 µs remainder
   as undifferentiated "MATH-bound".
2. `cmp_scale`'s 17.6 µs UNPACK is mostly that thread parked waiting on `cmp_rsqrt`. Found by the
   `scale_gamma_dest_fusion` subagent, whose isolated bench reproduced `cmp_gamma_mul` to within 1 %
   but `cmp_scale`'s unpack at 3 770 ns against the op's 17 634 — the gap *is* the artefact. **Stages
   5+6 are 8 886 ns of real work (11.8 %), not 21 550 (28 %)**, so the idea aimed at them was aimed at
   a mirage. Recorded because the same trap will recur: a `cb_wait_front` inside a zone charges the
   producer's latency to the consumer's stage.

**Roofline gate** (`/perf-ceiling-dm` in spirit; its NPE CLI is still an unconfigured test target in
this tree, so the empirical form from Refinement 5 was reused). The focus shape's own data movement is
**zero** — both tensors are zero-copy L1 shards — so the only DM is the combine's *self-inflicted*
traffic, whose volume is a design choice, not a floor: the necessary payload is 8 cores × 256
row-sums × 4 B = **8 KB per group**, against **1 MB actually moved**, 128× more. The interleaved
prefill column, by contrast, was already measured against `ttnn.clone` on the same tensor and found to
move bytes 4–6 % *more* efficiently than a plain DRAM copy — a genuine roofline, and the reason no
idea was aimed at prefill's input path.

**Ranked headroom: (1) phase 4's SFPU lane waste, (2) the combine's gather payload, (3) nothing else
above noise.** `cmp_square` and `cmp_gamma_mul` are at their FPU floor for the op count; the
reader/writer are at zero.

### Portfolio floated (6 ideas, deliberately overlapping)

| # | idea | target | verdict |
|---|---|---|---|
| A | `rsqrt_lane_and_window` — make `AddUnary(eps)`+`Rsqrt` cost 32 lanes, not 1024; and/or one DEST window for all `ht` tiles | stage 1 | **WIN 3.53×** → graduated |
| B | `gather_payload_shrink` — the gather ships 128× more bytes than it carries | stage 2 | **WIN 1.320×** → queued for Perf 2 |
| C | `gather_spread_topology` — spread the destination so no one core absorbs the group's inbound | stage 2 | **WIN 1.174×** → superseded by B |
| D | `scale_gamma_dest_fusion` — fuse phases 5+6 through DEST with a pre-expanded gamma | stages 3+5 | **REGRESSION 0.67×** |
| G | `gamma_broadcast_rowsplit` — the standing reuse-shared broadcast (Lamp L2) | prefill / `cw==1` | **WIN 1.14–1.21×** → graduated |
| F | `combine_compute_overlap` — software-pipeline the row-blocks | stage 2 | **not floated** (see below) |

B and C were floated as competing attacks on the same stage on purpose; D was floated alongside its
own components so a null on the fusion would not lose them. F was dropped once C's result showed the
combine's shape had already changed under it — it is a Perf 2 candidate against the new critical path.

### Per-idea results

**A — `rsqrt_lane_and_window`: WIN, 3.53× on phase 4. GRADUATED.**
The assigned window lever is a measured **NULL** (1.00×): ablation puts the whole per-tile
scaffolding at 80.9 ns/tile and blocking recovers ~17 ns of it, hidden behind the SFPU pass. The
906 ns/tile is **100 % per-tile SFPU lane work** — fitted at 23 ns per 32-lane accurate-rsqrt vector.
The waste: `cb_rms_sum` is a REDUCE_ROW statistic (column 0 only) whose sole consumer is
`mul_tiles_bcast<COL>`, so 24 of 32 vectors per pass computed lanes nobody reads — twice over, since
`AddUnary` and `Rsqrt` are separate elements hence separate passes.

| option | vectors/tile | ns/tile | vs base | col-0 PCC |
|---|---|---|---|---|
| baseline (the op's chain) | 64 | 912.7 | 1.00× | 0.9999967 |
| `chain_blocked` (the assigned lever) | 64 | 912.2 | **1.00× NULL** | 0.9999967 |
| `raw_fused_rc` (fuse only) | 32 | 864.0 | 1.06× | 0.9999968 |
| `raw_c` (`VectorMode::C`) | 32 | 482.8 | 1.89× | 0.9999967 |
| `raw_cskip` (parity stride) | 16 | 281.1 | 3.25× | 0.9999967 |
| **`chain_fused_cskip` (graduated)** | **8** | **258.3** | **3.53×** | **0.9999968** |
| *ablation* copy+pack only | 0 | 80.9 | *11.3×* | n/a |

**No option trades precision** — the fused variants are marginally *more* accurate because `x+eps` no
longer round-trips through the bf16 DEST. Body is the stock accurate rsqrt verbatim.
**Predicate: unconditional** (2.95×→3.60× across `ht` ∈ {1,2,4,8,16}, every CB format pair, both DEST
modes); the only guard is architectural, `!ARCH_QUASAR`, which lacks `_calculate_sqrt_body_`.
Safety verified **on device, not argued**: feeding `mul_tiles_bcast<COL>` a tile with poison in
columns 1..31 reproduces column 0 across the whole output at pure bf16 rounding error (max rel-err
0.0078), so leaving those lanes unwritten provably cannot change the output.
*Deferred option:* narrowing `cb_rms_sum`/`cb_rms_recip` fp32→bf16 is a further 2.4 % and halves
their L1 — free at `fp32_dest_acc_en=False` (DEST is bf16 anyway, so the fp32 container never held a
keepable bit) but **load-bearing at `True`**, so it needs its own guard. Perf 2.

**B — `gather_payload_shrink`: WIN, 1.320× whole-op. QUEUED FOR PERF 2.**
Each worker ships `ht` full 4 KB fp32 tiles whose only content is 32 row-sums each; the root receives
1 MB per group. `colpack_bf16` folds within-tile then column-packs `ht` row-sums into **one** bf16
tile: 75 490 → 57 181 ns, against a combine-fully-ablated floor of 56 070 — **94.3 % of the entire
combine cost recovered**, leaving 1 481 ns (2.6 %) on the table. It beats the pure-byte ceiling
(66 384 ns) because column-packing also moves the fold off the root: `ht·CW1 = 64` tile-reduces
become `CW1 + ht = 16`. bf16 partials are **bit-identical** to fp32 (same reason as A's deferred
option) — a free 2×, not a trade. PCC 0.99998256, all-ones exact.
Predicate: `w_split and cw2 == 1 and nw == 1 and 2 ≤ ht_block ≤ 16` for the column-pack (0.961× at
`ht_block == 1`; the ≤ 16 bound is a hard mechanism limit — the reduce scaler can only address
face-rows 0..15), and `w_split and not fp32_dest_acc_en` for the bf16 payload, which wins on **all 7**
geometries (1.020×–1.160×). Carries 3 documented raw-LLK bypasses (a non-canonical per-output-column
scaler; `reduce_uninit` between `tile_regs_commit` and `pack_tile` to clear the packer edge mask; a
raw L1 scaler-bank fill on the idle writer — a naive RISC-V store loop cost +20.5 µs and made the
whole idea read as 0.786×).
**Not graduated this round, for a measurable reason: its baseline no longer exists.** A cut 21 µs out
of the same kernel, so the 1.320× was measured against a 75 490 ns program that is now 54 986 ns and
whose combine share has changed. Perf 2's job is exactly to re-measure against the shifted critical
path; graduating a stale 1.320× would be claiming a number I had not measured on the current op.
*Trap worth carrying forward:* the first `colpack` cut scored **PCC 0.9998 (pass)** while corrupting
12.5 % of tile-rows — only the absolute check caught it. Cause: a CB whose per-row-block push count
did not divide its page count, so a multi-page `cb_reserve_back` straddled `fifo_limit`. **Any new
intermediate CB in this op must satisfy `pushes_per_row_block | num_pages`.**

**C — `gather_spread_topology`: WIN 1.174×, but SUPERSEDED by B. NOT GRADUATED.**
`row_rotate` (core *j* owns the fold for tile-rows ≡ *j* mod `cw`) gives 75 573 → 64 364 ns, and
61 373 (1.231×) once the L1 it frees buys a bigger block — **bit-exact** vs the current topology
(PCC 1.0000000, max |diff| 0), since it folds the same tiles in the same order. Scheme 1
(`two_stage_1d`, factorising `cw` in slot space) is 1.027× on the focus shape and null-to-0.93×
everywhere else. Superseded because B measures 1.320× and leaves only 2.6 % above the combine floor,
and the two are **mutually exclusive in pure form**: `colpack` packs all `ht` tile-rows into one tile,
which must go to one owner, so `row_rotate` has nothing left to distribute. B ≥ C on every geometry
measured. C's four integration notes (each a bug it hit and fixed) are preserved in its artifact dir
for Perf 2, in case re-measurement inverts the ranking.

**D — `scale_gamma_dest_fusion`: REGRESSION 0.67× (fusion) + NULL (components). NOT GRADUATED.**
8 886 → 13 176 ns, ratio invariant at 0.67× across focus / decode / no-gamma / streaming / RM. Cause
is structural, not tuning: all three TRISCs are already ~92 % busy (7 937 / 8 333 / 8 102 of an
8 886 ns kernel), so phases 5+6 sit at the TRISC-throughput floor for 2 FPU ops + 2 unpacks + 2 packs
per output tile and there is no idle engine for the fusion to reclaim; holding DEST live across two
dependent FPU ops costs more than the `cb_scaled` round trip it removes, in both reuse directions and
with reconfig on or off. Gamma pre-expansion is *not* the problem (484 ns once per core). Dropping
phase 6's `BroadcastDim::Row` is also a regression (0.92×) — the bcast unpack MOP is *cheaper* than a
plain two-operand unpack. This idea's real value was the stage-3 correction above, plus **two helper
bugs**, both silent and both invisible to an all-ones-gamma test:
- `ckl::UnaryBcast::exec` hardcodes `in_tile_index = 0` (`eltwise_chain.inl:1239`), so it always
  broadcasts the CB *front* tile and ignores the chain's walk index; only `InputLifecycle::Streaming`
  walks a multi-tile operand. A `CallerManaged` gamma silently expands tile 0 `WT` times.
- `DestReuseBinary` at `block_size == DEST_AUTO_LIMIT` corrupts one face of the highest DEST lane
  (PCC 0.988); `≤ DEST_AUTO_LIMIT − 1` is bit-exact, so the reuse path needs a spare slot that
  `chain_max_block_v` does not reserve.
Also quantified: the DEST-sync window costs **29 ns**, of which ~0.93 µs/core would be recoverable if
one window spanned two tile-rows — not expressible today, because the `1/rms` operand is
`OperandKind::Col` whose index *is* the row. A helper-surface gap, not an op choice.

**G — `gamma_broadcast_rowsplit`: WIN 1.14–1.21× on prefill. GRADUATED.**
op_design Lamp L2. Refinement 5 deferred this arguing prefill is DRAM-saturated so the gamma bytes are
free; that is now **measured false** — the ablation bound (gamma reads deleted, reserve/push kept) says
the 17.7 % byte share converts to ~19 % of wall clock, and the broadcast captures ~88 % of that bound.
Every clause of the predicate is measured: `cw == 1` (a W-split already gives each core a disjoint
slice), `gamma_resident`, `not is_rm_gamma`, `not sharded_in` (**0.735× on HEIGHT_SHARDED** — both
tensors are zero-copy L1 shards, so gamma is the only DRAM traffic and it is uncontended; ablation
4 307 → 4 319 ns, i.e. zero DM headroom), `active_cores ≥ 44` (110 → 1.386×, 55 → 1.170×,
44 → 1.139×, 33 → 1.054×, 22 → 1.004× NULL, 11 → **0.841× REGRESSION**), and at most two *dense*
rectangles, one per virtual-x run. Delivery stays in the **prologue** (1.09× slower when moved later:
the prologue is the one window where the injector's own read is uncontended) and one injector **per
run** beats one for the grid. Bit-exact vs baseline on all 8 cases. Cost added: 14.9 µs of handshake,
uniform across 110 cores (2.7 %), replacing a per-core prologue read that cost 243 µs of wall clock on
average (`rdr_gamma_resident` 243 525 → 14 877 ns).

### What graduated, and the whole-op result

Two changes, on **disjoint predicates** — A is unconditional and in a different stage; B (queued) is
`cw > 1` and G is `cw == 1`, so they can never both fire on one program.

`DEVICE KERNEL DURATION [ns]`, pinned perf config, one fresh-cache profiled run per column. **All
columns carry the new zone instrumentation, so they are comparable to each other**; the R5 column is
un-instrumented and is shown only for continuity (the zones cost ~500–800 ns/core *under the
profiler*, which is why the small cases read higher than R5 — they are free in a normal run).

| case | R5 (unzoned) | Perf 1 base | +A | +A+G | ref | vs ref now |
|---|---|---|---|---|---|---|
| **BLOCK 8192×1024** (focus) | 74 813 | **76 112** | **54 986** | **54 919** | 25 640 | **2.14×** (was 2.97×) |
| prefill 8192×1024 | 98 229 | 98 340 | 96 671 | **86 353** | 96 744 | **0.89× ahead** |
| prefill 8192×2304 | 217 673 | 218 244 | 217 987 | **181 141** | 211 345 | **0.86× ahead** |
| prefill 8192×5120 | 470 143 | 471 999 | 467 100 | **399 649** | 738 307 | 0.54× ahead |
| prefill 8192×7168 | 659 050 | 668 294 | 664 849 | **556 611** | 1 032 281 | 0.54× ahead |
| WIDTH 32×1024 | 4 900 | 5 715 | 5 080 | 5 030 | 4 110 | 1.22× |
| WIDTH 32×2304 | 5 503 | 6 356 | 5 667 | 5 637 | 4 617 | 1.22× |
| WIDTH 32×5120 | 5 664 | 6 530 | 6 096 | 6 158 | 5 267 | 1.17× |
| WIDTH 32×7168 | 6 079 | 7 064 | 7 109 | 7 183 | 5 481 | 1.31× |
| decode 32×1024 | 5 709 | 6 387 | 5 784 | 5 836 | 9 149 | 0.64× inside |
| decode 32×2304 | 6 219 | 7 054 | 6 536 | 6 533 | 17 003 | 0.38× inside |
| decode 32×5120 | 7 933 | 8 865 | 8 455 | 8 501 | 75 825 | 0.11× inside |
| decode 32×7168 | 8 917 | 9 709 | 9 354 | 9 446 | 14 894 | 0.63× — **11.0× against the 7.0× requirement** |

- **Focus shape 76 112 → 54 919 ns, 1.386×**, closing the gap to its reference from 2.97× to 2.14×.
  The 21.1 µs removed matches the 20.9 µs of MATH the bench predicted.
- **All four prefill cases are now ahead of their references** (8192×1024 and 8192×2304 previously sat
  1.02–1.03× over). 8192×1024 at 86 353 ns is essentially at the measured `ttnn.clone` DRAM-copy floor
  of 83 100 ns; 8192×7168 moves 235.8 MB at 424 GB/s, **1.056× faster than a plain clone of the same
  tensor**.
- **No regression** anywhere outside the 2–3 % noise band. WIDTH 32×7168 (7 109 → 7 183, 1.0 %) and
  decode 32×7168 (9 354 → 9 446, 1.0 %) are inside it; both are geometries G's predicate refuses, so
  the movement is noise, not the change.

**Liveness of the fast path was proven, not assumed.** A deliberate `static_assert(false)` inside
A's new element failed the build on trisc0/1/2 at the focus geometry's exact CT args
(`WT=4 WT_CHUNK=4 NW=1 HT_BLOCK=8 W_SPLIT=1 CW=8 CW1=8 CW2=1 FUSE_SQ=1`), so the measurement cannot
have been of dead code behind a stale cache. G's engagement was confirmed the same way, by having the
host print its resolved plan: 110 cores, `cw=1`, two dense families `[(0,0,6,9), (7,0,10,9)]` split
exactly at the Blackhole virtual-x seam, injectors at the low corners.

### Raw-LLK admitted

One bypass, in `rms_norm_compute.cpp`, with its justification at the kernel head so a later
helper-usage pass cannot "fix" it back and undo the win: `rsqrt_tile` hardcodes `VectorMode::RC` and
`ITERATIONS = 8` (as does `add_unary_tile`) and no vector-mode / iteration / DEST-stride knob exists
on the compute API or the chain elements; and `AddUnary`/`Rsqrt` are separate elements, so there is no
"unary with a pre-added scalar" element to compose. Measured authorization: 912.7 → 258.3 ns/tile.
The bypass is **only the SFPU body** — `RsqrtAddUnaryColZero` derives from `ckl::UnaryOp`, so
`eltwise_chain` still owns the CB lifecycle, the dtype reconfig and the dst-sync window, and the
call-site diff is two lines. G adds **no** raw LLK; it rides `mcast_pipe` exactly as the `1/rms`
broadcast does.

### Guard-set no-regression

One representative per distinct kernel path × layout × placement, and — for G — slices on **both
sides** of its predicate:

- `eval/golden_tests/rms_norm/`: `test_op_loose` (all 19, incl. all five pinned sharded geometries)
  + cross-product slices `1x1x2048x256` and `4x1x512x512` (64 tile-rows → 64 cores, G **engages**),
  `1x1x64x128`, `1x1x17x50`, `1x1x256x512` (G **refuses**) + all 5 `test_regression.py` numerics
  cases → **574 passed / 0 failed / 0 errors / 228 xfailed, no XPASS drift**, across TILE/RM ×
  gamma/no-gamma × every dtype × both `fp32_dest_acc_en` × interleaved/WIDTH/BLOCK/HEIGHT.
- `tests/ttnn/unit_tests/operations/rms_norm/`: **698 passed / 79 skipped**, including
  `test_rms_norm_fused_reduce`'s absolute all-ones checks and `test_rms_norm_wsplit`'s
  topology-agreement checks. Every file also green individually under `--dev` (watcher + NoC
  sanitizer), which R2's changelog flags as non-optional for the W-split scheme.

### Issues encountered

- **A pre-existing `--dev` whole-suite precompile failure, attributed rather than inherited.** Running
  the *entire* `tests/.../rms_norm/` directory under `--dev` produces BRISC linker errors
  (`non constant or forward reference address expression for section .text`). It is **not** a
  regression: the pre-tournament op produces **1136** such failures on the same command against
  **2** with these changes, every individual test file passes under `--dev`, and the whole directory
  passes without it. An environment/precompile-pass issue over a large heterogeneous suite; recorded
  as a finding, not fixed here.
- **The experiment dirs cannot live under `ttnn/`.** `ttnn/ttnn/operations/__init__.py`
  `exec_module()`s every `.py` it walks at `import ttnn`, so a module-level side effect in a bench
  file breaks `import ttnn` for the whole repo — which happened during the round (the shared
  `zone_report.py` parsed `sys.argv` at import and broke *every* pytest in the clone until it was
  wrapped in `main()`). Repo policy also forbids global `torch` imports there. All five artifact dirs
  therefore live at `tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/<idea_slug>/`.
- The pre-commit `clang-format` hook rewrites the kernels *after* a measurement run, so the committed
  source differed from what had been measured; re-verified green afterwards (whitespace only).

### Artifacts

`tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/` — one dir per idea, all five with their
benches, kernels, measured tables and correctness gates, plus `README.md` (the round's breakdown) and
`zone_report.py`. `probes/probe_037..040.py` — the blocking survey and G's engagement + ramp-gamma
gates. Breadcrumbs: `agent_logs/blocking-perf-coordinator_breadcrumbs.jsonl`.

### Perf 2 queue (measured, not speculative)

1. **B `colpack_bf16`** — re-measure against the post-A baseline (54 919 ns), then graduate on
   `w_split and cw2==1 and nw==1 and 2 ≤ ht_block ≤ 16`. Was 94.3 % of the combine.
2. **A's deferred CB narrowing** — `cb_rms_sum`/`cb_rms_recip` fp32→bf16, guarded on
   `fp32_dest_acc_en == False`: +2.4 % on phase 4 and halves their L1.
3. **Phase 4's remaining floor** — folding `+eps`/`rsqrt` into the reduce's `post_reduce_op` slot is
   bounded *exactly* by the ablation at 81 ns/tile, a further 1.44× on the stage, and would retire
   `cb_rms_sum` entirely.
4. **C `row_rotate`** as B's fallback if re-measurement inverts the ranking.
5. **F combine/compute overlap**, against whatever the critical path is after (1).

---

## Perf 2 — fan-out perf tournament, round 2 (5 ideas measured, 2 graduated as one fusion)

- **Date**: 2026-07-28
- **Device**: blackhole_p150b, 11 × 10 = 110-core compute grid, measured AICLK **1349.98 MHz** — the
  references' `reference_aiclk_mhz`, so `scaled_ns == achievable_ns` and nothing below is scaled.
- **`SUPPORTED` is byte-identical** to Perf 1 (`git diff` on `rms_norm.py` is empty). A perf
  tournament moves nothing in the registry; the signal is device-ns.
- **Headline: the focus shape went 54 377 → 27 091 ns, 2.007×**, closing its gap to the
  perf-flagged reference from **2.12× over to 1.06× over**. Every other perf case moved inside the
  noise band or improved 1.03–1.10×; nothing regressed.

### Round shift — why the round-1 queue could not just be graduated

Perf 1 ended with a five-item queue. It could not be cashed in as written, because Perf 1's own
graduations moved the critical path underneath it: idea **B** (`colpack`) had measured 1.320× against
a **75 490 ns** program that was now **54 377 ns**, with a different combine share. Round 2 therefore
re-ran the measured breakdown first, on the now-instrumented, partly-optimized op, and re-measured
every carried-over idea against the current baseline. Two of the round-1 queue items turned out to be
worth *more* than queued, one turned out to be **self-defeating** on top of the other, and one was
killed by its own sensitivity study. None of that was visible from round 1's numbers.

### Focus shape

`feature_spec.LOOSE_CASES` carries no literal `attention:` note, so — as in Perf 1 — the perf-flagged
entries are the thirteen `_perf_case` rows with `extras.achievable_ns`, and the mandatory primary
target is the one **furthest from its reference**: **`(1,1,8192,1024)` `BLOCK_SHARDED`, shard
`[1024,128]`, grid `(8,8)`** at 54 377 ns against a 25 640 ns reference — **2.12× over**, the largest
gap in the set by a wide margin (the next worst is `WIDTH 32×7168` at 1.17×).

Its **full** config was re-checked against `SUPPORTED` before any measurement: bf16 / TILE /
`fp32_dest_acc_en=False` / `math_fidelity=HiFi2` / `math_approx_mode=False` / bf16 TILE gamma /
`BLOCK_SHARDED` — **every knob supported, no generality gap**, so it was optimized exactly, never via
a proxy. Derived geometry (read from the live device, not assumed): 64 cores, `cw=8 cw1=8 cw2=1`
(flat fold), per core 32 tile-rows × `Wt=4`, `nw=1`, `ht_block=8`, `nh_core=4` row-blocks,
`fuse_sq=1`, `x_resident=1`, `gamma_resident=1`.

Secondary column: the four `WIDTH_SHARDED` decode geometries, the only other cases over their
references (1.06–1.17×).

### Measured breakdown, and the ranked bottleneck

Cumulative ablation (`RMS_NORM_ABLATE=combine[,gamma]`, `test_rms_norm_ablate_sharded`), one
fresh-cache profiled run per variant — peeled **cumulatively**, not one stage at a time:

| variant | ns | Δ | share |
|---|---|---|---|
| full | **54 377** | — | — |
| `−combine` | 35 026 | **19 351** | **35.6 %** |
| `−combine −gamma` | 28 403 | 6 623 | 12.2 % |

**The ablation is not a pure peel, and reading it as one would have understated the prize.** With
`cw = 1` every core must do its own local reduce, which *adds* 11 793 ns of `cmp_rowsum` MATH that
the real program does only on the root, inside `cmp_combine`. So the combine round trip actually
costs **~27 300 ns of member wait** — visible as the gap between `cmp_rsqrt`'s 35 112 ns MATH zone
and the 7 839 ns of real work the same 32 tiles take in the ablated run — and the ablation nets only
19 351 because it re-adds the rowsum. Per-stage real work, measured in the combine-ablated run so it
is work and not waiting:

| stage | MATH | UNPACK | PACK | note |
|---|---|---|---|---|
| `cmp_rsqrt` | 7 839 | 9 594 | 8 366 | 32 tiles, 245 ns/tile |
| `cmp_square` | 5 714 | 3 540 | 4 942 | 128 tiles, 49 ns/tile |
| `cmp_gamma_mul` | 3 952 | 4 175 | 4 004 | 128 tiles, 31 ns/tile — FPU floor |
| `cmp_scale` | 3 772 | 7 074 | 5 119 | 128 tiles, UNPACK-bound |
| reader/writer DM | — | — | — | `rdr_shard_publish` 77, `wtr_write` 54 |

**Roofline gate.** The focus shape's own data movement is **zero** — input and output are both
zero-copy L1 shards, so there is no DRAM traffic to optimize and the reader/writer are saturated at
zero. The only DM is the combine's *self-inflicted* traffic, and its volume is a design choice, not a
floor: the necessary payload is 8 cores × 256 row-sums × 4 B = **8 KB per group** against ~1 MB
actually moved, **128× more**. The interleaved prefill column was left alone for the opposite reason —
Refinement 5 measured it moving bytes 4–6 % *more* efficiently than a plain `ttnn.clone` of the same
tensor, a genuine roofline.

**The decisive observation.** Summing the three TRISCs' real work gives a compute floor of
**~22–24 µs** for the current decomposition — i.e. **below the 25 640 ns reference**. So the combine
round trip was not *a* bottleneck, it was **the entire gap to the goal**, and the round's portfolio
was aimed there almost exclusively.

**Ranked headroom:** (1) the combine round trip, ~27 300 ns of wait, 35.6 % net — nowhere near any
roofline; (2) `cmp_rsqrt` real work 7 839 ns MATH, ablation-bounded below by a measured 81 ns/tile
scaffolding floor, so ~5 200 ns reachable; (3) `cmp_square` at 49 ns/tile; (4) `cmp_scale`, already
measured at its TRISC-throughput floor in Perf 1 (a phases-5+6 fusion was 0.67×), so **excluded**;
(5) gamma's 6 623 ns, which is required math at the FPU floor; (6) reader/writer, saturated at zero.
The same combine lever also reaches the secondary column, where it is 38–52 % of each kernel
(`WIDTH 32×1024` 4 498→2 645, `32×2304` 5 157→3 197, `32×5120` 5 586→2 783, `32×7168` 6 410→3 090).

### Portfolio floated (5 ideas, deliberately overlapping and mutually exclusive)

`examples/master.md`'s `tensix_all_reduce` (T3) supplied the prior that made the overlap deliberate:
for an **isolated 1-D group with several tiles per core**, a tile-index reducer beats a flat root by
~2×; at **1 tile/core** it degenerates and the low-fan-in scheme wins instead. The focus shape is
exactly an 8-core 1-D group with `ht = 8` tiles/core — so *shrinking the payload* (fewer tiles/core)
and *parallelising the fold* (more owners per tile) pull in opposite directions. Both were floated.

| # | idea | target | verdict |
|---|---|---|---|
| 1 | `colpack_regraduate` — column-pack the `ht` row-sums into ONE tile; bf16 wire datum | combine payload | **WIN 1.507×** → graduated |
| 2 | `combine_parallel_fold` — sweep the pack factor `K` from pure-rotate to pure-pack | combine fold | **WIN but superseded** (K=1 *is* idea 1) |
| 3 | `combine_allgather_no_root` — one hop, no root: broadcast + fold everywhere | combine topology | **REGRESSION 0.42×** |
| 4 | `combine_latency_hiding` — software-pipeline the row-block loop | combine latency | **WIN 1.096× but superseded** |
| 5 | `phase4_packed_rsqrt_and_bf16_stats` — one SFPU pass on a packed statistic; bf16 stat CBs | stage 2 | **WIN 1.622–2.351×** → graduated **as a fusion with 1** |

Reader/writer levers came as pairs by construction here: the gather is the writer leg and the
multicast the reader leg, and ideas 1–3 all move both.

### Per-idea results

Every subagent's fork was gated for faithfulness first — its `baseline` variant had to reproduce
54 377 ns before any ratio it reported was allowed to mean anything. All five passed that gate
(54 270 / 54 355 / 54 344 / 54 381 / n-a), which is what makes the numbers below comparable.

**1 — `colpack_regraduate`: WIN 1.507×. GRADUATED.**
Each worker shipped `ht` RAW 4 KB fp32 tiles per row-block in which all 32 columns still held live
x² partial sums — 128 B of information in 4096. Folding within-tile and landing the `ht` results in
`ht` **distinct columns of ONE tile** cuts the payload `ht`-fold *and* turns the root's fold from
`ht·CW1` tile-reduces into `CW1 + ht` ops, because the pack moves the fold off the root as a side
effect.

| option | ns | vs base | PCC | precision |
|---|---|---|---|---|
| baseline | 54 270 | 1.000× | 0.99998402 | — |
| `bf16` | 46 533 | 1.166× | 0.99998402 | **bit-identical** |
| `colpack` | 36 581 | 1.483× | 0.99998256 | within contract |
| **`colpack_bf16`** | **36 014** | **1.507×** | 0.99998256 | within contract |

36 014 lands **988 ns above the combine-fully-ablated floor of 35 026 — 94.9 % of the whole 19 351 ns
combine cost recovered.** Zones: root `cmp_combine` 25.4 µs → 6.7 µs, `wtr_gather_hop` 43.2 → 23.8,
`rdr_mcast` 42.7 → 25.7. The bf16 payload is **free, not a trade** — re-verified on the current op,
not inherited: at `fp32_dest_acc_en=False` DEST is already 16-bit, so the fp32 container never held a
keepable bit. Predicates: `colpack` needs `w_split and cw2 == 1 and nw == 1 and 2 ≤ ht_block ≤ 16`
(monotone in `ht_block`, which **is** the pack factor: 1.507× / 1.343× / 1.141× / **0.967×** at
8 / 4 / 2 / 1 — hence the ≥ 2 guard; ≤ 16 is a hard mechanism limit, a reduce scaler can only address
face-rows 0..15); the bf16 payload needs only `w_split and not fp32_dest_acc_en` and won on **11 of
11** geometries (1.020–1.195×), which is why it is the wider guard and covers everything `colpack`
refuses.

**2 — `combine_parallel_fold`: WIN 1.403×, but SUPERSEDED. NOT GRADUATED.**
The bake-off swept the pack factor `K` — how many tiles a worker ships per row-block — from `K = ht`
(pure tile-index rotation, round 1's `row_rotate`) to `K = 1` (pure column-pack): 54 367 baseline →
43 029 (`row_rotate`, 1.264×) → **38 759 (`K=1`, 1.403×)** → 39 186 (`K=2`) → 41 392 (`K=4`) → 46 470
(`K=8`). **No interior `K` ever strictly beat both pure ends**, and the gap between them *widens* as
`ht_block` grows (at a 1 MB budget, `K=1` improves to 37 065 while `row_rotate` only reaches 40 167).
So the answer to master.md's payload-vs-parallelism tension is measured and one-sided here: `K = 1`
wins, and `K = 1` **is** idea 1's `colpack`. The idea's one unique offering — `row_rotate` winning
back the `ht_block ∈ [2,4]` regime by 1.05–1.11× on a smaller BLOCK shape — sits **at the 2–3 % noise
band** and would have cost a second mechanism with a *larger* CB footprint (412 vs 324 KB at
`ht_block=8`), so it was not graduated. Both schemes are bit-exact (they fold the same tiles in the
same order).

**3 — `combine_allgather_no_root`: REGRESSION on every geometry. NOT GRADUATED.**
Replacing the two-hop gather→root-fold→multicast with a single all-broadcast plus a local fold on
every core loses **1.11× to 6.93×**, monotonically worse in `cw · ht_block`: focus 54 344 → 129 426
(0.42×), and 61 166 against 8 832 at `cw = 56`. The mechanism is diagnosed, not guessed: the reader's
all-gather zone costs ~28 082 ns per row-block per core for `cw · ht_block = 64` individual 4 KB
`noc_async_read`s at **~439 ns each** — it is **transaction-count bound, not bandwidth bound**,
because the h-major landing layout the existing fold needs forces one transaction per (peer, tile-row)
pair. At 3.70× the combine-ablated bound versus the two-hop tree's 1.55×, **fewer logical hops did not
mean less overhead once each hop is many small point-to-point transactions instead of one
converge-then-broadcast tree.** A smaller payload would not rescue it (the floor is per-transaction,
not per-byte). Bit-exact throughout — correctness was never the problem. Recorded because the negative
result is load-bearing: it is the measured reason the op keeps its root-gather topology.

**4 — `combine_latency_hiding`: WIN 1.096× standalone, SUPERSEDED by its own sensitivity study. NOT GRADUATED.**
Prefetching the next row-block's pass A across the combine stall gives 54 381 → 49 636 (1.096×,
bit-exact), and the zones prove the mechanism rather than the wall clock: `rdr_gather_wait` drops
3 503→918 avg and 27 764→7 162 max (−74 %). It was asked for a **sensitivity curve**, and that is what
retired it:

| stall_wait (of CW1 = 8) | baseline | prefetch_a | ratio |
|---|---|---|---|
| unablated | 54 381 | 49 636 | **1.096×** |
| 6 | 48 385 | 49 023 | **0.987× — already a regression** |
| 4 | 45 111 | 48 322 | 0.933× |
| 1 | 44 124 | 47 679 | 0.925× |

Its fixed overhead (a doubled `cb_group_partials` = +262 144 B, a second semaphore, a parity branch,
and the `HT_BLOCK` 8→4 halving that the extra L1 forces) is only worth paying against the *current,
un-shrunk* stall. Idea 1 shrinks the stall by 94.9 %, i.e. far past the ~25 % point at which this
idea inverts — so it must be **superseded, not stacked**. That verdict is now confirmed twice over:
by its own ablation, and by the delivered stall (27 300 → 11 349 ns, −58 %). Two real bugs found on
the way, both worth keeping: the literal prefetch schedule has a **write-after-read race** on the
root's landing zone (a fast core remote-writes hb+1 into the slot the root is still reading for hb;
PCC 0.99982, and the all-ones gate did *not* catch it because uniform input hides a row mismatch),
and `Semaphore::wait(n)` is an **exact-match** wait, not a threshold, so it hangs forever if the
counter races past `n` — `wait_min` is the safe swap. `split_passA_all` is **BLOCKED**, not slow:
batching all four row-blocks needs 1 MB for `cb_group_partials` alone, which with the 524 288 B of
shard-resident CBs exceeds the entire 1 461 504 B L1 bank. Genuinely infeasible, not a tunable.

**5 — `phase4_packed_rsqrt_and_bf16_stats`: WIN on both sub-levers. (b) GRADUATED as a fusion with idea 1.**
The statistic for a tile-row is 32 scalars living in column 0 of its own tile, so the measured
81 ns/tile of per-tile copy+pack scaffolding was being paid `ht` times over for 32 live datums each.
Packing the row-block's statistics into ONE tile pays it once:

| option | ns | ns/tile | vs base | col-0 PCC |
|---|---|---|---|---|
| baseline | 8 679 | 271.2 | 1.000× | 0.9999898 |
| `baseline_bf16` (sub-lever a) | 8 461 | 264.4 | 1.026× | 0.9999898 |
| `pack_here_c` | 5 350 | 167.2 | 1.622× | 0.9999797 |
| `pack_here_cskip` | 4 610 | 144.1 | 1.883× | 0.9999797 |
| **`pack_given_c`** | **4 497** | **140.5** | **1.930×** | 0.9999862 |
| `pack_given_cskip` | 3 692 | 115.4 | 2.351× | 0.9999862 |

Predicate `ht_block ≥ 4`, **sharper than the ≥ 2 the round guessed**: monotone
0.422 / 0.682 / 1.090 / 1.622 / 2.150× at `ht_block` 1 / 2 / 4 / 8 / 16, so `ht_block = 2` is a clear
regression, not a wash. Confirmed by reading the header rather than trusting the brief:
`mul_tiles_bcast<COL>` takes a CB and a tile **index**, never a column offset, so the extract back to
col-0 form is structurally required, not an implementation choice. **Sub-lever (a)** — narrowing
`cb_rms_sum`/`cb_rms_recip` to bf16 — is **bit-exact at `fp32_dest_acc_en=False` (max |diff| 0.0) and
measurably different at `True` (5.1e-2)**, confirming its own guard is load-bearing; at 1.026× on the
stage it is **inside the whole-op noise band**, so it was measured and **not graduated** rather than
claimed (its 32 KB/core of freed L1 is recorded as available headroom for a future round). Four
silent bugs found and documented, all of the believable-but-wrong kind: a zero-copy sharded CB needs
an explicit publish before any `cb_wait_front` (hung the device); a selector keyed on the loop
variable instead of a fixed row (`h = 0` accidentally worked); `compute_kernel_hw_startup` referencing
an undeclared CB corrupts *other* CBs' format tracking (PCC 0.889, no crash);
`copy_tile_to_dst_init_short` does **not** reconfigure the unpacker dtype (PCC 0.59 on a stale fp32
config).

### What graduated — and why it is one fusion, not two changes

Aggregation resolved the overlaps: **1 supersedes 2** (its `K=1` point *is* `colpack`, and no interior
`K` beat it), **1 supersedes 4** (measured, by 4's own sensitivity curve), **3 is discarded**, and
**5(b) composes with 1 — but only in its `pack_given` form.**

That last point is the round's sharpest aggregation finding. 5(b)'s *standalone* form (`pack_here`,
1.622×) would have been **self-defeating** on top of the graduated `colpack`: the root was
column-select-extracting `ht` col-0 tiles only for every receiving core to re-pack them. Graduating
the two as a fusion deletes that extract instead:

- the **root** elementwise-sums the `CW1` packed tiles straight into `cb_rms_mean` (one page) and
  **stops** — no extract;
- the **multicast** ships **1 page instead of `HT_BLOCK`** (and a single-page CB has a constant write
  pointer, so the fixed-offset contract no longer needs its reserve-then-pop-the-tail dance);
- **phase 4** does ONE SFPU pass over the packed tile (`RsqrtMeanColPacked`, `VectorMode::C` with
  `NVEC=8/STRIDE=1` — the stock full-face walk, covering packed columns 0..15 for all 32 tile rows),
  then `ht` cheap FPU column-selects to produce the col-0 tiles `BroadcastDim::Col` consumes.

**The `1/N` had to move, and that is the load-bearing correctness detail.** The extract now runs
*after* the rsqrt, and rsqrt is non-linear, so the mean can no longer ride the column-select scaler.
It folded into the rsqrt body as one fp32 SFPU multiply ahead of the `+eps` — which removes a bf16
DEST round trip the old column-select pack had, so the fused result is marginally **more** accurate,
not less. `cb_colsel`'s content goes 1/W → 1.0, and the body's `scale` is 1.0f on the unchanged
per-tile-row path.

Predicate, routed through **one** CT flag word (`_payload_flags`, bit0 colpack / bit1 bf16) read
verbatim by reader, writer and compute so the CB plan and every kernel's trip count can never
disagree — never duplicated literals:

```
colpack       : w_split and not two_stage and nw == 1 and 2 <= ht_block <= 16
partial_bf16  : w_split and not fp32_dest_acc_en
```

Everything outside them keeps the byte-identical fallback, so those cases provably cannot regress.

### Whole-op result

`DEVICE KERNEL DURATION [ns]`, pinned perf config, one fresh-cache profiled run per column. The final
column was re-measured **after** the pre-commit `clang-format` pass rewrote the kernels, so the number
is of the committed source (round 1 recorded this exact trap):

| case | Perf-2 base | +colpack | +fusion (final) | round × | ref | vs ref |
|---|---|---|---|---|---|---|
| **BLOCK 8192×1024** (focus) | **54 377** | 35 973 | **27 091** | **2.007×** | 25 640 | **1.06×** (was 2.12×) |
| WIDTH 32×1024 | 4 498 | 4 195 | 4 190 | 1.074× | 4 110 | 1.02× |
| WIDTH 32×2304 | 5 157 | 4 802 | 4 776 | 1.080× | 4 617 | 1.03× |
| WIDTH 32×5120 | 5 586 | 5 095 | 5 097 | 1.096× | 5 267 | **0.97× — now beats its ref** |
| WIDTH 32×7168 | 6 410 | 6 113 | 6 051 | 1.059× | 5 481 | 1.10× |
| decode 32×1024 | 5 274 | 4 818 | 4 904 | 1.075× | 9 149 | 0.54× |
| decode 32×2304 | 5 896 | 5 633 | 5 563 | 1.060× | 17 003 | 0.33× |
| decode 32×5120 | 7 804 | 7 505 | 7 561 | 1.032× | 75 825 | 0.10× |
| decode 32×7168 | 8 774 | 8 533 | 8 415 | 1.043× | 14 894 | 0.56× — **12.4× against the 7.0× requirement** |
| prefill 8192×1024 | 86 538 | 86 951 | 86 533 | 1.000× | 96 744 | 0.89× |
| prefill 8192×2304 | 180 641 | 182 100 | 182 350 | 0.991× | 211 345 | 0.86× |
| prefill 8192×5120 | 398 196 | 395 169 | 395 142 | 1.008× | 738 307 | 0.54× |
| prefill 8192×7168 | 555 021 | 553 021 | 560 487 | 0.990× | 1 032 281 | 0.54× |

- **The focus shape is essentially at its reference** — 27 091 against 25 640, from 2.12× over. The
  isolated bench predicted 36 014 for the colpack step and the real op landed 35 973 (0.1 %), so the
  measurements transferred exactly rather than by luck.
- **All four prefill cases are unchanged within noise**, which is the correct outcome: `cw == 1` there,
  so neither guard engages and the emitted program is byte-identical. Their 0.990–1.008× spread is the
  noise band, not the change.
- **9 of 13 perf cases improved; none regressed** outside the 2–3 % band.
- Final focus zones: `cmp_rsqrt` 10 052 MATH (from 35 112 at round start), `cmp_combine` 1 600 and
  **only on the 8 root cores** (from 6 054 → and 25 668 MAX before round 2), `cmp_colpack` 477,
  `cmp_square` 6 332, `cmp_scale` 4 595, `cmp_gamma_mul` 3 957. MATH is saturated at 25 413 of a
  26 461 ns TRISC kernel: **the op is now compute-bound at its own decomposition's floor.**

### Liveness proven, not assumed

A deliberate `static_assert(false)` inside the new phase-4 branch failed the build on **trisc0, trisc1
and trisc2** at the focus geometry's exact CT args
(`...,4,4,4,1,8,1,32,1024,1,8,8,1,1,3` — `WT=4 HT_BLOCK=8 CW=8 CW1=8 CW2=1 NW=1`, payload flag word
**3** = colpack|bf16), then was reverted and the case reconfirmed green. So the 1.332× cannot be a
measurement of dead code behind a stale cache.

### Raw LLK admitted

Three bypasses, each with its measured authorization at the use site so a later helper-usage pass
cannot "fix" it back and undo the win:

1. **A per-output-column reduce scaler.** Blackhole's REDUCE_ROW SUM is an MVMUL with the scaler in
   SrcA, so `dest[i,j] = Σ_k data[i,k]·scaler[j,k]` and the scaler's **face-row index picks the output
   column**. `ckl::reduce` / `ckl::reduce_mean` take ONE canonical scaler tile with no per-output
   index and cannot express a non-canonical scaler at all — a per-output-column scaler *is* the whole
   mechanism.
2. **`reduce_uninit()` between `tile_regs_commit` and `pack_tile`.** `reduce_init` programs a packer
   **edge mask** that zeroes every datum outside column 0, which would erase a column-packed tile.
   The pack needs the mask defeated; the extract needs it left ON (column-0-only output is exactly
   what `BroadcastDim::Col` reads) and cleared once afterwards. No helper exposes that seam.
3. **A raw L1 scaler-bank fill on the IDLE WRITER.** The two non-canonical banks are built with a NoC
   memset (`async_write_zeros`) plus ~528 word stores, on BRISC — because a naive RISC-V store loop
   cost **+20.5 µs** and made the whole idea read as 0.786×, and because on the reader it would sit in
   front of the shard publish compute's first pass waits on. Measured as fully hidden in the final
   profile: `wtr_selectors` 4 206 ns against a saturated MATH thread, so removing it would only
   lengthen `wtr_gather_hop`'s wait.

The SFPU body remains the stock accurate rsqrt (`_calculate_sqrt_body_<APPROX, RECIPROCAL, !FAST>`)
verbatim, one multiply wider. **The precision contract (`fp32_dest_acc_en`, `math_fidelity`,
`math_approx_mode`, dtypes) is untouched throughout** — no idea in this round was allowed to tune it,
and the one option that would have traded precision (sub-lever 5a at `fp32_dest_acc_en=True`) is
recorded above with its cost and explicitly guarded out.

### Instrumentation

Extended, never removed: **`cmp_colpack`** (compute) and **`wtr_selectors`** (writer) join the
permanent `MaybeDeviceZoneScope` set, and both new predicate-guarded paths report through the existing
`cmp_combine` / `cmp_rsqrt` zones, so per-stage observability covers every branch.

### Guard-set no-regression

One representative per distinct kernel path × layout × placement, plus slices on both sides of each
new predicate:

- `eval/golden_tests/rms_norm/` — run as four `memory_layout` slices (the full directory exceeds the
  10-minute foreground budget): INTERLEAVED **1500 passed / 420 xfailed**, HEIGHT_SHARDED **1167 /
  315**, WIDTH_SHARDED **870 / 630**, BLOCK_SHARDED **870 / 630** → **4407 passed, 0 failed,
  0 errors, 1995 xfailed, no XPASS drift**, counts **identical** to the pre-change run. Plus
  `test_op_loose` **19/19** (all five pinned sharded geometries) and `test_regression.py` **15/15**.
  Coverage spans TILE/RM × gamma/no-gamma × every dtype × **both `fp32_dest_acc_en`** (the axis
  `partial_bf16` guards on) × all four memory layouts.
- `tests/ttnn/unit_tests/operations/rms_norm/` — `test_rms_norm_fused_reduce.py` **14/14** including
  the **absolute all-ones checks** and `test_rms_norm_wsplit.py` **33/33** including the
  topology-agreement checks. The absolute gate is not ceremony here: this round's most dangerous
  failure mode is a per-row **rescale** from a mis-placed `1/N`, and PCC scores exactly that class
  ≥ 0.9998 (round 1 shipped a cut that scored 0.9998 while corrupting 12.5 % of tile-rows).
- Both W-split suites also green under **`--dev`** (watcher, NoC sanitizer, CB sanitization, LLK
  asserts) — non-optional for this scheme per Refinement 2's changelog, and doubly so for a round that
  changed a multicast payload size and a semaphore-gated landing window.

### Issues encountered

- **The remote JIT compile server ran out of disk** mid-round (`No space left on device` on
  `/tmp/tt-metal-cache` at `bgdepyc01:54778`), failing the precompile warm pass. Worked around with
  `--no-jit-server` (local compiles); recorded as an environment finding, not an op problem.
- **The golden directory no longer fits one foreground run** (> 10 min). Split by `memory_layout`
  into four `-k` slices, which is also a cleaner attribution boundary.
- The pre-commit `clang-format` hook again rewrote the kernels *after* a measurement run, so the
  committed source differed from what had been measured. Both graduations were **re-measured after
  the reformat** (focus 27 003 → 27 091, 0.3 %) rather than assumed equivalent.

### Perf 3 queue (measured, not speculative)

1. **The residual combine latency**, now ~5 µs of wait inside `cmp_rsqrt` (down from 27 300). The
   remaining transport is a `cw`-into-1 gather plus a 1-page broadcast; `combine_allgather_no_root`'s
   measured diagnosis says any further attack must cut **transaction count**, not bytes — its own
   untested follow-up (peer-major landing + `Accumulate::at` across peers, or a true HW multicast
   push via `mcast_pipe`'s `SenderPipe`) is the concrete next experiment.
2. **`pack_given_cskip`** — 2.351× on the stage against the graduated `pack_given_c`'s 1.930×, needing
   only the pack to place its values at EVEN columns (`ht_block ≤ 8`) so the parity-stride body covers
   the packed tile in 8 vectors instead of 16.
3. **Sub-lever 5a**, the bf16 statistic CBs: bit-exact at `fp32_dest_acc_en=False`, 1.026× on the
   stage (inside whole-op noise today) and **32 KB/core of freed L1** — worth revisiting when a lever
   exists that can spend that L1.
4. **`cmp_scale`'s UNPACK** (4 264–4 595 ns, still UNPACK-heavy) and **`cmp_square`'s 49 ns/tile**
   against the plain muls' 31 — the only two compute items now above their apparent floors.
5. **`row_rotate`** stays on the shelf as the `ht_block ∈ [2,4]` fallback if a future geometry makes
   that regime matter; its integration notes are preserved in
   `perf_experiments/combine_parallel_fold/`.

### Artifacts

`tests/ttnn/unit_tests/operations/rms_norm/perf_experiments/` — five idea dirs with their benches,
forked descriptors + kernels, measured tables and correctness gates:
`gather_payload_shrink/` (refreshed against the current op, ROUND 2 section appended to
`bench.py` + `measurements/results.tsv` + raw zone dumps), `combine_parallel_fold/`,
`combine_allgather_no_root/`, `combine_latency_hiding/`,
`phase4_packed_rsqrt_and_bf16_stats/` (incl. `graduation_snippet.cpp`, the design + raw-LLK
justification the fusion was built from). `probes/probe_p2_plan.py` — the host-side predicate
resolution table (which geometry gets which guard, at both `fp32_dest_acc_en` values). Breadcrumbs:
`agent_logs/blocking-perf-coordinator_breadcrumbs.jsonl`.

**Summary: all 5 ideas measured; 2 graduated (as one fusion), 2 superseded with measured reasons,
1 measured regression; focus shape 2.007× faster and now within 1.06× of its reference; no regression
on any cell.**
