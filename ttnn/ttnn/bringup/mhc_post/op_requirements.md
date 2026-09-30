# Operation Requirements: mhc_post

## Definition
- **Formula**: for every token t and output stream j, `X'[t, j·C:(j+1)·C] = post[t, j]·F[t, :] + Σ_i comb[t, i·n + j]·X[t, i·C:(i+1)·C]` (comb applied transposed).
- **PyTorch Reference**:
  ```python
  def mhc_post_ref(f, x, post, comb):
      n = post.shape[-1]; lead, C = tuple(f.shape[:-1]), f.shape[-1]
      ff = f.float().reshape(-1, 1, C); xx = x.float().reshape(-1, n, C)
      pp = post.float().reshape(-1, n, 1); mm = comb.float().reshape(-1, n, n)
      return (pp * ff + torch.einsum("tij,tic->tjc", mm, xx)).reshape(*lead, n * C)
  ```
- **Import Path**: `from ttnn.bringup.mhc_post import mhc_post, default_compute_kernel_config`
- **Function Signature**: `mhc_post(input_tensor: ttnn.Tensor, residual: ttnn.Tensor, post: ttnn.Tensor, comb: ttnn.Tensor, *, compute_kernel_config: ttnn.ComputeConfigDescriptor = None) -> ttnn.Tensor`. The inputs are F `(..., T, C)`, X `(..., T, n·C)`, post `(..., T, n)` float32 and comb `(..., T, n·n)` float32, all TILE and DRAM interleaved. The output X' has X's shape and dtype.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [float32] (X and X')
- **SUPPORTED sublayer_dtype**: [float32] (F)
- **SUPPORTED layout**: [TILE]
- **SUPPORTED fp32_dest_acc_en**: [True] (False is refused, and is outside TARGET)
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, h_non_aligned}
- **Cores**: full grid: regime `flat_stream`, flattened (token-tile row, column tile) units split with `split_work_to_cores(..., row_wise=True)` (110 cores on Blackhole p150)
- **Compute config**: caller's `math_fidelity` / `math_approx_mode` honoured; fp32 DEST required; fp32 SFPU mix (`MulBinary` + n× `Addcmul`), `UnpackToDestFp32` on fp32 CBs
- **Golden baseline**: 84 / 208 passing: 84 supported_pass, 123 xfail_expected (all bf16 cells), 0 supported_fail / xpass_drift / xfail_wrong_mode
- **Perf**: fp32 loose sweep at 1.22–2.39× the DRAM roofline (T640 C7168 fp32: 537 µs vs 403 µs target; the composite is 3781 µs)

### [x] Refinement 1 — bfloat16 residual streams and sublayer output

**Goal**: add `ttnn.bfloat16` to `SUPPORTED["dtype"]` (X and X') and to `SUPPORTED["sublayer_dtype"]` (F), independently, so all four (dtype × sublayer_dtype) combinations run. Both stay with fp32 DEST, and post / comb stay applied as fp32 values. This moves the 75 `test_op` bf16 cells and the 48 bf16 loose cases from xfail to passing, and un-skips `test_regression.py::test_depth_chain[bf16]` (122 wraps, drift ≤ 1e-3).
- **Formats:** each CB's data format (F, X, X') and its `UnpackToDestFp32` tag are already derived from the tensor dtype on the host. A bf16 CB stays `Default`, because bf16 enters DEST exactly through srcA.
- **Block size:** `block_col_tiles_fit` rederives automatically: 23 at bf16/bf16, 12 at bf16 F / fp32 X.
- **Expected changes:** the expected change is dropping the `validate()` gate. Any kernel change should be limited to making the fp32 → bf16 output pack round-to-nearest-even (below).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- **Perf-1 anchor.** This single refinement lands the full perf-focus contract: bf16/bf16, TILE, DRAM interleaved, `fp32_dest_acc_en=True`, HiFi4 (`feature_spec._PERF_FOCUS`: (640, 7168), (640, 1792), (1280, 4096) bf16). No second generality refinement exists, because TARGET has no other gap.
- **Rounding risk (design Key Risks).** Packing the fp32 DEST result to a bf16 page must be RNE. If the packer truncates, the bf16 signed-bias gate (1e-5 + 6σ) and the 122-wrap drift bound fail with a *systematic* shrink. The fix is then an SFPU RNE rounding in DEST before the pack. It must not be an EXCLUSION: those cells produce an interpretable bias metric.
- **No wrapper.** Never convert F or X with `ttnn.typecast` / `to_dtype` at the entry point. The prompt's one-dispatch rule makes that a hard violation, not a partial-tick option.
- **Performance bar.** The bf16 path must keep the Phase 0 performance properties: full grid, coarsest-fit B, one barrier per block on both reader and writer. Refinement 2 optimises exactly this path.
- **Ledger.** Update `l1_ledger.md` page formats (bf16 rows) and the B_fit numbers if anything moves.

**Done when**: all 4 dtype combos pass on all 25 INPUTS shapes (both alignments), all 48 bf16 loose cases pass, `test_depth_chain[bf16]` passes, `verify_supported` shows 0 xfail_expected and 0 loud categories, and the precision baseline gains bf16 rows (PCC, rel-RMS, signed bias, ratio spread).

**Outcome**: no kernel change: SUPPORTED widened, and the packer's fp32 → bf16 rounding is RNE (bias ≤ 1.3e-5 rel; 122-wrap drift passes). The golden bf16 slice is 123/123 (75 test_op + 48 loose), and `test_depth_chain[bf16]` passes. Device time on 110 cores (bf16 / fp32): T640 C7168 388.0 / 537.6 µs, T640 C1792 151.0 / 186.6 µs, T1280 C4096 435.3 / 628.1 µs. Compute is now the bound, which is Refinement 2's job.

### [x] Refinement 2 — Speed up the compute-bound SFPU mix on the perf-flagged bf16 profiles

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags **T=640, C=7168, bf16/bf16** (DeepSeek-V4 per device, unsharded) as the mandatory perf target, along with T=1280 C=4096 bf16 (GLM-5.3-Flash) in the same regime. Both carry `perf_regime="dram"` and `target_ns` = the DRAM roofline at 80% of 512 GB/s (~202 µs and ~231 µs respectively). Neither changes SUPPORTED.
- **Why it's compute-bound:** after Refinement 1 these cells are compute-bound. The Phase 0 ablation at T640 C7168 fp32 measured the compute kernel at ~3.3 µs per output tile (556 → 415 µs with compute stubbed, 163 output tiles per core), and the SFPU cost does not halve at bf16 while the DRAM time does.
- **What to measure:** reach roofline on the flagged cells by cutting per-output-tile compute cost ~2.5×. Measure the exact flagged config (bf16/bf16, fp32 DEST, HiFi4).
- **Candidate levers** (the planner's Perf lamps; the implementer owns the choice and measures each):
  - (a) Keep the n+1 coefficient tiles of one output stream resident in DEST across the column walk, dropping n+1 of the 2(n+1) `copy_tile`s per output tile. SyncFull fp32 DEST has 8 slots.
  - (b) Hoist the per-tile element re-inits by making the chain uniform: all n+1 terms as `Addcmul` onto an accumulator, or a thin raw-LLK block op with inits hoisted.
  - (c) bf16 X only: FPU multiply-accumulate with each fp32 coefficient split into exactly representable bf16 parts (hi + mid + lo), accumulating in fp32 DEST.
- **Catalog pointers:** `ttnn/ttnn/operations/examples/master.md`: `compute_fusion` (SFPU mul is ~0.58× FPU; DEST reuse is only a win for SFPU consumers), `tensix_all_reduce_compute` / `eltwise_l1_vs_dest_accumulate` (DEST-resident accumulation, init once per batch), `compute_block_size`.

**Verifier notes**:
- Depends on Refinement 1: it must measure the real bf16 config, never an fp32 stand-in.
- This is a compute-kernel restructure (⭐⭐–⭐⭐⭐). Keep it to the compute levers. The reader / writer overlap and expansion levers are Refinement 3, and they only matter once compute stops being the bottleneck.
- **Hard precision constraint (prompt MUST):** lever (c) must pass the signed-bias gate. FPU operands must be exactly representable (bf16 X is; the coefficient parts must be), and no fp32 operand may enter srcA/srcB.
- Any change to `cb_coef_bcast`'s consumer (e.g. coefficients kept in DEST) should also re-justify its size and depth in `l1_ledger.md`. This is the folded-in ledger item from verification.
- The mixed bf16-F / fp32-X chain reconfigures the unpacker per element; the new schedule should not regress it.

**Done when**: measured device-ns on T640 C7168 bf16 (and T1280 C4096 bf16) improves toward `target_ns`, with every bf16 golden and regression cell still green (signed-bias gate and 122-wrap drift included) and bitwise determinism intact. There must be no device-ns regression across the config-spanning guard set: one representative per distinct path, i.e. {fp32/fp32, bf16/bf16, bf16 F / fp32 X} × {tile_aligned, h_non_aligned} × {small C (1792), large C (7168)}.

**Outcome**:
- **Measured** (device-ns, 110 cores):
  - T640 C7168 bf16: 388 → 306 µs.
  - T1280 C4096 bf16: 429 → 368 µs.
  - Every guard-set cell is faster (2–19%, fp32 and mixed included).
  - Precision, the 122-wrap drift and determinism are unchanged.
- **Bottleneck:** it was the per-element fp32-unpack-to-DEST ↔ bf16-srcA switch. There is now one fused `WeightedSum` SFPU pass per output tile, in a SyncFull DEST window with half-packed coefficient tiles. Compute's excess over the DM floor (224 µs with compute stubbed) fell from ~155 to ~80 µs. What remains is the serialized per-output-tile window: 3 coefficient unpack-to-DEST copies + 5 data copies + SFPU + pack, with no pack overlap under SyncFull.
- **Next:** find a coefficient form that survives DEST release, or a denser coefficient layout that allows K > 1 columns per window. Not attempted: every pack release ZEROACCs all of DEST, and at n=4 the SFPU lane geometry needs half a tile per coefficient.
- **DM floor vs target** (224 vs 202 µs) is Refinement 3's scope.

### [x] Refinement 3 — Reader / writer overlap and coefficient-expansion cost on the perf-flagged bf16 profiles

**Type**: perf

**Goal**: `feature_spec.LOOSE_CASES` flags **T=640, C=1792, bf16/bf16** (DeepSeek-V4 under TP4, `target_ns` ≈ 50 µs) as a mandatory perf target. The large flagged cells (T640 C7168, T1280 C4096 bf16) become data-movement-bound once Refinement 2 removes the compute bottleneck. Speed up the data-movement side of all three flagged shapes toward `target_ns`. No SUPPORTED change.
- **Where the time goes at small C:**
  - The Phase 0 ablation at T640 C1792 fp32 measured 223 µs baseline vs 101 µs with compute + expansion stubbed. The reader's per-segment coefficient load (a serial raw-tile read + barrier, then (n+n²)·1024 L1 stores) is a large share of it.
  - Each core owns ~10 units, so every segment is a single block of B = 10–11 columns: fill and drain are never overlapped.
- **Candidate levers** (planner Perf lamps; the implementer measures and picks). These are ⭐ T1 / ⭐⭐ T2 knob-turns on the exposed block surface:
  - co-tune `block_col_tiles` (`MAX_BLOCK_COL_TILES`) with `DEPTH_IN` / `DEPTH_OUT`. The chunk floor is one whole column (n+1 tiles); coarser amortises. Smaller blocks buy overlap only when a segment has ≥ 2 blocks.
  - hide or shrink the coefficient load: overlap the raw read with the first data block instead of barriering first, read only faces 0/2 of the raw tiles (2 × 1 KB), or move the expansion to BRISC.
  - `row_wise` placement re-measured against the default.
  - raise `L1_BUDGET_BYTES` toward the arch's usable L1, only if a deeper buffer measurably helps.
- **Catalog pointers:** `master.md` `double_buffer`, `split_reader`, `noc_placement`, `tile_reorder`.

**Verifier notes**:
- Runs after Refinement 2. At the large-C flagged shapes the data-movement levers are invisible while compute dominates (they measured as a slowdown in Phase 0: B=4/2 → 665/699 µs vs 560 µs fp32), so they must be re-measured on the post-Refinement-2 kernel.
- The expansion rewrite must keep coefficient values bit-exact. They are applied as fp32; any srcB path truncates to tf32.
- Moving the expansion to BRISC makes the writer the producer of `cb_coef_bcast`. Keep a single producer per CB, and keep the shared `SegmentWalker` derivation as the one source of segment and block boundaries.

**Done when**: measured device-ns improves on T640 C1792 bf16 toward its `target_ns`, and on T640 C7168 / T1280 C4096 bf16 beyond the Refinement 2 result. The golden suite and regression tests must stay green, with no device-ns regression across the same config-spanning guard set as Refinement 2.

**Outcome**:
- **Measured** (device kernel ns, 110 cores, Blackhole, bf16/bf16):
  - T640 C1792: 131.5 → 66–71 µs (target ≈ 50).
  - T640 C7168: 306.6 → 238 µs.
  - T1280 C4096: 370.0 → 267 µs.
  - Guard set: every cell is faster than after Refinement 2 (C1792 T640 fp32/bf16/mixed: 171/133/153 → 138/67/128 µs; C7168 T1000: 788/460/760 → 706/398/677 µs).
- **Levers:**
  - A tight, unrolled expansion store loop. This was the dominant cost: 54 of 131 µs at C1792, at ~3.5 cycles per store before and ~1.1 after.
  - The expansion moved to the writer (`COEF_EXPANDER` knob, default `"writer"`), with look-ahead for the next segment.
  - Per-stream coefficient pushes, so compute starts on stream 0 while the later streams are still being expanded.
  - The raw coefficient read folded into block 0's barrier.
  - A block-size policy: `MAX_BLOCK_COL_TILES = 8`, `MIN_BLOCKS_PER_CORE = 3`.
  - Deeper `DEPTH_IN` (3) was measured slower and was not adopted. `row_wise` = False measured the same.
- **Bottleneck now:**
  - The ablations put the DM floor with compute stubbed at 187 / 50 / 209 µs, which is at or below target_ns, and compute with DM stubbed at 184 / 62 / 207 µs. The stages are balanced.
  - The wall exceeds both because per-core DRAM service is uneven. At C7168 there is a physical-row gradient of about 170 µs (grid row 11) to 215 µs (grid row 2) that follows core position, not unit order. The kernel time is the slowest core.
  - At C1792, the expansion on the writer is still on the DM path of cores that straddle a row boundary (two expansions each).
- **Next (not attempted):**
  - A position-weighted work split, or a compute-side cut (R2's remaining DEST-window cost), to take the balanced stages below the imbalance.
  - Splitting the expansion across both DM RISCs by segment parity. This needs two coefficient CBs and a duplicated compute instantiation, a TRISC code-size risk.
