# Operation Requirements: rms_norm

## Definition
- **Formula**: `out[..., j] = x[..., j] * gamma[j] / sqrt(epsilon + (1/W) * sum_k x[..., k]^2)`, with `gamma = weight.reshape(-1)[:W]` and W the logical last dimension
- **PyTorch Reference**:
  ```python
  def reference(input_tensor, weight, epsilon=1e-12):
      x = input_tensor.double()
      gamma = weight.double().reshape(-1)[: x.shape[-1]]
      return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + epsilon) * gamma
  ```
- **Import Path**: `from ttnn.operations.rms_norm import rms_norm`
- **Function Signature**:
  ```python
  def rms_norm(
      input_tensor: ttnn.Tensor,
      *,
      epsilon: float = 1e-12,
      weight: ttnn.Tensor,                          # [1, 1, ceil(W/32), 32], ROW_MAJOR
      memory_config: ttnn.MemoryConfig = None,      # None only (rule R13)
      program_config=None,                          # never read (rule R14)
      compute_kernel_config: ttnn.ComputeConfigDescriptor = None,  # or a device config (rule R4)
  ) -> ttnn.Tensor
  ```
- **Specification**: `eval/specs/rms_norm/spec.py` (authoritative). Target: Quasar on ttsim (no clock: perf gates are structural counts, see `.claude/references/backends/ttsim.md`, *Performance measurement*).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED placement**: memory_config ∈ {DRAM, L1}, sharding = interleaved, memory_argument = None
- **SUPPORTED weight**: gamma_dtype = bfloat16, gamma_layout = ROW_MAJOR
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, h_non_aligned}, rank = 4
- **Cores**: full grid (`compute_with_storage_grid_size()`, 8×4 here); regime `row_resident`: tile-rows dealt to clusters in whole rounds of `C_eff`, thread plan `THREAD_PLAN = (4, 4, 2)` with per-call `C_eff ∈ {1, 2, 4}` (ring-cap and row-count bound)
- **Compute config**: `default_compute_kernel_config()` = HiFi4, `fp32_dest_acc_en=True`, `math_approx_mode=False`; any fidelity / approx mode accepted (R3); device configs accepted (R4)
- **Golden baseline**: 42 / 633 passed (40 `supported_pass` = 34 PHASE0 cells + 6 numeric regression cases, + 2 rejections), 590 `xfail_expected`, 0 `supported_fail` / `xpass_drift` / `xfail_wrong_mode`

### [ ] Refinement 1 — Numerical configurability: float32 input and weight, 16-bit DEST

**Goal**: add `ttnn.float32` to `SUPPORTED["dtype"]`, `False` to `SUPPORTED["fp32_dest_acc_en"]`, and `ttnn.float32` to `SUPPORTED["gamma_dtype"]` — all four input/weight dtype combinations at both DEST widths, except the float32 + `fp32_dest_acc_en=False` corner that rule R1 refuses in every phase (already enforced in `validate()`, ahead of the SUPPORTED loop; keep it there). Includes bringing the float32 rows of W = 8192 inside the ring extent cap so every float32 shape of `INPUTS` runs (no SUPPORTED cell may be refused by `select_regime`).

**Implementation skill**: /numeric-formats-metal, /memory-budget-metal

**Verifier notes**:
- First because Refinement 2's padding mask (w_non_aligned) is dtype-dependent: with both dtypes in place it is written once.
- Already dtype-aware: the per-row statistic rings follow DEST width (`_stat_dtype`), the input/output rings take the tensors' dtypes, `tile_bytes` comes from `buffer_page_size()`, `c_fit` follows `tile_bytes`.
- **float32 gamma**: Quasar's `UnaryBcast` refuses a 32-bit page (`bcast.h:113`), so the gamma ring is pinned to `GAMMA_FORMAT = bfloat16` and the reader's `static_assert(gamma_page_bytes == 2 * face_row_bytes)` will trip on a 128-B float32 weight page — that is where the change lands. Prefer an exact path (the raw `sfpu_mul_bcast_row` from `sfpu_binary_bcast.h`, exact in fp32 on the simulator) over narrowing gamma to bf16 in the reader; if narrowing is taken, say so in the changelog (bf16 rounding of gamma, ≈0.2% per element).
- **float32 at W = 8192**: a float32 tile-row of Wt = 256 is 1,048,576 B, 16 B over the 1,048,560-B ring cap even at C = 1 — cells (1,1,32,8192), (1,32,8192), (64,8192). Two ways: (a) keep the row resident across two input rings (half a row each; the input still crosses DRAM once — rank 1 of the design's traffic ranking, preferred), or (b) the design's deferred `row_streaming` regime (input crosses DRAM twice). Either way the regime is picked by the predicate in `select_regime`; record the choice and its DRAM-crossing cost in `l1_ledger.md`'s regime / data-movement rows.
- **16-bit DEST**: `sum_squares` folds Wt tiles into one DEST tile. The spec's DECISIONS measured a truncating bf16 fold at 2% (W=1024) to 21% (W=8192) off, beyond the 0.13 relative RMS of `legacy_tolerances[(bfloat16, False)]`. If wide bf16/False cells fail as `numerical-precision`, leave them failing (no EXCLUSIONS) and file `Refinement 1b` for a blocked fold (Σx² over blocks of k tiles into DEST, partials packed to `row_stat` and added) — the `row_reduce_accumulate` catalog entry.
- Keep the thread plan and the grid spread on the new paths (C_eff derives from the per-dtype `tile_bytes`); the thread-plan tests (`test_rms_norm_plan.py`) gain float32 cases at C_eff = 1, 2, 4.

**Done when**: every golden cell with `dtype=float32`, `fp32_dest_acc_en=False` or `gamma_dtype=float32` whose other axes are in SUPPORTED passes (R1 cells stay refused); the numeric regression cases still pass; 0 `xpass_drift` / `xfail_wrong_mode`; any remaining `supported_fail` is `numerical-precision` and named in `Refinement 1b`.

### [ ] Refinement 2 — Shape generality: W not a multiple of 32, ranks 2 and 3

**Goal**: add `"w_non_aligned"` to `SUPPORTED["alignment"]` — the padding columns of the last tile column enter neither Σx² nor the count (rule R6), and the weight elements past W are never applied to a logical column (R5) — and add `2` and `3` to `SUPPORTED["rank"]`.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- Ranks 2 and 3 should be validate-only: `num_tile_rows = buffer_num_pages() // Wt` and `Wt = padded_shape[-1] // 32` are rank-agnostic. Confirm on (17, 64), (1, 17, 128) and (4, 128, 47) (all three axes at once).
- The mean already divides by the logical W (`inv_w_bits`); what is missing is the mask. Tile padding is not guaranteed zero: mask, don't assume. The input ring is implicit sync, whose credits post when the read lands — a reader-side zeroing of tile Wt−1 needs that entry on explicit sync (and the ring's other entries stay implicit, the `reader_mixed.cpp` pattern), or mask in compute before the square (a column-mask tile the reader generates once, like the scaler, multiplied into the last tile of each row). Either must hold for both dtypes from Refinement 1 and under every `C_eff`.
- The output's padding columns are don't-care; gamma's elements past W multiply only padding columns.
- Performance-conformance: the mask must not add a per-tile handshake to the reader's input stream (it touches one tile per row).

**Done when**: every golden cell with `alignment=w_non_aligned` or `rank ∈ {2, 3}` passes, the R6 check (`{"alignment": "w_non_aligned"}` cells) included; golden suite otherwise green with 0 loud categories.

### [ ] Refinement 3 — Spread the few-tile-row shapes (the model calls) across NEOs and clusters

**Type**: perf

**Goal**: the four traced model calls — (1,1,32,2048), (1,1,32,128), (1,1,8,128), (1,1,32,2560) — and every shape with fewer tile-rows than NEO slots (all one- and two-tile-row shapes, e.g. (1,1,32,8192), (32,4096), (64,8192)) run on **one NEO of one cluster**: 1 of 128 compute threads, because `row_resident` cannot split the dependent `w_tiles` axis. Build the design's deferred `w_split` regime (op_design.md, Regimes): the row's Wt tiles dealt over a group of G clusters × C NEOs; each NEO's `sum_squares` tile (`row_stat`) is the partial; partials leave through a compute→writer ring to the group root, the root sums them, runs `inv_rms`, multicasts the tile back; each member normalises its chunk. Selected by predicate (`num_tile_rows < C_eff · num_cores` and Wt ≥ 2); `row_resident` stays for every other shape. A ⭐⭐⭐ T3 scheme-change: the whole phase. Relevant catalog entries in `ttnn/ttnn/operations/examples/master.md`: `width_split`, `tensix_all_reduce`. No SUPPORTED change.

**Verifier notes**: no loose case is perf-flagged in the specification, so the region is free-selected: the model calls are the op's reason to exist and leave >99% of the grid idle. Probe first that the multicast helper (`ttnn.Mcast` + `McastArgs`) and the semaphores it needs build and run on Quasar/ttsim (16 semaphores per cluster); if not, the gather is a unicast to the root and the broadcast G−1 unicasts — same topology, record it. The combine changes the summation order: re-run the precision baseline. Under `w_split`, a float32 W=8192 row splits into chunks well under the ring cap, so this regime also serves the shapes Refinement 1 had to fit.

**Done when** (structural, measured by a test reading the program artifacts; ttsim has no clock, so the gain in time is unconfirmed until `ttsim-perf` or hardware measures it): compute threads at work on (1,1,32,2048) go from **1 to ≥ 16** (and on (1,1,32,128) from 1 to the Wt-limited maximum, 4), with input DRAM crossings still 1 and cross-core traffic ≤ G·C partial tiles + one multicast tile per row; golden suite green; precision baseline within its gates; session tests (after another program, before `to_layout`) pass for the new program; no regression on the guard set — one shape per `row_resident` C_eff (1, 2, 4), DRAM and L1 — whose structural counts are unchanged.

### [ ] Refinement 4 — Row-statistics path and cluster occupancy on `row_resident`

**Type**: perf

**Goal**: two ⭐ T1 levers on the built `row_resident` path, from the design's perf lamps (op_design.md, *Perf lamps*) and `ttnn/ttnn/operations/examples/master.md`:
1. **Fold `inv_rms` into the reduce** as its `post_reduce_op` (multiply by 1/W, add epsilon, rsqrt on the DEST tile) and drop the `row_sum` ring, with exactly one output tile per `reduce<>` call, which is this op's case per tile-row (call per row if `rows_per_block` > 1). Count: packs of row statistics per tile-row 3 → 2; rings 6 → 5.
2. **Spread rounds over more clusters**: `C_eff` is today the largest c that fits, so (2,4,128,512) — 32 tile-rows — runs on 8 clusters × 4 NEOs. Pick the smallest c with `ceil(units / c) ≤ cores` instead (design lamp "grid synchronization / occupancy"): the same 32 NEOs on 32 clusters, 4× the NoC endpoints and DM harts for the reads and writes. Count: clusters engaged on (2,4,128,512) 8 → 32 at an unchanged NEO count.
No SUPPORTED change.

**Verifier notes**: independent of Refinement 3 (it touches `row_resident` only), but keep it after so `w_split` is built against the final row-statistics path. If the post-op inside `reduce<>` gives wrong values on this part, record it in the changelog and ship lever 2 alone (`[~]`). Lever 2 is a one-line change to `select_regime`; the thread-plan tests that pin `C_eff` per shape (`REGIME_CASES`) change with it — update them to the new rule, keeping one case per C_eff.

**Done when** (structural; gain in time unconfirmed on ttsim): both counts reached and read by a test; golden suite green; precision baseline within its gates; no regression on the guard set (C_eff = 1, 2, 4; DRAM and L1; and after Refinement 3, one `w_split` shape).
