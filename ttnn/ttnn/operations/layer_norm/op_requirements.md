# Operation Requirements: layer_norm

## Definition
- **Formula**: for every row `x[..., :]` of the input (W = last dimension):
  `mean = (1/W)·Σ_k x_k`, `var = (1/W)·Σ_k (x_k − mean)²` (biased),
  `out_j = (x_j − mean) · rsqrt(var + epsilon) · gamma_j + beta_j`, with `gamma = weight[0, 0, :]`, `beta = bias[0, 0, :]`
  (the first row of the `[1, 32, W]` tile-layout weight and bias; the other 31 rows are never read, rule R8).
- **PyTorch Reference**: `eval.specs.layer_norm.spec.reference` —
  `torch.nn.functional.layer_norm(x, [W], weight=weight.reshape(-1, W)[0], bias=bias.reshape(-1, W)[0], eps=epsilon)`, in float64.
- **Import Path**: `from ttnn.operations.layer_norm import layer_norm`
- **Function Signature**:
  ```python
  def layer_norm(input_tensor: ttnn.Tensor, *, epsilon: float = 1e-12, weight: ttnn.Tensor, bias: ttnn.Tensor,
                 compute_kernel_config: ttnn.ComputeConfigDescriptor = None) -> ttnn.Tensor
  ```
  `compute_kernel_config` also takes tt-metal's device config (`ttnn.WormholeComputeKernelConfig`, rule R4);
  `None` resolves through `default_compute_kernel_config()` (HiFi4, approx off, `fp32_dest_acc_en=True`, rule R2).

The specification (`eval/specs/layer_norm/spec.py`) governs. Target: Quasar on the ttsim simulator — Metal 2.0
`ttnn.ProgramSpec` only; perf gates are **structural counts** (the backend has no clock, `.claude/references/backends/ttsim.md`).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True, False]
- **SUPPORTED layout**: [TILE]; **affine_dtype**: [bfloat16]; **affine_layout**: [TILE]
- **SUPPORTED placement**: memory_config = [DRAM], sharding = [interleaved]
- **SUPPORTED shape-derived axes**: alignment = [tile_aligned], rank = [4]
- **Regime**: `row_resident` — a whole tile-row resident on one NEO (two-pass centred variance, no DRAM re-read)
- **Cores**: full grid (8×4 clusters on Quasar), thread plan `THREAD_PLAN = (4, 4, 2)` clamped per call by
  `select_blocking()` (C ∈ {1, 2, 4}: spread rows over clusters first, then NEOs)
- **Compute config**: caller's `math_fidelity` / `math_approx_mode` / `fp32_dest_acc_en` (any fidelity, R3); intermediates in Float32 under fp32 DEST, bfloat16 otherwise
- **Golden baseline**: 26 supported_pass (22 PHASE0 matrix cells + 4 numeric regression cases), 0 supported_fail, 522 xfail_expected, rejections X1/X2 passing, 0 xpass_drift / 0 xfail_wrong_mode (per `verifier_report.json`)

### [ ] Refinement 1 — Input shape and placement generality: W padding mask, ranks 2–3, H non-aligned, L1 input

**Goal**:
- add `w_non_aligned` and `h_non_aligned` to `SUPPORTED["alignment"]`;
- add `2` and `3` to `SUPPORTED["rank"]`;
- add `"L1"` to `SUPPORTED["memory_config"]` (interleaved; output in the input's buffer, rule R12).

All three on the existing `row_resident` kernel, native in-kernel (no `ttnn.to_layout` / `fill_implicit_tile_padding` wrapper).
The real kernel work is rule R6: the padded columns of the last w-tile must enter neither the row sum nor the
variance. Today `centre` writes `0 − mean` into the padded columns and `sq_sum` adds them. Fix it with a
constant mask tile (1 for logical columns, 0 for padding) applied to the last w-tile in `centre` (so padded
`c` = 0 before `sq_sum` and `affine`) and in `row_sum` (padding is not guaranteed to be zero). Keep the scaler
at the true count `1/W`, as it is now.

**Implementation skill**: /memory-layouts

**Verifier notes**:
- Evidence (verifier probes, `validate()` bypassed, `tests/.../layer_norm/probes/`):
  - h_non_aligned ((17,64), (4,8,47,256)), rank 2 ((512,1024), (17,64)), rank 3 ((4,128,512), (1,512,1024)) and an L1 input ((1,1,128,1024), (1,512,1024), (1,1,64,4096) fp32 acc) already compute correctly (PCC ≥ 0.99999). The output lands in L1 for an L1 input. For these values the work is the gate, the fit predicate and the tests.
  - w_non_aligned **is wrong today**. On standard-normal inputs it scores PCC 0.99999, because the row mean is near 0. On uniform [0, 1) inputs W=17 gives PCC 0.955 / rel-RMS 0.44 and W=50 gives 0.991 / 0.24, with the output scaled down (ratio median 0.87). Test the mask on uniform data; the golden normal cells alone will not catch a missing mask.
- L1 fit (ledger finding folded here): `fits()` budgets `get_max_worker_l1_unreserved_size() − 32 KiB`, which ignores L1 tensor pages. The largest INPUTS L1 cell holds 128 KiB per bank of input + output (float32, after Refinement 2). Before claiming `"L1"`, make the predicate subtract what the allocator holds in L1, or prove the 32 KiB reserve covers every INPUTS cell. No skill covers placement; nothing is sharded here.
- Rank folding is host-only: `num_tile_rows = buffer_num_pages / Wt` already counts per-image padded H. Keep `row_of` (in `kernels/layer_norm_rows.hpp`) the single source of the dealing.
- Hardest-first ordering (alignment is a data-access change; Refinement 2 is a precision knob). Refinement 2's float32 path must then carry the mask, so build the mask in the intermediate format (`t_acc`), not in a bf16-only form.

**Done when**: every golden cell with alignment ∈ {w_non_aligned, h_non_aligned}, rank ∈ {2, 3} or memory_config = L1 (bf16 input, bf16 affine, both accumulators) passes. A uniform-input test at W ∈ {17, 50} meets the PCC 0.999 gate with the median got/true ratio at 1.0. The thread-plan, session and precision-baseline tests stay green.

### [ ] Refinement 2 — Numerical configurability: float32 input and float32 weight/bias

**Goal**: add `ttnn.float32` to `SUPPORTED["dtype"]` and `ttnn.float32` to `SUPPORTED["affine_dtype"]` (all four mixes of input × affine dtype, both accumulators for bf16 input). Keep rule R1: a float32 input with `fp32_dest_acc_en=False` is refused (`validate()` already raises before the SUPPORTED check). Page formats are derived from the dtypes in one place (`pages_of` / `fmt`), and so are the unpack modes of Float32 rings.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**:
- Quasar blocks the obvious path in two places (see the compute kernel's header):
  - The γ/β prologue expands `weight`/`bias` row 0 with `UnaryBcast<Row>`. A **32-bit unary broadcast is refused on Quasar** (`bcast.h` LLK_ASSERT), so a float32 `weight` cannot go through the current expansion. Two replacements:
    - (a) the reader replicates row 0 into all 32 rows while filling an explicit-sync ring. The DM harts are idle, and this is the design's "β expansion on compute" perf lamp.
    - (b) an FPU path that takes row 0 through srcB with a Row broadcast.

    Either way the γ/β rings stay in `t_acc`, because the affine chain has one ALU format.
  - FPU binaries read their operands through the source registers (tf32 on Quasar), and the FPU reduce accumulates through tf32. The float32 cells are graded by the `('float32', 'default')` entry (PCC 0.99975, frob 0.02, ULP ≤ 2.5e6 for W ≤ 200), which tf32 intermediates should meet. If a cell misses it, the exact float32 route is the raw SFPU calls (`sfpu_reduce`, `sfpu_*_bcast_*`).
  - `UnpackToDest` applies to 32-bit pages only.
- `fits()` already re-evaluates with 4096 B input pages: the `cb_input_block` ring cap gives Wt ≤ 255 at C = 1, which covers W = 4096.
- No `bfloat8_b`: the part has no path for it (target page).

**Done when**: every golden cell with dtype = float32 or affine_dtype = float32 that is not refused by rule R1 passes, with Refinement 1's alignment / rank / L1 values included. Float32 + `fp32_dest_acc_en=False` is refused with `NotImplementedError`. The precision baseline is extended to float32 and recorded in the changelog.

### [ ] Refinement 3 — Fill the grid on the model's few-row wide calls (W-split with a cross-cluster combine)

**Type**: perf

**Goal**: the model's two traced calls, (1,1,64,4096) and (1,1,128,1024) (bf16, 16-bit DEST, DRAM, rank 4, already supported), have 2 and 4 tile-rows. Today 2 and 4 of the machine's 128 NEOs do all the work, and every engaged NEO reads all of γ and β. Build the design's deferred `w_split_combine` regime: split Wt across clusters; each slice owner reduces its columns to per-row partials, (Σx, Σc²) or Chan's `(mean_i, M2_i)`; one combiner per row merges them in one round (`M2 = Σ M2_i + Σ n_i (mean_i − mean)²`) and sends `(mean, rstd)` back; then the slices run `centre` and `affine` on their own columns. Select it when `num_tile_rows < num_clusters` (op_design.md, Regimes). Relevant catalog patterns, in `ttnn/ttnn/operations/examples/master.md`: `width_split` (the split) and `tensix_all_reduce` / `mcast_topology` (the combine). No SUPPORTED change.

**Verifier notes**: this is a scheme change (a cross-core combine on a dependent axis), so it stands alone. The mechanism has no worked example on Quasar at this base. `ttnn.experimental` does expose `SemaphoreSpec` / `SemaphoreBinding` and `ScratchpadSpec` in `ProgramSpec` (checked), but no multicast between clusters is demonstrated in `.claude/references/quasar_op_examples/`. Start from unicast NoC writes plus a semaphore increment per (row, slice), and keep the tagged/untagged transaction-id rules of the target page (an explicit-sync kernel tags every transfer). The per-row partial tiles in `cb_row_partial` are the combine payload, and `row_resident` stays as the per-slice kernel and as the regime for many-row shapes. If the combine cannot be made to run here, ship `[~]` with the split's host side and its structural test, and file `Refinement 3b` naming the blocker.

**Done when** (structural counts, read off the op by a test; the gain in time is **unconfirmed** until a timed backend or hardware measures it):
- **NEOs computing**: on (1,1,64,4096) from 2 to ≥ 32, and on (1,1,128,1024) from 4 to ≥ 32.
- **DRAM tile reads of weight + bias**: on (1,1,64,4096) from 512 (2 NEOs × 2 × 128) to ≤ 256 (each slice reads only its columns).
- x is still read once and the output written once.
- Both model calls still meet PCC 0.999 with the 16-bit accumulator.
- The golden suite is green.
- No structural count regresses on the guard set: one shape per kernel path × placement × dtype (a many-row aligned DRAM shape, a w_non_aligned shape, an L1 shape and a float32 shape).

### [ ] Refinement 4 — Stop re-reading γ/β per NEO on many-row shapes

**Type**: perf

**Goal**: in `row_resident` every engaged NEO reads all of `weight` and `bias` from DRAM (E× crossings). On (1,1,2048,256), 64 NEOs read 1,024 γ/β tiles to normalise 512 input tiles, which is 2× the input's own traffic (`l1_ledger.md`, Data-movement budget). Use the design's deferred `affine_fanout` successor (reader-only; the compute kernel and ring layout are unchanged):
- (a) Intra-cluster: one reader hart reads each γ/β tile once and fills the other NEOs' STRIDED slices with L1→L1 NoC copies, since no ALL ring is used here.
- (b) Cross-cluster: once per op, multicast or forward using Refinement 3's semaphore machinery (`shared_input_reuse` in `master.md`).

Optionally move the row-0 replication to the reader (perf lamp "β expansion on compute"), which removes the compute prologue's 2·Wt broadcast ops per NEO. No SUPPORTED change.

**Verifier notes**: a reuse scheme change. Variant (a) stands on its own; variant (b) reuses Refinement 3's mechanism, so take it after Refinement 3. If Refinement 2 moved the γ/β expansion to the reader, build on that rather than adding a second path. Validate by building the variant and counting reads, not by ablation.

**Done when** (structural counts; time gain **unconfirmed** on this backend):
- **DRAM tile reads of weight + bias on (1,1,2048,256)**: from 1,024 to ≤ 512 (variant a: once per cluster) or to 16 (variant b: once per op).
- On every guard-set shape no count is worse: one per kernel path × placement × dtype, as in Refinement 3.
- The golden suite is green and the session tests still pass.

### [ ] Refinement 5 — Compute-side block tuning: pair-add row sum and input overlap

**Type**: perf

**Goal**: the op is compute-bound (6 FPU tile ops plus 2 packs per input tile). Two T2 levers from `ttnn/ttnn/operations/examples/master.md` apply to the supported 16-bit-DEST model shapes and to the many-row shapes:
- (1) Pair-add the row sum: `x[i] + x[i + Wt/2]` with Strided/Offset operands of the same ring, copy-seeding an odd tile (`eltwise_l1_vs_dest_accumulate`, `row_reduce_accumulate`). This halves the FPU steps of `row_sum`, and with them the bf16 rounding steps of the 16-bit accumulator.
- (2) Co-tune `INPUT_BLOCK_DEPTH` with `block_tile_rows` (`compute_block_size`, `double_buffer`): with depth 2, blocks of bh/2 where L1 permits, so the read of block k+1 overlaps passes 1–2 of block k and is not exposed.

Keep every resident window starting at a ring base: every non-last block consumes exactly the capacity (op_design.md, Mechanism caps). No SUPPORTED change.

**Verifier notes**: knob-turns on the exposed block surface (`INPUT_BLOCK_DEPTH`, `block_tile_rows`, both single-sourced in `layer_norm_program_artifacts.py`). The pair-add loses the Cumulative wait's overlap with the read, so take (1) and (2) together and keep the variant whose counts are better. Take it after Refinements 3 and 4, which change which shapes reach `row_resident`.

**Done when** (structural counts; time gain **unconfirmed** on this backend):
- **Row-sum FPU tile ops per tile-row**: from Wt to ⌈Wt/2⌉ (+1 seed when Wt is odd).
- **Input blocks in flight per NEO**: from 1 to 2 on at least one many-row shape where L1 permits (e.g. (1,1,2048,256)).
- The 16-bit-DEST model calls still meet PCC 0.999, and their relative RMS does not rise.
- The golden suite is green, with no regression on the guard set (as in Refinement 3).
