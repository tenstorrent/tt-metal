# Operation Requirements: softmax

Specification: `eval/specs/softmax/spec.py` (governs). Design: `op_design.md`. L1 ledger: `l1_ledger.md`.
Target: Quasar on ttsim (no clock: every perf gate below is a **structural count**, per the ttsim backend page;
the gain in time is unconfirmed until a timed backend or hardware measures it).

## Definition
- **Formula**: `out[..., i, ...] = exp(x_i − max_j x_j) / Σ_j exp(x_j − max_j x_j)` along `dim` (always the stable form, R1)
- **PyTorch Reference**: `def reference(input_tensor, dim=-1): return torch.softmax(input_tensor.double(), dim=dim)`
- **Import Path**: `from ttnn.operations.softmax import softmax`
- **Function Signature**: `softmax(input_tensor: ttnn.Tensor, dim: int = -1, *, compute_kernel_config: ttnn.ComputeConfigDescriptor = None) -> ttnn.Tensor`
  (`compute_kernel_config` also takes tt-metal's `WormholeComputeKernelConfig`, R10; `None` → `default_compute_kernel_config()`, R4)

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Thread plan**: every refinement keeps `THREAD_PLAN = (2, 4, 2)` (one host constant, counts as CT args) on the paths it adds, and extends the plan tests (every NEO ran; one device session before/after a tt-metal program) to each new kernel path.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]; **fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED placement**: memory_config ∈ {DRAM, L1}, sharding = interleaved
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}, rank = 4
- **SUPPORTED op-specific axes**: dim = -2
- **Regime**: `column_resident` (whole tile-column resident per NEO through max / exp-sum / normalize; raw SFPU, Float32 stats)
- **Cores**: full grid — tile-columns dealt in whole rounds of C = 4 to `split_work_to_cores(row_wise=True)` clusters; thread plan (R, C, W) = (2, 4, 2)
- **Compute config**: `default_compute_kernel_config()` = HiFi4 / approx off / fp32 DEST; fidelity and approximation never gated (R5)
- **Golden baseline**: 72 supported cells pass, 0 supported_fail, 1468 xfail_expected, 592 invalid_skipped (per `verifier_report.json`)

### [ ] Refinement 1 — Reduce along W (`dim=-1`) and ranks 2, 3

**Goal**: add `-1` to `SUPPORTED["dim"]` and `2`, `3` to `SUPPORTED["rank"]` (all three alignments, DRAM and L1, bfloat16 TILE). Add `dim=-1` as the transposed regime `row_resident` of the same scheme. It is a knob-turn of the reduce direction (a compile-time flag on the same three block operations): the unit becomes a tile-row of `Wt` tiles dealt in whole rounds of C to the clusters; `sfpu_reduce<…, REDUCE_ROW>`; `sfpu_{sub,mul}_bcast_col`; and an additive `NEG_BIG` mask on the padding **columns** of the last tile-column (the mirror of `cb_h_mask`). Ranks 2 and 3 are a SUPPORTED widen: the kernels already see only `NC = prod(padded_shape[:-2])`; pin them with tests.

**Verifier notes**: lands first because Refinement 2's split-and-combine regime splits the axis this refinement adds. The resident predicate stays `Wt · tile_in · C ≤ DFB_RING_MAX_BYTES`, so bf16 rows of `Wt ≥ 128` (128 tiles × 2 KB × 4 NEOs = 1 MiB, 16 B over the ring cap) are refused with the resident-fit `SupportRefusal`. Those are the 16 cells named under **Done when**: they become `supported_fail` (a resource boundary, like an OOM) and Refinement 2 moves them to passing. Do not silence them with `EXCLUSIONS`, and do not lower `C` per call to squeeze them resident: Refinement 2 serves them at 1× DRAM with the grid filled. Keep the row unit spread over the whole grid and the reader and writer one tagged transfer per entry, as in Phase 0. Keep `Ht`/`Wt`/`block_cols` single-sourced (one CT arg each; the reduce direction is one more CT flag, not a second copy of the kernels).

**Done when**: every `dim=-1` and rank-2/3 golden cell inside SUPPORTED passes, except these 16, which fail only with the resident-fit refusal: `dim=-1`, bfloat16, TILE, fp32_dest_acc_en=True, memory_config ∈ {DRAM, L1}, shapes `(1,1,32,4096)`, `(1,1,32,8192)`, `(1,1,128,4096)`, `(2,1,64,4096)`, `(1,32,4096)`, `(1,32,8192)`, `(32,4096)`, `(128,8192)`. The translated ULP tests for `dim=-1` pass. The precision baseline gains a `dim=-1` case. The thread-plan tests cover `dim=-1`.

### [ ] Refinement 2 — Split the reduced axis across cores with a stats combine (`reduced_split_combine`)

**Goal**: move the 16 `dim=-1` wide-row cells Refinement 1 leaves failing (named there: rows of `Wt ≥ 128`) from `supported_fail` to passing. Do it by splitting the **dependent** (reduced) axis into S slices over several clusters, each slice small enough to stay resident. This is the design's deferred `h_split_combine` regime, generalised to both reduce directions. Each slice computes its partial `(m_i, s_i)`. The partials are combined at the row's or column's leader (`m = max m_i`, `s = Σ s_i · exp(m_i − m)`), and `(m, 1/s)` goes back to the S − 1 other slices before their `normalize_block`. Select the regime when the reduced extent fails the resident predicate, **or** when the independent units under-fill the grid (`num_units < grid · C`) while the reduced extent is ≥ 2 slices. The second case also fills the grid on the tall, narrow `dim=-2` shape `(1,1,2048,256)` (2 of 32 clusters engaged today).

**Verifier notes**: scheme-change, so it stands alone: the cross-core topology is the work. No skill covers cross-core combines yet, so the topology is spelled out here:
- **Mechanism.** Gather each slice's two Float32 stat tiles to the leader with semaphores; broadcast `(m, 1/s)` back with the multicast helper (host `ttnn.Mcast` + `attach()`, kernel `McastArgs` `sender(noc).send()` / `receiver(noc).receive()`, `mcast/kernel/mcast_args.hpp`). First confirm that multicast and semaphores build and run under a Metal 2.0 `ProgramSpec` on ttsim (`quasar_op_examples/` has no example). If they don't, use unicast writes plus semaphores.
- **Where the combine goes.** The stats already exist as tiles in `cb_col_max` / `cb_col_recip` between `column_expsum_block` and `normalize_block`, so the combine inserts there without touching the three block operations.
- **Ring discipline.** Every ring that a NoC transfer fills from another core follows the explicit-sync rules on the target page: `disable_dfb_implicit_sync_for`, and a transaction id on every transfer.
- **Traffic.** DRAM stays 1× in and 1× out. The added cross-core traffic is 2 stat tiles per slice per unit in and 2 out.
- **Ledger.** Update the data-movement budget to match.

**Done when**:
- The 16 named cells pass.
- No other golden cell regresses.
- Structural counts, read off the program spec by a test, before → after:
  - clusters engaged on `(1,1,32,8192)` `dim=-1`: refused → ≥ 8;
  - clusters engaged on `(1,1,2048,256)` `dim=-2`: 2 → ≥ 8;
  - input DRAM crossings: 1×, unchanged.
- The session tests (op after another program, op before `ttnn.to_layout`) pass on the new regime.

### [ ] Refinement 3 — Speed up the YOLOv8 shapes (`Ht = 1` columns)

**Type**: perf

**Goal**: no `LOOSE_CASES` entry is perf-flagged, so the region is chosen here: the callers' two shapes. These are `[1,4,16,8400]` bf16 DRAM `dim=-2` (the flagship: 1,052 columns over 32 clusters × 4 NEOs) and `[1,4,16,400]` bf16 L1 at LoFi with approximation on (the loose case `yolo_dfl_l1_lofi`). Their reduced extent is one tile (`Ht = 1`), so each output tile pays the full three-operation cost. Per output tile (masked) that is 9 `copy_tile` unpacks with 9 unpack reconfig + `copy_init`, 3 DEST sections, 3 packs (2 of them into the Float32 stat rings and back), 2 exponentials and 3 mask adds, plus 3 `pack_init` re-aims per block. Speed them up using the relevant patterns in `ttnn/ttnn/operations/examples/master.md`. `compute_fusion` and `compute_block_size` apply: fewer phase boundaries, and fixed costs amortised over more tiles. The design's perf lamp names the likely lever: a single-section path for reduced extents `≤ DEST_AUTO_LIMIT − 2`, which keeps `m` and `1/s` in DEST and leaves the stat rings unbound. No SUPPORTED change.

**Done when**: structural counts on `[1,4,16,8400]`, read by a test from the program spec / compile-time args, before → after:
- `copy_tile` unpacks per output tile: 9 → ≤ 3;
- packs per output tile: 3 → 1;
- Float32 stat rings bound on the `Ht = 1` path: 2 → 0;
- exponentials per element: 2 → 1.

The gain in time is unconfirmed on ttsim. Also required:
- `yolo_dfl_l1_lofi` still passes at its cell's tolerances, and so do the `large_magnitude_bfloat16` / `uniform_bfloat16` regression cases;
- the golden suite stays green;
- no regression across the config-spanning guard set (one representative per kernel path × reduce direction × placement: resident `dim=-2` multi-tile, resident `dim=-1`, `reduced_split_combine`, DRAM and L1).

### [ ] Refinement 4 — Row-major layout

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]` for the input and output (both reduce directions, ranks 2–4, DRAM and L1, every alignment), natively in the kernels. A tilize-wrapped reader turns sticks into tiles before the three block operations, and an untilizing writer emits sticks; the math stays on tiles. The non-aligned edge (H or W not a multiple of 32) is zero-padded in the reader. The existing additive masks then keep the padding out of `m` and `s`. Carry Phase 0's whole-grid unit split onto the RM path unchanged.

**Implementation skill**: /memory-layouts, /interleaved-parallel

**Verifier notes**: ordered before the precision bundle (layout is the harder, more invasive axis), after the split regime, so the RM legs are built once against the final structure. The op's kernels already hand the transaction tag back; keep it on the new writer. A ring that the reader fills by assembling sticks into tiles is an explicit-sync ring on this part (target page, *Ring protocols*). The partial-tick escape hatch, only if the in-kernel tilize/untilize is more than a focused pass, is `ttnn.to_layout` at the entry point plus on the output. That breaks the one-dispatch-per-invocation rule of this run, so it is **not** available here: ship `[~]` with the RM reader done and file `Refinement 4b` for the RM writer instead.

**Done when**: every `ROW_MAJOR` golden cell inside SUPPORTED passes (bf16; bfloat8_b + ROW_MAJOR stays INVALID) with one dispatch per call, and the thread-plan and session tests cover the RM path.

### [ ] Refinement 5 — Numerical configurability: float32 and a 16-bit accumulator

**Goal**: add `ttnn.float32` to `SUPPORTED["dtype"]` and `False` to `SUPPORTED["fp32_dest_acc_en"]`, honouring the whole `compute_kernel_config` (`math_fidelity`, `math_approx_mode`, `fp32_dest_acc_en`). Derive the input and output ring formats from the dtype: float32 tiles are 4 KB, so the resident caps halve, and `select_regime` and `select_blocking` must take the cap from `buffer_page_size()`, never a literal. Rule R3 is already implemented: keep refusing float32 with `fp32_dest_acc_en=False`.

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: cheapest axis, last of the generality refinements.
- **16-bit DEST.** The SFPU row/column broadcasts and SFPU binaries need fp32 DEST (`sfpu_binary_bcast.h`: "Both source registers must contain FP32 data"). For `fp32_dest_acc_en=False` either keep the kernel's DEST at 32 bits (precision ≥ requested; the (bfloat16, False) limits (0.99, 0.13) are looser) or move the broadcasts to the FPU. Record the choice in the changelog.
- **float32 precision.** The float32 input reloads through SrcA (tf32). Check the (0.999, 0.01) limit on the baseline shapes before claiming float32; `UnpackToDest` of a Float32 page is the lever if tf32 falls short.
- **Tall and wide float32 shapes.** float32 `(1,1,2048,256)` `dim=-2` (Ht = 64) and the `Wt ≥ 64` `dim=-1` rows exceed the float32 resident cap at C = 4. They must route to Refinement 2's split regime, not fail.
- `Omitted: SUPPORTED["dtype"] bfloat8_b — no path for bfloat8_b on Quasar (target page, Family facts): the golden suite skips those cells as unbuildable, validate() keeps refusing it, no refinement is queued.`

**Done when**:
- every float32 and every `fp32_dest_acc_en=False` golden cell inside SUPPORTED passes;
- the float32 numeric regression cases (`small_magnitude`, `large_magnitude`, `uniform`, `dim=-1`) pass;
- R3's refusal is still tested;
- the precision baseline gains float32 and 16-bit-DEST rows.

### [ ] Refinement 6 — Cut the per-tile fixed cost on multi-tile reduced extents

**Type**: perf

**Goal**: speed up the supported cells whose reduced extent spans several tiles, using the relevant patterns in `ttnn/ttnn/operations/examples/master.md`. Representatives: `(2,4,128,512)` and `(1,1,2048,256)` at `dim=-2` (Ht = 4, 64), and `(4,128,512)` / `(1024,1024)` at `dim=-1`. Applicable entries: `compute_block_size`, `sfpu_tile_scope`, `double_buffer`. Today:
- `normalize_block` recomputes `exp` (2 exponentials per element);
- under half-sync fp32 DEST, `m` and `1/s` are reloaded every 1–2 output tiles (`outs_per_section = DEST_AUTO_LIMIT − d_first_out`);
- every Phase 0 shape runs one block per NEO at `in_depth = 1`, so its read overlaps its compute only through `column_max_block`'s cumulative wait.

The design's perf lamps name the candidate levers: full-sync DEST (8 fp32 slots), a Float32 `exp` ring (halves the resident cap; tf32 reload), and two blocks per NEO at `in_depth = 2`. Co-tune `block_cols` × `in_depth` × DEST sync, and tighten `cb_output_tiles` to the section size the chosen DEST mode gives (L1 ledger finding folded here). No SUPPORTED change.

**Done when**: structural counts on the representatives, read by a test or probe, before → after:
- stat reloads per output tile: 1 (masked) / 0.5 → ≤ 0.2;
- exponentials per element: 2 → 1, or the lever's documented reason for keeping 2;
- blocks per NEO on the multi-block guard `[1,4,16,8400]` with `BLOCK_COLS_CAP`: unchanged or better.

Also required:
- `cb_output_tiles` capacity matches the section size × `OUT_DEPTH` (ledger updated);
- the gain in time is stated as unconfirmed on ttsim;
- the golden suite stays green, with no regression across the config-spanning guard set (one representative per kernel path × layout × placement × dtype).
