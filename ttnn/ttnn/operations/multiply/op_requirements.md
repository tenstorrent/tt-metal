# Operation Requirements: multiply

## Definition
- **Formula**: `out = a * b`, elementwise, with torch broadcasting (a dimension of size 1 repeats to the other
  operand's size; a lower-rank operand gets leading 1s); the product rounded once into the output dtype.
- **PyTorch Reference**: `torch.mul(input, other)`; graded against `a.double() * b.double()` (spec `reference`)
  at 0 ulp (spec `NUMERIC_RESOLVED`, `default` mode, every operand/output dtype pair).
- **Import Path**: `from ttnn.operations.multiply import multiply`
- **Function Signature**:
  `multiply(input_tensor_a: ttnn.Tensor, input_tensor_b: ttnn.Tensor, *, dtype: ttnn.DataType = None,
  compute_kernel_config: ttnn.ComputeConfigDescriptor = None) -> ttnn.Tensor`

Specification: `eval/specs/multiply/spec.py` (the only source of requirements). Target: Quasar on ttsim — perf
gates in this file are **structural counts** (backend page, *Performance measurement*); any gain in time stays
unconfirmed until a timed backend or hardware measures it.

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED fp32_dest_acc_en**: [True]
- **SUPPORTED layout**: [TILE]
- **SUPPORTED memory_config / sharding**: [DRAM, L1] / [interleaved]
- **SUPPORTED shape-derived axes**: alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}, rank ∈ {2, 3, 4},
  broadcast ∈ {none, over_h}
- **SUPPORTED op-specific axes**: output_dtype ∈ {None, bfloat16}
- **Cores**: full grid (32 clusters); `flat_stream` (broadcast none, flat tile split) and `row_bcast_strips`
  (over_h: column strips, b window held across the block's rows, H split into G row groups only to fill the
  grid). Thread plan (1, 1, 1) — bound by data movement, stated in the design.
- **Compute config**: `default_compute_kernel_config()` = HiFi4, fp32_dest_acc_en=True, approx off; the caller's
  config passed through whole.
- **Golden baseline**: 98 / 98 supported cells passing (92 PHASE0 cells + 6 numeric regression), X1 rejection
  passing, 1540 out-of-phase cells correctly refused (per `verifier_report.json`).

### [ ] Refinement 1 — ROW_MAJOR layout, in-kernel, on both built regimes

**Goal**: add `ttnn.ROW_MAJOR_LAYOUT` to `SUPPORTED["layout"]` for **both** operands and the output (the spec
gives both operands the cell's layout and the output a's layout), natively in the kernels: a tilize stage on
the input side and an untilize stage before the writer, the math staying on tiles, for both built regimes
(`flat_stream` and `row_bcast_strips`). Unlocks 92 cells blocked by the layout alone (816 with the later
axes).

**Implementation skill**: /memory-layouts, /interleaved-parallel

**Verifier notes**: hardest remaining generality item (difficulty tier 3), so it goes first. Refinement 2's new
broadcast regimes and Refinement 4's dtypes then extend one RM structure instead of re-doing it. The work unit
becomes a block of whole tile rows (32 sticks × the block's width). It must keep Phase 0's grid fill (the
design's tile-row blocks, split over every cluster) and batch both halves: several stick reads per barrier,
several stick writes per barrier, never one stick per handshake. RM sticks of a non-aligned W are not 32-element
multiples, and an RM operand of H = 1 (over_h's b) is one stick: tilize it into row 0 of a tile, so
`BroadcastDim::Row` keeps working. The `row_major` row of op_design.md's Regimes table is the starting point.
`ttnn.to_layout` wrappers at the entry point are **only** the partial-tick escape hatch. Use them only if the
in-kernel tilize/untilize stage cannot be done in one focused pass, and then file `Refinement 1b` to delete
them. (They also break the one-dispatch rule, so a wrapper is never a `[x]`.) Pack-target gap
(`quasar-library-gaps.md`): an untilize stage packs into a different ring from the multiply, so call
`pack_init(<ring>)` before each helper call whose output ring changes. The new path also needs the
session-order test (another program before, `ttnn.to_layout` after).

**Done when**: every golden cell with `layout=ROW_MAJOR` and otherwise-SUPPORTED axes passes, the
TILE cells still pass, and the RM path's work spans all 32 clusters on a multi-block shape (asserted off the
plan, like `test_regime_pinned`).

### [ ] Refinement 2 — Broadcast completeness: over_w, over_hw, over_leading, first_operand

**Goal**: add `over_w`, `over_hw`, `over_leading` and `first_operand` to `SUPPORTED["broadcast"]`, on both
layouts Refinement 1 leaves in SUPPORTED. Today 44 cells are blocked by the broadcast axis alone (over_w 16, over_hw 8, over_leading 12,
first_operand 8), out of 528 cells that carry one of these values. Each is a knob-turn on the existing structure, per op_design.md's Regimes
table:
- `over_w`: `col_bcast_strips`, the same walker with roles transposed. b holds one column tile per tile
  row, b mapping `Col`, `BroadcastDim::Col`.
- `over_hw`: `scalar_bcast`, a 1-tile b window per (core, image) run, `BroadcastDim::Scalar`.
- `over_leading`: the built regimes plus b's per-image strides (`img_b(n)`, already computed on the host for
  over_h).
- `first_operand`: host-side operand swap, rule R5. FPU and SFPU multiply are commutative, so the swap is exact.

**Verifier notes**: no generality skill covers new broadcast regimes, so this entry is verifier-authored
(difficulty tier 4: a regime extension with no cross-core traffic). It comes after R1 so each regime is
built once on both layouts. RM `over_w` reads a b whose sticks are one element wide (a 2-byte page under 16 B
L1 / 64 B DRAM alignment). Read it into an aligned staging slot and tilize from there; do not assume the
stick lands at a 16 B boundary. If RM over_w alone is out of reach, ship `[~]` with the TILE values plus RM
for the other three, and file `Refinement 2b` for RM over_w. The new regimes must fill the grid the way
Phase 0 does (`split_work_to_cores(..., row_wise=True)` over units that b does not force onto one core). Use
the per-operand image strides for `over_leading` now. Reusing b across images is Refinement 5's job, so do
not build a multicast here. A call where **both** operands need an in-tile broadcast after the swap (e.g.
`[32,1] × [1,64]`) is the deferred `dual_in_tile_bcast` regime. The FPU broadcasts srcB only. No INPUTS entry
reaches that call, so refuse it in `validate()` with `UnsupportedAxisValue` naming the reason, and record
the refusal in the changelog.

**Done when**: every golden cell with these four broadcast values and otherwise-SUPPORTED axes passes, and
each new regime is pinned by a test that reads the host's regime choice off the plan, as
`test_regime_pinned` does for the Phase 0 regimes.

### [ ] Refinement 3 — First-output latency of the YOLO over_h shapes

**Type**: perf

**Goal**: speed up the YOLOv8 detect-head calls, the shapes the spec's first phase exists for:
`[1,4,8400]×[1,1,8400]`, `[1,4,33600]×[1,1,33600]` and `[1,4,400]×[1,1,400]`, at bf16, TILE, DRAM/L1, fp32
DEST, all already in SUPPORTED. `LOOSE_CASES` flags no perf case, so this region is the verifier's pick.
These shapes have `Ht = 1`, so every `row_bcast_strips` block has `block_rows = 1` and the held b window
**reuses nothing**. It still serializes the core: the reader issues the block's whole b window and waits on
an explicit barrier before issuing any a read, and compute waits for the whole window before its first
multiply. On `[1,4,33600]` that window is 33 tiles (26 clusters × 33 columns); on `[1,4,8400]` it is 9. This
matches the catalog's "keep bytes in flight" situation (`ttnn/ttnn/operations/examples/master.md`:
`double_buffer`, `trid_double_issue`; `compute_block_size` for the opposite pull).

Candidate levers (the implementer picks):
- (a) when `block_rows == 1`, stream b one tile at a time in step with a, on an implicit-sync ring. The
  helper accepts a Row-broadcast operand with `InputTileMapping::Scalar` + `PerTile` wait/pop
  (`chain.inl` `is_legal_input_policy_for_mapping`).
- (b) cap `block_cols` at a pipelining granularity (whole tiles, ~4–8) so that with `B_WINDOW_DEPTH = 2` the
  next window loads while compute works on the current one.

No SUPPORTED change.

**Done when** (structural counts, read off the op by a test or probe; record before/after in the changelog):
- On `[1,4,33600]×[1,1,33600]`, the b tiles that must land on a cluster before its first multiply drop from
  33 to ≤ 4. On `[1,4,8400]` they drop from 9 to ≤ 4.
- The a reads that wait behind an explicit b barrier drop from all of the core's a reads to 0 (lever a) or
  to at most one window's worth (lever b).
- Tensor-tier traffic stays at a, b and out once each (b `G` times).
- The b ring does not grow beyond Phase 0's `2·block_cols` tiles.
- The golden suite stays green.
- No regression on the guard set: one representative per kernel path × layout × placement. At least:
  `flat_stream` TILE DRAM, `row_bcast_strips` with `block_rows > 1` (`[2,4,128,512]×[2,4,1,512]`, whose b
  reuse must survive), the G = 2 shape `[128,512]×[1,512]`, an L1 cell, and the RM paths from R1/R2.
- The entry and changelog state that the gain in time is unconfirmed on ttsim.

### [ ] Refinement 4 — Numerical configurability: float32, fp32_dest_acc_en=False, float32 output

**Goal**: add `ttnn.float32` to `SUPPORTED["dtype"]`, `False` to `SUPPORTED["fp32_dest_acc_en"]` and
`ttnn.float32` to `SUPPORTED["output_dtype"]`, together, on every regime and layout in SUPPORTED by then.
That covers 92 + 92 + 46 cells blocked by one of these values alone (816 / 816 / 544 in total). Rules R1 and R2
stay in force: float32 operands, or `dtype=float32`, with `fp32_dest_acc_en=False` are refused (the
validate() check already exists). Intermediate and output ring formats derive from the tensors' dtypes.
Each ring's own page size already feeds `block_cols_cap` (verifier fix).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: difficulty tier 5 (precision knobs), so it comes last among the generality refinements and
extends the RM and broadcast structure once. It is **not** zero-kernel-change. SrcA/SrcB hold 19 bits, so a
0-ulp float32 product needs the SFPU path on 32-bit `UnpackToDest` pages: the design's `fp32_sfpu` regime,
`binary_sfpu<MulBinary>` or the raw `sfpu_mul_bcast_{row,col}` calls. The library-gaps reference lists those
raw calls as exact in float32 on ttsim. Mind the gaps entries:
- load every operand of a DEST section before the SFPU op;
- `DataFormatReconfig::Disabled` on a chain read with `UnpackToDest`;
- never `UnpackToDest` a bf16 page into 32-bit DEST: it aborts ttsim. So bf16 operands stay on the FPU path
  in both DEST widths, including bf16 → float32 output, which is exact in fp32 DEST packed to a Float32 page.

The scalar broadcast (over_hw) has no `sfpu_mul_bcast_scalar` listed. Check the header before choosing a
mechanism. A bf16 × bf16 → bf16 cell at `fp32_dest_acc_en=False` puts the FPU product into 16-bit DEST. If
that write is not round-to-nearest-even, those cells miss 0 ulp by one ulp on about half the elements. Leave
any such cells failing as `numerical-precision`, with the ULP figure, and report them. Do **not** put them in
EXCLUSIONS. If 0 ulp is unreachable on this part's datapath, that is a target finding for the report.
Float32 rings double the ring bytes: re-read `l1_ledger.md`'s footprint at float32.

**Done when**: every golden cell with these values and otherwise-SUPPORTED axes passes, or fails only as
`numerical-precision` with its ULP recorded. R1/R2's refused combinations still raise `NotImplementedError`.
The precision baseline test gains a float32 row and a `fp32_dest_acc_en=False` row.

### [ ] Refinement 5 — Read b once under leading-dim broadcast

**Type**: perf

**Goal**: after Refinement 2, `over_leading` (and over_h with b broadcast in a leading dim) reaches b through
per-image strides. Every output image re-reads b's image, so b crosses the tensor tier `N_out / N_b` times:
32× on `[4,8,32,256]×[1,1,32,256]` and on `[4,8,32,64]×[32,64]`, and 2× on `[2,512,1024]×[1,512,1024]`. Bring b
to one crossing. The blocking model tags `n` as reuse-shared for b. Cut the axes b varies along, (h, w),
across clusters, and keep `n` innermost in each cluster's run with b's tiles held. This is the same move
`row_bcast_strips` makes for h, and it needs no cross-core traffic. Split `n` into groups only when the
(h, w) units under-fill the grid, which is the analogue of G. The design's deferred `leading_reuse` row
names a multicast successor instead; either is acceptable if the count below is met. See the catalog
(`ttnn/ttnn/operations/examples/master.md`, `shared_input_reuse`, `width_split`). No SUPPORTED change.

**Done when** (structural counts; changelog records before/after):
- b tile reads per call reach `|b| · G_n`, with `G_n` the n-groups needed for occupancy and 1 when the
  (h, w) units fill the grid. Concretely: `[4,8,32,256]×[1,1,32,256]` 256 → 8 (or `8·G_n`);
  `[2,512,1024]×[1,512,1024]` 1,024 → 512; `[4,8,32,64]×[32,64]` 64 → `2·G_n`.
- a and out still cross once.
- Every cluster with work in Phase 0 still gets work.
- The golden suite stays green, with no regression on the guard set named in Refinement 3.
- The time gain is stated as unconfirmed on ttsim.

### [ ] Refinement 6 — Per-block fixed compute costs

**Type**: perf

**Goal**: in the block regimes (`row_bcast_strips` and the R2 regimes), compute calls the `mul` helper once per
block. Each call repeats the broadcast init and the data-format reconfig, even when a, b and out share one
format and nothing changed since the previous block. Hoist the setup to once per kernel
(`eltwise_chain<InitReconfigOwner::Caller>`, `chain.hpp:179-191`). Disable the format reconfig when the three
formats are identical. This is the catalog's `compute_block_size` second lever, and it is safe only when the
dtype is constant across the boundary. Region: shapes with several blocks per cluster, e.g.
`[4,16,64,64]×[4,16,1,64]` (2 blocks per cluster), `[1,40,64,96]×[1,40,1,96]` (short blocks), and the R2
regime shapes with the most blocks per cluster. ⭐ T1-sized, so take every applicable fixed-cost lever in one
pass. No SUPPORTED change.

**Done when**:
- Helper init + reconfig sequences per cluster drop from `blocks_this_core` to 1 on the named shapes. Count
  them off the kernel's call structure, or with a counter the test reads back.
- The golden suite stays green, every dtype combination from R4 still passes (the reconfig is kept wherever
  formats differ), and the guard set from Refinement 3 does not regress.
- The time gain is stated as unconfirmed on ttsim.
