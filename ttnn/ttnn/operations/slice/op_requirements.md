# Operation Requirements: slice

## Definition
- **Formula**: `output = input[slice_start[0]:slice_end[0], ..., slice_start[r-1]:slice_end[r-1]]`; a negative index counts from the end of its dimension (`i + size`, spec R2); no step. Every output element is an input element, unchanged (R1): the op moves values and computes nothing.
- **PyTorch Reference**: `reference(input_tensor, slice_start, slice_end)` in `eval/specs/slice/spec.py` (float64 `narrow` per dimension after normalizing negatives).
- **Import Path**: `from ttnn.operations.slice import slice`
- **Function Signature**: `slice(input_tensor: ttnn.Tensor, slice_start, slice_end) -> ttnn.Tensor` (bounds as lists or tuples of ints, one per dimension, R3). Output: same rank, dtype and layout, interleaved in the input's buffer (R7).

## Phases

> **Non-regression rule**: Every refinement must pass all tests from prior phases.
> **Drift signal**: XPASS-strict failures mean the implementer added support but forgot to update SUPPORTED. The implementer fixes by updating SUPPORTED.
> **Checkbox protocol**: Implementer marks `[x]` when the refinement is complete and all tests pass, `[~]` when real work landed but at least one named axis value is deferred (treated as completed by the queue, surfaced as partial), `[ ]` only when nothing usable was produced.
> **Refinement ID + follow-up naming (mandatory — the runner parses this)**: Primary refinements are `Refinement N` (e.g. `Refinement 1`, `Refinement 2`). When you ship `[~]` partial and file the sharper follow-up the partial-tick protocol requires, name it by appending a lowercase letter to the parent's number: `Refinement 1b`, `Refinement 1c`, … (never `Refinement 1.5`, `Refinement 1 (follow-up)`, or a fresh number). Order follow-ups immediately after their parent so the queue runs them before later refinements — a partial's remaining-blocker follow-up must be picked next, not leapfrogged. The runner's parser matches exactly `Refinement \d+[a-z]?`; any other shape is invisible to the queue and silently skipped.
> **Backend (this run: Quasar on ttsim, untimed)**: a perf refinement is gated on the **structural count** it names, read off the op by a test; the gain in time is unconfirmed until a timed backend or hardware measures it. Never report a simulator timing as a speedup.

### [x] Phase 0 — Core Implementation

- **SUPPORTED dtype**: [bfloat16]
- **SUPPORTED layout**: [TILE, ROW_MAJOR]
- **SUPPORTED placement**: memory_config ∈ {DRAM, L1}, sharding = interleaved
- **SUPPORTED shape-derived axes**: rank = 3; alignment ∈ {tile_aligned, w_non_aligned, h_non_aligned}; sliced_region ∈ {outer, h}; start_alignment = tile_aligned
- **Regime built**: `page_relay` — whole pages relayed through one DM→DM DFB ring per cluster (implicit sync for pages ≤ 8 KiB, explicit block-batched sync above), page map on the reader, contiguous output-page ranges over the full 8×4 grid (`split_work_to_cores(row_wise=True)`), one `ttnn.generic_op` dispatch
- **Thread plan**: (R, W) = (1, 1), no compute kernel (bound by data movement)
- **Compute config**: none (the op computes nothing; spec DEFINITION)
- **Golden baseline**: 69 / 69 supported cells pass (68 PHASE0 cells + loose `bounds_as_lists`), 412 xfail_expected, 3 rejections pass, 0 loud (`verifier_report.json`)

### [ ] Refinement 1 — Row cuts: `rm_column_cut` + tile-layout W narrowing

**Goal**: add `w` to `SUPPORTED["sliced_region"]` for both layouts (start on a tile boundary; a start inside a tile is Refinement 2). TILE: the page map already carries the W tile axis (`page_start[3]`, `out_page_shape[3] = ceil(W_out/32)`) — validate-only, measured exact on the current kernels by verifier probe `probes/probe_005.py` (`[4,128,512]` from `(0,32,96)` to `(4,96,352)`, `[1,1,32,4096]` W 0–2048). ROW_MAJOR: build the deferred `rm_column_cut` regime of `op_design.md` — each output row is the byte range `[w_start·e, w_end·e)` of its input row: per-page source byte offset, a transfer size ≠ the ring entry, read the 16-B (L1) / 64-B (DRAM) aligned superset, write the output row from the right offset (realign on the writer hart in L1 where the NoC cannot write from a misaligned source). Unlocks the 16 bfloat16 rank-3 `w` + `tile_aligned` cells; the RM column-cut kernel must already accept any `w_start` (byte offset), because Refinement 2 relies on it for RM rows cut from inside a tile.

**Verifier notes**: scheme-change (new reader transfer shape + ring protocol), stands alone. Must precede Refinement 2: every TILE cell whose W start lies inside a tile is also `sliced_region = w`, so the retile's W-shift half cannot be tested until `w` is in SUPPORTED (true dependency). The column-cut ring is **explicit sync** (opted out via `disable_dfb_implicit_sync_for`, every transfer `NocOptions::TXN_ID` with user trid 0, block-batched reserve/barrier/push like the Phase 0 large-page path) — an implicit-sync transfer always moves exactly `entry_size` bytes. Keep the Phase 0 full-grid `out_page` split and block batching on both halves (no per-row barrier); the new regime is selected by the predicate in the design's regime table and needs regime-pinned tests (one TILE-`w` cell, one RM-`w` cell per placement). If the RM path is out of reach in one pass, partial-tick with TILE `w` only, an `EXCLUSIONS` entry `{"layout": ROW_MAJOR, "sliced_region": "w"}`, and a `Refinement 1b` for the RM kernel — never a host-side `ttnn.to_layout` round-trip.

**Done when**: every bfloat16 rank-3 golden cell with `sliced_region=w`, `start_alignment=tile_aligned` passes (both layouts, both placements); no `EXCLUSIONS` entry remains; prior cells still pass; the session-order tests in `test_slice_plan.py` cover the new regime.

### [ ] Refinement 2 — Starts inside a tile: `tile_retile` + row-major H start

**Goal**: add `non_tile_aligned` to `SUPPORTED["start_alignment"]`. ROW_MAJOR with W whole: validate-only (a row is the page; the `rm_unaligned_h_start` row of the design), measured exact by verifier probe `probes/probe_005.py` (`[1,96,128]` rows 16–80; `[4,8,47,256]` rows 15–47); ROW_MAJOR with W cut from inside a tile is Refinement 1's byte-offset path. TILE: build the deferred `tile_retile` regime — each output tile gathers its rows/columns from 2 (one axis shifted) or 4 (both) source tiles with an intra-tile shift across faces; the page map gives the top-left source tile. Unlocks the 12 bfloat16 rank-3 `non_tile_aligned` cells (8 `h`, 4 `w`).

**Verifier notes**: scheme-change, the hardest refinement in the queue (face-aware shift; up to 2×/4× source re-reads unless a core keeps the shared source tile-row resident across its block — a residency knob on `block_pages`, see the design's regime row). Depends on Refinement 1 (W-shift cells are `sliced_region = w`). **Mechanism conflict to resolve up front**: the design leaves open "DM-hart byte shuffle vs compute stage (untilize → shift → tilize)". A compute stage passes every element through DEST; R1 forbids any element passing through another dtype, and Refinement 4 adds float32 and uint32 — a compute retile must then be bit-exact for 32-bit values (fp32 DEST, no uint32 → float conversion). Prefer the DM-hart shuffle unless the compute path is proven bit-exact for all three TARGET dtypes. If the shuffle makes the DM harts the bottleneck, restate the thread plan in `op_design.md` (the (1,1) plan was justified for a pure relay) and size it with the target page's *Choosing the plan*. Spec `NUMERIC["special_values"]` is UNRESOLVED (tt-metal excludes a bf16 NaN in TILE with a W start inside a tile); no golden cell passes one.

**Done when**: every bfloat16 rank-3 golden cell with `start_alignment=non_tile_aligned` passes (both layouts, both placements, `h` and `w`); regime-pinned tests for `tile_retile` (H shift, W shift, both); prior cells still pass.

### [ ] Refinement 3 — Finer ISR batches on large per-core shares

**Type**: perf

**Goal**: free-selected region (no `LOOSE_CASES` entry is perf-flagged): the cells whose busiest core holds ≥ 16 pages — target `[2, 512, 1024]` rows 256–512, bfloat16, TILE **and** ROW_MAJOR, **L1** (16 pages per core on 32 cores; L1 keeps the 2 DRAM channels from capping the measurement). Phase 0 sets `block_pages = ceil(max_core_pages / RING_DEPTH_BLOCKS)`, so the ring spans the core's whole share and the DM0 ISR credits half of it per batch: the writer issues nothing until 8 of the core's 16 pages have landed, and every core runs exactly 2 batches whatever its share. Cap the block independently of the share (a named host knob, catalog `double_buffer`: 4–8 pages in flight is the sweet spot, whole pages minimum), and co-tune it with the depth — see `ttnn/ttnn/operations/examples/master.md` (`double_buffer`, `trid_double_issue`). No SUPPORTED change. Also discharges the L1-ledger finding (ring sized to the core's share rather than a streaming window).

**Verifier notes**: on the implicit-sync path the ISR batch is `num_entries / n` with `n` the smallest of {2, 3, 4} dividing `num_entries / max(R, W)` — with an even ring `n = 2` whatever `RING_DEPTH_BLOCKS` is (`op_design.md`, verifier note above *Regimes*; `DataflowBuffer.md`, "How many txn ids"). So raising `RING_DEPTH_BLOCKS` alone does **not** make batches finer; the block-size knob does. Keep `block_pages` the single source (ring size, `thread_block_pages` and the divisibility rounding all derive from it). Bounded by the 2-channel DRAM: on DRAM placements the change may be a no-op — run `/perf-ceiling-dm` (with `--aiclk-mhz` given explicitly) before choosing the value.

**Done when**: on the target shape (both layouts, L1), a test reading `create_program_artifacts` shows the busiest core's pages per ISR batch go from 8 to ≤ 4 (≥ 4 batches per core) with the ring's bytes no larger than Phase 0's; the golden suite is green; and no regression in the same structural counts (pages per batch, ring bytes, cores engaged, harts at work) across the config-spanning guard set — one representative per kernel path × layout × placement: implicit TILE, implicit RM, explicit RM (pages > 8 KiB), each in DRAM and in L1, plus one `rm_column_cut` and one `tile_retile` cell. The gain in time is unconfirmed on ttsim; the changelog records the counts before and after.

### [ ] Refinement 4 — dtype and rank widening (float32, uint32, ranks 2 and 4)

**Goal**: add `ttnn.float32` and `ttnn.uint32` to `SUPPORTED["dtype"]` and `2`, `4` to `SUPPORTED["rank"]`, across every regime built by then. float32 and the ranks are validate-only on `page_relay`: verifier probe `probes/probe_005.py` measured them exact on the Phase 0 kernels (fp32 TILE; fp32 RM `[1,144,8400]` = 33,664-B rows on the explicit path; rank 2 TILE/RM; rank 4 `h`/`outer`). uint32 needs one host change: Quasar refuses uint32 as a `DataflowBufferSpec.data_format_metadata` (`TT_FATAL dfb.cpp:148`, measured) — carry it as `ttnn.int32` (same width; the bytes are never unpacked) or leave the metadata unset. Unlocks every remaining TARGET cell (float32 / uint32 at ranks 2–4, bfloat16 at ranks 2 and 4).

**Implementation skill**: /numeric-formats-metal

**Verifier notes**: from the skill, only the dtype half applies: the spec gives this op **no compute config** (DEFINITION, "It therefore has no compute config") and it has no compute kernel, so add no `compute_kernel_config` and no intermediate-CB precision. R1 is the binding rule: every value must move bit-exact (the numeric requirement of every cell is `exact`), so float32/uint32 must never pass through a 16-bit buffer or through DEST — on `page_relay` and `rm_column_cut` they do not; on `tile_retile` that is Refinement 2's choice of mechanism (see its notes) and must be re-checked here for the 4-byte element width. Wider elements double `entry_bytes`: RM float32/uint32 rows over 2,048 elements cross the 8 KiB implicit-sync limit and take the explicit path, and rows over 16,384 elements hit the 64 KiB burst cap (no INPUTS entry does; widest is 8,400 × 4 = 33,600 B). Placed after the structural refinements (hardest first): the dtypes then extend kernels that no longer change underneath them.

**Done when**: every golden cell of TARGET passes (0 `xfail_expected` left), with float32 and uint32 bit-exact; the precision baseline (`test_slice_precision_baseline.py`) gains one float32 and one uint32 case.

### [ ] Refinement 5 — Keep reads on the wire across the explicit-sync barriers

**Type**: perf

**Goal**: the explicit-sync path (row-major pages > 8 KiB) serves YOLOv8's traced calls — `[1, 144, 8400]` RM, rows 0–64 and 64–144 — and the wide row-major cells. Each reader hart issues a block, then drains the NoC to empty with one barrier before it can issue the next block (and the writer likewise); with 2–5 pages per core that is one full drain per block. Target: `[1, 144, 8400]` RM bfloat16 rows 64–144 and the whole tensor (0–144), in **L1** and in DRAM. Lever: transaction-id rotation — barrier on the oldest block's user trid while the next block is in flight (catalog `trid_double_issue`, ⭐⭐ T2; see `ttnn/ttnn/operations/examples/master.md`), which needs `RING_DEPTH_BLOCKS ≥ 3` on this path (the explicit path owns its batching, so the depth knob works there). No SUPPORTED change.

**Verifier notes**: user trids are `[0, 7]` (`DataflowBuffer.md`, Transaction IDs); every transfer of an explicit-sync kernel stays tagged and the kernel still hands trid 0 back at exit (target page, *Ring protocols*); `test_a_tt_metal_program_follows_explicit_sync` must stay green at (1,1) and (2,2). Read the catalog README before writing it: the ring's write pointer lags the in-flight blocks. Run `/perf-ceiling-dm` (with `--aiclk-mhz`) first: if the DRAM placement is at the 2-channel floor, the L1 placement is the measured target. After Refinement 4, float32 rows over 2,048 elements also take this path — include one in the guard set.

**Done when**: on the target shapes a test or probe shows the reader and writer harts each drain the NoC to empty (no later block outstanding) once per hart — at the end — instead of once per block; golden green; no regression in the structural counts of Refinement 3 across the same guard set. Gain in time unconfirmed on ttsim; the changelog records the counts before and after.
