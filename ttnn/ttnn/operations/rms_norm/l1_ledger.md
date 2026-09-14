# L1 Ledger: rms_norm

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions: `op_design.md` (Blocking Model, §H).

Block axes: `row` (extent `block_rows` = B), `w` (extent `core_w_tiles` = Wc — the whole per-core W slice), `w_split` (extent `num_partials` = Np — the cross-core face of `w`: the group's ACTIVE W slices; `num_w_splits` = Cw is the group rectangle's core count, ≥ Np, and only sizes the multicast rectangle / passive-core roster). Every row accounts for all three.

Implementation note (Phase 0): CB capacities are **per W-group** — each distinct `core_w_tiles` value gets its own CB descriptors sized with that group's `Wc`, so every capacity is an exact multiple of its push/pop quantum (a ragged quantum wraps a CB illegally). The address-stable collective CBs (`cb_gather`, `cb_rstd`) and the other block-only CBs are created first, uniformly over the whole program range, so they sit at identical addresses on every core.

`rows` in a push/wait count is the runtime extent (`block_rows`, or `last_block_rows` on the tail); capacities are sized with `block_rows` and `core_w_tiles_max`.

## Circular buffers (per core, uniform across the program's core range)

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_tiles` | TILE interleaved: `DEPTH_X · B · Wc_max` (`DEPTH_X = 2`); RM: `1 · B · Wc_max`; R3: `Rt · Wc` (globally allocated on the input shard) | `B · Wc` (one block resident from `sumsq_block` through `normalize_block`) — R3: `Rt · Wc` (the shard) | `{row: spans → B (R3: Rt), w: spans → Wc, w_split: streams → this core's slice only}` | input dtype (`Float16_b` / `Float32`) — relayout only, no accumulation crosses it | reader (TILE) / compute-tilize (RM) / reader-published (R3) | compute | per block; R3 whole kernel | **Not shared.** Capacity > live set by `DEPTH_X` = double buffering (reader prefetch of block k+1) — RM sets depth 1 because its producer is compute. R3: the shard itself, no allocation. |
| `cb_x_sticks` (RM only) | `DEPTH_X_STICKS_BLOCKS · B · Wc` tile-sized pages (`DEPTH_X_STICKS_BLOCKS = 2`) — same bytes as `32 · DEPTH · B` stick pages; TILE granularity (`read_sticks_for_tilize<TILE>`, one NoC barrier per 32 sticks, `Wc` pages pushed per tile-row) | `B · Wc` pages (one block; tilize consumes `Wc` pages = 32 sticks per tile-row as they arrive) | `{row: spans → B (×2 depth), w: spans → Wc (bytes per stick), w_split: streams}` | input dtype | reader | compute (tilize) | per block | **Not shared.** Capacity = 2 blocks: this is the RM overlap knob (`depth_x_sticks_rows`) — the only buffer through which the DRAM read overlaps compute on the RM path. |
| `cb_scaler` | 1 | 1 | `{row: —, w: —, w_split: —}` (constant) | `Float16_b` (bf16 packed scaler, 1.0 for SUM) | reader (once) | compute | whole kernel | **Not shared:** live for the whole kernel (every `reduce` call waits on it). Bare literal legal: genuinely constant. |
| `cb_sumsq_partial` | `B` | `rows` | `{row: spans → B, w: collapsed (one tile per tile-row), w_split: streams}` | `Float32` | compute (A) | compute (B) | per block | **Not shared:** its live range (A→B) is concurrent with `cb_x_tiles` (resident) and immediately precedes `cb_partial_collapsed`/`cb_gather` (B's output) — no disjoint peer of ≥ its size exists except `cb_normed`, whose lifetime (D→E) is disjoint; not aliased because the saving is `B · 4 KiB` (≤ 0.4 % of a wide block) and aliasing would couple the gamma/no-gamma builds. |
| `cb_partial_collapsed` (Cw>1 builds only) | `B` | `rows` | `{row: spans → B, w: collapsed, w_split: streams (this core's one partial)}` | `Float32` | compute (B) | writer (gather send) | per block | **Not shared:** it is a cross-kernel handoff (compute → writer) and must be distinct from any compute-internal accumulator (CB ownership invariant). Absent in the `Cw = 1` build (B packs straight into `cb_gather`). |
| `cb_gather` | `B · Np` — **exactly one round** (`Np = num_partials`, the group's active W slices; sizing it `B · Cw` over-allocated by the passive bounding-box cores AND made the root combine sum never-written slots — fixed in Phase 0) | `rows · Np` (root); 0 on non-root cores | `{row: spans → B, w: collapsed, w_split: spans → Np (one slot per active W-split core)}` | `Float32` | compute (B) in the `Cw = 1` build / writer (after `Cw − 1` remote arrivals + own local copy) | compute (C) | per block | **Not shared:** remote cores write into it by base + slot offset, so its address must be stable and identical on every core — no aliasing, no depth (Mechanism cap). Over-allocated on non-root cores by design (uniform descriptors; `B · Cw · 4 KiB`, bounded by §H1). |
| `cb_rstd_handoff` (Cw>1 builds only) | `B` | `rows` (root) | `{row: spans → B, w: collapsed, w_split: —}` | `Float32` | compute (C) | writer (mcast source) | per block | **Not shared:** cross-kernel handoff (compute → writer), must not be the mcast landing buffer or an accumulator. |
| `cb_rstd` | `B` — **exactly one round** | `rows` | `{row: spans → B, w: collapsed, w_split: —}` | `Float32` | compute (C) in the `Cw = 1` build / writer (mcast receive or sender loopback) | compute (D) | per block | **Not shared:** mcast landing address must be identical on all cores and stable across rounds (Mechanism cap); single round because the pre-handshake enforces round-k consumption before round-k+1 delivery. |
| `cb_gamma_tiles` (gamma builds only) | `Wc_max` | `Wc` | `{row: — (constant along rows), w: spans → Wc, w_split: streams (this core's slice)}` | gamma dtype — relayout only | reader (TILE gamma) / compute-tilize (RM gamma) | compute (E) | whole kernel | **Not shared:** resident for the whole kernel (every block's `scale_block` reads it). |
| `cb_gamma_sticks` (RM-gamma builds only) | `Wc` tile-sized pages (= 32 sticks of `Wc·32·e_g` B) | row 0 written from DRAM, rows 1..31 zeroed by the DM engine (`noc.async_write_zeros`), all `Wc` pages read once by the tilize LLK | `{row: — (one stick), w: spans → Wc, w_split: streams}` | gamma dtype | reader | compute (tilize) | start-up only | **Shares allocation with `cb_normed`** (second format descriptor on the same `CBDescriptor`): lifetimes disjoint (start-up vs. per block), and `32 · Wc·32·e_g = Wc·T_g ≤ B · Wc · 4096` for every B ≥ 1 and every gamma dtype. Both exist iff gamma is present. |
| `cb_normed` (gamma builds only) | `B · Wc_max` | `rows · Wc` | `{row: spans → B, w: spans → Wc, w_split: streams}` | `Float32` (DEST width; the x·rstd intermediate carries fp32 into the gamma multiply — no bf16 downcast) | compute (D) | compute (E) | per block | Hosts `cb_gamma_sticks` (above). **Not shared with `cb_x_tiles` / `cb_output_tiles`:** both are live concurrently with D→E (x is popped at the end of D; the output block is being produced by E while the writer may still be draining the previous block at depth 2). Removing it entirely is the `normed_roundtrip` lamp (precision-gated). |
| `cb_output_tiles` | TILE interleaved: `DEPTH_OUT · B · Wc_max` (`DEPTH_OUT = 2`); RM: `1 · B · Wc_max`; R3: `Rt · Wc` (globally allocated on the output shard) | `rows · Wc` | `{row: spans → B (R3: Rt), w: spans → Wc, w_split: streams}` | output dtype | compute (D or E) | writer (TILE) / compute-untilize (RM) / none (R3 — it is the output tensor) | per block; R3 whole kernel | **Not shared.** Capacity > live set by `DEPTH_OUT` = writer drains block k while compute fills k+1; RM depth 1 (compute → compute, sequential helpers require the full block). |
| `cb_out_sticks` (RM only) | `DEPTH_OUT_STICKS_ROWS · Wc_max` tile-sized pages (`DEPTH_OUT_STICKS_ROWS = 2`) | `Wc` (one tile-row of untilized sticks) | `{row: streams → one tile-row window, w: spans → Wc, w_split: streams}` | output dtype | compute (untilize) | writer | per block | **Not shared:** live concurrently with `cb_output_tiles` (untilize reads one, writes the other). Depth 2 = writer/untilize overlap. |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `B` = `block_rows` | tile-rows per block | `1 ≤ B ≤ min(core_row_tiles, block_rows_max_l1(Wc_max, Cw))` | §H3 closed form; `block_rows_max_l1 ≥ 1` asserted (else the inventory, not the knob, is the finding — Rule 1/2) |
| `Wc` = `core_w_tiles`, `Wc_max` | tile-columns in this core's slice / the largest slice in the program | interleaved: `Wc_max = ceil(Wt / Cw) ≤ core_w_tiles_max_l1`; R3: `shard_w / 32` (fixed by the caller) | §H2 residency floor `Cw ≥ ceil(Wt / core_w_tiles_max_l1)`; R3: `block_rows_max_l1(Wc, Cw) ≥ 1` with shards subtracted from the budget, else CB-OOM refusal |
| `Cw` = `num_w_splits` | cores per row-group rectangle (incl. passive bbox cores in R3) — the multicast rectangle | `1 ≤ Cw ≤ min(Wt, grid_x · grid_y)`; `Cw = a·b`, `a ≤ grid_x`, `b ≤ grid_y` | §H2; R3: `|bbox(shard_spec.grid)|` |
| `Np` = `num_partials` | active W slices per group = gather slots per row; `num_partials_expected = Np − 1` | interleaved: `Np = Cw`; R3: `Np = |shard_spec.grid| ≤ Cw` | R3 predicate; `Np ≥ 1` |
| `Rt` = `tensor_row_tiles` | whole-tensor tile-rows (R3 capacities only) | appears **only** in the zero-copy shard CBs, which are the caller's buffers (already resident, not allocated by the op) | predicate: `memory_layout == WIDTH_SHARDED` |
| `T_in`, `T_out`, `T_g` | tile bytes of input/output/gamma dtype | 2048 (bf16) or 4096 (fp32) | dtype ∈ SUPPORTED |
| `e_in`, `e_g` | element bytes | 2 or 4 | dtype ∈ SUPPORTED |
| `DEPTH_X`, `DEPTH_OUT`, `DEPTH_X_STICKS_BLOCKS`, `DEPTH_OUT_STICKS_ROWS` | buffer-depth knobs | 2, 2, 2, 2 | host constants (§H0) |
| `L1_CB_BUDGET` | bytes available to CBs | `min(ttnn.get_max_worker_l1_unreserved_size() (− 2·shard bytes in R3), largest contiguous free L1 block per bank at build time) − 64 KiB` | device query at descriptor build (`ttnn.get_memory_view(device, L1).largest_contiguous_bytes_free_per_bank`): the CB region is carved from the bottom of L1 up to the lowest live buffer, so a caller's other live L1 tensors shrink it below the nominal size |

## Total per-core footprint (bytes)

TILE interleaved (R1/R2):

```
F = Wc_max · B · (DEPTH_X·T_in + HG·4096 + DEPTH_OUT·T_out)          # scales with B·Wc — the block
  + HG · Wc_max · T_g                                                   # scales with Wc — resident gamma slice
  + 4096 · B · (Np + 2 + 2·[Cw>1])                                      # scales with B and Np — partial/gather/rstd chain
  + 2048                                                                # constant — scaler
```

RM interleaved: replace `DEPTH_X·T_in` by `(DEPTH_X_STICKS_BLOCKS + 1)·T_in` and `DEPTH_OUT·T_out` by `T_out`, and add `DEPTH_OUT_STICKS_ROWS · Wc_max · T_out` (scales with Wc). `cb_gamma_sticks` adds nothing (aliased on `cb_normed`). Per-W-group sizing replaces `Wc_max` by that group's `Wc` for every W-scaled term; `Wc_max` remains the bound used by §H3.

R3 (WIDTH_SHARDED): `F = HG·B·Wc·4096 + HG·Wc·T_g + 4096·B·(Np + 2 + 2·[Cw>1]) + 2048`, with the input and output shards (`2 · Rt · Wc · T_in`) already resident and subtracted from `L1_CB_BUDGET` before the fit.

Worked points (bf16, gamma, `L1_CB_BUDGET ≈ 1.3 MiB`): `Wc = 32, Cw = 1` → `B_max = 3` (8192×1024 prefill: 2–3 rows per core → one block each); `Wc = 75, Cw = 3` → `B_max = 1` (8192×7168 prefill: 7 blocks of 1 row); decode 32×7168 at `Cw = 14..26` → `Wc ≤ 16`, `F ≈ 0.3 MiB`.

## Data-movement budget (chosen split: rows × W with root combine; R1 is its Cw = 1 point)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input` (x) | **1** (R1/R2: each tile read by exactly one core, once — the block is resident across both passes); **0** (R3: already in L1) | x stays resident in `cb_x_tiles` from `sumsq_block` to `normalize_block`, so the normalize pass never re-reads it — this is what forbids sub-chunking `w` (R5 rejected) | none |
| `output` | **1** (R1/R2); **0** (R3) | one write per tile | none |
| `gamma` | **`active_cores / Cw`** full-tensor equivalents (each active core reads its `Wc`-tile slice once; the `num_row_groups` groups holding the same slice each read it) | gamma is resident per core for the kernel; sharing it across row-groups is regime R4 (deferred) | none in Phase 0 (R4 would add one `Wc·T_g` mcast per grid column) |
| per-row Σx² partials | 0 | never leave L1 | **`(Np−1) · rows · 4 KiB` unicast into the root + `Np−1` semaphore incs per block per group** (0 when `Cw = 1`) |
| rstd | 0 | never leaves L1 | **`rows · 4 KiB` mcast to `Cw−1` receivers + 2 handshake semaphore ops per block per group** (0 when `Cw = 1`) |

Totals per tier: DRAM = `bytes(x) + bytes(out) + (active_cores/Np)·bytes(gamma)`; cross-core = `num_row_groups · num_blocks · (Np + Cw) · rows · 4 KiB` (gather unicasts + the rstd multicast landing on every rectangle core, passive ones included) (+ `Np + 2·Cw` semaphore ops per block per group); core-local = the CB traffic of one block per pass (x unpacked twice: A and D; `cb_normed` packed and unpacked once).

> Cheapest-traffic split considered: **rows × W with root combine (R2/R3), and its Cw = 1 point R1** — x and out at the DRAM minimum (1 crossing each), cross-core `Cw·rows·4 KiB` per block per group (≤ ~4 % of x for W ≥ 1 K). Implemented: **the same split.** The only unbuilt traffic reduction is orthogonal to the split — gamma broadcast (R4), deferred because its saving is `(num_row_groups − 1)·bytes(gamma)` — ≤ ~14 % of x at wide prefill and zero whenever rows under-fill the grid — and the built structure (every core lands gamma in the same `cb_gamma_tiles`) keeps it reachable as a second `mcast_pipe` family.
