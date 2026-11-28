# L1 Ledger: g6_unsqueeze

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes from `op_design.md`:
`w_tile` (extent `block_w_tiles`), `tile_half` (extent `halves_per_risc` = 1 per RISC), leading
unit dims (extent 1), `in_tile_row` (only row 0), new axis (extent 1).
`fp32_dest_acc_en`: not applicable — no compute kernel, no DEST; pages are a dataflow-only payload.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_face_stage_half0` | `buffer_depth · block_w_tiles` pages of `face_bytes` (1024 B), `total_size = buffer_depth·block_w_tiles·face_bytes + 64` | `resident`: `block_w_tiles` faces (all in flight between the read barrier and the write barrier); `streaming`: `2·block_w_tiles` (block b being written while b+1 lands) | `{w_tile: spans → block_w_tiles; tile_half: spans → 1 (half 0 only); leading dims: streams → extent 1; in_tile_row: spans → row 0 lands in face row 0, scatter fills column 0 of the same face; new axis: streams → extent 1; blocks: streams → buffer_depth window}` | Float32 (raw fp32 bytes; dataflow-only payload, never through DEST — width must equal the tensor's 4 B element to keep bit-exactness) | reader-config kernel (`tile_half=0`) | same kernel (private scratch; no hand-off) | whole kernel | Not shared with `cb_face_stage_half1`: both live concurrently on two different RISCs (the point of the split). Input row and output face share one allocation already (in-place scatter: the 64 B row lands at row 0 of the face it becomes) — no separate input CB. +64 B: alignment slack for the 64 B DRAM-read landing |
| `cb_face_stage_half1` | same expression | same | `{w_tile: spans → block_w_tiles; tile_half: spans → 1 (half 1 only); leading dims: streams → 1; in_tile_row: spans → row 0; new axis: streams → 1; blocks: streams → buffer_depth window}` | Float32 (same reason) | writer-config kernel (`tile_half=1`) | same kernel | whole kernel | Not shared with `cb_face_stage_half0`: concurrent lifetime on the other RISC. In-place as above. +64 B alignment slack |

Implementation note (implementer): `CircularBufferConfig` requires `total_size` to be a whole
number of pages, so each staging CB is declared with `page_size = 64 B` (the read-alignment unit;
the CB is private scratch with no push/pop, so the page size only sizes the allocation). The byte
capacity is unchanged: `buffer_depth·block_w_tiles·halves_per_risc·entry_bytes + 64`, where `entry_bytes` follows the
`WRITE_SHAPE` knob (Refinement 2): `face` 1024 B (production default, unchanged), `segment` 256 B,
`tile` 2048 B (face 2h of a staged 4096 B tile image; forces `halves_per_risc = 2`). With
`HALVES_PER_RISC = 2` only `cb_face_stage_half0` exists and holds both halves of each tile
(`halves_per_risc` faces per tile); the per-CB L1 share then is the whole budget.

Capacity vs live set: capacity equals the live set (+64 B alignment slack) in both regimes; a face
(1024 B) is staged per tile half because the output write is a whole face (2 transactions per tile);
the 16 B-segment alternative (`WRITE_SHAPE="segment"`, a live knob) shrinks it to 32·16 B = 512 B
per tile but measured 2.2–2.5x slower (see the Refinement 2 note below).

### Symbol table

| Symbol | Bound | Established by |
|---|---|---|
| `face_bytes` | 1024 (16·16·4, fp32) | dtype = float32 (SUPPORTED) |
| `buffer_depth` | 1 (`resident`) or 2 (`streaming`) | regime predicate in `select_plan()` |
| `block_w_tiles` | `≤ cap(buffer_depth) = floor((L1_BUDGET_BYTES/2 − 64)/(buffer_depth·face_bytes))` = 658 / 329 | `select_plan()` clamp; `resident` additionally `block_w_tiles = core_w_tiles ≤ cap(1)` |
| `core_w_tiles` | `ceil(tensor_w_tiles / num_cores)`; unbounded in W, absorbed by `streaming` | `split_work_to_cores` |
| `L1_BUDGET_BYTES` | 1347776 | model environment (BB-05) |

### Total per-core footprint

`L1_op = (2 / halves_per_risc) · (buffer_depth · block_w_tiles · halves_per_risc · face_bytes + 64)` bytes
(`halves_per_risc = 1` in production: `2 · (buffer_depth · block_w_tiles · face_bytes + 64)`). The code's cap
generalizes the symbol-table bound to `floor((L1_BUDGET_BYTES/(2/halves_per_risc) − 64) / (buffer_depth · halves_per_risc · face_bytes))`
(`_block_cap()` in the program descriptor).
Scales with `block_w_tiles` and `buffer_depth` (both terms); nothing scales with W directly.
Maximum: `resident` 2·(658·1024 + 64) = 1347712 B; `streaming` 2·(2·329·1024 + 64) = 1347712 B; both ≤ 1347776.
Production: p1/p2 (1 tile per core) → 2·(1024 + 64) = 2176 B per core.

## Data-movement budget (p1: W = 512, 16 tiles; p2 halves every number)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input (65536 B allocated, 2048 B logical) | 1 × the logical row only: 16 tiles × 2 × 64 B = 2048 B | only row 0 of faces 0/1 is read; nothing is re-read (no operand shared across blocks) | none |
| output (65536 B allocated, 2048 B logical) | 1 × column-0 faces: 16 tiles × 2 × 1024 B = 32768 B | each face holding column 0 written once; padding faces 1/3 never written | none |

Totals per call (p1): DRAM read 2048 B, DRAM write 32768 B, NoC cross-core 0 B, core-local L1: 2048 B landed + 15·4 B·32 moved by RISC stores per core-pair.

> Cheapest-traffic split considered: `w_tile` across cores with 16 B column-segment output writes — write 8192 B instead of 32768 B (−24576 B DRAM write), same 2048 B read, 0 B cross-core, but 32 write transactions per tile instead of 2. Implemented: `w_tile` across cores + `tile_half` across the 2 RISCs with 1024 B face writes. The byte-cheaper variant is the same split with a different transfer shape, recorded as the "output write shape" perf lamp (transaction count, not bytes, is expected to dominate at 16 B); the structure (per-tile scatter into private staging) keeps it reachable by changing only the staging stride and the write size.

> Verifier audit (Phase 0): the deferred cheapest-traffic variant above (16 B column-segment writes,
> −24576 B DRAM write on p1) has an *expectation* ("transaction count is expected to dominate"), not a
> measurement, as its reason. Folded into the output-write-shape perf refinement in
> `op_requirements.md` (Refinement 2), which must measure it and either build it or record the
> measured reason it loses.
>
> **Refinement 2 (measured, settles it):** all three output write shapes are built behind
> `WRITE_SHAPE` and measured on device (`--profile`, DEVICE KERNEL DURATION, median of 3 fresh runs;
> shapes above the 32-tile floor, 110 cores, `resident`):
>
> | variant | W=14080 (440 tiles, 4/core) | W=112640 (3520 tiles, 32/core) |
> |---|---|---|
> | (a) `segment`, 16×16 B stateful one-packet, halves on 2 RISCs | 9263 ns | 70349 ns |
> | (a) `segment`, halves on 1 RISC | 13687 ns | 98596 ns |
> | **(b) `face`, 2×1024 B, halves on 2 RISCs (kept)** | **4193 ns** | **28642 ns** |
> | (b) `face`, halves on 1 RISC | 8051 ns | 56544 ns |
> | (c) `tile`, 1×4096 B, both halves on 1 RISC | 13225 ns | 97407 ns |
>
> (a) with plain `noc_async_write` instead of the stateful API: 9219–9589 ns / 69263–70254 ns, the
> same as stateful. Issue cost does not bind it. The 16 B DRAM write transactions do: they carry
> 4x fewer bytes and still take 2.2–2.5x longer. (c) writes 2x the bytes of (b) from one RISC and is
> ~2x slower than (b) on one RISC. At these shapes the op is bound by output-write DRAM bytes once
> each transaction is at least a face. (b) is the fastest of the three and stays the default.
> Cheapest-traffic line, measured: 8192 B (segment) loses to 32768 B (face) on p1's transfer shape.
