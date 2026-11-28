# L1 Ledger: g6_unsqueeze

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes are those of `op_design.md` → Blocking Model: `lead`, `w_tile`, `tile_half`, `face_lane`, `new_axis`.

Inventory. Each mover needs exactly one buffer.
- No input-row buffer: the 64-B gather lands in row 0 of the destination face slot, and the row → column move happens in place inside that slot (Rule 3, patterns 1 and 2).
- No output buffer: the slot is the source of the face write.
- No compute CBs: there is no compute kernel.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_mover0_face_slots` | `staging_depth · block_w_tiles · block_tile_halves` pages of `face_bytes`. Phase 0, p1 and p2: `1 · 1 · 1` = 1 page = 1024 B. | = capacity. A block's slots are gathered together, moved, and written; each slot is live from its gather until its write has left L1. With `staging_depth = 2`, two blocks are live at once by construction. | `{lead: streams → 1 (extent 1), w_tile: spans → block_w_tiles · staging_depth, tile_half: spans → block_tile_halves, face_lane: spans → face_h rows of one face (16 values in column 0), new_axis: streams → 1 (extent 1)}` | Float32. A dataflow-only payload that never passes through DEST (there is no compute kernel), so audit 2 does not apply. The output ABI is float32. | `mover0` (RISCV_1, NoC0) | `mover0` (RISCV_1, NoC0) | whole kernel run | Cannot share with `cb_mover1_face_slots`: that CB is live at the same time on another thread, and CB ownership allows one thread per role. A page is a whole face, 16× the 64 B of values it carries. Reason: transfer quantum. The output face is the smallest single write holding its 16 column-0 rows. A 64-B page would turn each face write into 16 writes (regime `column_strip_writes`, rejected). |
| `cb_mover1_face_slots` | `staging_depth · block_w_tiles · block_tile_halves` pages of `face_bytes`. Phase 0, p1 and p2: 1 page = 1024 B. | = capacity, for the same reason as `cb_mover0_face_slots` | `{lead: streams → 1 (extent 1), w_tile: spans → block_w_tiles · staging_depth, tile_half: spans → block_tile_halves, face_lane: spans → face_h rows of one face (16 values in column 0), new_axis: streams → 1 (extent 1)}` | Float32, dataflow-only (as above) | `mover1` (RISCV_0, NoC1) | `mover1` (RISCV_0, NoC1) | whole kernel run | Cannot share with `cb_mover0_face_slots`: the two lifetimes coincide and the owning threads differ. The page is a whole face for the same transfer-quantum reason. With `movers_per_core = 1` this CB does not exist and `cb_mover0_face_slots` carries `block_tile_halves = 2`. |

## Symbol table

| Symbol | Bound | Predicate / source that establishes it |
|--------|-------|----------------------------------------|
| `face_bytes` | `face_h · face_w · elem_bytes` = 1024 | `face_h = face_w = 16` (`tile_constants.h:21-22`); `elem_bytes = 4` because SUPPORTED dtype = float32 |
| `face_row_bytes` | `face_w · elem_bytes` = 64 | as above; must be a multiple of the 64-B DRAM read alignment (host assert) |
| `block_w_tiles` | `1 ≤ block_w_tiles ≤ block_w_tiles_cap` | plan function: `min(max_core_w_tiles, block_w_tiles_cap)` |
| `block_w_tiles_cap` | `floor(op_l1_budget_bytes / (movers_per_core · staging_depth · block_tile_halves · face_bytes))` = 658 at Phase 0 | L1 mechanism cap (op_design.md → Mechanism caps) |
| `staging_depth` | ∈ {1, 2}; Phase 0 = 1 | host constant |
| `movers_per_core` | ∈ {1, 2}; Phase 0 = 2 | host constant |
| `block_tile_halves` | `tile_halves / movers_per_core` ∈ {2, 1}; Phase 0 = 1 | derived |
| `op_l1_budget_bytes` | 1347776 | spec rule BB-05 (L1 the model leaves to the op) |
| `tensor_w_tiles`, `core_w_tiles` | unbounded (W has no upper limit) | in no capacity expression. They reach the kernel only as the per-core page count, which the block loop divides into `num_blocks_this_core` blocks. |

## Total per-core footprint

`L1(core) = movers_per_core · staging_depth · block_w_tiles · block_tile_halves · face_bytes = tile_halves · staging_depth · block_w_tiles · face_bytes`

- Scales linearly with `block_w_tiles` and `staging_depth`. It does not depend on `movers_per_core`, which only redistributes the halves between RISCs, nor on any op dimension.
- Phase 0, p1 and p2: `2 · 1 · 1 · 1024` = **2048 B** per mover core. Cores outside the mover rectangle use 0 B.
- Largest reachable (huge W, block clamped): `2 · 1 · 658 · 1024` = 1347584 B ≤ 1347776 B.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input `x` | Once, for row 0 only: `2 · face_row_bytes` = 128 B per page, in 2 reads. p1: 2048 B of 65536 B allocated; p2: 1024 B of 32768 B. | Each face row is read once by the mover that owns its half. The 64-B read granule is exactly one face row, so no padding row (rows 1–31) crosses. | none |
| output | Once, for faces 0 and 2 only: `2 · face_bytes` = 2048 B per page, in 2 writes. p1: 32768 B of 65536 B; p2: 16384 B of 32768 B. | Each output face that holds a value is written once from its slot. Faces 1 and 3 are pure padding and are never written. | none |
| (core-local L1) | — | 15 word moves (60 B read, 60 B written) per (page, half), done by RISC-V loads and stores | — |

Totals per tier:

| Case | DRAM | Cross-core NoC | Core-local L1 (RISC-V moves) |
|------|------|----------------|------------------------------|
| p1 `(1,512)` | 34816 B in 64 transactions (32 reads × 64 B, 32 writes × 1024 B) | 0 B | 1920 B read + 1920 B written over 16 cores |
| p2 `(1,256)` | 17408 B in 32 transactions | 0 B | 960 B + 960 B over 8 cores |

> Cheapest-traffic split considered: `w_tile` across cores × `tile_half` across each core's two data-movement RISCs — 0 B delta per tier; it is the DRAM-boundary minimum for one-transaction face writes. Implemented: the same split, so the implemented split is the cheapest. Deferred: nothing on traffic grounds. `role_split_pipeline` (a deferred regime row) moves the same bytes and changes only which NoC carries reads and writes at DRAM-bound W, which no golden shape reaches.
