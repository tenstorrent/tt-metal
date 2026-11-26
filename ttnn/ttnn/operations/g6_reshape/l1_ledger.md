# L1 Ledger: g6_reshape

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes: `op_design.md` → Blocking Model
(`t_s` → `block_tiles`, `r` → rows of a tile, `fh` → `segments_per_strip`, `e` → `SEGMENT_ELEMS`, map-side axes
`t_m`/`r_m`/`c_m`, staged granules → `staging_granules`). No compute kernel: no DEST, no `fp32_dest_acc_en`; every page holds
raw bytes that never pass an unpacker or a packer (audit 2: dataflow-only payload, page format = tensor dtype as a CB label).

**Inventory (Rule 1).** `scatter_aligned`: one buffer. `gather_*`: two. A pure move needs at least one landing buffer —
the NoC cannot move DRAM → DRAM (a read lands in local L1, a write sources from local L1). "Pack into the destination" is
applied in both directions: `scatter_aligned` reads input face runs into the buffer the writer drains, `gather_aligned`
reads every congruent segment straight into the output tile images, and there is no separate input-tile buffer in any
regime. The staging buffer exists only because a 32-B face row cannot always land at its destination under the 64-B DRAM
read congruence (mechanism cap), and because `gather_general` pieces are not face-row aligned.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_tile_images` (page = `tile_bytes`) | `slot_depth · block_tiles` = `min(TILE_SLOT_DEPTH, max_blocks_per_core) · min(max_core_tiles, BLOCK_TILES_CAP)` | `slot_depth · block_tiles` = capacity: with ≥ 2 blocks per core the reader fills slot `b+1` while the writer drains slot `b`; with 1 block `slot_depth = 1` | `{t_s: spans → block_tiles, r: spans → 32 (a tile image holds every row position; valid rows ≤ 32), fh: spans → 2, e: spans → 16, t_m/r_m/c_m: streams (never resident), staged granules: streams (not in this buffer)}` | input dtype, raw bytes (bfloat16 in Phase 0) | reader | writer | whole kernel; ring of `slot_depth` block slots | cannot share with `cb_granule_staging`: both are live during `assemble_block` (staging is the source, the slot the destination of the same copies). Valid rows per tile can be 1 of 32, but compacting the image would break the identical-offset face-run transfers that make the tile side congruent and coalesced; the cost is bounded by `BLOCK_SLOT_BYTES_CAP` |
| `cb_granule_staging` (page = `granule_bytes`; `gather_*` only) | `staging_granules + ceil(8 · copy_entries / granule_bytes)`, with `staging_granules = min(STAGING_GRANULES_CAP, staged_pieces_bound)`, `copy_entries = min(COPY_TABLE_CAP, staged_pieces_bound)` | ≤ staged pieces between two flushes; = capacity whenever `staged_pieces_bound` is below the caps (every golden shape: one flush per block). `staged_pieces_bound` counts every segment as staged; `gather_aligned` stages ~half — the gap is the cost of a bound that needs no per-segment parity count on the host | `{t_s: streams (flushed per block or when full), r: streams, fh: streams, e: streams, t_m/r_m/c_m: streams → one granule per distinct staged map-side face-row pair, staged granules: spans → staging_granules}` | raw bytes | reader (NoC reads it issues; copy-table stores) | reader (copies out after its own barrier) | whole kernel | cannot alias `cb_tile_images` (concurrent, above); cannot live in the slot's padding rows (a full tile image has none, and the writer drains the slot while the next block stages). Not a pipe: no reserve/push/wait/pop, so it carries no cross-kernel credit |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `tile_bytes` | 2048 (bf16), 4096 (fp32) | dtype ∈ TARGET TILE dtypes |
| `BLOCK_SLOT_BYTES_CAP` | 262,144 B | module constant |
| `BLOCK_TILES_CAP` | `max(1, BLOCK_SLOT_BYTES_CAP div tile_bytes)` = 128 (bf16) / 64 (fp32) | — |
| `block_tiles` | 1 ≤ `block_tiles` ≤ `BLOCK_TILES_CAP` | `min(max_core_tiles, BLOCK_TILES_CAP)`; `max_core_tiles ≥ 1` because `n_cores ≤ U` |
| `max_core_tiles` | ≤ `ceil(U / n_cores) + 1` | strip runs are contiguous (a run touches at most one tile more than its rows fill) |
| `TILE_SLOT_DEPTH`, `slot_depth` | 2; `slot_depth ∈ {1, 2}` | `slot_depth = min(TILE_SLOT_DEPTH, max_blocks_per_core)` |
| `max_block_strips` | ≤ `32 · block_tiles` | `min(max_core_strips, 32 · block_tiles)` |
| `P` | 1 (`gather_aligned`); ≤ 16 (`gather_general`) | `P = min(16, ceil(16 / W_in) + 2)` — pieces per segment |
| `staged_pieces_bound` | `2 · max_block_strips · P` | — |
| `STAGING_GRANULES_CAP`, `COPY_TABLE_CAP` | 512, 1024 | module constants |
| `granule_bytes` | 64 (BH, every TARGET dtype); 32 (WH bf16) | `max(ttnn.get_dram_alignment(), face_row_bytes)` |
| `U`, `N`, `W`, `H`, `IMG` | unbounded by the contract (`N < 2**32` by the index-width cap) | appear in **no** capacity expression except inside `min(·, cap)` — otherwise only in trip counts and RT args |

## Total per-core footprint

`L1 = slot_depth · block_tiles · tile_bytes + [gather_*] (staging_granules · granule_bytes + ceil(8 · copy_entries / granule_bytes) · granule_bytes)`

- `cb_tile_images` scales with `block_tiles` (the `t_s` knob) and `slot_depth`; `cb_granule_staging` scales with
  `max_block_strips · P` up to its caps and is absent in `scatter_aligned`.
- Worst case, any shape: `2 · 128 · 2048 + 512 · 64 + 128 · 64 = 524,288 + 32,768 + 8,192 = 565,248 B` ≤ 1,347,776 B
  (BB-05) — no shape can exceed the budget, so no host refusal by size exists.
- Production (110-core grid, one block per core, `slot_depth = 1`):

| Case | Regime | `block_tiles` | `staging_granules` | L1 per core |
|------|--------|---------------|--------------------|-------------|
| `[16,1,5120]→[1,1,16,5120]` | `gather_aligned` | 3 | 48 | 9,600 B |
| `[8,1,5120]→[1,1,8,5120]` | `gather_aligned` | 3 | 24 | 7,872 B |
| `[1,16,6,256]→[1,1,16,1536]`, `[16,1,1536]→[1,1,16,1536]` | `gather_aligned` | 2 | 14 | 5,120 B |
| `[1,1,16,4608]→[16,1,4608]`, `[1,1,8,4608]→[8,1,4608]` | `scatter_aligned` | 3 | — | 6,144 B |
| `[1,16,64]→[16,1,64]`, `[1,1,6,256]→[1,1,1536]` | `scatter_aligned` | 1 | — | 2,048 B |
| `[16,1,64]→[1,1,16,64]` | `gather_aligned` | 1 | 2 | 2,240 B |
| `[1,4096]→[4096,1]` (bf16 `gather_general` pin) | `gather_general` | 3 | 228 | 22,592 B |

## Data-movement budget

Minimum (boundary DRAM): every logical input byte read once, every logical output byte written once. Bytes for bf16 on
Blackhole (64-B read congruence: a 32-B face row whose source and destination row parities differ is read as its 64-B
granule).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | 1 × its valid face rows; padding rows never read. `gather_aligned`: staged face rows (~50 %) cost a 64-B granule ⇒ ~1.5 × valid bytes. `scatter_aligned`: exactly 1 × (face runs). `gather_general`: each distinct granule once per flush window | the index map is a bijection and the strip runs partition the tile side, so each map-side segment is named by exactly one core; the tile side moves each valid face row once | none (only a `gather_general` granule straddling a core boundary is read by both cores: ≤ 1 granule per boundary) |
| `output` | 1 × its valid face rows; padding rows never written. `gather_general`: padding columns of partial face rows ride along inside their face row | `gather_*` writes one face run per (tile, face, row run) of the slot; `scatter_aligned` writes each valid face row once | none |

Per production shape (valid bytes, transactions issued over the whole op):

| Case | Regime | input valid (whole tiles) | reads | output valid (whole tiles) | writes |
|------|--------|---------------------------|-------|----------------------------|--------|
| `[16,1,5120]→[1,1,16,5120]` | `gather_aligned` | 160 KiB (5.24 MB) | 5120 face rows, ~2560 staged → ~240 KiB | 160 KiB (320 KiB) | 320 face runs of 512 B |
| `[1,1,16,4608]→[16,1,4608]` | `scatter_aligned` | 144 KiB (288 KiB) | 288 face runs of 512 B | 144 KiB (4.7 MB) | 4608 face rows of 32 B |
| `[1,16,6,256]→[1,1,16,1536]` | `gather_aligned` | 48 KiB (256 KiB) | 1536 face rows, ~768 staged → ~72 KiB | 48 KiB (96 KiB) | 96 face runs of 512 B |
| `[2,1,5120]→[1,1,2,5120]` | `gather_aligned` | 20 KiB (640 KiB) | 640 face rows → ~30 KiB | 20 KiB (320 KiB) | 320 face runs of 64 B |
| `[1,1,6,256]→[1,1,1536]` | `scatter_aligned` | 3 KiB (16 KiB) | 16 face runs of 192 B | 3 KiB (96 KiB) | 96 face rows |
| `[16,1,64]→[1,1,16,64]` | `gather_aligned` | 2 KiB (64 KiB) | 64 face rows | 2 KiB (4 KiB) | 4 face runs of 512 B |

Total per tier (chosen split): DRAM ≈ 1.0–1.5 × logical bytes read + 1.0 × logical bytes written; cross-core 0;
core-local: one L1 → L1 RISC-V copy per staged piece.

> Cheapest-traffic split considered: contiguous tile-side strip runs with the owner moving each segment, the tile side
> chosen as the tensor with fewer face runs — it is the minimum on both tensors in bytes (up to granule rounding of staged
> face rows) and the minimum transaction count among the minimum-byte splits. Implemented: that split
> (`gather_aligned` / `scatter_aligned` / `gather_general`). The implemented split is also the cheapest.
