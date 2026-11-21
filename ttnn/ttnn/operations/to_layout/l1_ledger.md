# L1 Ledger: to_layout

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking: `op_design.md` → Blocking Model.
Target: Quasar (`.claude/references/targets/quasar.md`), worked values for bfloat16 (`tile_bytes = 2048`).
Thread plan (1, 1, 1), so every capacity below is per NEO × C with C = 1.

Named block axes (from the Blocking Model): `tile_row` (extent `block_height_tiles`), `tile_col` (extent
`block_width_tiles` = `bw`), `row_in_tile` (fixed 32), `col_in_tile` (fixed 32), `page` (identity regime,
extent `copy_block_pages`).

Each program holds the rings of **one** regime. Regimes are separate `ProgramSpec`s and are never
co-resident.

## Inventory decisions (Rule 1, before any budget)

| Phase boundary | Could it be storage-free? | Decision |
|----------------|---------------------------|----------|
| DRAM sticks → tilize | Not without a buffer: the unpacker reads only L1 | `cb_row_block` |
| tilize → DRAM tiles | Pack into the destination: the helper packs straight into the ring the writer drains, with no scratch ring between them | `cb_tile_block` (the destination) |
| tilize input vs output in one ring (transform in place) | No: `tilize_helpers.inl:89-90` `static_assert(input_dfb != output_dfb)`. Tiles and rows are different byte orders of the same block, and the unpacker reads the whole 32-row region while the packer writes tile by tile | two rings |
| DRAM tiles → untilize → DRAM sticks | same reasoning, mirrored (`untilize_helpers.inl:153-154`) | `cb_tile_block_in`, `cb_row_block_out` |
| padding rows of a partial tile-row | zeroed in place inside `cb_row_block` (`async_write_zeros`), with no zero source buffer | no extra ring |
| identity copy | DM → DM: one ring, no compute | `cb_pages` |
| scaler / mask / constant | none (no arithmetic) | — |

Minimum inventory: 2 rings for tilize, 2 for untilize, 1 for identity. No budget solve is needed on any
PHASE0 shape. Step 2 of `select_block_width_tiles` is a **cap**, not a search: occupancy (step 4) sets
`bw`, and the cap only bites for `tensor_w_tiles > 255` with `total_tile_rows ≥ 32`.

## Ledger

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_row_block` (tilize) | `BUFFER_DEPTH · block_height_tiles · bw` = `2·bw` (pages of `tile_bytes`) | `block_height_tiles · bw` pages: one whole 32-row band, because the unpack-tilize MOP reads all 32 rows of the band as one region (`llk_unpack_tilize_api.h:71-107`). Capacity − live = one more block, for double buffering (reader fills k+1 while k is tilized) | `{tile_row: streams → block_height_tiles per block; tile_col: spans → bw; row_in_tile: spans → 32 (bw entries = 32 rows × chunk_bytes); col_in_tile: spans → 32; page: n/a (identity only)}` | Float16_b (bfloat16): a dataflow-only payload into the unpacker, and the input dtype. DEST is 16-bit (`enable_32_bit_dest=False`), so the width matches | reader | compute | whole program | `cb_tile_block`: lifetimes concurrent (pipelined: block k is packed into `cb_tile_block` while block k+1 lands here), and the helper forbids in-place (`tilize_helpers.inl:89-90`) |
| `cb_tile_block` (tilize) | `2·bw` | `bw` tiles (one tile-row band). Capacity − live = double buffer (writer drains k while k+1 is packed) | `{tile_row: streams; tile_col: spans → bw; row_in_tile: spans → 32; col_in_tile: spans → 32; page: n/a}` | Float16_b: output dtype, packed from 16-bit DEST | compute | writer | whole program | `cb_row_block`: concurrent (see above) |
| `cb_tile_block_in` (untilize) | `2·bw` | `bw` tiles: `pack_untilize` writes each output row across all `bw` tiles of the band (`untilize_helpers.inl:264-270`). In the block-based path the helper's live input is `sub_block_width` ≤ `bw`, and the capacity stays `bw`-sized because the reader posts per block. Capacity − live = double buffer, which is also the two implicit-sync txn-id halves | `{tile_row: streams; tile_col: spans → bw; row_in_tile: spans → 32; col_in_tile: spans → 32; page: n/a}` | Float16_b: input dtype, read by the unpacker into 16-bit DEST | reader | compute | whole program | `cb_row_block_out`: concurrent (pipelined), and the helper forbids in-place (`untilize_helpers.inl:153-154`) |
| `cb_row_block_out` (untilize) | `2·bw` | `bw` pages = one band of 32 rows × `chunk_bytes`, all written before the writer may start (the band is complete only after the last tile's untilize). Capacity − live = double buffer | `{tile_row: streams; tile_col: spans → bw; row_in_tile: spans → 32; col_in_tile: spans → 32; page: n/a}` | Float16_b: output dtype, packed from 16-bit DEST | compute | writer | whole program | `cb_tile_block_in`: concurrent (see above) |
| `cb_pages` (identity) | `BUFFER_DEPTH · copy_block_pages` = 8 (pages of `buffer_page_size()`) | 1 page per transfer in flight. Capacity − live = the in-flight window: `copy_block_pages` pages per implicit-sync txn-id half × 2 halves (transfer quantum + double buffering, catalog `double_buffer` 4–8) | `{tile_row: n/a (regime has no tile-row block); tile_col: n/a; row_in_tile: n/a; col_in_tile: n/a; page: streams → window copy_block_pages}` | none (DM-only ring: metadata omitted; bytes never pass DEST) | reader | writer | whole program | nothing else in the program. It is the identity regime's only ring |

## Symbol table

| Symbol | Meaning | Bound | Predicate / source establishing it |
|--------|---------|-------|------------------------------------|
| `bw` = `block_width_tiles` | tile-columns per block | `bw_min ≤ bw ≤ bw_max`, `bw` divides `tensor_w_tiles` | `select_block_width_tiles` steps 1–4 (`op_design.md`). Quasar bf16: ≤ 255. PHASE0 values 2–16 |
| `block_height_tiles` | tile-rows per block | = 1 (Phase 0) | host constant `BLOCK_HEIGHT_TILES` |
| `BUFFER_DEPTH` | blocks per ring | = 2 | host constant |
| `copy_block_pages` | pages per txn-id half (identity) | = 4 | host constant `COPY_BLOCK_PAGES` |
| `tile_bytes` | TILE-side page size | 2048 (bf16 32×32) | `buffer_page_size()` of the TILE-side tensor |
| `page_bytes` | identity page size | ≤ `W·elem_size` aligned (RM stick; ≤ 16 KiB for W = 8192 bf16) or `tile_bytes` | `input_tensor.buffer_page_size()` |
| `chunk_bytes` | bytes of one row of one block | `bw·32·elem_size`; multiple of the DRAM alignment (64) | derived from `bw`, step 3 |
| `DFB_RING_MAX_BYTES` | one ring's size limit | 1,048,560 B | target page, *Cluster model* |
| `dfb_budget` | per-cluster bytes the rings may take | `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES`. The query's value on Quasar is not recorded in the profile (≤ 4,194,304 B); `L1_RESERVE_BYTES = 1 MiB` leaves room for L1-resident input/output tensors (R3/R5) | run-time query + host constant |
| `tensor_w_tiles`, `total_tile_rows`, `num_pages` | op dimensions | never in a capacity expression: they only set unit counts and the `bw` choice | — |

## Total per-cluster footprint

- tilize / untilize: `2 rings × BUFFER_DEPTH × block_height_tiles × bw × tile_bytes` = `8192 · bw` B (bf16).
  This scales linearly with the `bw` and `BUFFER_DEPTH` knobs and with `block_height_tiles`, and is independent of every op dimension.
  PHASE0 worst case `bw = 16`: **128 KiB**. Cap at `bw = 255`: 2,088,960 B, under the ring cap per ring
  (1,044,480 B ≤ 1,048,560 B) and under `dfb_budget`.
- identity: `BUFFER_DEPTH × copy_block_pages × page_bytes` = `8 · page_bytes`; PHASE0 worst case (RM stick, W = 8192
  bf16 = 16 KiB) **128 KiB**. Scales with `copy_block_pages`, `BUFFER_DEPTH` and the page size.

## Data-movement budget

Boundary: the tensor's home (DRAM banks, or L1 banks for an L1 tensor). `S_rm = num_images·H·W·elem` (logical
row-major bytes), `S_tile = num_images·ceil(H/32)·ceil(W/32)·tile_bytes` (tile bytes including padding rows).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input (tilize, RM) | 1 × `S_rm` | each stick chunk is read once into the block that tilizes it. Padding rows are zeroed in L1, never read | none |
| output (tilize, TILE) | 1 × `S_tile` | each tile is packed once and written once | none |
| input (untilize, TILE) | 1 × `S_tile` | each tile is read once. The padding rows of each image's last tile-row cross too (the tile page is the transfer unit) | none |
| output (untilize, RM) | 1 × `S_rm` | only valid rows are written, each once | none |
| input / output (identity) | 1 × size / 1 × size | page copy, each page once | none |

Totals per tier — tilize: DRAM `S_rm + S_tile`, NoC cross-core 0, core-local L1: every block byte is written into
`cb_row_block` once, read by the unpacker once, packed once and read by the writer once. Untilize is the mirror. Identity:
DRAM `2 × size`, cross-core 0.

> Cheapest-traffic split considered: `tile_row × tile_col` (2-D, independent, no combine): `S_rm + S_tile`
> DRAM, 0 cross-core. Implemented: the same split. **The implemented split is the cheapest.** Every independent
> split moves the same bytes, and the dependent split (`row_in_tile` across cores) adds one cross-core crossing
> of the whole tensor. Within the cheapest class the choice is occupancy (step 4), then transfer size (step 1
> floor).
