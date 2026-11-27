# L1 Ledger: g6_reshape

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes: `op_design.md` → Blocking Model.

| Axis | Block knob |
|------|------------|
| `t_s` | `block_tiles` |
| `strip` | strips of the lane's run in the block |
| `r` | `strip_rows` |
| `fh` | segments per row |
| `e` | 16 |
| map-side `t_m` / `r_m` / `c_m` | none |
| staged granules | `staging_slots`, `copy_entries` |

There is no compute kernel, so there is no DEST and no `fp32_dest_acc_en`. Every page holds raw bytes that never pass an unpacker or a packer (audit 2: a dataflow-only payload; the page format is the tensor dtype as a CB label). Every CB is lane-private scratch with no CB credit: one RISC-V fills and drains it.

**Inventory (Rule 1).**

- `scatter_aligned` needs one buffer per lane; `gather_*` needs two per lane.
- A pure move needs at least one landing buffer: the NoC cannot move DRAM → DRAM, because a read lands in local L1 and a write sources from local L1.
- "Pack into the destination" applies in both directions. `scatter_aligned` reads input face runs into the buffer its own writes drain. `gather_aligned` reads every congruent segment straight into the output tile images.
- No regime has a separate input-tile buffer.
- The staging buffer exists for two reasons only: a 32-B face row cannot always land at its destination under the 64-B DRAM read congruence (a mechanism cap), and `gather_general` pieces are not face-row aligned.
- The lane count doubles every row. That is the price of two independent issue streams per core; the alternative, a reader/writer pipeline, needs the same two slots per core plus CB credit. Total stays below 12 KiB per core on every production shape.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_lane0_tile_images` (page = `tile_bytes`) | `slot_depth · block_tiles` = `min(SLOT_DEPTH, max_blocks_per_lane) · min(max_lane_tiles, BLOCK_TILES_CAP)` | `slot_depth · block_tiles` = capacity: one slot per block in flight; with `slot_depth = 2` (read-ahead) a slot drains its writes while the next fills | `{t_s: spans → block_tiles, strip: spans → all strips of a tile (a tile image holds every row position), r: spans → strip_rows, fh: spans → 2, e: spans → 16, t_m/r_m/c_m: streams (never resident), staged granules: streams (not in this buffer)}` | input dtype, raw bytes (bfloat16 in Phase 0) | lane 0 (NCRISC) | lane 0 (NCRISC) | whole kernel; ring of `slot_depth` block slots | Cannot share with `cb_lane0_staging`: both are live during `assemble_block` (staging is the source, this slot the destination of the same copies). Cannot share with `cb_lane1_tile_images`: the two lanes run concurrently on disjoint strips. A tile image keeps all 32 row positions even when a lane holds 2 of them, because compacting it would break the identical-offset transfers that keep the tile side congruent and coalesced. The cost is bounded by `BLOCK_SLOT_BYTES_CAP` |
| `cb_lane1_tile_images` (page = `tile_bytes`) | as `cb_lane0_tile_images` | as `cb_lane0_tile_images` | as `cb_lane0_tile_images` | raw bytes | lane 1 (BRISC) | lane 1 (BRISC) | whole kernel | concurrent with lane 0's buffers (above); not allocated when `LANES_PER_CORE = 1` |
| `cb_lane0_staging` (page = `granule_bytes`; `gather_*` only) | `staging_slots + ceil(8 · copy_entries / granule_bytes)`, with `staging_slots = min(STAGING_SLOTS_CAP, staged_bound)` and `copy_entries = min(COPY_ENTRIES_CAP, staged_bound)` | ≤ staged pieces between two flushes. It equals capacity whenever `staged_bound` is below the caps (every golden shape: one flush per block). `staged_bound` counts every segment as staged, but `gather_aligned` stages about half; the gap is the cost of a bound that needs no per-segment parity count on the host | `{t_s: streams (flushed per block or when full), strip: streams, r: streams, fh: streams, e: streams, t_m/r_m/c_m: streams → one slot per distinct staged map-side granule, staged granules: spans → staging_slots}` | raw bytes | lane 0 (the NoC reads it issues; copy-table stores) | lane 0 (copies out after its own barrier) | whole kernel | Cannot alias `cb_lane0_tile_images` (concurrent, above). Cannot live in the slot's padding rows: a full tile image has none, and a staged granule must be read before the slot rows it feeds are known to be free. Cannot share with lane 1 (concurrent). Not a pipe: no reserve/push/wait/pop |
| `cb_lane1_staging` (page = `granule_bytes`; `gather_*` only) | as `cb_lane0_staging` | as `cb_lane0_staging` | as `cb_lane0_staging` | raw bytes | lane 1 (BRISC) | lane 1 (BRISC) | whole kernel | concurrent with lane 0's buffers; not allocated when `LANES_PER_CORE = 1` |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `tile_bytes` | 2048 (bf16), 4096 (fp32) | dtype ∈ the TARGET TILE dtypes |
| `granule_bytes` | 64 (BH, every TARGET dtype); 32 (WH bf16) | `max(ttnn.get_dram_alignment(), face_row_bytes)` |
| `strip_rows` | 1 or 2 | `granule_bytes div face_row_bytes` |
| `BLOCK_SLOT_BYTES_CAP` | 131,072 B | module constant |
| `BLOCK_TILES_CAP` | `max(1, BLOCK_SLOT_BYTES_CAP div tile_bytes)` = 64 (bf16) / 32 (fp32) | — |
| `block_tiles` | 1 ≤ `block_tiles` ≤ `BLOCK_TILES_CAP` | `min(max_lane_tiles, BLOCK_TILES_CAP)`; `max_lane_tiles ≥ 1` because `n_lanes ≤ U` |
| `max_lane_tiles` | ≤ `ceil(max_strips_per_lane · strip_rows / 32) + 1` | strip runs are contiguous in tile-major order; the host computes it exactly per lane |
| `SLOT_DEPTH`, `slot_depth` | `SLOT_DEPTH ∈ {1, 2}`; `slot_depth ∈ {1, 2}` | `slot_depth = min(SLOT_DEPTH, max_blocks_per_lane)` |
| `max_block_rows` | ≤ `32 · block_tiles` | `min(max_lane_rows, 32 · block_tiles)` |
| `P` | 1 (`gather_aligned`); ≤ 16 (`gather_general`) | `P = min(16, ceil(16 / W_in) + 2)`, pieces per segment |
| `staged_bound` | `2 · max_block_rows · P` | — |
| `STAGING_SLOTS_CAP`, `COPY_ENTRIES_CAP` | 512, 1024 | module constants |
| `LANES_PER_CORE` | 1 or 2 | module constant |
| `U`, `N`, `W`, `H`, `IMG` | unbounded by the contract (`N < 2**32` by the index-width cap) | they appear in **no** capacity expression except inside `min(·, cap)`; otherwise they appear only in trip counts and RT args |

## Total per-core footprint

```
L1 = LANES_PER_CORE · ( slot_depth · block_tiles · tile_bytes
                        + [gather_*] (staging_slots + ceil(8 · copy_entries / granule_bytes)) · granule_bytes )
```

- `cb_laneK_tile_images` scales with `block_tiles` (the `t_s` knob) and `slot_depth`. `cb_laneK_staging` scales with `max_block_rows · P` up to its caps and is absent in `scatter_aligned`. Both scale linearly with `LANES_PER_CORE`.
- Worst case for any shape and any N: `2 · (2 · 64 · 2048 + 512 · 64 + 128 · 64) = 2 · (262,144 + 32,768 + 8,192) = 606,208 B` ≤ 1,347,776 B (BB-05). No shape can exceed the budget, so there is no host refusal by size. With the Phase 0 `SLOT_DEPTH = 1` the worst case is 344,064 B.
- Production (110-core grid, 220 lanes, one block per lane, `slot_depth = 1`):

| Case | Regime | max strips / lane | `block_tiles` | `staging_slots` | L1 per core |
|------|--------|-------------------|---------------|-----------------|-------------|
| `[16,1,5120]→[1,1,16,5120]` | `gather_aligned` | 6 | 2 | 24 | 11,648 B |
| `[8,1,5120]→[1,1,8,5120]` | `gather_aligned` | 3 | 2 | 12 | 9,984 B |
| `[1,16,6,256]→[1,1,16,1536]`, `[16,1,1536]→[1,1,16,1536]` | `gather_aligned` | 2 | 1 (every 2-strip run starts on an even strip, so no lane crosses a tile; exact per-lane `max_lane_tiles`) | 8 | 5,248 B |
| `[1,8,6,256]→[1,1,8,1536]`, `[8,1,1536]→…`, `[2,1,5120]→…`, `[16,1,64]→…`, `[8,1,64]→…` | `gather_aligned` | 1 | 1 | 4 | 4,736 B |
| `[1,1,16,4608]→[16,1,4608]`, `[1,1,8,4608]→[8,1,4608]` | `scatter_aligned` | 6 / 3 | 2 | — | 8,192 B |
| `[1,16,64]→[16,1,64]`, `[1,8,64]→…`, `[1,1,6,256]→[1,1,1536]` | `scatter_aligned` | 1 | 1 | — | 4,096 B |
| `[1,4096]→[4096,1]` (bf16 `gather_general` pin) | `gather_general` | 10 | 2 | 120 | 25,472 B |

## Data-movement budget

Minimum (boundary DRAM): every logical input byte read once and every logical output byte written once. Bytes below are bf16 on Blackhole. With the 64-B read congruence, a 32-B face row whose source and destination row parities differ is read as its 64-B granule.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | 1 × its valid face rows; padding rows are never read. `gather_aligned`: staged face rows (~50 %) cost a 64-B granule ⇒ ~1.5 × valid bytes. `scatter_aligned`: exactly 1 × (merged face runs). `gather_general`: each distinct granule once per flush window per lane | the index map is a bijection and the strip runs partition the tile side, so exactly one lane names each map-side segment; the tile side moves each valid face row once | none. Only a `gather_general` granule that straddles a lane boundary is read by both lanes: ≤ 1 granule per boundary |
| `output` | 1 × its valid face rows; padding rows are never written. `gather_general`: padding columns of partial face rows ride along inside their face row | `gather_*` writes one merged run per (lane, tile, face run); `scatter_aligned` writes each valid face row once | none |

Per production shape (valid bytes; transactions issued over the whole op):

| Case | Regime | input valid (whole tiles) | reads | output valid (whole tiles) | writes |
|------|--------|---------------------------|-------|----------------------------|--------|
| `[16,1,5120]→[1,1,16,5120]` | `gather_aligned` | 160 KiB (5.24 MB) | 5120 face rows, ~2560 staged → ~240 KiB | 160 KiB (320 KiB) | ≤ 880 merged runs (≤ 4 per lane) |
| `[1,1,16,4608]→[16,1,4608]` | `scatter_aligned` | 144 KiB (288 KiB) | ≤ 880 merged runs | 144 KiB (4.7 MB) | 4608 face rows of 32 B |
| `[1,16,6,256]→[1,1,16,1536]` | `gather_aligned` | 48 KiB (256 KiB) | 1536 face rows, ~768 staged → ~72 KiB | 48 KiB (96 KiB) | ≤ 880 merged runs |
| `[2,1,5120]→[1,1,2,5120]` | `gather_aligned` | 20 KiB (640 KiB) | 640 face rows, 320 staged → ~30 KiB | 20 KiB (320 KiB) | 320 runs of 64 B |
| `[1,1,6,256]→[1,1,1536]` | `scatter_aligned` | 3 KiB (16 KiB) | 48 runs of 64 B | 3 KiB (96 KiB) | 96 face rows |
| `[16,1,64]→[1,1,16,64]` | `gather_aligned` | 2 KiB (64 KiB) | 64 face rows, ~32 staged | 2 KiB (4 KiB) | 32 runs of 64 B |

Total per tier (chosen split):

| Tier | Traffic |
|------|---------|
| DRAM | ≈ 1.0–1.5 × logical bytes read + 1.0 × logical bytes written |
| Cross-core | 0 |
| Core-local | one RISC-V L1 → L1 copy per staged piece |

> Cheapest-traffic split considered: contiguous tile-side strip runs over two lanes per core, each lane moving its own
> segments, with the tile side chosen as the tensor with fewer tile-side transfers. It is the minimum on both tensors in
> bytes (up to granule rounding of staged face rows), and the minimum transaction count among the minimum-byte splits.
> Implemented: that split (`gather_aligned` / `scatter_aligned` / `gather_general`). The implemented split is also the
> cheapest.
