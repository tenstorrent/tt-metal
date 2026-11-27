# L1 Ledger: g6_interleaved_to_sharded

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes are from `op_design.md` → Blocking Model: `image` (folded), `tile_rows`, `tile_cols`, `tile_face_rows`.

The op allocates **no CB and no scratch buffer**. The only per-core L1 involved is the output tensor's own shard. BB-05 counts that shard inside the 1347776 B budget, so it is the single row below. It is the destination the reads land in, not a buffer the op adds.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `output_shard` (the output tensor's L1 shard; **not a CB**: its base address is a common RT arg) | `shard_rows_tiles · shard_cols_tiles` (caller's placement; p1 1·5 = 5, p2 1·4 = 4) | `block_rows · block_cols` pages, all resident at once because the shard is the result (p1 5, p2 4). Pages beyond the tensor on a ragged last shard are allocated but unwritten, and the capacity covers them by the shard contract | `{image: folded into tile_rows, tile_rows: spans → shard_rows_tiles, tile_cols: spans → shard_cols_tiles, tile_face_rows: spans → whole page (32 rows); a trimmed read writes only face rows < trim_span_bytes / 32 of faces 0/1 and leaves the rest unspecified}` | `Float16_b` (bfloat16 32×32 tile, 2048 B). A dataflow-only payload that never passes through DEST, so audit 2 does not apply (no `fp32_dest_acc_en`) | reader (NoC read responses from DRAM) | none in this program: the result tensor, read by the next program | allocated by the entry point before dispatch; outlives the program | It is the result. No other buffer exists to share with, and every read lands in it directly (Rule 3, pattern 1: pack into the destination) |

### Buffers considered and not allocated

| Candidate | Would have held | Why it does not exist |
|-----------|-----------------|-----------------------|
| `cb_output_shard` (globally allocated on the output buffer, `ttnn.cb_descriptor_from_sharded_tensor`) | the same bytes as `output_shard` (0 B added) | Its only role would be to carry the shard address, and no kernel would pop it. A common RT arg carries the same address with one patch per call on the cached descriptor (BB-07), and it adds no CB config to the kernel-config region (BB-04/BB-18) |
| `cb_input_tiles` (staging between a reader and a writer) | `depth · page_size` | Not needed: a NoC read's destination is the core's own L1, and the owner core is the destination, so the page lands in its final place in one step |
| `cb_forward_staging` | `forward_depth · page_size` on helper cores | belongs to the `deferred` `reader_and_forward` regime only |
| compute-path CB pair (`copy` through unpack/pack) | 2 × depth pages | the engine path is forbidden by spec rule 4, and the op computes nothing |

### Symbol table

| Symbol | Bound | Predicate / source establishing it |
|--------|-------|------------------------------------|
| `shard_rows_tiles`, `shard_cols_tiles` | `shard_rows_tiles · shard_cols_tiles · page_size ≤ 1347776 B` | caller's placement. The op cannot change it, and the allocator enforces it at `allocate_tensor_on_device`. Production: 5 and 4 pages (10240 B, 8192 B) |
| `block_rows ≤ shard_rows_tiles`, `block_cols ≤ shard_cols_tiles` | the clipped shard rectangle | `derive_plan`: `min(nominal, tensor extent − start)` |
| `page_size` | 2048 B | bfloat16 32×32 tile (`input_tensor.buffer_aligned_page_size()`) |
| `trim_span_bytes` | `≤ face1_offset_bytes = 512 B` | trim predicate `v ≤ 16` ⇒ `round_up(32·v, 64) ≤ 512` |

### Total per-core footprint

```
L1(core) = shard_rows_tiles · shard_cols_tiles · page_size      # output shard: caller-fixed, scales with no op knob
         + 0                                                    # everything the op adds
p1: 5 · 2048 = 10240 B     p2: 4 · 2048 = 8192 B     (budget 1347776 B, BB-05)
```

No term scales with an op knob: `block_rows`/`block_cols` select pages inside the caller's shard, and `trim_span_bytes` changes bytes moved, not bytes resident. Cores of the grid without a shard still hold the shard allocation, because L1 buffers are lockstep across the grid, and the op adds nothing there.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x (input, DRAM interleaved) | 1 (partial pages for the last tile row of each image) | Owner-fetch: each page is read once, by the core that owns its shard. A trimmed page moves 2 spans of `trim_span_bytes` (rows 0..1 of faces 0 and 1 at the 64 B DRAM read granularity); an interior-row page moves whole | 0 |
| y (output, L1 sharded) | 0 | Lives in L1. The DRAM read response lands at its shard address on its owner | 0 |

Totals per tier:

| Shape | DRAM bytes read | Logical bytes | Cross-core NoC bytes | Core-local L1 bytes written |
|-------|-----------------|---------------|----------------------|-----------------------------|
| p1 (1, 5120) → 32 width shards | 160 × 128 = 20480 | 10240 | 0 | 20480 |
| p2 (1, 1, 1, 128) → 1 height shard | 4 × 128 = 512 | 256 | 0 | 512 |

> Cheapest-traffic split considered: owner-fetch with trimmed last-row pages — 20480 B of DRAM and 0 cross-core bytes on p1 (whole pages: 327680 B; single 576 B span: 92160 B; reader-and-forward: + one NoC write per forwarded page). Implemented: owner-fetch with trimmed last-row pages. **The implemented split is the cheapest.** Nothing is deferred on traffic grounds; `reader_and_forward` is deferred on occupancy grounds only (see the regime table in `op_design.md`).
