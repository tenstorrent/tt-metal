# L1 Ledger: g6_unsqueeze

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes: `op_design.md` → Blocking Model (`w_tiles`, `n_rows`, `lane`, `new_axis`).

## CB table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_output_tiles` (page = `tile_bytes`) | `staging_depth_blocks * block_w_tiles` | `staging_depth_blocks * block_w_tiles` pages at depth 2: slot `b` is still being flushed out while slot `b+1` receives its loads. `block_w_tiles` pages at depth 1. Production (p1, p2): 1 page. | `{w_tiles: spans → block_w_tiles × staging_depth_blocks; n_rows: spans → 1 (contract domain, host-validated); lane: spans → one whole tile page (32 lanes land in column 0 at a 64 B stride across faces 0 and 2, and the store writes the page as one 4096 B transfer); new_axis: none (extent 1, no data)}` | Float32. This is a dataflow-only payload that never passes through DEST (no compute kernel, `fp32_dest_acc_en` does not apply). The format is the tensor's own, as the output ABI requires. | mover (NoC reads of the row segments, then the in-place word transform) | mover (NoC whole-tile writes) | whole kernel. Slot `s` is live from `load_row_block(b)` until `store_block(b)`'s writes are flushed. | Shares with nothing, because no other CB exists. The row staging buffer was eliminated: segments land at their own tile offsets in this page (pattern 1, pack into the destination) and the transform runs in place (pattern 2), since the source bytes {4..60, 1024..1084} and the destination bytes {64..960, 2048..3008} are disjoint. Capacity exceeds the 128 B of logical values per page (32 × 4 B) by the transfer quantum: the store moves the whole 4096 B tile page in one transaction (lamp `write_shape` in `op_design.md`). |

## Symbol table

| Symbol | Bound | Predicate / source that establishes it |
|--------|-------|----------------------------------------|
| `tile_bytes` | 4096 | float32 32×32 tile; `ttnn.tile_size(ttnn.float32)`; SUPPORTED dtype is float32 only |
| `L1_BUDGET_BYTES` | 1347776 | contract (BB-05, `sources/local/contract.md` §5 `l1_free_bytes`) |
| `max_single_block_tiles` | `floor(L1_BUDGET_BYTES / tile_bytes)` = 329 | derived from the two above |
| `tensor_w_tiles` | `ceil(W / 32)`, unbounded (the contract sets no upper limit on W) | appears in no capacity expression. It only sets the core count and the per-core trip count. |
| `core_w_tiles_max` | `ceil(tensor_w_tiles / min(tensor_w_tiles, max_cores))` | `ttnn.split_work_to_cores` group 1; appears in no capacity expression directly |
| `block_w_tiles` | `core_w_tiles_max` if `core_w_tiles_max <= max_single_block_tiles`, otherwise `floor(L1_BUDGET_BYTES / (2 * tile_bytes))` = 164 | `_plan()` on the host, the single source; passed as a CT arg |
| `staging_depth_blocks` | 1 if `core_w_tiles_max <= max_single_block_tiles`, otherwise 2 | `_plan()` on the host, the same predicate |

## Total per-core footprint

`L1_op = staging_depth_blocks * block_w_tiles * tile_bytes`

- The predicate `core_w_tiles_max <= 329` holds for every W up to `32 * 329 * max_cores`. Then `L1_op = core_w_tiles_max * 4096 <= 1347584 <= L1_BUDGET_BYTES`.
- Otherwise `L1_op = 2 * 164 * 4096 = 1343488 <= L1_BUDGET_BYTES`.
- Production: p1 and p2 both give `1 * 1 * 4096 = 4096` B per core.
- Scaling: linear in `block_w_tiles` (the `w_tiles` extent knob) and in `staging_depth_blocks` (the depth knob). It is independent of `tensor_w_tiles` beyond the clamp, so no unbounded op dimension appears.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | Its logical row crosses once: `2 * tensor_w_tiles` reads of 64 B = `tensor_w_tiles * 128` B (p1 2048 B, p2 1024 B). Padding rows 1..31 cross 0 times. | Only row 0 is read, and it lands directly in the destination staging page, so nothing is re-read. Each tile is owned by exactly one core. | 0 |
| `output` | once: `tensor_w_tiles` writes of 4096 B (p1 65536 B, p2 32768 B) | Every output tile is written once, as a whole page, by the core that loaded its input tile. | 0 |

Totals per call:

| Case | DRAM read | DRAM write | Cross-core | Core-local L1 (transform) |
|------|-----------|------------|------------|---------------------------|
| p1 `[1, 512]` | 2048 B (32 transactions) | 65536 B (16 transactions) | 0 | 16 × 31 word loads + stores |
| p2 `[1, 256]` | 1024 B (16 transactions) | 32768 B (8 transactions) | 0 | 8 × 31 word loads + stores |

> Cheapest-traffic split considered: `w_tiles` across cores. Its byte-cheapest write shape (faces 0 and 2 only) writes `tensor_w_tiles × 2048` B instead of `× 4096` B (p1: −32768 B DRAM write, +16 write transactions; p2: −16384 B, +8). Implemented: `w_tiles` across cores, whole-tile writes. That is the same split, so the implemented split is the cheapest split: no candidate split moves fewer bytes, and none has cross-core traffic. The byte-cheaper write shape inside it trades bytes for transactions and is recorded as perf lamp `write_shape` in `op_design.md`, not as a deferred split.
