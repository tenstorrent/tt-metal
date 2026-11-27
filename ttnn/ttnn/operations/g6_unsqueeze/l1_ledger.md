# L1 Ledger: g6_unsqueeze

The schema and audits are defined in `.claude/references/l1-footprint-discipline.md`. The blocking axes come from `op_design.md` → Blocking Model: `w_tiles`, `lane`, `n_rows`, `new_axis`.

## CB table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_output_tiles` (page = `tile_bytes`) | `staging_depth_blocks * block_w_tiles` | `staging_depth_blocks * block_w_tiles` pages. At depth 2, slot `b` drains its writes while slot `b+1` receives its reads. At depth 1, `block_w_tiles` pages. Production (p1, p2): 1 page. | `{w_tiles: spans → block_w_tiles × staging_depth_blocks; lane: spans → one whole tile page (the 32 lanes land at a 64 B stride in faces 0 and 2, and the store writes the page as one 4096 B transfer); n_rows: spans → 1 (contract domain, host-validated); new_axis: none (extent 1, carries no data)}` | Float32: a dataflow-only payload that never passes through DEST (no compute kernel, so `fp32_dest_acc_en` does not apply). It is the tensor's own format, as the output ABI requires. | mover (NoC reads of the row segments, then the in-place word move) | mover (NoC whole-tile writes) | Whole kernel. A slot is live from `load_row_block(b)` until `store_block(b)`'s writes are flushed (depth 2) or barriered (`drain_stores`). | Shares with nothing, because no other CB exists. The row-segment buffer was removed by landing each segment at the base of the output face that receives it: pattern 1, pack into the destination. The move then runs in place (pattern 2), with sources `{F+4 .. F+60}` and destinations `{F+64 .. F+960}` disjoint for each `F ∈ {0, 2048}`. Capacity exceeds the 128 B of logical values per page by the transfer quantum: the store moves the whole 4096 B page in one transaction (lamp `write_shape` in `op_design.md`). |

## Symbol table

| Symbol | Bound | Predicate / source that establishes it |
|--------|-------|----------------------------------------|
| `tile_bytes` | 4096 | 32×32 float32 tile, `ttnn.tile_size(ttnn.float32)`. SUPPORTED holds float32 only, and the host refuses any tile shape other than 32×32. |
| `L1_BUDGET_BYTES` | 1347776 | contract BB-05, `docs/explorations/g6_unsqueeze/sources/local/contract.md` section 5 (`l1_free_bytes`) |
| `max_single_block_tiles` | `floor(L1_BUDGET_BYTES / tile_bytes)` = 329 | derived from the two above |
| `max_double_block_tiles` | `floor(L1_BUDGET_BYTES / (2 * tile_bytes))` = 164 | derived from the same two |
| `tensor_w_tiles` | `ceil(W / 32)`, unbounded: the contract sets no upper limit on `W` | Appears in no capacity expression. It sets only the core count and the per-core extent. |
| `core_w_tiles_max` | `ceil(tensor_w_tiles / num_work_cores)` | the larger group of `ttnn.split_work_to_cores`. It appears in no capacity expression directly. |
| `block_w_tiles` | `core_w_tiles_max` if `core_w_tiles_max <= max_single_block_tiles`; otherwise `ceil(core_w_tiles_max / ceil(core_w_tiles_max / max_double_block_tiles))`, which is `<= 164` | `_plan()` on the host, the single source. Passed as a CT arg. |
| `staging_depth_blocks` | 1 if `core_w_tiles_max <= max_single_block_tiles`, otherwise 2 | `_plan()` on the host, using the same predicate |

## Total per-core footprint

`L1_op = staging_depth_blocks * block_w_tiles * tile_bytes`

- **Single-block branch** (`core_w_tiles_max <= 329`). This holds for every `W <= 32 * 329 * MAX_CORES`. Then `L1_op = core_w_tiles_max * 4096 <= 1347584 <= L1_BUDGET_BYTES`.
- **Multi-block branch.** `L1_op = 2 * block_w_tiles * 4096 <= 2 * 164 * 4096 = 1343488 <= L1_BUDGET_BYTES`.
- **Production.** p1 and p2 both give `1 * 1 * 4096 = 4096` B per core.
- **Scaling.** The footprint is linear in `block_w_tiles` (the `w_tiles` extent knob) and in `staging_depth_blocks` (the depth knob). Beyond the clamp it is independent of `tensor_w_tiles`, so no unbounded op dimension appears.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` | The logical row crosses once, as `2 * tensor_w_tiles` reads of 64 B = `tensor_w_tiles * 128` B (p1 2048 B, p2 1024 B). Padding rows 1..31 cross 0 times. | Only face row 0 of faces 0 and 1 is read, and it lands directly in the destination staging page, so nothing is re-read. Each tile is owned by exactly one core. | 0 |
| `output` | once: `tensor_w_tiles` writes of 4096 B (p1 65536 B, p2 32768 B) | Every output tile is written once, as a whole page, by the core that loaded its input tile. | 0 |

Totals per call:

| Case | DRAM read | DRAM write | Cross-core | Core-local L1 (move) |
|------|-----------|------------|------------|----------------------|
| p1 `[1, 512]` | 2048 B (32 transactions) | 65536 B (16 transactions) | 0 | 16 × 30 word loads and stores, 30 per core |
| p2 `[1, 256]` | 1024 B (16 transactions) | 32768 B (8 transactions) | 0 | 8 × 30 word loads and stores, 30 per core |

> Cheapest-traffic split considered: `w_tiles` across cores. Inside that split, the byte-cheapest write shape (faces 0 and 2 only) writes `tensor_w_tiles × 2048` B instead of `× 4096` B (p1: −32768 B DRAM write, +16 write transactions; p2: −16384 B, +8). Implemented: `w_tiles` across cores with whole-tile writes. The implemented split is the cheapest split: no candidate split moves fewer bytes, and none has cross-core traffic. The byte-cheaper write shape inside it trades bytes for transactions; it is recorded as perf lamp `write_shape` in `op_design.md`, not as a deferred split.
