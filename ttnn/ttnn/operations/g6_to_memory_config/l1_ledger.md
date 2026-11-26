# L1 ledger: g6_to_memory_config

Block axes (from op_design.md Blocking Model): `tile_row`, `tile_col`, `page`, `run`.
No compute kernel and no DEST use: `fp32_dest_acc_en` does not apply; every page format is the
tensor's own dtype carried as bytes (never unpacked).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_staged_pages` (STAGE) | `stage_depth * stage_block_pages`, `stage_block_pages = min(core_pages_max, STAGE_BLOCK_PAGES_CAP)` | `2 * stage_block_pages` (one block landing, one draining) | `{tile_row: streams → flat range, tile_col: streams → flat range, page: spans → whole page, run: streams → 1 page (interleaved ends)}` — the flat block spans `stage_block_pages` of the folded `tile_row×tile_col` range | input dtype (Float16_b / Bfp8_b / Bfp4_b bytes) | reader | writer | whole STAGE kernel | none: only CB the op allocates; PULL/PUSH do not instantiate it (regimes are disjoint programs). Capacity = live set × 2 because of double buffering (overlap of NoC0 reads with NoC1 writes) |
| `cb_output_shard` (PULL) | output shard pages (`shard_tile_rows * shard_tile_cols`, nominal) | the whole shard (every page written in place) | `{tile_row: spans → block_tile_rows, tile_col: spans → block_tile_cols, page: spans, run: streams}` | output dtype | pull kernel (lands NoC reads) | none in-program | program | aliases the **output tensor's shard** (zero-copy `cb_descriptor_from_sharded_tensor`) — no allocation of its own; it is the output, counted once below as output L1 |
| `cb_input_shard` (PUSH) | input shard pages (nominal) | the whole shard (read in place) | `{tile_row: spans → block_tile_rows, tile_col: spans → block_tile_cols, page: spans, run: streams}` | input dtype | none in-program (model's tensor) | push kernel | program | aliases the **input tensor's shard** — no allocation; the input is the model's tensor and is not counted (contract) |

### Symbol table

| Symbol | Bound | Predicate establishing it |
|---|---|---|
| `page_size` | ≤ 2048 B (bf16), 1088 (bf8), 576 (bf4) | TILE layout, dtype ∈ TARGET |
| `STAGE_BLOCK_PAGES_CAP` | 16 | host constant |
| `stage_depth` | 2 | host constant |
| `core_pages_max` | ≤ `ceil(num_pages / num_stage_cores)` | `split_work_to_cores`; only `min` with the cap enters the CB |
| output shard pages | ≤ 20 bf16 pages (40960 B) in the golden INPUTS (largest: [32, 640]) | output shard spec; BB-05 predicate below |
| output L1-interleaved pages per core | `ceil(num_pages / num_l1_banks)`; ≤ 94 bf16 pages (192512 B) in the golden INPUTS (2078) | allocator round-robin; BB-05 predicate below |

### Total per-core footprint

```
op_l1(regime) = (regime == STAGE) ? stage_depth * stage_block_pages * page_size : 0
output_l1     = (out in L1) ? (out sharded ? shard_pages * page_size
                                           : ceil(num_pages / num_l1_banks) * page_size) : 0
BB-05 predicate (host refusal if false):  op_l1 + output_l1 <= 1347776
```
Scaling: `op_l1` scales with `stage_block_pages` and `stage_depth` only (≤ 2·16·2048 = 65536 B); `output_l1`
scales with the placement, not with any op knob. Worst golden case: 2078 STAGE, 65536 + 192512 = 258048 B.
PULL and PUSH allocate nothing.

## Data-movement budget

Phase 0 (bfloat16). DRAM crossings counted per whole tensor.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input, DRAM-interleaved source (PULL 16 cases, STAGE 27) | 1 | each page read by exactly one mover, never re-read | PULL: 0 beyond the read itself (lands in the owner's shard); STAGE: +1 NoC write per page (mover → L1 bank) |
| input, L1 source (PULL from L1-int/L1-sharded 15, PUSH 28) | 0 | L1 resident | PULL: 1 NoC read per maximal run; PUSH: 0 reads (local shard) |
| output, DRAM-interleaved target (PUSH 19) | 1 | each page written once by its source owner | 1 NoC write per page (interleaved → run 1) |
| output, L1 target (PULL 31, PUSH 9 to L1-int, STAGE 27) | 0 | — | as above; STAGE pays 2 NoC crossings per page |
| output, DRAM width-sharded (refinement, STAGE) | 1 | written once | 1 NoC read (DRAM) + 1 NoC write (DRAM) per page — structural minimum DRAM→DRAM |

Totals per tier: DRAM = `bytes(x) [x in DRAM] + bytes(y) [y in DRAM]` (the minimum, every regime); NoC = `bytes(x)` per page moved for PULL/PUSH, `2·bytes(x)` for STAGE.

> Cheapest-traffic split considered: L1-bank owner (BANK-OWNER) for the 27 DRAM-interleaved → L1-interleaved cases — saves one NoC crossing per page (`bytes(x)` of NoC traffic, e.g. 20 MB for 2078), DRAM unchanged. Implemented: STAGE for those cases (PULL/PUSH, which are the cheapest split, for every sharded-L1 end). Deferred because the L1-bank→logical-core map needed to place movers on bank cores is not exposed to the Python program-descriptor layer (`ttnn/cpp/ttnn-nanobind/device.cpp:811` exposes the DRAM map only); STAGE's reader with a local landing (`is_local_page`) is that regime with a different core-assignment, so it stays reachable.
