# L1 Ledger: g6_to_memory_config

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking Model: `op_design.md`.

Block axes named in the Blocking Model: `tile_row`, `tile_col`, and the flattened page index
(`tile_row × tile_col`) used by the L1-bank owners (`bank_pages`) and the bounce workers
(`bounce_block_pages`).

## CB table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_staging_pages` (regime R3 bounce only) | `bounce_depth · bounce_block_pages`, with `bounce_block_pages = min(ceil(max_chunk_pages / bounce_depth), floor(MODEL_L1_BUDGET_BYTES / (bounce_depth · aligned_page_size)))` | `bounce_depth · bounce_block_pages` — reader fills slot k+1 while writer drains slot k; capacity = live set (the two slots together hold the worker's whole chunk) | `{tile_row: streams (flattened into the target-order chunk), tile_col: streams (flattened), page index: spans → bounce_block_pages per slot × bounce_depth slots}` | the input dtype (bfloat16 in Phase 0; bfloat8_b / bfloat4_b later). Dataflow-only payload that never passes through DEST — the format is the tensor's own page format, not a DEST-width choice | reader (RISCV_1) | writer (RISCV_0) | the whole R3 kernel | nothing — it is the op's only CB. Pack-into-destination is impossible in R3: neither side is in L1 on any core, and a NoC read lands only in the issuer's own L1, so a page bound DRAM→DRAM must rest in a worker slot. |
| *(R1 pull, R2 push: no CB)* | 0 | 0 | `{tile_row: n/a, tile_col: n/a, page index: n/a}` — the owned block is the output (R1) or the input (R2) itself | — | — | — | — | Inventory minimized to zero: R1 reads each page straight into its final target slot (pattern 1, pack into the destination); R2 writes from the resident source slot. A zero-copy `cb_descriptor_from_sharded_tensor` alias was considered and dropped: one kernel per core has no handshake partner, so the alias would be bookkeeping only. |

## Symbol table

| Symbol | Meaning | Bound | Predicate that establishes it |
|--------|---------|-------|-------------------------------|
| `aligned_page_size` | bytes per tile page, from the buffer | 2048 (bf16), 1088 (bf8), 576 (bf4) | dtype ∈ TARGET; TILE layout |
| `tensor_pages` | `prod(shape[:-2]) · ceil(H/32) · ceil(W/32)` | `< 2^32` | host assert |
| `num_l1_banks` | `get_device_info(device).l1_num_banks` | 110 on the campaign's Blackhole (11×10 grid) | `l1_bank_owners` (`num_l1_banks == grid.x · grid.y`) |
| `shard_rows_tiles`, `shard_cols_tiles` | target/source shard extents in tiles | given by the shard spec; one shard per core | validation (shard dims multiple of 32, cores ≥ shards) |
| `max_chunk_pages` | largest R3 worker chunk | `ceil(tensor_pages / min(tensor_pages, grid.x · grid.y))` | R3 predicate `h_in == 0 and h_out == 0` |
| `bounce_depth` | staging slots | 2 (Phase 0) | host constant |
| `bounce_block_pages` | pages per staging slot | `≤ floor(MODEL_L1_BUDGET_BYTES / (bounce_depth · aligned_page_size))` (329 bf16, 619 bf8, 1169 bf4) | clamp in the plan |
| `MODEL_L1_BUDGET_BYTES` | L1 the model leaves to the op per core (BB-05) | 1347776 | spec, Model environment |

## Total per-core footprint

```
L1_core = out_l1_bytes_on_core + [regime == bounce] · bounce_depth · bounce_block_pages · aligned_page_size

out_l1_bytes_on_core = ceil(tensor_pages / num_l1_banks) · aligned_page_size        (L1 interleaved output)
                     = shard_rows_tiles · shard_cols_tiles · aligned_page_size        (L1 sharded output; full slot, also a ragged last shard)
                     = 0                                                               (DRAM output)
```

- Scales with the input shape: `out_l1_bytes_on_core` only — the output's own pages, which BB-05
  counts and the op cannot avoid. The op's own buffers add nothing in R1/R2.
- Scales with a knob: the R3 staging term, linear in `bounce_block_pages` and `bounce_depth`,
  clamped so `L1_core ≤ MODEL_L1_BUDGET_BYTES` (R3 has a DRAM output, so the first term is 0).
- Golden maximum: Qwen 2078 `[16,128,5120]` DRAM→L1 interleaved, `ceil(10240/110) = 94` pages =
  192512 B per core (14 % of the bound). PERF FOCUS Qwen 2183 (refinement, R3): `max_chunk_pages =
  745`, `bounce_block_pages = 373`, staging 2 · 373 · 576 = 429696 B.
- Host refusal (`ValueError`) when `L1_core > MODEL_L1_BUDGET_BYTES` — no golden shape reaches it.

## Data-movement budget

`B = tensor_pages · aligned_page_size` bytes.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | 1 if the input is in DRAM, else 0 | each page is read exactly once, by its single mover (R1 target holder, R3 worker) or sent once from its holder (R2); no re-read, no intermediate placement | 0 beyond the move itself |
| output | 1 if the output is in DRAM, else 0 | each page is written exactly once by its single mover; never staged through a third placement | 0 beyond the move itself |
| the move itself (both sides in L1 on different cores) | — | one NoC transaction per segment, landing in the final slot | `B` (one hop per page); 0 when source and target sit on the same core (NoC loopback) |

Totals per tier: DRAM `B · ([input in DRAM] + [output in DRAM])`; cross-core NoC `B` only for
L1↔L1 moves between different cores; core-local staging `0` (R1/R2) or `B` through worker slots (R3,
DRAM→DRAM only, where it is structural).

> Cheapest-traffic split considered: owner split (R1/R2; R3 where no side is in L1) — every page
> crosses each tier it must cross exactly once, `+0` bytes per tier over the minimum. Implemented:
> owner split. It is also the cheapest; the occupancy alternative R4 (helper fan-out, `+helper_pages
> · aligned_page_size` cross-core bytes) is the `deferred` regime row in `op_design.md`, reachable
> without restructuring because a helper is an R3 worker on a subset of the owner's pages.
