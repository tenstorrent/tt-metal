# L1 Ledger: g6_to_memory_config

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md`):
`leading` (folded into `tile_row`), `tile_row`, `tile_col`, `page` (the flattened `tile_row × tile_col` view).

## Circular buffers

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_bounce_pages` (bounce regime only) | `bounce_depth · bounce_block_pages`, with `bounce_block_pages = min(ceil(core_pages_max / bounce_depth), floor(L1_OP_BUDGET_BYTES / (bounce_depth · page_size)))` | `bounce_depth · bounce_block_pages` — both slots are live at once by design (reader fills block `k+1` while the writer drains block `k`); equals capacity | `{leading: folded into tile_row, tile_row: folded into page, tile_col: folded into page, page: spans → bounce_block_pages per slot × bounce_depth}` | input dtype (`Float16_b` Phase 0; `Bfp8_b` / `Bfp4_b` refinement) — a dataflow-only byte payload that never passes through DEST, so the DEST-width rule does not apply | reader | writer | bounce kernels, the whole call | nothing to share with: it is the op's only buffer. Not aliased onto the input or output: in `bounce` both live in DRAM, and an L1 staging slot is the only place a DRAM→DRAM page can land (a NoC read lands only in the issuer's L1) |

Pull and push regimes allocate **no CB**. Considered and decided per the reuse patterns: the
phase boundary "page arrived" → "page stored" is not storage — pull packs straight into the
destination (pattern 1: the read lands at the page's final address in the output buffer), and push
reads the resident source in place (the shard / bank stripe is consumed where it lies, never re-read
over the NoC). There is no transform, so patterns 2–4 have nothing to act on.

## Other L1 the BB-05 bound counts (tensor-backed, not CBs)

| Term | Bytes on the most loaded core | Axis accounting | Bound / predicate |
|------|-------------------------------|-----------------|-------------------|
| output share, L1-sharded target (`pull_shard`, `push_shard` into L1 shards) | `shard_h_tiles · shard_w_tiles · page_size` (the full shard allocation, also on the ragged last shard) | `{tile_row: spans → shard_h_tiles, tile_col: spans → shard_w_tiles, page: n/a}` | the caller's shard spec; golden max 40960 B (20 pages, Qwen 2073's [32, 640] target shard) |
| output share, L1-interleaved target (`pull_bank`, `push_shard` into L1 interleaved) | `ceil(P / nb) · page_size` | `{page: spans → ceil(P / nb)}` | golden max 94 pages · 2048 B = 192512 B (Qwen 2078, nb = 110) |
| output share, DRAM target | 0 | — | — |
| input | not counted (the model's tensor, per spec) | — | — |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `P` | total tile pages, `prod(padded_shape[:-2]) · ceil(H/32) · ceil(W/32)` | ≤ 81920 in INPUTS (bf4b 2183); ≤ 10240 for bf16 (Qwen 2078) | none needed for CB sizing: `P` enters only through `core_pages_max`, itself clamped below |
| `G` | compute grid cores, `compute_with_storage_grid_size().x · .y` | device constant (110 on this Blackhole board) | — |
| `nb` | L1 banks = `G` (one per compute core) | = `G` | `bank_cores_cover_grid`; otherwise bank regimes become `bounce` |
| `core_pages_max` | `ceil(P / min(P, G))` — largest bounce assignment | ≤ 745 in INPUTS (bf4b 2183, G = 110) | `split_work_to_cores` balance |
| `page_size` | `input_tensor.buffer_aligned_page_size()` | ∈ {2048, 1088, 576} B | TILE layout; dtype ∈ TARGET |
| `bounce_depth` | bounce buffer depth | 2 | host constant |
| `L1_OP_BUDGET_BYTES` | L1 the model leaves the op per core | 1347776 B | model environment (BB-05) |
| `bounce_block_pages` | bounce block | ≤ `floor(1347776 / (2 · page_size))` = 329 (bf16), 619 (bf8b), 1169 (bf4b) | the `min(...)` in its definition |

## Total per-core footprint

```
L1_op(core) = output_share(regime, core)                                   # scales with shard_h_tiles·shard_w_tiles, or ceil(P/nb); 0 for DRAM targets
            + [regime == bounce] · bounce_depth · bounce_block_pages · page_size   # scales with bounce_depth and bounce_block_pages; ≤ L1_OP_BUDGET_BYTES by construction
```

Only one term is non-zero per regime: owner regimes into L1 carry the output share and no CB;
`push_*` into DRAM carry nothing; `bounce` carries only the CB (its target is DRAM). Golden maxima:
192512 B (`pull_bank`, Qwen 2078), 40960 B (L1-sharded targets), 429696 B
(`bounce`, bf4b 2183, refinement: `bounce_block_pages` = ceil(745 / 2) = 373, 2 · 373 · 576 B, as
`plan_of` reports) — all under 1347776 B. A shape whose `L1_op` exceeds the budget is
refused on the host before allocation (BB-05).

## Data-movement budget

`B = P · page_size` bytes per tensor.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input, DRAM placement | 1 (each page read once) | every page has exactly one mover, which reads it once — owner regimes into its final slot, `bounce` into its CB | none beyond that read |
| input, L1 placement | 0 | read in place by its owner (`push_*`) or read once over the NoC by the target owner (`pull_*`) | `pull_*`: 1 NoC read per page (B bytes); `push_*`: the read is local (0) |
| output, DRAM placement | 1 (each page written once) | one mover writes each page once | none beyond that write |
| output, L1 placement | 0 | written once, by the pull owner's read landing in place or the push owner's write | `push_*` into L1: 1 NoC write per page (B bytes); `pull_*`: 0 extra |

Totals per tier for the chosen splits:

| Regime | DRAM bytes | NoC transfers (bytes) |
|--------|-----------|------------------------|
| `pull_shard`, `pull_bank` | `B` if the source is DRAM, else 0 | `B` (one transfer per page) |
| `push_shard`, `push_bank` | `B` if the target is DRAM, else 0 | `B` |
| `bounce` | `2B` (read once, write once) | `2B` — the minimum for DRAM→DRAM |

> Cheapest-traffic split considered: owner moves (pull by the target's owner or push by the source's
> owner) — one NoC transfer per page, each tensor crossing its tier once. Implemented: the owner
> regimes for every placement pair with an L1 side, and `bounce` for DRAM→DRAM, where two transfers
> per page are the minimum. The implemented split is the cheapest-traffic split for every placement
> pair. The only above-minimum path is `bounce` taken as the fallback of a bank regime when
> `bank_cores_cover_grid` is false (+`B` NoC bytes, same DRAM bytes); `plan_of` reports it.
