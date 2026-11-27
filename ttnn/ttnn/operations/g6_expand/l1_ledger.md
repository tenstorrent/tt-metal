# L1 Ledger: g6_expand

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions: `op_design.md` → Blocking Model.

Block axes: `q` (x page = out tile column, images `n` folded in), `r` (out tile row), `i` (row inside a tile, fixed 32).

## Buffers

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_row_bcast_tiles` | `t_block_depth · block_pages` | `block_pages` while one block is loaded/replicated/published and written. `t_block_depth · block_pages` only when a group has more than one block (the injector fills block k+1 while block k is written). Production: 1 page. | `{q: spans → block_pages; r: streams → each resident page is written core_rows_tiles times from the same bytes, no capacity; i: spans → inside the page (32 rows, fixed by the tile)}` | `Float16_b` (bf16). A dataflow-only payload: no unpacker, DEST or packer touches it, so no DEST-width rule applies; the bytes move verbatim. | reader | writer | whole kernel; one slot per block | **Sole buffer — nothing to share with.** The x page read lands in it at its final offset (no staging buffer), the row replication is in place, the multicast sends from it and lands in it at the same address on receivers, and the writer writes out pages straight from it (no output buffer). |

## Symbol table

| Symbol | Bound | Predicate / source |
|--------|-------|--------------------|
| `page_bytes` | 2048 | bf16 32×32 tile; `x.buffer_aligned_page_size()`; TARGET dtype is bf16 only |
| `group_pages_max` | `ceil(Q / num_groups)` | `select_plan` |
| `block_pages` | `1 ≤ block_pages ≤ min(group_pages_max, floor(L1_OP_BUDGET / (t_block_depth · page_bytes)))` | `select_plan` clamp; production value 1 |
| `t_block_depth` | `∈ {1, 2}`: 1 iff `ceil(group_pages_max / block_pages) == 1` | `select_plan`; production value 1 |
| `L1_OP_BUDGET` | 1,347,776 B | model environment (BB-05) |
| `core_rows_tiles` | `≤ ceil(Ht / row_splits)` | does not appear in any capacity (streams) |

## Total per-core footprint

`L1_op = t_block_depth · block_pages · page_bytes` (+ two mcast semaphore words: data-ready, consumer-ready)

- Scales with `block_pages` and `t_block_depth` only. Independent of `Ht`, `row_splits` and the core count; depends on `W` only through `group_pages_max` and is clamped.
- Production (all five shapes): `1 · 1 · 2048 = 2048 B` per core, against a budget of 1,347,776 B.

## Audits

| Audit | Result |
|-------|--------|
| 1 Capacity vs live set | Production: capacity = live set = 1 page. Multi-block groups: depth 2 is the pipelining reason (stated). Under: the live set spans `q` and capacity scales with `block_pages`. |
| 2 Page format vs DEST | No DEST in the op (no compute kernel). `Float16_b` for a dataflow-only payload, which is the format of the bytes themselves. |
| 3 Disjoint lifetimes | Single buffer; no pair exists. |
| 4 Bounds and closed form | Every symbol above is bounded; the total is closed-form in `block_pages`, `t_block_depth`, `page_bytes`. |

## Data-movement budget (chosen split: `column_group_bcast`)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x | 1 — each page read once, by its group's injector, as a whole 2048 B page (64 B logical) | Groups partition `q`. Receivers get the built page over the NoC instead of re-reading DRAM. | `(row_splits − 1) · page_bytes` per x page: one 1-D multicast down the strip, plus one data-ready flag multicast and `row_splits − 1` ready-ack increments per block |
| out | 1 — each page written once, by the one core owning `(q, r)` | The built page stays resident while all `core_rows_tiles` copies are written | none |

| W | x DRAM read | out DRAM write | NoC multicast delivered |
|---|-------------|----------------|-------------------------|
| 512 | 32 KiB | 512 KiB | 128 KiB (16 pages × 4 receivers) |
| 256 | 16 KiB | 128 KiB | 112 KiB (8 × 7) |
| 207, 196 | 14 KiB | 98 KiB | 84 KiB (7 × 6) |
| 185 | 12 KiB | 72 KiB | 60 KiB (6 × 5) |

> Cheapest-traffic split considered: `q`-only (`row_splits = 1`, one core per group writing all `Ht` rows). Same DRAM bytes per tier, 0 NoC bytes (saves the last column above). Implemented: `column_group_bcast`, whose `row_splits = 1` setting **is** that split — built and test-pinned, not deferred. Phase 0's default `row_splits > 1` deliberately trades those NoC bytes, on the cheapest non-local tier, for write-issue parallelism and a core count chosen by the plan instead of by `Q` (blocking-model §4: an explicit trade, stated). Perf lamp L1 measures it.
