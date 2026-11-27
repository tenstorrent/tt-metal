# L1 Ledger: g6_indexed_fill

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md`
→ Blocking Model): `u` (work units: x pages ++ value pages, cut into row chunks; extent knob
`block_units`), `w` (row chunk; knob `row_chunk_bytes`), `j` (entries of batch_ids scanned by the
decision; knob `ids_chunk_entries`). All three CBs belong to the one data-movement kernel `dm`
(fills and drains them; no CB credit, ordering by NoC barriers). No payload passes through DEST and
there is no compute kernel, so `fp32_dest_acc_en` does not apply: every page format is a
dataflow-only payload (audit 2's mechanism reason), labelled UInt32.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_batch_ids` | 1 page of `ids_window_bytes = align_up(4*ids_chunk_entries, A) + (A if ids_chunks > 1 and (4*ids_chunk_entries) % A else 0)` | 1 window. Capacity exceeds the `4*ids_chunk_entries` logical bytes by the DRAM-read alignment superset (mechanism: a window must start and end on `A`) | `{u: streams (one window serves every unit of every block), w: streams (decisions are per page; chunks of a page share them), j: spans → ids_chunk_entries}` | UInt32, dataflow-only | dm (NoC read) | dm (scalar loads) | `ids_resident`: the whole kernel; `ids_streamed`: one window inside `select_block`, re-filled per window | none: live together with `cb_staged_pages` (block 0's loads land during the ids read — the stall-shadow overlap) and with `cb_block_dest` (the table is updated while the window is scanned) |
| `cb_staged_pages` | `staging_depth * block_units` pages of `row_chunk_bytes` | `staging_depth * block_units` slots: at depth 2 block `k+1` lands while block `k`'s writes drain; at depth 1 exactly one block. Production: 1 × 1 | `{u: spans → block_units (× staging_depth slot sets), w: spans → row_chunk_bytes (the page size), j: streams (not involved)}` | UInt32, dataflow-only | dm (NoC read) | dm (NoC write source) | one block per slot set, from `load_block` until `store_block`'s writes are flushed | none: concurrent with both other CBs; cannot pack into the destination because the result is in DRAM and the NoC has no DRAM → DRAM copy (tensix.md, Gaps), so every page is staged through L1 once |
| `cb_block_dest` | 1 page of `align_up(4*block_units, A)` | `block_units` words (one destination or SKIP per unit). Capacity exceeds it by alignment to `A` only | `{u: spans → block_units, w: streams (each chunk unit carries its own word), j: streams (the decision accumulates over the K entries / the windows)}` | UInt32, dataflow-only | dm (`select_block` stores) | dm (`store_block` loads) | one block: written by `select_block(k)`, read by `store_block(k)` | none: concurrent with `cb_batch_ids` (written while it is scanned) and with `cb_staged_pages` (read while that block's slots are written out); not kept in registers or on the stack because `block_units` is unbounded by the stack and `ids_streamed` must carry each unit's state across windows; one table for both regimes keeps one kernel code path (BB-18) |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `A` | 64 (Blackhole DRAM read alignment) | `ttnn.get_dram_alignment()`; constant per architecture |
| `row_chunk_bytes` | `A <= row_chunk_bytes <= max(A, align_down(PLAN_CAPS["max_row_chunk_bytes"], A))` = 65536; multiple of `A` | `derive_plan` clamp (mechanism caps: transfer size, alignment) |
| `block_units` | `block_units * row_chunk_bytes * staging_depth <= PLAN_CAPS["staging_budget_bytes"]` (1 MiB), except `block_units = 1` when even one unit per slot exceeds it, where the staging term is `<= max_staging_depth * row_chunk_bytes <= 2 * 65536`. Hence `block_units <= 1 MiB / A = 16384` | `derive_plan`: depth-1 whole-assignment branch only when `n1 * row_chunk_bytes <= budget` |
| `staging_depth` | `1 <= staging_depth <= PLAN_CAPS["max_staging_depth"]` (2) | `derive_plan` |
| `ids_chunk_entries` | `1 <= ids_chunk_entries <= PLAN_CAPS["ids_chunk_cap_entries"]` (16384) | `min(K, cap)`; the regime predicate `ids_chunks == 1` selects `ids_resident` |
| `K, B, d1, d2, W` | unbounded (contract) | appear in no capacity expression except through `min()` with a cap above |

## Total per-core footprint

```
L1(core) = ids_window_bytes                                  # scales with ids_chunk_entries (j)
         + staging_depth * block_units * row_chunk_bytes     # scales with block_units (u), row_chunk_bytes (w), staging_depth
         + align_up(4 * block_units, A)                      # scales with block_units (u)
        <= (4*16384 + 64) + 1048576 + 4*16384 = 1179712 B   <= 1347776 B (BB-05), for every shape
```

| Case | `ids_window_bytes` | staging | dest | Total |
|------|--------------------|---------|------|-------|
| production `[1]`, `[7,1,1,32]`, `[1,1,1,32]` | 64 | 1 × 1 × 128 | 64 | **256 B** |
| rule-9 shape `[2]`, `[5,2,1,64]`, `[2,2,1,64]`, default caps | 64 | 1 × 1 × 256 | 64 | 384 B |

No budget solve exists beyond the two clamps in `derive_plan`; the inventory is three buffers, each
live concurrently with the other two, none an intermediate between phases.

## Data-movement budget (production, chosen split `ids_resident`)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `batch_ids` (4 B logical, one 64-B granule) | 8 (one granule per work core) | every work core decides from every entry; the replicated read is issued in parallel with that core's page load, off the serial chain | none |
| `input_tensor` (7 pages × 128 B) | 1 (all 7 pages; the named page is loaded and dropped) | loads are unconditional so they can be issued before the ids land (stall shadow) | none |
| `value_tensor` (1 page × 128 B) | 1 | each value page is owned by exactly one unit | none |
| result (7 pages × 128 B) | 1 (each page written exactly once) | the exactly-one-writer invariant of `select_block` | none |

Totals per call: DRAM read 8 × 64 + 896 + 128 = **1536 B on the wire** (1028 B logical), DRAM
write **896 B**, cross-core **0 B**. Transactions: 8 ids reads + 8 page reads + 7 writes, at most
3 per core, 2 serial DRAM round trips per core.

> Cheapest-traffic split considered: `gather_after_index` with the ids multicast (`ids_mcast`) —
> DRAM read 960 B instead of 1536 B (−128 B: the named x page is not loaded; −448 B: 7 of the 8
> ids granules), plus 7 × 64 B of multicast payload and a semaphore
> round, and 4 serial round trips per core instead of 2. Implemented: `ids_resident` (scatter,
> unconditional loads, replicated ids). Deferred because the op is bound by per-transfer latency,
> not bandwidth (~23 transfers of ≤ 128 B, tensix.md), and both savings put a serial hop (ids →
> chosen load; injector → multicast → semaphore) on every core's critical path for under 600 B;
> each half is its own `deferred` regime row with the structure that keeps it reachable.
