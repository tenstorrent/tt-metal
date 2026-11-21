# L1 Ledger: slice_write

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Target: `quasar` (4 MiB L1 per cluster), on ttsim.

Block axes, from op_design.md's Blocking Model: `src_row` (extent `rows_per_block`, which is the whole shard in Phase 0), `row` (extent `row_bytes`, source stride `src_row_stride_bytes`), `write_batch` (extent `rows_per_write_batch`).

## Built regime: `shard_owner_writes`

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `dfb_src_shard` | `src_shard_entries = W` (writer harts of `THREAD_PLAN`; **1** at the Phase 0 plan) entries of `(shard_rows / W) · src_row_stride_bytes` B, i.e. the whole shard. **Borrowed** from the input tensor's shard (`borrowed_from = "src"`), so it allocates 0 B. | 1 entry (the whole shard is resident for the whole store, because every row is a NoC write source until its batch barrier) | `{src_row: spans → shard_rows (allocation; rows_this_core valid), row: spans → src_row_stride_bytes, write_batch: streams → rows_per_write_batch rows in flight, no buffer}` | `input.dtype` (bfloat16 in Phase 0). Metadata only: a dataflow-only payload that never passes through DEST, so audit 2 does not apply | `reader` | `writer` | the whole program | **Shares with the input tensor's own shard**: it is that memory (zero-copy). There is no other buffer to share with. Pack-into-destination does not apply, because the destination is in DRAM and is written directly from this entry. |

No other CB exists. Inventory reasoning, per Rule 3:

| Candidate buffer | Exists? | Why |
|---|---|---|
| A staging copy of the source | no | the source is resident, so the ring borrows it |
| An output staging buffer | no | rows go from the shard straight to the DRAM pages; there is no transform between them |
| An intermediate between phases | no | there is one phase and no compute |

## Deferred regimes' buffers

Not allocated in Phase 0. Recorded here so the refinement inherits their bounds.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `dfb_src_rows` (`interleaved_stream` only, replacing `dfb_src_shard`) | `src_rows_depth · 1` entries of `rows_per_block · src_row_stride_bytes` B; `src_rows_depth = 2` | 2 entries (one filling, one draining) | `{src_row: spans → rows_per_block, row: spans → src_row_stride_bytes, write_batch: streams}` | `input.dtype`, metadata only | `reader` | `writer` | the whole program | Cannot borrow: an interleaved source is not resident on this core. Depth 2 for read/write overlap (catalog `double_buffer`). |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|---|---|---|
| `shard_rows` | the input's shard spec height. Already allocated by the caller, so it costs nothing here | `validate()`: `memory_axes(input) == {L1, height}` |
| `rows_this_core` | `0 < rows_this_core ≤ shard_rows` | host: `min(shard_rows, total_rows − i·shard_rows)`. Cores with 0 rows are excluded from the work unit |
| `src_row_stride_bytes` | the shard width × element size, 16 B-aligned (`ttnn.get_l1_alignment()`) | taken from `dfb_spec_from_sharded_tensor(...).entry_size` |
| `row_bytes` | `≤ src_row_stride_bytes`; `≤ 32 KiB` over TARGET (8192 × 4 B) `< NOC_MAX_BURST_SIZE` (64 KiB) | `input.logical_shape[-1] · element_size` |
| `rows_per_write_batch` | `≤ WRITE_BATCH_ROWS = 32` | mechanism cap: per-trid outstanding counter (op_design.md Mechanism caps) |
| `src_shard_entries` | `= W` (1 at the Phase 0 plan (1, 0, 1)); entries together span exactly the shard | host: `W | shard_rows`, else the plan runs at `W = 1` (`slice_write_program_artifacts.py`; op_design.md Perf lamps) |
| `rows_per_block` (deferred regime) | `2 · rows_per_block · src_row_stride_bytes ≤ budget`, and `≤` the core's share | `budget = ttnn.get_max_worker_l1_unreserved_size() − l1_reserve_bytes` (the target page records no value for the query: unknown here) |

**Total per-core footprint added by the op (built regime):** `0 B`. The only buffer, `dfb_src_shard` (`shard_rows · src_row_stride_bytes`), is the caller's resident shard, and no term scales with any knob. In the deferred `interleaved_stream` regime the footprint is `2 · rows_per_block · src_row_stride_bytes`, which scales with `rows_per_block` and the depth knob.

## Data-movement budget

`B` = block bytes = `n_s · h_s · w_s · row_bytes` (the source's logical bytes).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` (L1 height-sharded) | 0 reads | resident in L1. The borrowed ring makes the shard itself the NoC write source, with no re-read. | 0 |
| `output_tensor` (DRAM interleaved), block pages | 1 write of `B` | each block page is written exactly once by the core that owns its source row | 0 |
| `output_tensor`, pages outside the block | 0 | never addressed (R2) | 0 |

Totals per call: DRAM `B` (written), with 0 read; cross-core NoC 0; core-local copies 0. Write transactions: `n_s·h_s·w_s`, one per row. That is the minimum for an interleaved row-major destination, since consecutive pages sit in different banks. Worked on this target for traced call 1: `B = 434,240 B` and 6,785 transactions, spread over the 32 cores of the golden-suite shard config (213 rows per core).

> Cheapest-traffic split considered: cutting `src_row` along the source's own shards (owner writes) — 0 source DRAM reads, 0 cross-core bytes, `B` written once. Implemented: the same split (`shard_owner_writes`). The implemented split **is** the cheapest-traffic split. The occupancy alternative `spread_beyond_shard_grid` adds up to `B` of cross-core NoC traffic. It is a `deferred` regime row, kept reachable because its helper cores run the built writer unchanged.
