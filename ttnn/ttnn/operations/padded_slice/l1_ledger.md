# L1 Ledger: padded_slice

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Target for worked numbers: `quasar`
(ttsim), 4,194,304 B L1 per cluster, 32 clusters.

## Inventory

The op's program allocates **no** DataflowBuffer and **no** scratchpad. Inventory reasoning (Rule 1, Rule 3):

| Candidate buffer | Disposition |
|------------------|-------------|
| reader → writer staging ring | eliminated by *pack into the destination*: the NoC read lands directly in the output shard (`LocalTensorAccessor` is a legal read destination, `internal/local_mem/noc_traits.h:104-127`), so no writer and no ring |
| alignment staging (mid-page `start[3]`, odd shard-row offsets) | eliminated: the Quasar NoC reads at 1-byte alignment (`noc_parameters.h:486-491`) |
| zero scratch for padded rows (regime B) | eliminated: `async_write_zeros` zeroes the shard in place through iDMA, no source buffer (`noc.h:816-849`) |
| untilize ring + compute output ring (regime D, deferred) | not in this phase; would be added with D and ledgered then |

The single row below is the caller-owned output tensor region the kernel writes; it is listed so the
axes are accounted for, not because the op allocates it.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `output_shard` (tensor region `output`, not a DFB) | `shard_h` pages of `out_row_stride` B (caller's shard spec; allocated by `ttnn.allocate_tensor_on_device`) | `num_rows` pages (= `shard_h` in every cell: configs hold exactly the result's rows) — all simultaneously resident: it is the result | `{out_row: spans → block_rows (= shard_h), n/h/w: folded into out_row, channel: spans → width (row_bytes + pad_bytes), thread: streams → each hart writes [lo_t, hi_t)}` | input dtype as raw bytes (bfloat16 in Phase 0); dataflow-only, never through DEST, so the DEST-width audit does not apply | `gather` (reader harts, NoC reads) | none in this program (the op's result) | the program and beyond (it is the output) | shares with nothing: it is the only L1 the op touches; no other buffer exists to alias with |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `shard_h` | ≤ result rows `n*h*w`; shard bytes ≤ the allocator's free L1 | the caller's `memory_config`; `allocate_tensor_on_device` refuses a shard that does not fit `ttnn.get_max_worker_l1_unreserved_size()` |
| `out_row_stride` | `align_up(width * elem_bytes, ttnn.get_l1_alignment())` | `output.buffer_aligned_page_size()` |
| `width` | ≥ `c` (X6 refuses narrower) | `canonical_to` |
| `block_rows` | = `shard_h` (Phase 0: `MAX_BLOCK_ROWS = 0`, whole shard) | host constant |
| `R` (`GATHER_THREADS`) | 1..6 (R + W ≤ 6, W = 0) | host constant, checked at program build |
| L1 budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes`, `l1_reserve_bytes = 0` | run-time query (no op predicate spends it: op-owned footprint is 0) |

## Total per-core footprint

`op_owned = 0 B` (no DFB, no scratchpad; independent of every knob, including `R` and `block_rows`).
`output_region = shard_h × out_row_stride` B, scaling with the caller's shard only.

Worked (this target): golden-suite configs at most 295 × 64 = 18,880 B per core (stem entry, 23 cores);
the traced single-core stem config 6,785 × 64 = 434,240 B.

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` (the block's bytes only: `n*h*w*c*elem_bytes`) | 1 | each output row is read once, by the hart of the core that owns its shard; nothing is re-read (no second pass, no shared operand) | 0 |
| `output` | 0 (L1 tensor) | reads land in their final location; no write-back | 0 |

Totals (Phase 0): DRAM = block bytes once (e.g. 434,240 B for the stem entry, 65,536 B for
`(1,64,32,32)`); cross-core NoC = 0; core-local L1 = block bytes written once (by the NoC read itself).
Transaction count = one read per output row.

> Cheapest-traffic split considered: owner-computes row gather into each core's own shard (regime A) —
> it is the minimum on every tier (DRAM once, cross-core 0, no staging). Implemented: the same split.
> The implemented split is the cheapest. The alternatives in the regime table (staged ring E: +1 L1
> write and read per byte; helper cores F: +1 cross-core crossing per byte; channel split G: 2×
> transactions) all move more.
