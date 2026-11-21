# L1 Ledger: upsample

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes: `stick`
(N·H·W flattened, extent `block_sticks`), `channel` (whole stick), `rep_h`, `rep_w` (whole factors) —
see `op_design.md` → Blocking Model. Worked values are for **quasar** (32 clusters, 3,963,328 B
unreserved L1 per cluster, probe 2026-11-21) and the thread plan `(R, C, W) = (2, 0, 4)`.

## Inventory

One buffer. Every candidate second buffer was considered and not created:

| Candidate | Decision |
|-----------|----------|
| an output-side ring (replicated sticks / assembled output rows) | not created: the writer writes every replica straight from the input entry (pattern 1, "pack into the destination" — here the destination is the output tensor itself); an interleaved row-major page is one NoC write either way, so an assembled row saves no transfer (`l1_row_replicate` rejected in Regimes) |
| a compute-side ring pair (reader → compute → writer) | not created: no compute stage (spec DEFINITION, R1) |
| a scratch for misaligned sticks | not created: `stick_stride` aligns each stick's slot to `max(l1_align, dram_align)` inside the one entry |

## Ledger

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_sticks` | `RING_ENTRIES = RING_ENTRIES_PER_THREAD · max(READER_THREADS, WRITER_THREADS)` = 2 · 4 = 8 entries of `entry_bytes = block_sticks · stick_stride` | peak: one entry per writer thread being written + one per writer thread being filled = `2 · WRITER_THREADS` = 8 entries (in Phase 0 INPUTS a cluster has ≤ 8 blocks, so its whole share can be resident at once) | `{stick: spans → block_sticks per entry, streams over num_blocks_this_core; channel: spans → whole stick (stick_stride); rep_h: streams (never stored: replicas are re-writes of the entry); rep_w: streams (same)}` | input dtype (bfloat16 in Phase 0; float32 when unlocked) — a dataflow-only payload that never passes through DEST, so the DEST-width audit does not apply; `fp32_dest_acc_en` does not exist for this op (no compute config, spec DECISIONS) | reader (2 threads, STRIDED, explicit sync) | writer (4 threads, STRIDED, explicit sync) | whole program | no other buffer exists to share with; capacity above one entry per writer thread is the double-buffering depth (`RING_ENTRIES_PER_THREAD = 2`) that lets readers fill block `b + W_t` while block `b` is written; must be a multiple of `max(R, W_t)` (STRIDED dealing) |

## Symbol table

| Symbol | Bound | Predicate / source |
|--------|-------|--------------------|
| `stick_bytes` | `C · elem_size`; ≤ 16,384 B bf16 / 32,768 B fp32 over TARGET × INPUTS (C ≤ 8192) | `input.buffer_page_size()`; bounded by the `stick_stream` regime predicate below |
| `stick_stride` | `align_up(stick_bytes, max(ttnn.get_l1_alignment(), ttnn.get_dram_alignment()))` = `stick_bytes` in Phase 0 (C % 32 == 0 → 2C multiple of 64) | queries |
| `max_core_sticks` | `ceil(N·H·W / num_clusters)` | `ttnn.split_work_to_cores(device.compute_with_storage_grid_size(), N·H·W, row_wise=True)` |
| `READER_THREADS`, `WRITER_THREADS` | 2, 4; R + W ≤ 6, W % R == 0 | host constant `THREAD_PLAN`; target page *Thread plan* |
| `RING_ENTRIES_PER_THREAD` | 2 | host constant (depth knob) |
| `BLOCKS_PER_WRITER_THREAD` | 2 | host constant (cut-count knob) |
| `RING_MAX_BYTES` | 1,048,560 B | target page *Cluster model* (no Python query); host constant |
| `L1_RESERVE_BYTES` | 0 in Phase 0 | host constant (reserve knob) |
| `l1_budget` | `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES` = 3,963,328 B on this target | query |
| `ring_bytes_cap` | `min(RING_MAX_BYTES, l1_budget)` = 1,048,560 B on this target | derived |
| `block_sticks` | `max(1, min(ceil(max_core_sticks / (WRITER_THREADS · BLOCKS_PER_WRITER_THREAD)), floor(ring_bytes_cap / (RING_ENTRIES · stick_stride))))` | derived once on host; the single source for the ring's `entry_size` and both kernels' CT arg |
| regime predicate | `RING_ENTRIES · stick_stride ≤ ring_bytes_cap` (else `NotImplementedError`, deferred `channel_chunk`) | `validate()` |

## Total per-core footprint

`L1_ops = RING_ENTRIES · block_sticks · stick_stride`
`      = RING_ENTRIES_PER_THREAD · max(R, W_t) · block_sticks · stick_stride ≤ ring_bytes_cap`

- scales with the depth knob (`RING_ENTRIES_PER_THREAD`) and the plan (`max(R, W_t)`) linearly;
- scales with `block_sticks` (the `stick` extent), which itself shrinks as `BLOCKS_PER_WRITER_THREAD` or
  `WRITER_THREADS` grow, and is clamped by `ring_bytes_cap`;
- scales with `stick_stride` (`channel` extent, whole) and the element size (dtype);
- independent of `scale_h`, `scale_w` (replicas are never stored).

Phase 0 maximum over INPUTS: 131,072 B (`[1,4,4,8192]`, bf16); 57,344 B for the model's
`[1,40,40,512]`. Never above 1,048,560 B by construction.

## Data-movement budget

`B_in = N·H·W·C·elem_size` (input bytes); output bytes `= scale_h·scale_w·B_in`.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | 1 (`N·H·W` page reads) | the split gives each input stick to exactly one cluster, and the stick stays resident in the ring until all its `sh·sw` replicas are written | none |
| output | 1 (`N·H·sh·W·sw` page writes) | every output page is written once, from the resident stick | none |

For an L1-placed input/output the same counts cross the NoC to the L1 banks instead of DRAM.

Totals per tier (model call `[1,40,40,512]`, bf16, (2,2)): DRAM 1,638,400 B read + 6,553,600 B written
= 8,192,000 B; cross-core 0 B; core-local: each input byte written into L1 once and read out of L1
`sh·sw` = 4 times by the writer's NoC transfers.

> Cheapest-traffic split considered: input sticks across clusters (`stick_stream`) — 1 input crossing
> + 1 output crossing, 0 cross-core bytes. Implemented: the same split. It is the cheapest: no
> candidate moves fewer bytes (the `channel_chunk` split ties on bytes with more transactions; output,
> row-of-output and replica splits re-read the input up to `sh·sw` times).
