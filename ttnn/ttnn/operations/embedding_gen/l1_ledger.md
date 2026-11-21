# L1 Ledger: embedding_gen

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Target: `quasar` (8×4 clusters, 4 MiB L1 per cluster), worked values for bfloat16. Thread plan (R, C, W) = (1, 0, 1): there is no compute kernel, so no buffer is sliced per NEO.

Named block axes: **token** (`block_tokens`), **column** (`block_cols`), **lookup** (V: not blocked, rows selected by index), **unit** (blocks walked per core).

## Inventory (minimized before any solve)

Two buffers in total.
- The output-layout assembly happens **in the destination**: pieces land directly at their final position in the ring entry the writer drains. That is Rule 3, pattern 1. There is no staging row and no tilize scratch.
- The indices need a reader-private buffer. The reader both writes it (by NoC) and reads it. A DM self-loop ring is forbidden (`dfb.cpp:651-657`), so it is a scratchpad.
- No compute kernel means no DEST, no intermediates and no scaler.

## Table

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_gathered_block` (DFB, entry = one block) | `RING_DEPTH · max(READER_THREADS, WRITER_THREADS)` entries (implementation: `plan.ring_entries`; equals the design's `RING_DEPTH · READER_THREADS` at the P0 plan) of `entry_bytes = block_tokens · block_cols · 32 · esize` B | 2 entries: one being assembled by the reader, one being drained by the writer. Capacity = live set at depth 2. | `{token: spans → block_tokens, column: spans → block_cols, lookup: streams → rows selected per token (not resident), unit: streams → 1 entry per block}` | none (DM-only byte payload; it never passes through DEST, so audit 2 does not apply; `data_format_metadata` unset, `dfb.cpp:272-282`) | reader | writer | whole program | Cannot share with `scratch_token_indices`: concurrent lifetimes (the index slot is read while the entry is being filled). The two are also different binding kinds: a ring is for 1P/1C between kernels, a scratchpad is reader-private. |
| `scratch_token_indices` (ScratchpadSpec, reader-only) | `READER_THREADS · INDEX_SLOTS · BLOCK_TOKENS_MAX · 4` B = 1·2·32·4 = **256 B** (constant: `BLOCK_TOKENS_MAX` caps `block_tokens`) | 2 slots per reader thread: the current block's indices (being read) and the next block's (in flight) | `{token: spans → ≤ BLOCK_TOKENS_MAX per slot, column: streams → constant (same indices for every column unit), lookup: streams → constant, unit: streams → 2 slots}` | none (uint32 bytes; uint32 is not a buffer format, `dfb.cpp:148`; unset is legal for a scratchpad only DM kernels touch) | reader (NoC writes) | reader (loads) | whole program | Cannot share with `cb_gathered_block`: concurrent lifetime. It cannot live inside a ring entry either, because the entry is pushed to the writer while the next block's indices are still needed. |

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `esize` | 2 (bf16, P0); 4 (float32, later) | validate(): dtype ∈ SUPPORTED |
| `block_tokens` | `1 ≤ block_tokens ≤ BLOCK_TOKENS_MAX = 32`; `= 32` for TILE out | `plan_blocks` step 1 |
| `block_cols` | `1 ≤ block_cols ≤ min(Dt, cols_cap)` | `plan_blocks` steps 3–5 |
| `cols_cap` | `floor(min(DFB_MAX_BYTES, l1_budget − scratch_bytes) / (RING_DEPTH · block_tokens · 32 · esize))`; bf16, TILE out: 255; RM out at `block_tokens = 1`: 8,191 | `plan_blocks` step 4 |
| `RING_DEPTH` | 2 (P0 knob) | host constant |
| `INDEX_SLOTS` | 2 | host constant |
| `READER_THREADS` | 1 (P0, `THREAD_PLAN`); ≤ 6 − `WRITER_THREADS` | host constant; target page R + W ≤ 6 |
| `DFB_MAX_BYTES` | 1,048,560 B per ring | target page cluster model (mechanism cap; Python has no query) |
| `l1_budget` | `ttnn.get_max_worker_l1_unreserved_size() − L1_RESERVE_BYTES` (`L1_RESERVE_BYTES` = 64 KiB, host constant in `embedding_gen_program_artifacts.py`). On quasar this is below 4,194,304 B, and the ring cap binds first. | run-time query |
| `V`, `D`, `B`, `S` | unbounded by L1: no capacity depends on them. `Dt = ceil(D/32)` enters only through `block_cols ≤ Dt`. | — |

## Total per-core footprint

```
L1(core) = RING_DEPTH · READER_THREADS · block_tokens · block_cols · 32 · esize      // scales with token × column knobs and depth
         + READER_THREADS · INDEX_SLOTS · BLOCK_TOKENS_MAX · 4                       // constant (256 B in P0)
         ≤ DFB_MAX_BYTES + 256 B  (by step 4)
```

Phase 0 worst case over PHASE0 INPUTS: `2 · 16,384 + 256 = 33,024 B`, from `(4,1,1,128) × (1,1,2048,512)` and `(2,32) × (1,1,64,4096)`. The largest value the cap admits is ≈ 1 MiB (D = 8192, TILE out, 32 tokens per block: `block_cols` clamped to 128 → 2 × 256 KiB = 512 KiB).

## Data-movement budget

Gather-direct with the token split as primary, plus the column split when `n_token_groups < num_cores`. `N_tok = B·S` and `nC = num_col_chunks`.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `input_tensor` (indices) | `nC` (1 when the column split is off) | Each column chunk of a token group reads its own index slice: 4 B per token per chunk. It is not broadcast, because a multicast plus a rendezvous costs more than ≤ 128 B parallel reads that are prefetched under the gather. | none |
| `weight` (table) | **not as a whole.** Selected-row bytes `N_tok · D · esize` cross once per token occurrence; a repeated index re-reads its row. | Nothing is resident across tokens (the regime *resident rows* is deferred). A TILE table is read as 32 B face-row pieces: exactly the selected bytes, `2·Dt` transactions per token. | none |
| output | 1 | Written once from the ring entry: whole tiles, or row slabs at page offsets | none |

**Totals per tier.**
- DRAM read: `N_tok · (D · esize + 4 · nC)` B.
- DRAM write: `N_tok · D · esize` B (TILE out also writes the padding rows of the last tile-row per batch: `B · ceil(S/32) · 32 · D · esize` in all).
- Cross-core: 0.
- Core-local: each byte lands in L1 once and leaves once. There are no local copies.

> Cheapest-traffic split considered: **token split** (column chunks = 1): it saves `(nC − 1) · N_tok · 4` B of index reads and keeps RM rows in single transfers, at the cost of idle cores when `n_token_groups < num_cores`. Implemented: **token split, with the column split engaged only to fill the grid** (`plan_blocks` step 3). The implemented split *is* the cheapest-traffic split whenever `n_token_groups ≥ num_cores`. Below that, the column split trades ≤ 124 B of index reads per group for `nC×` more busy cores. That is an occupancy trade, stated, and the perf lamp *grid synchronization: column split at small widths* measures it.
