# L1 ledger: nlp_concat_heads_decode

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Target: quasar (8×4 clusters, 4 MiB L1 each).

**The op allocates no buffer of its own**: no DataflowBuffer, no scratchpad, no semaphore. The only
phase boundary ("fetch a row" → "place the row") is expressed as Rule 3 pattern 1, *pack into the
destination*: each NoC read lands directly in the output shard. The two rows below are the
tensor-owned L1 regions the kernel touches, listed so that every block axis is accounted for and
so that the op's own footprint (0 B) is visible as a decision.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `output` shard (tensor-owned, `LocalTensorAccessor`; not a DFB) | `head_tiles` tiles of `tile_bytes` (allocated by ttnn for the output tensor, not by the op) | `head_tiles` tiles: the whole shard is the destination of the single block | `{head: spans → 1 (the core's head), head_dim_tiles: spans → head_tiles (block_tiles per thread), user: spans → rows 0..num_users-1 of each tile, head_row: streams → 1 row selected, face_half: spans → 2 runs per row}` | the input dtype (bfloat16 Phase 0; float32 TARGET), raw bytes — never passes DEST | gather kernel (iDMA zero + NoC reads) | the caller (next op / host read-back) | whole op and beyond (it is the result) | Not shareable: it is the op's output tensor. No staging buffer shares it because none exists (reads land in place). |
| `input` shard (tensor-owned, remote, `TensorAccessor`; not a DFB) | `head_tiles` tiles per user on that user's core (the caller's tensor) | rows `h < num_heads` of `head_tiles` tiles are read once, by `num_heads` different cores | `{head: streams → row head_idx read by core head_idx, head_dim_tiles: spans → head_tiles, user: spans → one shard per user, head_row: streams → 1 row per reading core, face_half: spans → 2 runs}` | input dtype, raw bytes | the caller | gather kernels on `num_heads` cores (read only) | the op's read phase | Not shareable: it is the op's input, read in place over the NoC (no local copy). |

**Op-allocated per-core footprint:** `0 B` — no term scales with any knob (`block_tiles`,
`num_users`, `READER_THREADS`, `zero_padding_rows` change transfer counts only). The tensor-owned
output shard is `head_tiles · tile_bytes` per core (worked, quasar: ≤ 8 · 4096 = 32 KiB at
`head_dim = 256` float32), allocated by ttnn before the program runs. No L1 budget predicate,
safety fraction or blocking search exists, because nothing of the op's needs to fit.

### Symbol table

| Symbol | Bound | Predicate that establishes it |
|--------|-------|-------------------------------|
| `head_tiles` | `head_dim / 32`; ≤ 8 in INPUTS | `head_dim % 32 == 0` (alignment gate); output shard allocated by ttnn (its allocator refuses what does not fit) |
| `tile_bytes` | 2048 (bfloat16), 4096 (float32) | dtype gate (SUPPORTED); `tile_shape == [32, 32]` check |
| `num_users` | 1 in TARGET; ≤ 32 structurally | `users` axis gate; 32 output rows (R3) |
| `num_heads` | ≤ `grid.x · grid.y` (32 on quasar) | R5 check before allocation |
| `reader_threads` | `min(READER_THREADS, head_tiles)`, ≤ 6 | host clamp; target page R + W ≤ 6 |
| L1 budget | `ttnn.get_max_worker_l1_unreserved_size() - l1_reserve_bytes` (value on quasar not recorded) | unused: no op-owned allocation |

## Data-movement budget

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input | 0 | lives in L1 (height-sharded); each needed row is read straight from its shard | `num_heads · users · head_dim · elem_bytes` B (only rows `h < num_heads`; R8), in `2 · head_tiles · users` reads of `16 · elem_bytes` B per output core |
| output | 0 | lives in L1 (width-sharded); written in place by the reads that land there | none beyond the reads above; padding zero fill is core-local iDMA (`head_tiles · tile_bytes` per core), no NoC |

Totals (worked, quasar, bfloat16, `num_heads = 32`, `head_dim = 128`, `users = 1`): DRAM 0 B;
NoC 8 KiB in 256 transfers of 32 B (8 per cluster, one barrier per cluster); core-local zero fill
8 KiB per cluster.

> Cheapest-traffic split considered: `head` across cores, rows pulled by the output cores —
> each defined input byte crosses the NoC once, 0 B DRAM. Implemented: the same split
> (`pull_rows`). The implemented split is the cheapest; no deferred alternative exists (push moves
> the same bytes from fewer issuers; whole-tile pull moves 32× the bytes).
