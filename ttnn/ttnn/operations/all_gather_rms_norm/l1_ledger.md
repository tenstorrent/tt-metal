# L1 Ledger: all_gather_rms_norm

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking axes come from `op_design.md` → Blocking Model:
**row** (`block_row_tiles`, blocks `num_blocks_this_core`), **W** (`core_w_tiles`, pitch `w_pitch`; regime N:
`land_w_tiles` / `out_w_tiles`), **device/slice** (`G`, pinned), **item** (`xfer_rows`), **chunk** (`run_pages`).

Inventory decisions made before any sizing (Rule 1 / Rule 3):

| Candidate buffer | Decision | Pattern |
|---|---|---|
| a gathered-row staging buffer between the gather and the norm (the composite's DRAM round trip) | **does not exist**: remote slices land directly in the consumer's `cb_x_landing` | pack into the destination (the fabric router writes the final consumer slot) |
| a copy of the own slice for the ports | **does not exist**: ports read the members' `cb_x_landing` | reuse of the resident block |
| an x² block buffer | **does not exist**: Σx² is accumulated in DEST | fold into the accumulator |
| an `x·inv_rms` intermediate before ·gamma | **does not exist**: gamma is a DEST-reuse row-broadcast multiply in the same chain | fold into DEST |
| regime N: a resident dense-aligned copy of the row | **does not exist**: dense tiles are composed into a 2-unit stream `cb_x_aligned` at pass 2. Pass 1 runs on the landing tiles directly, with the slice tails zeroed in place. | stream instead of storing |
| port staging | one 2-batch ring per port (`cb_port_chunks`) | required by the sender (one producer per router channel, rule 8: no local copies on the sender) |

## Ledger

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_landing` (0) | `num_blocks_max · block_row_tiles · w_pitch` (regime N: `w_pitch = land_w_tiles`) | = capacity: every block of the core is resident from its landing until the end of the call, because peers write it without credit | {row: spans → `num_blocks_max·block_row_tiles`; W: spans → `w_pitch`; device: spans (slices nested in the member range); item: streams (lands per item); chunk: streams} | input dtype (Float32 / Float16_b / Bfp8_b) | reader (credit; bytes also written by fabric routers and the reader's DRAM reads) | compute | program | none: it is declared **first on workers ∪ ports** so its base address is identical everywhere (the senders address peers with it). It cannot alias anything: landed rows must survive until their own normalize, and a relay may still read them after. |
| `cb_gamma` (1) | `w_pitch_out + RMS_GAMMA_TAIL_PAGES` (base op compact pages; `w_pitch_out = w_pitch` in A, `out_w_tiles` in N), only when `has_gamma` | = capacity (program-resident) | {row: absent (gamma has no row extent); W: spans → `w_pitch_out`; device: spans (full-W gamma, member columns only); item, chunk: absent} | gamma dtype, compact row-0 faces 0/1 (`P_g` stride) | writer | compute | program | none: live for every block of the program, which overlaps every other CB |
| `cb_ones` (2) | 1 (constant: the 1.0 reduce scaler) | 1 | {row: absent; W: absent; device: absent; item, chunk: absent} | Float16_b | reader (`calculate_and_prepare_reduce_scaler<SUM, REDUCE_ROW>`) | compute | program | none: constant for every collapse of the program |
| `cb_sumsq_tiles` (3) | `block_row_tiles` | `block_row_tiles` | {row: spans → `block_row_tiles`; W: absent (reduced in DEST); device: absent; item, chunk: absent} | T_acc (Float32 iff `fp32_dest_acc_en`) | compute | compute | pass 1 → collapse of one block | could alias `cb_partial_sumsq` (disjoint within a block), but the collapse **reads** sumsq while **writing** the partial, so the lifetimes overlap during the collapse: not disjoint |
| `cb_partial_sumsq` (4) (k > 1) | `block_row_tiles` | `block_row_tiles` | {row: spans; W: absent; device: absent; item, chunk: absent} | T_acc | compute | writer | collapse → exchange | not with `cb_inv_rms`: the writer may still be shipping partial(b) while compute holds inv(b−1) for normalize(b−1) |
| `cb_gathered_sumsq` (5) (k > 1) | symmetric: `depth_g · k · block_row_tiles` (`depth_g = 2` if `num_blocks_max > 1` else 1); owner: `ceil(block_row_tiles/k) · ceil(k/32)` row-form gather tiles (uniform per block: lockstep slots) | symmetric: `k·block_row_tiles` per block, 2 blocks when a peer runs one block ahead | {row: spans; W: spans → `k` (one partial per member); device: absent; item, chunk: absent} | T_acc | writer (peers' NoC writes, credit by my writer) | compute | exchange → combine | none: peers write it without my credit (lockstep cap, op_design Mechanism caps) |
| `cb_inv_owned` (6) (owner combine) | `ceil(block_row_tiles/k)` (owned rows per block) | owned rows | {row: spans → `ceil(block_row_tiles/k)`; W: absent; device: absent; item, chunk: absent} | T_acc | compute | writer | owner combine → return | none: the writer returns it while compute may already wait on `cb_inv_rms` |
| `cb_inv_rms` (7) | `block_row_tiles` | `block_row_tiles` | {row: spans; W: absent (column 0, broadcast over W); device: absent; item, chunk: absent} | T_acc | compute (k = 1, symmetric) / writer (owner return) | compute | finalize → end of normalize of the block | not with `cb_sumsq_tiles`: in the k = 1 path the collapse reads sumsq while it writes inv, and in the symmetric path the combine of block b can overlap pass 1 of nothing else but is popped only after normalize(b) |
| `cb_x_aligned` (8) (regime N) | `compose_depth · out_unit` (2 · `min(out_w_tiles, 16)`) | `out_unit` (+1 unit in flight) | {row: streams; W: streams → window `out_unit`; device: streams (composed from all slices); item, chunk: absent} | input dtype (Float16_b when staged: bfp8 regime N) | reader | compute | per unit | not with `cb_out_tiles`: both are live at once (compose of unit u+1 while unit u is normalized and packed) |
| `cb_x_stage` (12) (regime N, bfloat8_b only; Refinement 2) | `STAGE_ROWS · land_w_tiles` (`STAGE_ROWS` = 1) | one landed row | {row: streams (one row at a time); W: spans → `land_w_tiles` (every slice, own included); device: spans (all G slices); item, chunk: absent} | Float16_b | compute (copy_tile of the row's bfp8 landing / own-shard tiles → bf16 pack, before that row's normalize) | reader (the compose reads its 2-byte datums; one wait / pop per row) | one row: compute → reader → compute round trip | none: live together with `cb_x_landing` (its source) and `cb_x_aligned` (the compose's target). A finer window (units of `out_unit`) would need compute to interleave the copy with the normalize per unit; one row is ≤ 256 KiB on every INPUTS shape (`(1,1,64,1000)`, G = 4: 128 tiles) and the k = 1 split's L1 fit check includes it. |
| `cb_out_tiles` (9) | `out_depth · out_unit` (2 · `min(core_w_tiles or out_w_tiles, 16)`) | `out_unit` (+1 unit in flight) | {row: streams; W: streams → `out_unit`; device: absent; item, chunk: absent} | output dtype | compute | writer | per unit | no reuse of the landing slot: packing over `cb_x_landing` in place is illegal because a relay port may still read the raw landed row Native sharded placement (Refinement 1): **backed on the output shard** (zero-copy, whole shard, not CB budget). Compute packs every row at its final place; the writer only waits / pops (no NoC write). |
| `cb_port_chunks` (10) (ports only) | `2 · group` chunk slots of `run_pages · T_in` (`group = min(8, port_cb_bytes / (2·run_pages·T_in))`, `port_cb_bytes = 112 KiB`) | `2·group` chunks (one batch read, one sent) | {row: streams (items); W: streams (link part); device: streams (entries); item: streams; chunk: spans → `run_pages`} | input dtype | port reader | port sender | per batch | none on the port core: it is the only CB a port uses (besides the address-only declaration of `cb_x_landing`) |
| `cb_x_in` (11) (native HEIGHT only) | the input shard (`rows_per_shard · Wt_lp` pages), **backed on the input shard** (zero-copy, not CB budget) | = capacity (resident for the call) | {row: spans → the shard rows; W: spans → `Wt_lp` (the own slice); device: absent; item, chunk: absent} | input dtype | reader (one reserve/push of the resident shard; regime N zeroes the own tail tile's W padding in place) | compute (Σx² + normalize of the own-slice columns) and the ports (NoC reads of the own slice, read-only) | program | none: it IS the input tensor. Replaces the own-slice load into `cb_x_landing` (those landing columns stay unused on native HEIGHT members). |

Semaphores (not L1 CBs; listed for completeness): `sem_landed[0..2L−1]`, `sem_partials` / `sem_partials_odd` (one per block parity, verifier fix: see op_design.md Mechanism caps), `sem_inv`, `sem_local_loaded`, `sem_arrival` (program semaphores on workers ∪ ports, ≤ 2L + 5 ≤ 9 ids), plus one cached global `ready` semaphore on the union of port cores.

## Symbol table (non-block parameters)

| Symbol | Meaning | Bound | Predicate establishing it |
|---|---|---|---|
| `T_in` = `T_out` | input / output tile bytes | ≤ 4096 (fp32) | dtype ∈ {float32, bfloat16, bfloat8_b} (validate) |
| `T_acc` | intermediate tile bytes | 4096 (fp32 DEST) / 2048 | `fp32_dest_acc_en` |
| `P_g` | compact gamma page stride | ≤ base op `_gamma_page_bytes(float32)` | gamma dtype ∈ {float32, bfloat16} |
| `k` | members per row group | ≤ `K_CAP = min(164, n_workers)` | `select_split()` candidate set |
| `w_pitch` | max member width (tiles) | ≤ `land_w_tiles = G·Wt_lp` | nested member ranges |
| `num_blocks_max` | blocks on the busiest core | `ceil(R_t / n_active_groups)` | round-robin rows |
| `run_pages` | tiles per packet | `floor(payload / T_in)` ≤ 13 | active fabric router payload (14 KiB + 64 B) |
| `group` | chunks per port batch | ≤ 8 | `port_cb_bytes = 112 KiB` |
| `out_unit` | writer / compose unit | ≤ 16 | `OUT_UNIT_CAP` (base op) |
| `L` | links | ≤ discovered | validate |
| budget | CB bytes per core | `cb_l1_budget(device)` (base op `rms_norm_program_descriptor.py:888`): ≈ 1.4 MiB on BH with no L1 tensors | allocator state at call time (L1-sharded input / output lower it) |

`select_split()` admits a split only if both totals below are ≤ budget. If none is admitted, the call raises `RuntimeError` (the `dram_landing` regime is the deferred fallback).

## Footprint per core

```
worker(k) = num_blocks_max·block_row_tiles·w_pitch·T_in          # cb_x_landing      — scales with row blocks × W member width
          + has_gamma·(w_pitch_out + tail)·P_g                     # cb_gamma          — scales with W
          + 2048                                                   # cb_ones           — constant
          + 3·block_row_tiles·T_acc                                # sumsq, partial, inv — scale with block_row_tiles
          + [k>1]·depth_g·k·block_row_tiles·T_acc                  # cb_gathered       — scales with k × block_row_tiles (symmetric)
          + [owner]·owner_terms (base op cb_plan)
          + [N]·2·out_unit·T_in                                    # cb_x_aligned      — constant window (bf16 tiles when staged)
          + [N, bfp8]·STAGE_ROWS·land_w_tiles·2048                 # cb_x_stage        — scales with W (one landed row)
          + 2·out_unit·T_out                                       # cb_out_tiles      — constant window
port      = num_blocks_max·block_row_tiles·w_pitch·T_in            # cb_x_landing (address-only, same CB)
          + 2·group·run_pages·T_in                                 # cb_port_chunks ≈ 112 KiB
```

The landing term dominates. It equals the core's share of the chip's whole gathered tensor (`≈ R_t·land_w_tiles·T_in / busy_cores`), which is the price of landing without flow control.

Largest INPUTS cases (G = 8, the Galaxy axis 0, worst over dtypes):

| Case | Regime, k*, blocks × w | worker footprint | port footprint |
|---|---|---|---|
| (1,1,2048,1280) fp32 | A, k = 16, 11 × 20 | 220·4 KiB + small ≈ 0.90 MiB | ≈ 0.97 MiB |
| (1,1,64,1000) fp32 | N, k = 1, 1 × 256 landing | 1 MiB + compose/out 2·16·4 KiB·2 + gamma ≈ 1.31 MiB | ≈ 1.11 MiB |
| (1,1,128,8192) fp32 | A, k = 24, 1 × 86 | ≈ 0.35 MiB | ≈ 0.45 MiB |
| focus 640×384 bf16, G = 4 | A, k = 8, 2 × 6 | ≈ 0.04 MiB | ≈ 0.14 MiB |

## Data-movement budget (focus case: 640 × 384 bf16 per chip, G = 4 ring, L = 2)

S = 640·384·2 B = 491,520 B.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x (own slice) | 1 (S) | read once by its member into the resident landing slot. Both ports re-send it from member L1. Native HEIGHT shards: 0 — consumed in place from the input shard (`cb_x_in`), the ports read it there. | on-chip NoC: 2·S port reads (fwd + bwd) |
| x (remote slices) | 0 | landed by the fabric directly into member L1 | links: busiest direction 1.5·S = 737,280 B (368,640 B per link); on-chip: 3·S landing writes + 1·S relay reads |
| out | 1 (G·S = 1,966,080 B) | written once from `cb_out_tiles` per block. Native sharded output: 0 NoC writes — packed into the resident output shard. | — |
| gamma (RM sticks) | 13 groups × 48 sticks × 64 B = 39,936 B (≈ 0.08·S) | reuse-shared across row groups, read once per group (the `gamma_mcast` row is deferred) | — |
| partials | 0 | in-chip only | symmetric k = 8: 64 · 2 KiB per row-group-block · 20 rows ≈ 2.6 MB on-chip NoC |

Totals per tier: DRAM ≈ 2.50 MB (x 0.49 + out 1.97 + gamma 0.04) → roofline 4.9 µs at 512 GB/s. Links: 0.74 MB on the busiest direction over 2 links → 7.6 µs at 48.5 GB/s. On-chip NoC ≈ 0.98 + 0.49 + 1.47 + 2.6 ≈ 5.5 MB, spread over ~100 cores and both NoCs.

> Cheapest-traffic split considered: raw-slice relay gather into member L1 with every chip normalizing the full row
> (minimum on DRAM: x once, out once; minimum on links: (G−1)/2·S per busiest direction). Implemented: the same split.
> It is also the cheapest. The nearest alternatives (the example's DRAM-sourced ports: +2·S DRAM; DRAM landing:
> +(2G−1)·S DRAM; stats-then-normalized gather: same bytes plus a serial exchange) are `rejected` or `deferred` rows
> in op_design.md's regime table, with their deltas.
