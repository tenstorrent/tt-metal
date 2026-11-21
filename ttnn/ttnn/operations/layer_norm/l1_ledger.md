# L1 Ledger: layer_norm (regime `row_resident`, Phase 0)

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md`,
Blocking Model): **tile_row** (extent `bh` = `block_tile_rows`) and **w_tile** (extent `Wt` =
`tensor_w_tiles`, whole extent). Every ring is STRIDED and sliced per NEO: the capacities below are
**per NEO** and the allocation is that times `C` (`C_eff` of the thread plan). Page sizes:
`t_in` (input/output page), `t_aff` (weight/bias page), `t_acc` (intermediate page: Float32 when
`fp32_dest_acc_en`, else bfloat16; also the page of the reduce scalers and ε, which share an FPU chain's
single ALU format with the intermediates). Inventory was minimised
before the fit predicate was written: no buffer below exists to make the op fit, and the predicate only
chooses `bh` and `C`.

> **As built (implementer, Phase 0).** The ring set below is the implemented one. It departs from the
> planner's inventory in four rows, all for Quasar library limits found while building:
> (1) `cb_gamma` (reader → compute, Wt) and
> `cb_beta_stream` became one `cb_affine_stream` window (γ then β, same producer and consumer, R9 makes
> their dtypes equal) feeding two compute-expanded rings `cb_gamma_rows` / `cb_beta_rows`: the affine step
> now applies rstd as the FPU's srcB column broadcast, so γ must reach DEST reuse as a full tile like β;
> (2) `cb_gamma_rows`, `cb_beta_rows`, the two scalers are in `t_acc`: on Quasar the ALU source formats
> are fixed within a chain, so every operand of the affine chain and of the
> reduces shares the intermediate format; (3) `cb_rstd_rows` became a column-0 rstd tile (no
> UnaryBcast of a 32-bit operand on Quasar), and the verifier folded it into `cb_row_partial` as that ring's
> third FIFO use (disjoint lifetimes; 12 → 11 rings, −bh·t_acc per NEO); (4) `cb_epsilon` holds ε at element [0, 0] only (scalar
> broadcast), zeroed by the NoC, so no whole-tile store loop.

| CB | Capacity (pages, per NEO) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|---------------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_block` | `INPUT_BLOCK_DEPTH·bh·Wt` (depth 1) | `bh·Wt` (whole block held through `row_sum` and `mean`, then drained tile by tile in `centre`) | `{tile_row: spans → bh, w_tile: spans → Wt}` | input dtype (bf16) — the DRAM format | reader | compute | per block (row_sum → centre) | Not shared: overlaps `cb_centred` (centre streams one into the other) and the reader refills it for block k+1 while `cb_centred` of block k is live. Capacity a whole number of blocks: resident windows never straddle the ring end. |
| `cb_affine_stream` | `AFFINE_STREAM_DEPTH` = 2 | 1 | `{tile_row: streams (prologue only), w_tile: streams → window}` | weight dtype (= bias dtype, R9) | reader | compute | prologue | Shared by γ and β (one producer, one consumer, consecutive uses). Depth 2: reader ↔ expansion double buffer. |
| `cb_gamma_rows` | `Wt` | `Wt` | `{tile_row: streams (re-read per row, Row mapping), w_tile: spans → Wt}` | `t_acc` (the affine chain's single ALU format) | compute | compute | kernel | Not shared: concurrent with `cb_beta_rows`, `cb_centred`, `cb_row_partial` (rstd) in every `affine`. Per-NEO copy (no ALL ring). |
| `cb_beta_rows` | `Wt` | `Wt` | as `cb_gamma_rows` | `t_acc` | compute | compute | kernel | As above. Exists because DEST reuse takes no broadcast on Quasar. |
| `cb_centred` | `bh·Wt` | `bh·Wt` (held through `sq_sum`, drained in `affine`) | `{tile_row: spans → bh, w_tile: spans → Wt}` | `t_acc` | compute | compute | per block (centre → affine) | Not shared: overlaps `cb_input_block` and `cb_output_tiles`. Replaces a second DRAM read of x and lets the input block go after `centre`. |
| `cb_row_partial` | `bh` | `bh` | `{tile_row: spans → bh, w_tile: streams (summed away; rstd: column 0 valid, broadcast by the FPU)}` | `t_acc` | compute | compute | per block, three uses (2Σx → mean reduce; Σc² → var reduce; rstd → affine) | FIFO reuse (disjoint lifetimes, one format): each use is popped before the next is pushed. Not shared with `cb_row_stat`: the reduce reads one while writing the other, and `rstd` reads var from `cb_row_stat` while writing here. |
| `cb_row_stat` | `bh` | `bh` | `{tile_row: spans → bh, w_tile: streams (column 0 valid)}` | `t_acc` | compute | compute | per block, two uses (mean, then var) | FIFO reuse. Not shared with `cb_row_partial` (see above). |
| `cb_output_tiles` | `OUTPUT_DEPTH` = 2 | 1 | `{tile_row: streams, w_tile: streams → window}` | input dtype (output ABI) | compute | writer | per block | Double buffering packer ↔ writer. Lifetime disjoint from `cb_affine_stream` (prologue), but not shareable: a ring has exactly one producer and one consumer kernel (reader → compute there, compute → writer here). |
| `cb_mean_scaler` | 1 | 1 | constant | `t_acc` — value `1/(2W)` | reader | compute | kernel | Not shared with `cb_var_scaler`: both needed every block; the reduce reads its scaler's front tile. |
| `cb_var_scaler` | 1 | 1 | constant | `t_acc` — value `1/W` | reader | compute | kernel | As above. |
| `cb_epsilon` | 1 | 1 | constant | `t_acc`, ε at [0, 0] | reader | compute | kernel | Not shared: constant for the kernel; read as a scalar broadcast (no `add_unary_tile` on Quasar). |

## Symbol table

| Symbol | Meaning | Bound | Predicate that establishes it |
|--------|---------|-------|-------------------------------|
| `Wt` | `ceil(W/32)`, whole reduced extent | ≤ 128 in INPUTS; on Quasar ≤ 255 (fp32 DEST, ring cap on `cb_centred`) / ≤ 477 (16-bit DEST, footprint) | `fits(1, 1)` in `validate()`; beyond it `NotImplementedError` (regime `row_streaming` deferred) |
| `bh` | `block_tile_rows` | 1 ≤ bh ≤ `rounds_max = max_c ceil(rows_c / C)` | `bh = min(rounds_max, bh_fit(C))`, `bh_fit(C)` = largest bh with `fits(C, bh)` |
| `C` | NEOs per cluster engaged (`C_eff`) | {1, 2, 4} ≤ `THREAD_PLAN.C` | `select_blocking()`: smallest c filling the grid, halved while `not fits(c, 1)` |
| `t_in`, `t_aff` | input / affine page bytes | bf16 2048, fp32 4096 | `tensor.buffer_page_size()` |
| `t_acc` | intermediate page bytes | 4096 (fp32 DEST) / 2048 | `ttnn.tile_size` of `acc_fmt` |
| `l1_budget` | per-cluster budget | `ttnn.get_max_worker_l1_unreserved_size() − L1_RESERVE_BYTES` = 3,963,328 − 32,768 = 3,930,560 B on Quasar (probe) | run-time query |
| (open) L1-resident tensors | bytes per bank of an input/output in L1 | not in `l1_budget` today: Phase 0 refuses `memory_config = L1`, so only the 32 KiB reserve stands between the rings and L1 tensor pages. Largest INPUTS L1 cell: 2 × 524,288 elements × 4 B / 32 banks = 128 KiB per bank (float32 in + out) | verifier finding, folded into Refinement 1: the predicate must subtract what the allocator holds in L1 before `memory_config = L1` is claimed |
| `DFB_RING_CAP_BYTES` | per-ring extent cap | 1,048,560 B | target page; every ring `C · capacity · page ≤ cap` inside `fits()` |

## Footprint

Per NEO:

```
F(bh) = bh·Wt·(t_in + t_acc)          # cb_input_block + cb_centred      — scales with bh and Wt
      + Wt·2·t_acc                     # cb_gamma_rows + cb_beta_rows     — scales with Wt
      + 2·t_aff + 2·t_in               # cb_affine_stream + cb_output_tiles — depth knobs (2)
      + 2·bh·t_acc                     # cb_row_partial (3 FIFO uses) + cb_row_stat — scales with bh
      + 3·t_acc                        # scalers + epsilon                — constant
```

Per cluster: `C · F(bh) ≤ l1_budget`, and per ring `C · bh·Wt · max(t_in, t_acc) ≤ DFB_RING_CAP_BYTES`,
`C · Wt · t_acc ≤ DFB_RING_CAP_BYTES`. That is `fits(C, bh)` (`layer_norm_program_artifacts.py`, computed
from `ring_entries_per_neo` × `ring_page_bytes`, the same tables the ring specs are built from).

Worked as built (Quasar, bf16 input/affine, 11 rings): (1,1,64,4096), Wt = 128, C = 1, bh = 1 → F = 1,863,680 B
(fp32 DEST) / 1,067,008 B (16-bit DEST), 47 % / 27 % of the budget; largest rings `cb_centred`,
`cb_gamma_rows`, `cb_beta_rows` 524,288 B each / 262,144 B. (1,1,128,1024), Wt = 32 → 487,424 / 280,576 B.
The resident bound is unchanged: Wt ≤ 255 with fp32 DEST (ring cap on `cb_centred` and the γ/β rows).

## Data-movement budget (chosen split: `row_resident`)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| input `x` | 1 | the block stays resident (`cb_input_block`, then `cb_centred`) across all passes; no pass re-reads DRAM | none |
| output | 1 (+ at most C−1 duplicate tile-rows per op, the padded last round, same bytes) | written once from `cb_output_tiles` | none |
| `weight` (γ) | E = engaged NEOs (incl. padded), ≤ num_tile_rows | one resident copy per NEO (`cb_affine_stream` → `cb_gamma_rows`), reused by every block of that NEO; no cluster-shared ring on this part | none |
| `bias` (β) | E | as γ (`cb_affine_stream` → `cb_beta_rows`) | none |

Totals for the chosen split: DRAM = `|x| + |out| + E·(|γ| + |β|)`; cross-core = 0; core-local
(L1 → unpacker): x twice (row_sum, centre), centred twice (sq_sum with the operand deduplicated, affine),
γ and β_rows once per row, per-row tiles once each.

| Shape (bf16) | `|x|` = `|out|` | `|γ|` = `|β|` | E | DRAM total | minimum (each tensor once) |
|--------------|-----------------|---------------|---|------------|---------------------------|
| (1,1,64,4096) | 512 KiB | 256 KiB | 2 | 2 MiB | 1.5 MiB |
| (1,1,128,1024) | 256 KiB | 64 KiB | 4 | 1 MiB | 640 KiB |
| (1,1,2048,256) | 1 MiB | 16 KiB | 64 | 4 MiB | 2.03 MiB |
| (1,1,12288,1024) (untraced model call) | 24 MiB | 64 KiB | 128 | 64 MiB | 48.1 MiB |

> Cheapest-traffic split considered: `w_split_combine` (split Wt across clusters, one-round Chan
> combine) — saves `(E − 1)·(|γ| + |β|)` DRAM bytes (e.g. 512 KiB of 2 MiB at (1,1,64,4096)) and adds
> 4 tiles of cross-core traffic per (row, slice) plus one rendezvous; for shapes with
> `num_tile_rows ≥ num_clusters·C` `row_resident` is itself the cheapest. Implemented: `row_resident`.
> Deferred because the cross-cluster combine needs multicast + semaphores in a Metal 2.0 ProgramSpec,
> which has no worked example on this part at this base, while `row_resident` is required in any
> design (it is the many-row regime and the per-slice kernel of `w_split_combine`); the per-row
> partial tiles in `cb_row_partial` are the combine payload, so the structure keeps it reachable.
> The γ/β fan-out (`affine_fanout`) is the reader-only successor that removes the E× term alone.
