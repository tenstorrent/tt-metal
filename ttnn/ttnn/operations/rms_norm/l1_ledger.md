# L1 Ledger: rms_norm (regime R1 `row_resident`)

Block axes (from `op_design.md`, Blocking Model): `row` (extent `R = block_row_tiles`) and `w` (block extent `Wt = tensor_w_tiles`, whole row; streaming sub-extent `C = w_chunk_tiles` inside `normalize_block`).
`fp32_dest_acc_en = True` in Phase 0, so every compute-packed intermediate is Float32.

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_input_tiles` | `in_depth * R * Wt` | `R * Wt` (one block resident across sumsq and normalize) + `R * Wt` in flight from the reader when `in_depth = 2` | `{row: spans -> R, w: spans -> Wt}` | input dtype (Float16_b / Float32), set by the tensor | reader (TILE) / compute tilize (RM) | compute | per block, `load_block` -> end of `normalize_block` | none: live across every compute phase of the block, so no other buffer has a disjoint lifetime with it. The `in_depth = 2` slot is the double buffer (reader overlap); RM uses `in_depth = 1` because `cb_input_rm` carries the overlap |
| `cb_scaler` | `1 + P` | `1 + P` (constant: full scaler 1.0, partial scaler) | `{row: streams (constant), w: streams (constant)}` | Float16_b (scaler 1.0 is exact; pool-type-aware generator) | reader | compute | whole kernel | none: live for the whole kernel; constant tile |
| `cb_gamma_stick` | 1 page of `align_up(W, 32) * gs` bytes (G only) | 1 | `{row: streams (shared across rows), w: spans -> align_up(W,32) elements}` (a single row vector, `Wt*32` elements, not tiles) | gamma dtype | writer | writer (scratch) | whole kernel | none: whole-kernel lifetime; it is the source of every gamma chunk. Bounded by the R1 predicate (`W` bounded, see symbol table) |
| `cb_gamma_tiles` | `gamma_depth * C` (G only) | `C` (one chunk read by N2) + `C` being built | `{row: streams (Row-mapped, reused by all R rows), w: streams -> window C}` | gamma dtype (a format conversion would need a compute pass) | writer | compute | per chunk | cannot alias `cb_x_normed`: concurrent within the chunk (N2 reads both). `gamma_depth = 2` is the double buffer (writer builds chunk c+1 during N2 of chunk c) |
| `cb_sumsq` | `(1 + P) * R` | `(1 + P) * R_b` (all sum tiles of the block exist before `row_rinv_block` starts: sequential helpers) | `{row: spans -> R, w: streams (collapsed to 1 tile, 2 with P)}` | Float32 (fp32 DEST; a 16-bit page would truncate the accumulated sum) | compute | compute | per block, `row_sumsq_block` -> `row_rinv_block` | cannot alias `cb_rinv`: both live inside `row_rinv_block` (reduce reads one, writes the other). Disjoint from `cb_x_normed` in lifetime, but aliasing would need lockstep pointers over different quanta (`(1+P)*R` per block vs `R*C` per chunk); its size (`2R` Float32 tiles at most) does not justify it |
| `cb_rinv` | `R` | `R_b` (held across every chunk of `normalize_block`) | `{row: spans -> R, w: streams (Col-broadcast over every chunk)}` | Float32 (feeds the broadcast multiply at full precision) | compute | compute | per block, `row_rinv_block` -> end of `normalize_block` | none: live across all of `normalize_block`, overlapping every chunk-scoped buffer |
| `cb_x_normed` | `R * C` (G only) | `R_b * C_c`: N1 must finish the chunk before N2 starts (both helpers own all TRISCs) | `{row: spans -> R, w: streams -> window C}` | Float32 (fp32 DEST product; a 16-bit page would add a rounding step before the gamma multiply) | compute | compute | per chunk, N1 -> N2 | cannot alias `cb_output_tiles` for bf16 outputs: page formats differ (Float32 vs Float16_b), so the two pointers cannot advance in lockstep. For fp32 outputs an alias group is legal; recorded as a perf lamp in `op_design.md`. Not needed at all without gamma: N1 packs straight into `cb_output_tiles` (pack-into-destination) |
| `cb_output_tiles` | `out_depth * C` | `C` being packed + `C` being drained | `{row: streams, w: streams -> window C}` | output dtype (the tensor's format; the pack converts from Float32 DEST) | compute | writer (TILE) / compute untilize (RM) | per chunk | none: concurrent with `cb_x_normed` and `cb_gamma_tiles` within the chunk. `out_depth = 2` is the double buffer against the writer |
| `cb_input_rm` (RM only) | `rm_in_depth * C` | `C` tile-equivalents (32 sticks of one C-wide chunk) + the next chunk in flight | `{row: streams (32-stick group at a time), w: streams -> window C}` | input dtype | reader | compute (tilize) | per chunk | none: concurrent with `cb_input_tiles` (tilize writes it from this buffer). Depth 2 = reader overlap |
| `cb_output_rm` (RM only) | `rm_out_depth * C` | `C` tile-equivalents being untilized + the previous chunk draining | `{row: streams, w: streams -> window C}` | output dtype | compute (untilize) | writer | per chunk | none: concurrent with `cb_output_tiles` (untilize reads that while packing this). Depth 2 = writer overlap |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing the bound |
|--------|---------|-------|----------------------------------|
| `R` | `block_row_tiles` | `1 <= R <= ceil(core_row_tiles_max / MIN_BLOCKS_PER_CORE)` | `select_blocking()` |
| `R_b` | rows of block b (runtime) | `1 <= R_b <= R` | ragged last block |
| `Wt` | `tensor_w_tiles = ceil(W / 32)` | `footprint(1, 1, 1) <= L1_CB_BUDGET` gives `Wt` <= about 600 (bf16) / 300 (fp32) | R1 predicate (regime table) |
| `W` | true last-dim length | `W <= 32 * Wt` | same |
| `C` | `w_chunk_tiles` | `1 <= C <= Wt`; RM: `C` divides `Wt` | `select_blocking()` |
| `C_c` | width of chunk c (runtime) | `1 <= C_c <= C` | ragged last chunk |
| `in_depth` | input block depth | `{1, 2}`; 1 on RM | `select_blocking()` |
| `gamma_depth`, `out_depth`, `rm_in_depth`, `rm_out_depth` | streaming depths | 2 | host constants |
| `P` | `1 if W % 32 else 0` | `{0, 1}` | shape |
| `G` | `has_gamma` | `{0, 1}` | call |
| `xt`, `ot` | input / output tile bytes | 2048 (bf16), 4096 (fp32) | dtype |
| `gt`, `gs` | gamma tile bytes / element bytes | 2048 / 2 (bf16), 4096 / 4 (fp32) | gamma dtype |
| `ft` | Float32 tile bytes | 4096 | constant |
| `st` | bfloat16 tile bytes | 2048 | constant |
| `L1_CB_BUDGET` | `ttnn.get_max_worker_l1_unreserved_size() - L1_HEADROOM_BYTES` | device query minus 64 KiB | host |

## Total per-core footprint

TILE (Phase 0):

```
F(R, C, in_depth) =
      in_depth * R * Wt * xt                       # cb_input_tiles      scales with R and Wt (and the depth knob)
    + (1 + P) * st                                 # cb_scaler           constant
    + (1 + P) * R * ft                             # cb_sumsq            scales with R
    + R * ft                                       # cb_rinv             scales with R
    + G * align_up(W, 32) * gs                     # cb_gamma_stick      scales with W only
    + G * gamma_depth * C * gt                     # cb_gamma_tiles      scales with C
    + G * R * C * ft                               # cb_x_normed         scales with R * C
    + out_depth * C * ot                           # cb_output_tiles     scales with C
```

RM (refinement 1): `in_depth = 1`, plus `rm_in_depth * C * xt + rm_out_depth * C * ot`.

Dominant terms: `in_depth * R * Wt * xt` (the resident block) and `G * R * C * ft`. Turning `R` scales the first, third, fourth and seventh terms; turning `C` scales the sixth to eighth; `in_depth` only the first.

Worked check, largest Phase 0 cases (`R = 1`):
- bf16, W = 8192 (Wt = 256), gamma bf16, `in_depth = 1` (Rt = 1): `512 KiB (x) + 2 KiB (scaler) + 4 KiB (sumsq) + 4 KiB (rinv) + 16 KiB (stick) + 2*C*2 KiB (gamma) + C*4 KiB (normed) + 2*C*2 KiB (out)`; C = 64 gives about 1.28 MiB (near the budget), so `select_blocking()` settles on the largest C that fits the device budget (C >= 8 always fits).
- fp32, W = 8192, gamma fp32, `in_depth = 1`: `1024 KiB (x) + 10 KiB (scaler, sumsq, rinv) + 32 KiB (stick) + 2*C*4 KiB (gamma) + C*4 KiB (normed) + 2*C*4 KiB (out)`; fits at `C = 8` (about 1.20 MiB).

## Data-movement budget (R1, chosen split: rows across cores)

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| `x` | 1 | the whole row stays resident in `cb_input_tiles` across `row_sumsq_block` and `normalize_block` | 0 |
| `out` | 1 | streamed once from `cb_output_tiles` | 0 |
| `gamma` | `A = min(tensor_row_tiles, grid_cores)` (once per active core) | the RM stick is held in `cb_gamma_stick` for the whole kernel; chunk tiles are rebuilt from L1, not DRAM | 0 |
| reduce scaler | 0 | generated on-core | 0 |

Totals per tier:
- DRAM read: `tensor_row_tiles * Wt * xt + A * W * gs` (TILE; RM reads `rows * W * xs`), with `A * W * gs <= (tensor_row_tiles * Wt * xt) / 32 * (gs / xs)` because each active core owns at least one 32-row tile-row.
- DRAM write: `tensor_row_tiles * Wt * ot`.
- Cross-core NoC: 0.
- Core-local L1: `x` block read twice by compute (sumsq, normalize); per block `2 * Wt` gamma row-0 segment copies; Float32 intermediates `cb_sumsq`, `cb_rinv` (one tile per row), `cb_x_normed` (one pass per tile, gamma only).

> Cheapest-traffic split considered: rows across cores with the whole row resident (R1). Implemented: R1. It is also the cheapest split: R2 (`w_split_combine`) ties it on DRAM and adds `~2 * R` tiles of cross-core traffic per group per block. R2 stays deferred because every Phase 0 SUPPORTED shape satisfies R1's footprint predicate. The `w` extent is already a CT knob (`core_w_tiles`), and R2 only inserts a combine stage between `row_sumsq_block` and `row_rinv_block`, so R1 keeps it reachable.
