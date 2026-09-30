# L1 Ledger: mhc_post

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Block axes (from `op_design.md` Blocking Model): `r` (token-tile row, extent `block_token_tiles = 1`), `c` (column tile, extent `B = block_col_tiles`), `j` (output stream, extent `n`), `i` (input stream / contraction, extent `n`), `k` (coefficient index of the expanded set, `n + n²` terms packed two per tile into `n · P` tiles, `P = ceil((n+1)/2)`; Refinement 2).

Tile bytes: `fB` = F tile bytes (4096 fp32 / 2048 bf16), `xB` = X / X' tile bytes (same values), coefficient tiles are always 4096 (fp32).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_sublayer_tiles` | `depth_in · B` | `2 · B` when the reader fills block k+1 while compute holds block k; `B` otherwise | `{r: streams → 1 row per block, c: spans → B, j: streams → the block stays resident across the n output-stream walk (re-read, not re-fetched), i: streams → not an axis of F, k: streams → not an axis of F}` | F dtype: Float32 / Float16_b (both live since Refinement 1). An **input** page carries the tensor's own dtype; nothing is packed into it from DEST, so the 16-bit page under fp32 DEST is exact (audit 2 *under* does not apply) | reader | compute | per block (`load_block` → end of `mix_block`) | not merged with `cb_residual_tiles`: F and X have independent dtypes (`sublayer_dtype` vs `dtype` axes), and a CB has one data format. Capacity − live set = 0 at depth 2 (the double buffer IS the live set during overlap) |
| `cb_residual_tiles` | `depth_in · n · B` | `2 · n · B` during overlap | `{r: streams → 1 row, c: spans → B, i: spans → n, j: streams → resident across the j walk, k: streams → not an axis of X}` | X dtype: Float32 / Float16_b (both live since Refinement 1); input page, as above | reader | compute | per block | cannot be the output buffer (in-place X → X'): every X'_j needs all n X_i, so no X_i slot is free until the last j is packed; and an in-place output would give this CB two consumers (compute + writer). Not merged with F (format, above) |
| `cb_coef_raw` | `ceil(n/32) + ceil(n²/32)` (= 2 for n ≤ 5) | 2 — both raw tiles are read by the expansion | `{r: streams → one row per fill, c: streams → coefficients are constant along c, j: spans → post columns (inside one raw tile), i: spans → comb columns (inside one raw tile), k: spans → all n + n² raw values (inside 2 tiles)}` | Float32 (fixed contract: post/comb are float32) | the writer (BRISC role of `mhc_post_dm.cpp`; since Perf 1 the only expander — the `COEF_EXPANDER` reader option was removed) | same kernel (private scratch) | per segment: read → expand → pop | not aliased onto `cb_coef_bcast`: the raw tiles are the source of every one of the n + n² expanded tiles being written, so its lifetime overlaps the whole expansion; in-place expansion would overwrite unread raw columns. 8 KB total |
| `cb_coef_bcast` | `coef_depth · n · P`, `P = ceil((n+1)/2)` (Refinement 2; was `coef_depth · (n + n²)`) | `n · P` for the row compute is mixing, `+ n · P` while the expander prepares the next row (pushed one stream, P tiles, at a time since Refinement 3) | `{r: streams → one row per set, coef_depth sets in flight, c: streams → one set serves every column of the segment (column-broadcast in-tile), j: spans → P tiles per output stream, i: spans → term 1+i of stream j, k: spans → n + n² terms, two per tile (half-packed: term t of stream j in faces 0/2 (t even) or 1/3 (t odd) of tile j·P + t/2; mhc_post_common.hpp)}` | Float32 — must be fp32 (coefficients applied as fp32, contract); read with `UnpackToDestFp32` | the writer (BRISC role of `mhc_post_dm.cpp`), single producer | compute | per segment (`load_coefficients` → `release_coefficients`) | no disjoint-lifetime partner: it is live for the whole segment, concurrently with every block buffer. Capacity − live set = one row set only while the next row is being prepared (explicit pipelining decision: row boundaries fall mid-range on most cores; `coef_depth` knob). **Refinement 2 re-justification (consumer changed):** compute now copies stream j's P tiles into DEST once per output column window (not 2 per term per tile), and the SFPU reads a coefficient half for both data faces of the same rows, so a half-tile per term is sufficient — the right-face duplicate the old layout carried was pure redundancy (12 vs 20 tiles at n = 4, 96 KB vs 160 KB at depth 2). Depth 2 kept for the same row-boundary reason, and it is now also **required** when the writer expands (host-asserted): the writer loads segment s+1's set right after writing segment s's first block, while compute still holds segment s's set; the tiles cannot be DEST-resident across windows because every pack releases DEST with a full ZEROACC (SyncFull) |
| `cb_output_tiles` | `depth_out · n · B` | `2 · n · B` during overlap | `{r: streams → 1 row, c: spans → B, j: spans → n, i: streams → contracted away in DEST, k: streams → not an axis of X'}` | X dtype: Float32 (Phase 0; fp32 DEST packed to fp32 page — audit 2 consistent) / Float16_b (Refinement 1: one RNE rounding of the fp32 result by the packer — measured signed bias ≤ 1.3e-5 rel, no systematic shrink) | compute | writer | per block (`mix_block` → `store_block`) | cannot alias an input CB: concurrent lifetime with `cb_residual_tiles` (the X block must stay until all n outputs are packed) and the writer is a second thread |

No intermediate CB: the whole per-output-tile expression (`post_j·F + Σ_i comb_ij·X_i`) folds into one DEST accumulator (reuse pattern 4, "fold into the accumulator") and packs straight into the destination (pattern 1). The phase boundary between "scale F" and "add the mixed streams" is not storage.

## Symbol table

| Symbol | Bound | Predicate establishing it |
|--------|-------|---------------------------|
| `n` | 1 ≤ n ≤ 5 | `validate` (mechanism cap: `n² ≤ 32`, comb row in one raw tile) |
| `B` = `block_col_tiles` | 1 ≤ B ≤ `min(B_fit, max_segment_col_tiles, MAX_BLOCK_COL_TILES, ceil(max_units_per_core / MIN_BLOCKS_PER_CORE))` | host (Refinement 3 policy, defaults 8 and 3): the coarsest L1 fit left 1–2 blocks per core with no read/mix/write overlap; `B_fit ≥ 1` asserted |
| `depth_in`, `depth_out`, `coef_depth` | Phase 0: 2, 2, 2 | host constants (knobs) |
| `fB`, `xB` | ∈ {2048, 4096} | F / X dtype ∈ {bfloat16, float32} (registry axes) |
| `L1_BUDGET_BYTES` | 1 MiB (1 048 576) | host constant; below the usable L1 on Wormhole (1464 KB) and Blackhole (1536 KB) with room for kernel binaries and the allocator base |
| `max_segment_col_tiles` | ≤ `tensor_col_tiles` = C/32 | derived from the work split |

## Footprint

```
coef_bytes  = coef_depth · n · ceil((n+1)/2) · 4096  +  (ceil(n/32) + ceil(n²/32)) · 4096
per_col     = depth_in · (fB + n · xB)  +  depth_out · n · xB
footprint   = B · per_col  +  coef_bytes
B_fit       = floor((L1_BUDGET_BYTES − coef_bytes) / per_col)
```

- Terms scaling with `B` (and `depth_in` / `depth_out`, `n`, dtypes): the three streaming CBs.
- Terms scaling with `coef_depth` and `n²`: the coefficient set only. Nothing scales with T or C.
- n = 4, depths 2 (Refinement 2 half-packed set, P = 3): `coef_bytes` = 98 304 + 8 192 = 106 496 (was 172 032 with n + n² full tiles).
  - fp32 / fp32: `per_col` = 2·(4096 + 16384) + 2·16384 = 73 728 → `B_fit` = 12 (was 11), footprint at B=12 = 991 232 B.
  - bf16 / bf16: `per_col` = 36 864 → `B_fit` = 25 (was 23), footprint at B=25 = 1 028 096 B.
  - bf16 F / fp32 X: `per_col` = 2·(2048 + 16384) + 32768 = 69 632 → `B_fit` = 13 (was 12).
  - fp32 F / bf16 X: `per_col` = 2·(4096 + 8192) + 16384 = 40 960 → `B_fit` = 23 (was 21).
  - (Refinement 1: all four combos run; `B` is the smaller of the row above and the longest per-core segment, e.g. 11 at T640 C1792.)
  - Refinement 3: the block policy caps `B` at 8 and at ceil(units per core / 3), so `B_fit` is now an upper bound rather than the chosen size (T640 C1792: B = 4; T640 C7168 and T1280 C4096: B = 8). At B = 8 the footprint is 8 · per_col + coef_bytes, i.e. 696 320 B fp32/fp32 and 401 408 B bf16/bf16. L1 headroom is deliberately left unused: deeper `DEPTH_IN` (3) measured slower (more DRAM burst at the start, not more overlap).

## Data-movement budget

Chosen split: `flat_stream` (flattened (token row, column tile) units, contiguous per core, full grid).

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| F | 1 | each unit's F tile is read by exactly one core, once; resident across the n-stream walk inside the block | none |
| X | 1 | each X_i tile read once by the one core owning its unit; resident across all n output streams | none |
| X' | 1 (write) | packed once from the DEST accumulator, written once | none |
| post | ≤ `(tensor_token_tiles + num_cores − 1) / tensor_token_tiles` per row (≈ 1 + cores per row) | each core reads the raw tile of every token row it touches (reuse-shared operand, re-read instead of multicast); 4 KB per read | none |
| comb | same as post | same | none |

Refinement 3: post / comb are now read by the writer over NoC1 instead of by the reader. The crossings are unchanged; the reader streams F / X only. Perf 1: when the busiest core has ≥ `HELP_MIN_BLOCKS` blocks, the writer also reads the F block of every block ≥ `HELP_FROM_BLOCK` into the reader's reserved `cb_sublayer_tiles` window (NoC0, dynamic-NoC mode; two L1 semaphores). No CB size changes.

Totals at T=640, C=7168, fp32, n=4 (grid 110): F 18.4 MB + X 73.4 MB + X' 73.4 MB + post/comb ≤ 129 × 8 KB ≈ 1.06 MB ≈ **166 MB DRAM** (bf16/bf16: F 9.2 + X 36.7 + X' 36.7 + 1.06 ≈ **83.7 MB**), 0 B cross-core, core-local L1: each F/X tile unpacked n times from L1 (re-read locally instead of refetched).

> Cheapest-traffic split considered: `flat_stream` — it is the implemented split; the only above-minimum term is the ≤ 1.06 MB coefficient re-read (0.6%). `height_split` ties on DRAM bytes (−1 MB) but occupies 20 of 110 cores at T=640; `coef_mcast` would save that ≤ 1 MB at the cost of ~10× more NoC bytes (80 KB expanded set per receiving core). Implemented: `flat_stream`. Nothing is deferred on traffic grounds.
