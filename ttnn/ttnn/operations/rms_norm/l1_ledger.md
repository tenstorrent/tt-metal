# L1 Ledger: rms_norm

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions: `op_design.md` → Blocking Model.

Named block axes: **R** (tile-rows; extent `block_rows` = `br`) and **W** (tiles of the core's W-slice; extent `block_w_tiles ≡ core_w_tiles` = `cw`).
Bounded non-block parameter in sizes: `w_split` = `k` (cores per line).
Flags (0/1, fixed per program variant): `[g]` γ present, `[g_rm]` γ ROW_MAJOR, `[ws]` regime W_SPLIT, `[tile]` / `[rm]` input layout.
Page bytes: `T_in` = input-dtype tile (2048 bf16 / 4096 fp32 / 1088 bf8b), `T_out = T_in` (output dtype = input dtype), `T_int` = intermediate tile (4096 Float32 when fp32 DEST on / 2048 Float16_b when off — Refinement 1; one `t_int` in `create_program_descriptor`), `T_g` = γ-dtype tile, `es_γ` = γ element bytes.

Page-format rule applied (Refinement 1, ledger audit 3): every compute-produced **intermediate** page follows the DEST width — `Float32` when `fp32_dest_acc_en = True`, `Float16_b` when it is off (16-bit DEST already rounded the value; a Float32 page would store no extra bits and cost 2× packer/unpacker, NoC gather/mcast and L1). One `int_fmt` / `t_int` value feeds the CB descriptors, `_per_row_bytes()` (so `block_rows` grows when it halves) and the writer's `int_tile_bytes` CT arg. Input-data CBs (`cb_x`, `cb_x_rm`, `cb_gamma*`) carry the tensor's own dtype (no compute rounding happened yet); output CBs (`cb_y`, `cb_out_rm`, RM `cb_x` after normalize) carry the output dtype by contract (the op's final rounding).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_rm` [rm] | `x_rm_depth · cw` = `2·cw` | `cw` per tile-row being tilized + `cw` being filled | `{R: streams → 1 tile-row window ×2, W: spans → cw}` | input dtype | reader | compute | per tile-row | capacity = 2× live: double-buffer so the reader fills tile-row i+1 during tilize of i. Not shared with `cb_out_rm`: both live concurrently (reader prefetch while writer drains) |
| `cb_x` | [tile] `x_block_depth · br · cw` = `2·br·cw`; [rm] `br·cw` | one block `br·cw` (resident from first push through the last normalize pop) | `{R: spans → br, W: spans → cw}` | input dtype (= output dtype; RM holds y after the in-place normalize) | [tile] reader / [rm] compute | compute | per block | [tile] capacity = 2× live: depth 2 so the reader fetches block b+1 during normalize(b) and during the W_SPLIT combine stall (op_design stall-shadow table). [rm] **is** the reuse: normalize transforms it in place and untilize reads it, replacing an RM `cb_y` of `br·cw` pages (Rule 3.2, real credit). [tile] in place is impossible: the reader is its producer |
| `cb_gamma_rm` [g_rm] | 1 page of `cw·32·es_γ` bytes | 1 page | `{R: neither (constant), W: spans → cw (one row of the slice)}` | γ dtype | writer | compute | prologue only | Disjoint from every per-block CB, but not aliased: it is ≤ `cw·128` bytes (≤ 1/32 of one tile-row of x) and aliasing it into `cb_y`/`cb_out_rm` would put a writer-produced and a compute-produced stream on one allocation, which the alias lockstep discipline forbids for different producers |
| `cb_gamma` [g] | `cw` | `cw` (whole kernel) | `{R: neither (constant, reuse-shared), W: spans → cw}` | γ dtype | [g_rm] compute (tilize) / [g, TILE γ] writer | compute | whole kernel | Live for the whole kernel → overlaps every other CB; nothing to share. Kept resident precisely so γ crosses DRAM once per core. TILE γ reads only face0/face1 row 0 per tile; block-float (bf8b) TILE γ has no face-row addressing, so its valid tiles are read whole (`w_valid·T_g`, T_g = 1088) — Refinement 1 |
| `cb_sumsq_raw` | `br` | `br` (one raw Σx² tile per tile-row; packed per tile-row, collapse waits the block) | `{R: spans → br, W: streams (summed into DEST, not stored)}` | `int_fmt` (Float32 / Float16_b) | compute | compute | sumsq → collapse | Not aliased with `cb_rinv_col` / `cb_partial`: the collapse reads it and writes them in the same reduce call (`reduce` asserts in ≠ out, `reduce_helpers_compute.inl:333`). This buffer is what remains of the square pass: x² is never materialised (squares fold into DEST, Rule 3.4) |
| `cb_partial` [ws] | `br` | `br` | `{R: spans → br, W: neither (collapsed)}` | `int_fmt` (Float32 / Float16_b) | compute | writer | collapse → gather send | Not aliased with `cb_rinv_col`: on the root the combine (writing `cb_rinv_col`) may start before the writer's pop of `cb_partial` retires; the ordering between those two threads is not CB-credit-guaranteed |
| `cb_gathered` [ws] | `k · br` | `k · br` on roots; 0 on non-roots (allocated for uniform addresses) | `{R: spans → br, W: spans → k (one partial per line core)}` | `int_fmt` (Float32 / Float16_b) | reader (root; data lands by remote writes) | compute (root) | gather → combine | Capacity = exactly one block: the landing address must be the constant CB base. Not shared: written remotely while local CBs are live. Non-root waste (`k·br·T_int`) is counted in the budget; removing it needs per-core CB layouts, which break the uniform-address rule |
| `cb_rinv_col` | `br` | `br` (LOCAL: every core; W_SPLIT: root only) | `{R: spans → br, W: neither}` | `int_fmt` (Float32 / Float16_b) | compute | compute | collapse/combine → finalize | Not aliased with `cb_rinv` / `cb_rinv_bcast`: the finalize chain reads it and packs them in the same call |
| `cb_rinv_bcast` [ws] | `br` | `br` (root only) | `{R: spans → br, W: neither}` | `int_fmt` (Float32 / Float16_b) | compute | writer | finalize → mcast send | Not shared with `cb_rinv`: the root's own `cb_rinv` receives the multicast while this buffer is still the send source; also two consumers (writer, compute) would violate single ownership |
| `cb_rinv` | `br` | `br` | `{R: spans → br, W: neither (broadcast across W)}` | `int_fmt` (Float32 / Float16_b) | LOCAL: compute / W_SPLIT: reader | compute | finalize/receive → normalize | Capacity = exactly one block (constant mcast landing address). Live across normalize, which overlaps every per-block CB |
| `cb_y` [tile] | `y_depth · y_chunk_tiles` = `2·8` | `y_chunk_tiles` draining + up to `y_chunk_tiles` being packed | `{R: streams, W: streams → window of 8}` | output dtype | compute | writer | per chunk | capacity = 2× live: double buffer (compute packs chunk c+1 while writer drains c). Does not span W, so it adds no `cw` term. Not in place on `cb_x`: the reader produces `cb_x` |
| `cb_out_rm` [rm] | `out_rm_depth · cw` = `2·cw` | `cw` being written + `cw` being untilized | `{R: streams → 1 tile-row window ×2, W: spans → cw}` | output dtype | compute | writer | per tile-row | capacity = 2× live: double buffer. Must span W: `untilize` emits a whole tile-row of sticks (`untilize_helpers.hpp:145-154`, block width is the CT `cw`). Not shared with `cb_x_rm` (concurrent) |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `br` (`block_rows`) | block extent along R | `1 ≤ br ≤ ⌈tensor_rows_tiles / num_lines⌉` | `select_blocking()`: `br = min(line_rows_tiles_max, ⌊(B − fixed_bytes)/per_row_bytes⌋)`, ≥ 1 because `k ≥ k_fit` |
| `cw` (`core_w_tiles`) | block extent along W = whole per-core slice | `1 ≤ cw ≤ tensor_w_tiles` | `cw = ⌈tensor_w_tiles / k⌉` with `k ≥ k_fit` ⇒ `footprint(1, cw, k) ≤ B` |
| `k` (`w_split`) | cores per line | `1 ≤ k ≤ grid_cores` (≤ 140 on any current part) | `k = max(k_fit, k_occ)`, `k ≤ C` by construction; host refuses if `k_fit` does not exist |
| `B` (`L1_CB_BUDGET_BYTES`) | per-core CB budget | `1 MiB` (constant) | host constant (perf lamp L8) |
| `x_block_depth`, `x_rm_depth`, `y_depth`, `out_rm_depth` | buffer depths | 2 each | host constants |
| `y_chunk_tiles` | writer chunk | 8 | host constant |
| `T_in`, `T_out`, `T_g` | tile bytes | ≤ 4096 | dtype ∈ {bf16, fp32} (SUPPORTED) |
| `es_γ` | γ element bytes | ≤ 4 | γ dtype ∈ {bf16, fp32} |

## Total per-core footprint

```
footprint(br, cw, k) = br · per_row_bytes(cw, k) + fixed_bytes(cw)

per_row_bytes(cw, k) =  [tile]· x_block_depth·cw·T_in            # cb_x            (scales with br·cw)
                      + [rm]  · cw·T_in                          # cb_x (one block, in place)
                      + 3·T_int                                    # cb_sumsq_raw, cb_rinv_col, cb_rinv   (scale with br)
                      + [ws]· (2 + k)·T_int                        # cb_partial, cb_rinv_bcast, cb_gathered (scale with br; gathered with br·k)

fixed_bytes(cw)      =  [g]   · cw·T_g                           # cb_gamma        (scales with cw)
                      + [g_rm]· cw·32·es_γ                       # cb_gamma_rm     (scales with cw)
                      + [tile]· y_depth·y_chunk_tiles·T_out      # cb_y            (constant)
                      + [rm]  · (x_rm_depth·cw·T_in + out_rm_depth·cw·T_out)   # cb_x_rm, cb_out_rm (scale with cw)
```

Which knob each term scales with: `br·cw` — `cb_x` (dominant); `cw` — `cb_gamma`, `cb_gamma_rm`, `cb_x_rm`, `cb_out_rm`; `br` — the five `int_fmt` per-row tiles; `br·k` — `cb_gathered`; constant — `cb_y`.

Worked values (bf16, γ RM bf16, `B = 1 MiB`):

| Shape | Regime | `k` | `cw` | `br` | footprint |
|-------|--------|-----|------|------|-----------|
| (1,1,32,64) TILE | LOCAL | 1 | 2 | 1 | `1·(2·2·2048 + 3·4096) + (2·2048 + 2·64 + 16·2048)` ≈ 57 KB |
| (8192, 1024) TILE | LOCAL | 1 | 32 | 3 | `3·(131072 + 12288) + (65536 + 2048 + 32768)` ≈ 530 KB |
| (1,1,32,8192) TILE | W_SPLIT | 32 | 8 | 1 | `1·(32768 + 12288 + 34·4096) + (16384 + 512 + 32768)` ≈ 233 KB |
| (8192, 7168) TILE | W_SPLIT | 2 | 112 | 1 | `1·(458752 + 12288 + 4·4096) + (229376 + 7168 + 32768)` ≈ 757 KB |
| (1,1,32,8192) RM | W_SPLIT | 32 | 8 | 1 | `1·(16384 + 12288 + 34·4096) + (16384 + 512 + 2·8·2048 + 2·8·2048)` ≈ 250 KB |

## Data-movement budget

Chosen split = LOCAL or W_SPLIT as selected (both are built). `S_x` = bytes of x (= bytes of y), `R` = `tensor_rows_tiles`, `G` = `num_lines`.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x | 1 | the block (`br × cw`) stays resident in `cb_x` from the sumsq pass through the normalize pass; a row (LOCAL) or W-slice (W_SPLIT) never leaves L1 between the two passes | 0 |
| y | 1 | written once by the writer | 0 |
| γ | `G` reads of its W elements in total (each core reads only its slice's row-0 lanes, once per kernel) | γ is reuse-shared along R: held resident in `cb_gamma` across all of a core's blocks, so per-block re-reads are 0; the remaining `G−1` extra copies are the GAMMA_MCAST deferral | 0 |
| Σx² partials (W_SPLIT) | 0 | — | `R·(k−1)·T_int` unicast core → root |
| rinv (W_SPLIT) | 0 | — | `R·T_int` multicast root → `k−1` receivers (one transfer per block per line) |

Totals per tier:

| Tier | LOCAL | W_SPLIT |
|------|-------|---------|
| DRAM | `2·S_x + G·W·es_γ` (`G = min(R, C)`) | `2·S_x + G·W·es_γ` (`G = min(R, ⌊C/k⌋)`, fewer γ copies than LOCAL) |
| cross-core NoC | 0 | `R·(k−1)·T_int + R·T_int·(k−1 receivers)` + 2 semaphore signals per block per line |
| core-local L1 | x read twice by compute (sumsq, normalize) from `cb_x` | same |

> Cheapest-traffic split considered: R × W split with γ multicast (W_SPLIT + GAMMA_MCAST) — saves `(G−1)·W·es_γ` DRAM bytes (≤ 1/16 of `S_x`, ≤ 1/32 when γ dtype = x dtype) and adds one γ rendezvous per core. Implemented: LOCAL / W_SPLIT, both at the x/y minimum (1 crossing each). GAMMA_MCAST is deferred because γ's row-0-only reads already bound its extra DRAM traffic to ≤ 1/16 of x's, and the structure keeps it reachable: `cb_gamma` is filled once in the writer prologue, which a multicast receive can replace without touching any per-block operation. The rejected STREAM_TWO_PASS split would add `S_x` DRAM bytes (x read twice).

## Implementation notes (ttnn-implementer, Phase 0)

The implemented CB set, page counts and formats match the table above row for row (`rms_norm_program_descriptor.py`
computes every page count from `select_blocking()`'s `block_rows` / `core_w_tiles` / `w_split` and the depth
constants). Non-CB L1 added: semaphore `SEM_RINV_LOCAL` (W_SPLIT only, one word) — the line root's writer bumps it
after its barriered loopback multicast so the root's reader can push its own `cb_rinv` (the Mcast sender has no
receiver role). The data-movement budget is unchanged: x and y cross DRAM once; γ is read once per core (row-0 lanes).

Measured (BH p150b, bf16, RM γ, DEVICE KERNEL DURATION): (1,1,32,4096) TILE 7.8 µs / 16 cores; (1,1,32,16384)
15.3 µs / 64 cores; (1,1,128,8192) 17.0 µs / 104 cores; (1024,1024) 15.8 µs / 96 cores; (4,1,512,512) 16.2 µs /
64 cores. RM is 15–30 % slower than TILE at the same shape.
