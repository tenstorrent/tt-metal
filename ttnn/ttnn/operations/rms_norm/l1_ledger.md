# L1 Ledger: rms_norm

Schema and audits: `.claude/references/l1-footprint-discipline.md`. Blocking decisions: `op_design.md` → Blocking Model.

Named block axes: **R** (tile-rows; extent `block_rows` = `br`) and **W** (tiles of the core's W-slice; extent `block_w_tiles ≡ core_w_tiles` = `cw`).
Bounded non-block parameter in sizes: `w_split` = `k` (cores per line).
Flags (0/1, fixed per program variant): `[xg]` = `select_blocking().xg_prescale` (Refinement 3: `[ws]` ∧ `[tile]` ∧ `[g]` ∧ ¬`[chunk]` ∧ the `cb_xg` term fits beside the chosen block without shrinking `br`), `[g]` γ present, `[g_rm]` γ ROW_MAJOR, `[ws]` regime W_SPLIT, `[tile]` / `[rm]` input layout, `[sh]` input L1-sharded (Refinement 2; `[sh·tile]` = resident TILE shard), `[mask]` = `[sh·tile]` ∧ W % 32 ≠ 0, `[chunk]` = `wc < cw` (γ streams; RM staging chunked), `[stage]` RM γ slice offset off the DRAM read grain (RM shards), `[dense]` = `select_blocking().dense` (Refinement 4: `[ws]` ∧ ¬`[mask]` ∧ `DENSE_RINV` ∧ the dense footprint fits a block of ≥ `DENSE_MIN_ROWS = 4` tile-rows; Σx² / rinv travel as `u = ⌈br/32⌉` dense tiles per block instead of `br`), `[rt]` = the ReduceTile collapse (`COLLAPSE_REDUCE_TILE` ∧ ¬`[mask]` ∧ ¬`[dense]`).
Page bytes: `T_in` = input-dtype tile (2048 bf16 / 4096 fp32 / 1088 bf8b), `T_out = T_in` (output dtype = input dtype), `T_int` = intermediate tile (4096 Float32 when fp32 DEST on / 2048 Float16_b when off — Refinement 1; one `t_int` in `create_program_descriptor`), `T_g` = γ-dtype tile, `es_γ` = γ element bytes.

Page-format rule applied (Refinement 1, ledger audit 3): every compute-produced **intermediate** page follows the DEST width — `Float32` when `fp32_dest_acc_en = True`, `Float16_b` when it is off (16-bit DEST already rounded the value; a Float32 page would store no extra bits and cost 2× packer/unpacker, NoC gather/mcast and L1). One `int_fmt` / `t_int` value feeds the CB descriptors, `_per_row_bytes()` (so `block_rows` grows when it halves) and the writer's `int_tile_bytes` CT arg. Input-data CBs (`cb_x`, `cb_x_rm`, `cb_gamma*`) carry the tensor's own dtype (no compute rounding happened yet); output CBs (`cb_y`, `cb_out_rm`, RM `cb_x` after normalize) carry the output dtype by contract (the op's final rounding).

| CB | Capacity (pages) | Live set | Axis accounting | Page format | Producer | Consumer | Lifetime | Shares with / why not |
|----|------------------|----------|-----------------|-------------|----------|----------|----------|-----------------------|
| `cb_x_rm` [rm] | `x_rm_depth · wc` = `2·wc` (`wc = cw` unless `[chunk]`) | `cw` per tile-row being tilized + `cw` being filled | `{R: streams → 1 tile-row window ×2, W: spans → cw}` | input dtype | reader | compute | per tile-row | capacity = 2× live: double-buffer so the reader fills tile-row i+1 during tilize of i. Not shared with `cb_out_rm`: both live concurrently (reader prefetch while writer drains) |
| `cb_x` | [tile] `x_block_depth · br · cw` = `2·br·cw`; [rm] `br·cw`; **[sh·tile] backed zero-copy on the input shard** (`cb_descriptor_from_sharded_tensor`; capacity = shard pages, **0 allocator bytes**) | one block `br·cw` (resident from first push through the last normalize pop) | `{R: spans → br, W: spans → cw}` | input dtype (= output dtype; RM holds y after the in-place normalize) | [tile] reader / [rm] compute | compute | per block | [tile] capacity = 2× live: depth 2 so the reader fetches block b+1 during normalize(b) and during the W_SPLIT combine stall (op_design stall-shadow table). [rm] **is** the reuse: normalize transforms it in place and untilize reads it, replacing an RM `cb_y` of `br·cw` pages (Rule 3.2, real credit). [tile] in place is impossible: the reader is its producer |
| `cb_gamma_rm` [g_rm] | 1 page of `wc·32·es_γ` bytes; [chunk] `γ_depth = 2` pages | 1 page | `{R: neither (constant), W: spans → cw (one row of the slice)}` | γ dtype | writer | compute | prologue only | Disjoint from every per-block CB, but not aliased: it is ≤ `cw·128` bytes (≤ 1/32 of one tile-row of x) and aliasing it into `cb_y`/`cb_out_rm` would put a writer-produced and a compute-produced stream on one allocation, which the alias lockstep discipline forbids for different producers |
| `cb_gamma` [g] | `cw`; [chunk] `γ_depth · wc` = `2·wc` (one W-chunk streaming per normalize pass) | `cw` (whole kernel) | `{R: neither (constant, reuse-shared), W: spans → cw}` | γ dtype | [g_rm] compute (tilize) / [g, TILE γ] writer | compute | whole kernel | Live for the whole kernel → overlaps every other CB; nothing to share. Kept resident precisely so γ crosses DRAM once per core. TILE γ reads only face0/face1 row 0 per tile and (Refinement 3) zeroes only the nominal tail tiles past the valid slice — rows 1–31 are never read by the Row-broadcast FPU op, so the former full-chunk zero fill (~0.7 µs of writer time per core) is gone; block-float (bf8b) TILE γ has no face-row addressing, so its valid tiles are read whole (`w_valid·T_g`, T_g = 1088) — Refinement 1 |
| `cb_sumsq_raw` | `parts · br` (`parts = 2` iff `[mask]`, else 1) | `br` (one raw Σx² tile per tile-row; packed per tile-row, collapse waits the block) | `{R: spans → br, W: streams (summed into DEST, not stored)}` | `int_fmt` (Float32 / Float16_b) | compute | compute | sumsq → collapse | Not aliased with `cb_rinv_col` / `cb_partial`: the collapse reads it and writes them in the same reduce call (`reduce` asserts in ≠ out, `reduce_helpers_compute.inl:333`). This buffer is what remains of the square pass: x² is never materialised (squares fold into DEST, Rule 3.4) |
| `cb_partial` [ws] | `br`; [dense] `u` | `br` (dense: `u`, the densify matmul's output) | `{R: spans → br, W: neither (collapsed)}` | `int_fmt` (Float32 / Float16_b) | compute | writer | collapse → gather send | Not aliased with `cb_rinv_col`: on the root the combine (writing `cb_rinv_col`) may start before the writer's pop of `cb_partial` retires; the ordering between those two threads is not CB-credit-guaranteed |
| `cb_gathered` [ws] | `k · br`; [dense] `k · u` (Refinement 4: one whole dense tile per sender per 32 tile-rows — 32× smaller, e.g. 512 KB → 16 KB at `k = 8`, `br = 32`) | `k · br` (dense `k · u`) on roots; 0 on non-roots (allocated for uniform addresses) | `{R: spans → br, W: spans → k (one partial per line core)}` | `int_fmt` (Float32 / Float16_b) | reader (root; data lands by remote writes) | compute (root) | gather → combine | Capacity = exactly one block: the landing address must be the constant CB base. Not shared: written remotely while local CBs are live. Non-root waste (`k·br·T_int`) is counted in the budget; removing it needs per-core CB layouts, which break the uniform-address rule. Refinement 3 (lamp L4, `GATHER_COL_FACES`): each sender transmits only faces 0 + 2 of its column-0-valid partial (half of `T_int`), so the **fan-in bytes** halve; the slots stay tile-pitched (faces 1/3 hold stale bytes the elementwise combine carries and nothing reads), so the **capacity** — and the over-capacity finding at large `k` (`k·br·T_int`, ~412 KB at `k = 103`, Float32) — is unchanged. Shrinking it needs compact slots that the combine reduce can still read as tiles (e.g. two partials per tile, left / right half, plus a half-fold) |
| `cb_rinv_col` | `br` | `br` (LOCAL: every core; W_SPLIT sparse: root only; [dense]: every core — the **expanded** rinv, one broadcast tile per tile-row, written by `expand_block`'s matmul and read by normalize) | `{R: spans → br, W: neither}` | `int_fmt` (Float32 / Float16_b) | compute | compute | collapse/combine → finalize | Not aliased with `cb_rinv` / `cb_rinv_bcast`: the broadcast chain reads it and packs them in the same call. Refinement 3: it is also the **in-place** target of finalize step 1 (`rsqrt(Σ·(1/W)+ε)` on the column-0 vectors, streaming: each tile is popped before its slot is re-reserved), so the column-scoped SFPU pass needs no extra CB |
| `cb_rinv_bcast` [ws] | `br`; [dense] `u` | `br` (root only; dense: `u` dense rinv tiles = the multicast payload) | `{R: spans → br, W: neither}` | `int_fmt` (Float32 / Float16_b) | compute | writer | finalize → mcast send | Not shared with `cb_rinv`: the root's own `cb_rinv` receives the multicast while this buffer is still the send source; also two consumers (writer, compute) would violate single ownership |
| `cb_rinv` | `br`; [dense] `u` | `br` (dense: `u` dense rinv tiles, popped by the expand matmul) | `{R: spans → br, W: neither (broadcast across W)}` | `int_fmt` (Float32 / Float16_b) | LOCAL: compute / W_SPLIT: reader | compute | finalize/receive → normalize | Capacity = exactly one block (constant mcast landing address). Live across normalize, which overlaps every per-block CB |
| `cb_xg` [xg] | `br · cw` | one block `br·cw` (prescale → normalize) | `{R: spans → br, W: spans → cw}` | `int_fmt` (Float32 / Float16_b: the same precision x·γ had in DEST before `DEST_TO_SRCA`, so y is bit-identical to the fused chain) | compute (prescale_block) | compute (normalize_block) | per block, prescale → normalize | Refinement 3 (W_SPLIT stall shadow): x·γ is computed while the core waits for the gather → combine → multicast round trip, leaving one FPU mul (`xg·rinv`) after it. Not in place on `cb_x`: TILE `cb_x` is reader-produced (or the caller's input shard). Not packed into `cb_y` and transformed in place: `cb_y`'s consumer is the writer (two consumers). Not aliased with `cb_sumsq_raw` / `cb_partial`: those are live (on the root, `cb_gathered` → combine) while `cb_xg` fills. Gated off where it would shrink `br` (`_xg_prescale`), so it never adds a rendezvous |
| `cb_reduce_scaler` [rt] | 1 (Float16_b) | 1 (whole kernel) | `{R: neither, W: neither}` | Float16_b (1.0: exact) | reader (`calculate_and_prepare_reduce_scaler<SUM, REDUCE_ROW>`, built after block 0's x is published) | compute (collapse reduce, ReduceTile datapath) | whole kernel | Refinement 4: the ReduceTile collapse (FPU reduce against ones) replaces AccumulateViaAdd's SFPU lane reduce (~305 → ~53 cycles per tile). Live for the whole kernel; not shared. Absent under `[mask]` (that collapse keeps the 0/1 lane mask in `cb_wmask`) and `[dense]` (no collapse) |
| `cb_dense_sum` [dense] | `u` | `u` (root only; allocated everywhere for uniform addresses) | `{R: spans → br (32 tile-rows per tile), W: neither}` | `int_fmt` | compute (combine) | compute (dense finalize) | combine → finalize | Refinement 4: the root's combined dense Σx². Not in place on `cb_rinv_bcast`: that CB's consumer is the writer, which would see the combine's push before the finalize ran. Not aliased with `cb_rinv_col`: the expand later writes `br` tiles there and a ring wrap past `u` would straddle |
| `cb_jsel` [dense] | `n_sel = min(br, 32)` | `n_sel` (whole kernel) | `{R: spans → min(br, 32) (one selector per tile-row of a dense group), W: neither}` | `int_fmt` (0 / 1.0: exact) | reader (`gen_jsel`: NoC zero fill, 16 one-hot faces by 256 stores, the rest by local 512 B NoC copies; after block 0's x) | compute (densify `D = Σ S_r @ J_r`, expand `R @ J_rᵀ` — matmul in1, `NoWaitNoPop`) | whole kernel | Refinement 4: the one-hot column selectors J_r. One set serves both matmuls (the expand transposes in1). Live for the whole kernel; not shared |
| `cb_y` [tile] | `y_depth · y_chunk_tiles` = `2·8`; **[sh·tile] backed zero-copy on the output shard** (0 allocator bytes; compute packs y at its final position) | `y_chunk_tiles` draining + up to `y_chunk_tiles` being packed | `{R: streams, W: streams → window of 8}` | output dtype | compute | writer | per chunk | capacity = 2× live: double buffer (compute packs chunk c+1 while writer drains c). Does not span W, so it adds no `cw` term. Not in place on `cb_x`: the reader produces `cb_x` |
| `cb_out_rm` [rm] | `out_rm_depth · wc` = `2·wc` | `cw` being written + `cw` being untilized | `{R: streams → 1 tile-row window ×2, W: spans → cw}` | output dtype | compute | writer | per tile-row | capacity = 2× live: double buffer. Must span W: `untilize` emits a whole tile-row of sticks (`untilize_helpers.hpp:145-154`, block width is the CT `cw`). Not shared with `cb_x_rm` (concurrent) |
| `cb_wmask` [mask] | 1 | 1 (whole kernel; pushed only on the core owning the partial-W tile) | `{R: neither, W: neither (one lane mask)}` | `int_fmt` (must equal `cb_sumsq_raw`'s: the masked fold unpacks both through one SrcA/SrcB format) | reader (`prepare_reduce_mask<REDUCE_ROW>(w_valid_last)`) | compute (collapse reduce scaler slot, `only_partial()`) | whole kernel | Exists because a resident shard's pad lanes belong to the caller's tensor (poison may sit there) and must not be zeroed in place: the partial-W tile's squares go to the 2nd per-row `cb_sumsq_raw` slot and the collapse folds them through this 0/1 mask. Not shared: live for the whole kernel |
| `cb_gamma_stage` [stage] | 1 page of `wc·32·es_γ + 2·dram_align` bytes | the aligned DRAM window of one γ chunk | `{R: neither, W: spans → wc}` | γ dtype | writer | writer | per γ chunk load | RM shards put the γ slice at a 16 B-grain byte offset; a DRAM read needs `src ≡ dst (mod 64)`, so the aligned window lands here and the slice is copied core-locally into `cb_gamma_rm`. Not merged into `cb_gamma_rm` (an overlapping self-copy is unsafe) |

## Symbol table

| Symbol | Meaning | Bound | Predicate establishing it |
|--------|---------|-------|---------------------------|
| `br` (`block_rows`) | block extent along R | `1 ≤ br ≤ ⌈tensor_rows_tiles / num_lines⌉` | `select_blocking()`: `br = min(line_rows_tiles_max, ⌊(B − fixed_bytes)/per_row_bytes⌋)`, ≥ 1 because `k ≥ k_fit` |
| `cw` (`core_w_tiles`) | block extent along W = whole per-core slice | `1 ≤ cw ≤ tensor_w_tiles` (+ `wc − 1` RM chunk padding) | interleaved: `cw = ⌈tensor_w_tiles / k⌉` with `k ≥ k_fit` ⇒ `footprint(1, cw, k) ≤ B`. Sharded: `⌈shard_w / 32⌉` (TILE: the shard's tile pitch, fixed); RM `[chunk]`: rounded up to a multiple of `wc` |
| `wc` (`w_chunk_tiles`) | W-chunk of the γ and RM staging buffers | `1 ≤ wc ≤ cw` | `select_blocking()` sharded branch: the largest `⌈cw/m⌉` (m = 1, 2, …) with `footprint(1, cw, k, wc) ≤ B_sh`; interleaved: `wc = cw` (byte-identical) |
| `B_sh` | sharded CB budget | `min(L1 addr of the sharded input / output) − L1 allocator base − 16 KiB` | host: the CB region grows up from the allocator base and must end below the resident shard buffers, which the allocator places at the L1 top |
| `k` (`w_split`) | cores per line | `1 ≤ k ≤ grid_cores` (≤ 140 on any current part) | interleaved: `k_occ = max(k_fit, …)`, then (Refinement 5 `BALANCED_W_SPLIT`) widened within `[k_occ (R<C) / k_fit (R≥C), k_hi]` by the busiest-core cost model; `k ≤ C` by construction; host refuses if `k_fit` does not exist. Every candidate is footprint-checked through `k_fit` (a wider line only shrinks per-core CBs) |
| `B` (`L1_CB_BUDGET_BYTES`) | per-core CB budget | `1 MiB` (constant) | host constant (perf lamp L8) |
| `x_block_depth`, `x_rm_depth`, `y_depth`, `out_rm_depth` | buffer depths | 2 each | host constants |
| `y_chunk_tiles` | writer chunk | 8 | host constant |
| `T_in`, `T_out`, `T_g` | tile bytes | ≤ 4096 | dtype ∈ {bf16, fp32} (SUPPORTED) |
| `es_γ` | γ element bytes | ≤ 4 | γ dtype ∈ {bf16, fp32} |

## Total per-core footprint

(Refinement 2) Sharded placements change the terms as follows: `[sh·tile]` drops the `cb_x` term from `per_row_bytes` and the `cb_y` term from `fixed_bytes` (both are backed on the shards, outside the CB region); `cb_sumsq_raw` contributes `parts·T_int` per row; every `cw` in the γ and RM staging terms becomes `γ_depth·wc` / `depth·wc`; `[mask]` adds `T_int`; `[stage]` adds `wc·32·es_γ + 2·dram_align`. The budget is `B_sh` instead of `B`.

```
footprint(br, cw, k) = br · per_row_bytes(cw, k) + fixed_bytes(cw)

per_row_bytes(cw, k) =  [tile]· x_block_depth·cw·T_in            # cb_x            (scales with br·cw)
                      + [rm]  · cw·T_in                          # cb_x (one block, in place)
                      + 3·T_int                                    # cb_sumsq_raw, cb_rinv_col, cb_rinv   (scale with br)
                      + [ws]· (2 + k)·T_int                        # cb_partial, cb_rinv_bcast, cb_gathered (scale with br; gathered with br·k)
                      + [xg]· cw·T_int                             # cb_xg (Refinement 3; added only after br is chosen, never shrinks it)

fixed_bytes(cw)      =  [g]   · cw·T_g                           # cb_gamma        (scales with cw)
                      + [g_rm]· cw·32·es_γ                       # cb_gamma_rm     (scales with cw)
                      + [tile]· y_depth·y_chunk_tiles·T_out      # cb_y            (constant)
                      + [rm]  · (x_rm_depth·cw·T_in + out_rm_depth·cw·T_out)   # cb_x_rm, cb_out_rm (scale with cw)
```

(Refinement 4) `[dense]` footprint (`_footprint_dense`, `u = ⌈br/32⌉`): `br·([¬sh·tile] x_block_depth·cw·T_in + 2·T_int) + u·(4 + k)·T_int + min(br, 32)·T_int + fixed_bytes` (cb_sumsq_raw + expanded cb_rinv_col per row; cb_partial / cb_gathered / cb_rinv_bcast / cb_rinv / cb_dense_sum per dense group; cb_jsel). `block_rows` is the largest `br ≤ line_rows` it fits (binary search), so a tall BLOCK shard (e.g. 8192×1024 on 8×8: 32 tile-rows) becomes ONE block — the sparse schedule needed 2 blocks there (`cb_gathered` = `k·br` tiles) — and the `[xg]` term now fits beside it. `[rt]` adds one Float16_b tile to `fixed_bytes`.

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

**Sharded placements (Refinement 2).**

| Tensor | DRAM crossings | Core-local | Cross-core |
|--------|----------------|------------|------------|
| x [sh·tile] | **0** (resident shard is `cb_x`; never re-read) | 0 copies | 0 |
| x [sh·rm] | **0** | 1 L1→L1 re-stride (shard stick pitch → `wc·32·es`), its own shard only, no accessor | 0 |
| y [sh·tile] | **0** (packed into the output shard) | 0 | 0 |
| y [sh·rm] | **0** | 1 L1→L1 stick write into its own output shard | 0 |
| γ | unchunked: as interleaved (once per core). `[chunk]`: `num_blocks·passes` reads of the slice (`passes` = tile-rows per block for RM, 1 for TILE). Only when a HEIGHT-sharded row is too wide to keep γ resident next to its two shards; there the shards hold 1–2 tile-rows, so it is ≈ one read | — | — |
| partials / rinv [ws] | 0 | — | as W_SPLIT, over the shard-grid line (WIDTH: whole grid; BLOCK: one shard-grid row, column for COL_MAJOR) |

(Refinement 5) No DRAM crossing changed and no CB was added: x and y still cross DRAM once. The interleaved split now picks `w_split` by a busiest-core cost model, so some prefill shapes moved LOCAL → W_SPLIT (e.g. 8192×5120: k 1 → 2) and the 640-row shapes moved k 5 → 11; their footprints are the existing W_SPLIT rows of this ledger at the chosen `k`. The interleaved `cb_y` DEST window is now the largest divisor of both `core_w_tiles` and the `Y_DEPTH·Y_CHUNK_TILES` ring (≤ `NORM_DEST_BLOCK`); it was `NORM_DEST_BLOCK` whenever `block_rows·core_w_tiles` was a multiple, which straddled the ring for `core_w_tiles % 8 ≠ 0` (latent bug, never reached by a golden shape). The reader / writer issue order within a chunk is rotated per core (bank stagger) — same pages, same CB slots.

(Refinement 4) No DRAM crossing changed. `[dense]` cross-core: the partial gather carries `⌈R/32⌉·(k−1)·T_int` (whole dense tiles) instead of `R·(k−1)·T_int/2`, and the rinv multicast `⌈R/32⌉·T_int` instead of `R·T_int` per line; each core re-expands rinv locally (compute only).

(Refinement 3) No DRAM crossing changed. Cross-core: the partial gather now carries `R·(k−1)·T_int/2` (faces 0 + 2 only, lamp L4); the rinv multicast drops its receiver-ready round trip (`handshake=False`, lamp L5: `k−1` fewer semaphore incs per block). γ's DRAM reads are issued at kernel start and committed after block 0's partial send (W_SPLIT), so they overlap the first sumsq instead of delaying the gather.


Chosen split = LOCAL or W_SPLIT as selected (both are built). `S_x` = bytes of x (= bytes of y), `R` = `tensor_rows_tiles`, `G` = `num_lines`.

| Tensor | DRAM crossings | Why that many | Cross-core traffic added |
|--------|----------------|---------------|--------------------------|
| x | 1 | the block (`br × cw`) stays resident in `cb_x` from the sumsq pass through the normalize pass; a row (LOCAL) or W-slice (W_SPLIT) never leaves L1 between the two passes | 0 |
| y | 1 | written once by the writer | 0 |
| γ | `G` reads of its W elements in total (each core reads only its slice's row-0 lanes, once per kernel) | γ is reuse-shared along R: held resident in `cb_gamma` across all of a core's blocks, so per-block re-reads are 0; the remaining `G−1` extra copies are the GAMMA_MCAST deferral | 0 |
| Σx² partials (W_SPLIT) | 0 | — | `R·(k−1)·T_int/2` unicast core → root (faces 0 + 2, Refinement 3; was `R·(k−1)·T_int`) |
| rinv (W_SPLIT) | 0 | — | `R·T_int` multicast root → `k−1` receivers (one transfer per block per line) |

Totals per tier:

| Tier | LOCAL | W_SPLIT |
|------|-------|---------|
| DRAM | `2·S_x + G·W·es_γ` (`G = min(R, C)`) | `2·S_x + G·W·es_γ` (`G = min(R, ⌊C/k⌋)`, fewer γ copies than LOCAL) |
| cross-core NoC | 0 | `R·(k−1)·T_int/2 + R·T_int·(k−1 receivers)` + `k` partial-arrival incs + 1 data-ready signal per block per line (no ready handshake) |
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
