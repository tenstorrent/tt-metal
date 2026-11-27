# Operation Design: g6_reshape

## Overview

| Field | Value |
|-------|-------|
| Classification | data_movement (no arithmetic, no compute kernel) |
| Goal | `torch.reshape` on the logical values, laid out again in the input's own layout under the new shape, bit-exact, as ONE generic_op dispatch that reads each logical input value from the input tensor and writes it to the output tensor (no DRAM or host staging). |
| Math | `out[j] = x[i]` whenever `lin(j; e) = lin(i; d)`, `lin(i; d) = Σ_k i_k · Π_{l>k} d_l`; one `-1` in `shape` resolved as `N / Π(other entries)`. |
| Phase 0 scope | `dtype = bfloat16`, `layout = TILE_LAYOUT`, DRAM interleaved, every rank 1..4 on both sides, any last two dims |
| References | Spec (task prompt); `eval/golden_tests/g6_reshape/feature_spec.py` (TARGET, INPUTS, LOOSE_CASES); `eval/golden_tests/g6_reshape/helpers.py` (oracle, exact gate, `TOLERANCES = ("exact", None)`); `.claude/references/blocking-model.md`; `.claude/references/l1-footprint-discipline.md`; `.claude/references/ttnn-cb-memory-fundamentals.md`; catalog `ttnn/ttnn/operations/examples/master.md` (entries cited in Dataflow Strategy); `tt_metal/hw/inc/internal/tt-1xx/blackhole/noc/noc_parameters.h:375-380` (NoC alignment) |

### Notation (used everywhere below)

| Symbol | Meaning | Source |
|--------|---------|--------|
| `e` | element bytes (2 bf16; 4 fp32/int32/uint32) | `tensor.element_size()` |
| `F` | face-row bytes = `16·e` (32 bf16, 64 fp32). The face-row (16 elements of one tile row inside one 16×16 face) is the largest unit contiguous in BOTH tile layouts | derived |
| `A` | DRAM read alignment (64 on Blackhole, `noc_parameters.h:379`) | `ttnn.get_dram_alignment()` |
| `(B_x, H_x, W_x)` | per tensor x ∈ {in, out}: `W = s[-1]`, `H = s[-2]` (1 for rank 1), `B = Π s[:-2]` (1 for rank ≤ 2) | logical shape |
| `Ht_x, Wt_x` | tile rows per image / tile columns: `padded_shape[-2]/32`, `padded_shape[-1]/32` (= `ceil(H/32)`, `ceil(W/32)`, per image) | padded shape; host asserts `buffer_num_pages() == B·Ht·Wt` |
| `R_x` | logical rows `B_x·H_x`; row `r = b·H + h` | derived |
| tile position | row `r=(b,h)`, column `c` → page `(b·Ht + h/32)·Wt + c/32`, byte offset `16F·face + (h%16)·F + (c%16)·e`, `face = 2·((h%32)/16) + (c%32)/16` | ttnn tile layout (4 faces of 16×16) |
| TRS | "tile-row-slice": one output logical row `r'` × one output tile column `tc'` = 2 output face-rows (`half` 0/1); half 1 exists iff `W_out − 32·tc' > 16` | derived |
| `atom_rows` | `max(1, A / F)` — 2 for bf16, 1 for fp32 on Blackhole | derived |
| atom | `atom_rows` consecutive rows `h ≡ 0 (mod atom_rows)` of ONE output image in ONE output tile column (an image's last atom is shorter when `H_out % atom_rows ≠ 0`). The indivisible ownership unit | derived |
| `G` | total atoms `= Wt_out · B_out · ceil(H_out / atom_rows)` | derived |
| `block_atoms`, `T` | atoms per block; `T = block_atoms · atom_rows` = TRS per block (upper bound) | host knob |
| `S` | bound on input face-rows one output face-row touches: 1 (face-row aligned), 3 (`W_in ≥ 16`), `ceil(15 / W_in) + 1` (`W_in < 16`) | derived |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes, positional | rank 1..4; Phase 0: bfloat16, TILE_LAYOUT (32×32 tiles), DRAM interleaved | — | geometry → CT; buffer address → common RT |
| `shape` | `list[int]` or `tuple[int]` | yes, positional | length 1..4; entries `> 0` except at most one `-1`; product (after resolution) `= N` | — | resolved `e` → CT geometry (host only; never a tensor) |

Host validation (raises `ValueError`) — after the support check:

| Condition | Error |
|-----------|-------|
| `len(shape) ∉ [1,4]` or input rank `∉ [1,4]` | `ValueError` |
| any entry `== 0` or `< -1` | `ValueError` |
| more than one `-1` | `ValueError` |
| `-1` present and `N % Π(other entries) ≠ 0`, or resolved product `≠ N` | `ValueError` |

Order in `validate(input_tensor, shape)`: (1) build `axes = {"dtype": input_tensor.dtype, "layout": input_tensor.layout}`; (2) `SUPPORTED` per-axis → `UnsupportedAxisValue`; (3) `EXCLUSIONS` (empty in Phase 0) → `ExcludedCell`; (4) placement guard: non-DRAM-interleaved input → `UnsupportedAxisValue` (placement is not a TARGET axis; no golden cell reaches it); (5) the shape checks above → `ValueError`. The public entry point calls `validate()` as its first line. `INPUT_TAGGERS = {"dtype": tag_dtype, "layout": tag_layout}` reading `inputs[0]["dtype"]` / `inputs[0]["layout"]` of the scenario dict; axis names and values exactly those of `feature_spec.TARGET`.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(d_0..d_{n-1})`, `n ∈ [1,4]`, any extents (no multiple-of-32 requirement on any dim) |
| Dtype | bfloat16 (Phase 0); fp32 (TILE), int32/uint32 (ROW_MAJOR) by refinement |
| Layout | TILE_LAYOUT (Phase 0); ROW_MAJOR by refinement |
| Memory | DRAM interleaved |
| Padding content | arbitrary (NaN included) — never read into a logical output position |

### Output

| Property | Value |
|----------|-------|
| Shape | `e` (resolved `shape`) |
| Dtype | `input_tensor.dtype` |
| Layout | `input_tensor.layout` |
| Memory | `input_tensor.memory_config()` (DRAM interleaved); allocated with `ttnn.allocate_tensor_on_device(ttnn.Shape(e), dtype, layout, device, memory_config)` — the ONLY device allocation |
| Padding content | unspecified (padding rows never written; padding columns of a partially valid face-row carry whatever the image slot held) |

Structural impossibilities: none to add (`INVALID = []` is right: TARGET holds no block format; every (dtype, layout) pair is a legal tensor).

## Blocking Model

The op is a pure permutation of the logical stream. Input and output differ only in where the stream is cut into rows (`W_in` vs `W_out`) and how rows are laid into tiles. The face-row is contiguous in both layouts; everything coarser is contiguous in at most one of them. The design is **output-stationary**: each mover owns whole output atoms, gathers their preimage with coalesced reads into a staging buffer in source byte order, permutes face-rows into an output image in output byte order with RISC-V copies, and writes contiguous output runs. No dependent axis, no combine, no reuse-shared operand, no cross-core traffic.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `out_tile_col` (`Wt_out`) | independent — each output tile column is filled from its own stream ranges; no cross-column result | outer component of the folded atom axis → `block_atoms` | the whole per-mover share | host `block_atoms` (CT) | **spread**: outer component of the column-major atom fold; contiguous atom ranges per core | knob-turn |
| `out_row` (`R_out = B_out·H_out`, leading dims folded with H — the op treats them identically and they are contiguous in the stream) | independent — each output row is a copy of its own stream range | inner component of the folded atom axis → `block_atoms` (unit = atom of `atom_rows` rows) | the whole per-mover share | host `block_atoms` (CT); `atom_rows` derived from `A/F` | **spread**: inner component; range boundaries snap to atoms | knob-turn |
| `half` (2 face-rows of a TRS) | independent; kept whole — both halves share the row's index state and the output tile; splitting would duplicate the index walk for no traffic gain | `trs_halves` | 2, or 1 where half 1 is pure output padding (`W_out − 32·tc' ≤ 16`) | derived per tile column | not split (decision: whole) | — |
| `face_row_elem` (16 elements) | independent; the transfer granule — the largest unit contiguous in both tile layouts | fixed 16 | 16 | tile layout constant | not split (decision: whole) | — |
| `movers` (2 data-movement RISC-Vs per core) | — (execution resource, not a data axis) | `movers_per_core` | 2: mover 0 = RISCV_1/NoC0, mover 1 = RISCV_0/NoC1 | host constant | each core's atom range split into two contiguous halves | knob-turn (1 ↔ 2) |
| input preimage (`R_in` rows × `Wt_in` cols) | not a work axis — the read set is the preimage of the block; each logical input face-row is needed by exactly one output face-row (aligned) or two (element-run straddle) | none (derived); `S` bounds face-rows per output face-row | — | host `S` | follows the output assignment | — |
| dependent axes | **none** — output is a permutation of input; nothing spans an axis | — | — | — | — | — |
| reuse-shared operands | **none** — operand-reuse check: the single input varies along the split (disjoint atoms need disjoint input elements) | — | — | — | — | — |

Phase 0 is multi-core from day 1: `P = min(max_cores, G)` cores with `max_cores = grid_x·grid_y`, so every production shape spreads over the grid (e.g. `(16,1,64)→(1,1,16,64)` has 2 output tiles but `G = 16` atoms → 16 cores). Every knob (`max_cores`, `movers_per_core`, `block_atoms`, `stage_depth`, `image_depth`) is a parameter with one source; all CB sizes, trip counts and RT ranges are derived from them.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_src_stage_m{0,1}` | `stage_depth` | 1 | depth 2 + trid-scoped barriers lets block k+1's reads fly while block k is permuted (only when a mover owns > 1 block; every production shape is one block per mover) |
| `cb_dst_image_m{0,1}` | `image_depth` | 1 | depth 2 lets block k's writes drain while block k+1 is permuted; with depth 1 the `noc_async_writes_flushed()` before the next permute is already covered by the next block's read round trip |
| `cb_move_plan_m{0,1}` | none (rebuilt per block) | 1 | — |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DRAM read congruence: src NoC addr ≡ dst L1 addr (mod `A`), `A = 64` on BH (`noc_parameters.h:379`, checked `sanitize.h:746`) | placement of every read run in `cb_src_stage` | each run's staging address = next address ≥ cursor with `addr ≡ src_noc_addr (mod A)`; phase taken from the actual source NoC address | `--dev`: `DebugSanitizeNocAlignment` hang; release: shifted bytes |
| DRAM write congruence (mod 16, `noc_parameters.h:380`) | write-run sources in `cb_dst_image` | image holds whole face-rows (`F` = 32/64 B) at offsets multiple of `F`; a partial face-row is written whole, never as a sub-face-row piece | sanitizer hang / shifted bytes |
| Output ownership atom (two face-rows of rows `2k, 2k+1` share one 64-B DRAM atom for bf16) | core and mover range boundaries, block boundaries | every boundary falls on an atom boundary (`atom_rows = max(1, A/F)` rows, aligned to `h ≡ 0 mod atom_rows` within an image) | two movers issue the two 32-B halves of one 64-B DRAM word concurrently — an unverified read-modify-write hazard; designed out at zero cost |
| Runs are page-local | read-run and write-run length | a run is extended only by the next byte-contiguous face-row of the SAME page | the transfer runs into the next bank's address space |
| L1 budget, BB-05: 1347776 B per core | `block_atoms` | `block_atoms = min(max_atoms_per_mover, floor(1347776 / (movers_per_core · atom_rows · bytes_per_trs)))`, `bytes_per_trs` from the ledger | launch refused / OOM on large `N` |
| Staging / plan capacity | entries per block | capacities derived from `S` (bound above); `--dev` `ASSERT` cursor ≤ capacity | silent overrun into the neighbouring CB |
| RV32 natural alignment | copy width | 32-bit copies only when `src`, `dst`, `len` ≡ 0 (mod 4); 16-bit copies otherwise (bf16 pieces are 2-B aligned) | misaligned load/store fault |
| 32-bit counters | linear index | stream index `L` computed in 64 bit once per (column segment, half); all per-row / per-piece steps are 32-bit incremental counters | wraparound for `N ≥ 2^32` |
| Inline DRAM writes unsupported (`dataflow_api.h:2292-2295`) | — | every DRAM write sources an L1 image slot | — |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `tile_face_row` | **built** (Phase 0; every bf16 production case) | `layout == TILE and W_in % 16 == 0 and W_out % 16 == 0` | `block_atoms` atoms (≤ `T` TRS) × `trs_halves` × 16 elements; every output face-row is exactly one whole input face-row | **minimum** against DRAM: each logical input face-row crosses once, each logical output face-row once; no padding row or padding column of either tensor crosses (face-rows are fully logical when both W are multiples of 16) | per block: one read barrier (one exposed DRAM round trip), one `invalidate_l1_cache` + one store drain, one `writes_flushed`, one 64-bit div per (segment, half); longer face-column runs (fewer read and write transactions); fewer output tiles split across two blocks (a split tile costs an extra write run) |
| `tile_element_run` | **built** (Phase 0 claims every bf16 TILE shape; also the fp32 production geometry `[1,W]→[1,1,W,1]`) | `layout == TILE and not (W_in % 16 == 0 and W_out % 16 == 0)` | same block; each output face-row assembled from ≤ `S` pieces of input face-rows | above minimum, bounded: a partially logical face-row carries its padding columns (≤ 15 elements per row end, both sides); an input face-row straddling two output face-rows that fall in different half-passes or blocks is read twice (≤ 2× on those face-rows); padding values never reach a logical output | same as above |
| `rm_row_gather` | **deferred** — SUPPORTED excludes ROW_MAJOR in Phase 0 (spec: a refinement adds int32/uint32 with ROW_MAJOR in one change). Reachable: same mover skeleton (`gather → await → permute → store`); swap the geometry (page = logical row, offset = `c·e`, atom = one output page slot of `align(W_out·e, A)` bytes, or one `A`-byte chunk of a wide row) and fold atoms **bank-affine** (page `p` ordered by `(p mod NB, p / NB)`), so a mover's block is consecutive slots of one DRAM bank and stores as ONE write per (block, bank) | `layout == ROW_MAJOR` | `block_atoms` page-slot atoms | reads: each logical input byte once (+ ≤ `A` per run); writes: the full `A`-byte slot of each page (page + its alignment padding) — for `[1,4096]→[4096,1]` int32 that is 16× the valid bytes in exchange for ~`NB` writes per mover instead of one 4-B write per page (the flagged PERF case) | fewer per-bank write runs; one input covering read per block |
| `page_copy` | **deferred** — no production shape keeps both last dims (`H_in == H_out and W_in == W_out`); the built tile regimes are correct there (runs coalesce to whole valid spans) | `H_in == H_out and W_in == W_out` | whole pages | minimum, and no permute | no shuffle at all |
| `input_stationary_scatter` | **rejected** — superseded by the two built regimes: same bytes, but an output tile fed by several movers fragments writes to per-row runs and puts two movers on one 64-B DRAM atom | — | input atoms | same bytes, more write transactions | — |
| `compute_tilize_untilize` (untilize → row shuffle → tilize in one program) | **rejected** — values pass unpacker/FPU/packer formats (denormal flush, NaN canonicalization, int32 through a float path) — breaks bit-exactness; two extra L1 passes; superseded by the dataflow permute | — | — | +2 L1 passes per value | — |
| `dram_staged_multi_dispatch` | **rejected** — forbidden (BB-01; "no further DRAM tensor") | — | — | 3× DRAM traffic | — |

Regime selection (host, exact): `regime = "tile_face_row" if (W_in % 16 == 0 and W_out % 16 == 0) else "tile_element_run"` for TILE. The two built regimes share one kernel source; the regime is a CT flag (`kFaceRowAligned`) that collapses the piece loop to one whole-face-row piece and selects 32-bit copies. Regime-pinned tests: the acceptance test asserts `describe_plan(...)["regime"]` on aligned production shapes and on two element-run shapes.

### Traffic ranking

Bytes per tier: DRAM is crossed once per logical face-row in every non-rejected split; cross-core traffic is zero in every split (no split needs a combine). The splits therefore differ in **transaction shape and occupancy**, accounted separately below (blocking-model §4).

| Candidate split | DRAM crossings (in / out) | Cross-core | Transaction shape | Verdict |
|-----------------|---------------------------|------------|-------------------|---------|
| A. output atoms, **column-major fold** (tile column outer, logical row inner) | 1 / 1 | 0 | consecutive units are consecutive rows of one face column on BOTH sides → read runs coalesce down input faces (e.g. `[1,1,16,4608]→[16,1,4608]`: a mover's rows of one input face column = one read per half), write runs coalesce down output faces (e.g. `[16,1,5120]→[1,1,16,5120]`: a mover's rows of one output tile = one write per half, one 1-KB write when it owns all 16 rows) | **chosen** |
| B. output atoms, row-major fold | 1 / 1 | 0 | consecutive units are different tiles on both sides → no coalescing | rejected: same bytes, more transactions |
| C. input atoms (input-stationary) | 1 / 1 | 0 | writes fragment per row; shared 64-B output atoms | rejected (`input_stationary_scatter`) |
| D. whole output pages per core | 1 / 1 | 0 | core count capped by output tiles (2 for `(16,1,64)→(1,1,16,64)`, 2 for `(8,1,64)→(1,1,8,64)`) | rejected: occupancy imposed by shape (Work geometry rule) |
| E. whole input pages per core | 1 / 1 | 0 | core count capped by input pages (1 for RM `[1,4096]`) | rejected (same rule) |
| F. split a dependent axis + combine | n/a | n/a | the op has no dependent axis | not applicable |
| G. multi-dispatch / DRAM staging | 3 / 3 | — | — | rejected (contract) |

Cheapest-traffic split = A = implemented.

### Block schedule

Each mover (both run the same kernel source on disjoint atom ranges of the same core) executes:

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_mover; ++block_idx) {
    gather_block(block_idx);    // walk the block's output face-rows; issue coalesced input reads into cb_src_stage; emit move plan + output tile table into cb_move_plan
    await_block(block_idx);     // one read barrier
    permute_block(block_idx);   // execute the move plan: cb_src_stage -> cb_dst_image (RISC-V copies)
    store_block(block_idx);     // one async write per contiguous output run of cb_dst_image; no wait
}
drain();                        // noc_async_write_barrier (BB-03)
```

| Block operation | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|-----------------|------------------------|--------------------|-----------------------------------|
| `gather_block` | the block's atoms, decomposed into **column segments** (maximal runs of atoms in one `tc'`: rows `[r'_a, r'_b)`); per segment: build its output tile entries (page id, in-tile rows `[lo,hi)`, `trs_halves`, image base) first, then walk **face-column order**: for `half` in `[0, trs_halves)`, for `r'` in `[r'_a, r'_b)` → one output face-row → ≤ `S` pieces. Per piece: locate the source face-row; reuse it if it is the last fetched face-row, extend the open read run if it is the next byte-contiguous face-row of the same page, else issue the open run and open a new one at the next phase-aligned staging address; append a plan entry `(stage addr + (c%16)·e, image addr + k·e, len·e)` | nothing from earlier blocks | one 64-bit `divmod(L, W_in)` (+ `divmod(r, H_in)`) per (segment, half); per row and per piece only adds/compares with host-precomputed steps `divmod(W_out, W_in)`, `divmod(·, H_in)`; one `noc_async_read` per read run (issued as the run closes) |
| `await_block` | the block | staging (landing), plan | exactly one `noc_async_read_barrier()` per block |
| `permute_block` | every plan entry of the block | staging, plan, tile table | one `noc_async_writes_flushed()` (image reuse), one `invalidate_l1_cache()` (Blackhole RISC L1 cache, `cache.h:12`) before the first staging load, one copy per plan entry (exactly one `F`-byte copy per output face-row in `tile_face_row`), one store drain (`ckernel::load_blocking` on the last written word) before `store_block` |
| `store_block` | every output tile entry of the block | image | one `noc_async_write` per contiguous run: per tile, regions `f0, f1, f2, f3` (each = the block's rows of that face) in DRAM byte order, merged when byte-contiguous (`lo == 0, hi ≥ 16` merges f0+f1; full 32 rows = one 2-KB write) |
| `drain` | — | — | one `noc_async_write_barrier()` per kernel (reads are already barriered) |

Image layout per tile entry (compacted, DRAM byte order): with `n0 = |[lo, min(hi,16))|`, `n1 = |[max(lo,16), hi)|`, `h2 = trs_halves`: regions `f0[n0] f1[n0 if h2==2] f2[n1] f3[n1 if h2==2]`, each `F` bytes per row. Destination of output face-row (in-tile row `ρ`, `half`): `ρ < 16` → `base + (half·n0 + ρ − lo)·F`; `ρ ≥ 16` → `base + (h2·n0 + half·n1 + ρ − max(lo,16))·F`. DRAM offset of region `fX` = `X·16F + (first row % 16)·F`.

Piece walk for output face-row (`r'`, `tc'`, `half`): `L0 = r'·W_out + 32·tc' + 16·half`, `n = min(16, W_out − 32·tc' − 16·half)`; input `(r, c) = divmod(L0, W_in)` (incremental); loop `k = 0..n`: `len = min(n − k, 16 − c%16, W_in − c)`; source face-row = (page of `(r, c)`, offset of `(h_in%32, c − c%16)`); `k += len; c += len; if c == W_in: c = 0, r += 1`. In `tile_face_row` this loop is exactly one piece of 16.

Stall shadow: the only waits are `await_block` and `drain`. Filling them: (1) the sibling mover runs its own gather/permute/store in that window (built); (2) the tile-table build and the walk itself overlap the reads already issued (runs are issued as they close, the barrier only covers the residual latency) (built); (3) staging depth 2 with trid-scoped barriers, or trid sub-batches inside one block so permute starts on the first batch (perf lamp). Permission comes from the op's algebra: blocks and pieces are independent copies, so any order is legal and no floating-point order exists to gate on.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| grid synchronization | full-grid spread leaves production movers 1–10 TRS (2–20 face-rows); kernel launch, RT-arg delivery and one DRAM round trip per mover may dominate | cap `max_cores` so each mover owns ≥ 16 TRS; and `movers_per_core = 1` on more cores vs 2 on fewer |
| mover roles / NoC direction | mover 1 reads DRAM on NoC1; catalog `noc_placement` measured reads-on-NoC1 2.5–4.8× slower for spread placements (bandwidth-heavy streams; ours are small) | asymmetric: RISCV_1/NoC0 runs gather + await and hands stage + plan through a CB pair to RISCV_0/NoC1, which runs permute + store |
| store realization | `permute_block` copies every face-row once more in L1 | `tile_face_row` only: write each face-row straight from `cb_src_stage` to DRAM (16-B congruent), dropping `cb_dst_image` and the copy, at one DRAM write per face-row instead of per run |
| read granule on sparse input pages | the forward family (`[B,1,W]→[1,1,B,W]`) reads `f0 row ρ` and `f1 row ρ` of each input page as two 32-B reads | one covering read `[off(f0,ρ), off(f1,ρ)+F)` = 544 B per page: half the read transactions at 8.5× the read bytes (reads may include unneeded bytes; writes may not) |
| overlap (large block) | one block per mover exposes one full DRAM round trip before any copy starts | trid sub-batches within the block (`set_read_trid` / `async_read_barrier_with_trid`), or `stage_depth = 2` with smaller blocks |
| copy engine | RISC-V load/store per face-row is on the critical path | self-aimed NoC L1→L1 reads for 16-B-congruent entries (`local_copy_helpers_dataflow.hpp:57`); the idle TRISCs (a compute kernel) as extra copiers |
| per-core RT args | the dispatcher writes 2 RT args per mover per core | common RT args only; each mover derives its atom range from its logical core index with the host's split formula |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM input pages → `cb_src_stage_mX` | input face-rows in source byte order; each read run phase-aligned mod `A` | `noc_async_read` on the mover's own NoC, addresses from `TensorAccessor(in).get_noc_addr(page, offset, noc)` | only logical input rows are ever addressed; `tile_face_row` reads only logical columns |
| `cb_src_stage_mX` → `cb_dst_image_mX` | output face-rows, compacted per output tile in DRAM byte order | RISC-V copies per plan entry | `invalidate_l1_cache()` after `await_block`; store drain before `store_block` |
| `cb_dst_image_mX` → DRAM output pages | contiguous face-band runs | `noc_async_write` on the mover's NoC, `TensorAccessor(out).get_noc_addr(page, offset, noc)` | output padding rows never written |
| Tensix ↔ Tensix | — | none | no semaphores, no multicast, no combine — no deferred regime needs one |

TARGET as regimes (TARGET has no placement axis: DRAM interleaved only, nothing sharded):

| TARGET value | When | Cost class | How |
|--------------|------|------------|-----|
| `dtype = bfloat16`, `layout = TILE_LAYOUT` | Phase 0 | — | built regimes, `e = 2` |
| `dtype = float32` (+ TILE) | refinement | knob-turn | `e = 4` → `F = 64`, `atom_rows = 1`; same kernel; FLUX `[1,W]→[1,1,W,1]` runs `tile_element_run` |
| `dtype = int32 / uint32` + `layout = ROW_MAJOR_LAYOUT` | refinement (one change, per the spec rule) | regime addition on the same skeleton | `rm_row_gather` (deferred row); `e` generic, so bf16 RM and int32 TILE fall out of the same two geometries |

Catalog entries that set knobs:

| Entry | Knob it informed |
|-------|------------------|
| `tile_reorder` | move maximal contiguous runs, batch barriers; whole pages only where all their bytes are owned and useful (full-height tiles become one 2-KB write automatically) |
| `double_buffer` | issue every read of a block, then ONE barrier; `stage_depth` lamp |
| `trid_double_issue` | overlap lamp (trid-scoped barriers) |
| `split_reader` | both data-movement RISC-Vs issue transactions (`movers_per_core = 2`), because the op is per-transaction issue-bound, not bandwidth-bound |
| `noc_placement` | cores taken row-major from the grid (`row_wise` spread); NoC-direction lamp |
| `width_split` | tile columns are the OUTER fold component, so wide-short outputs spread over columns instead of stranding on a few rows |
| `read_coalescing` | face-column walk turns consecutive rows of one face into one read |
| `bank_stagger` | no extra mechanism: the fold already starts each core at a different tile column / page |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | atom (`atom_rows` output rows of one image × one output tile column × both halves) |
| Grid | `grid = device.compute_with_storage_grid_size()`, `max_cores = grid.x·grid.y` (knob); `P = min(max_cores, G)`; core `i` = logical `(i % grid.x, i // grid.x)` → ≤ 2 rectangular core ranges (full rows + one partial row), one range set for both mover kernels (BB-17) |
| Per-core work | atoms `[⌊i·G/P⌋, ⌊(i+1)·G/P⌋)` (differ by ≤ 1 atom) |
| Per-mover work | mover 0 = first `⌈n_i/2⌉` atoms of core `i`, mover 1 = the rest (may be 0); RT args per mover `(atom_begin, atom_count)` |
| Atom → rows | `a → tc' = a / (B_out·A_img)`, `q = a % (B_out·A_img)`, `b = q / A_img`, `k = q % A_img`, rows `r' ∈ [b·H_out + k·atom_rows, b·H_out + min(H_out, (k+1)·atom_rows))`, `A_img = ceil(H_out / atom_rows)` |
| Remainder | alignment-aware throughout: per-image `ceil` for tile rows and atoms; ragged last atom of an odd-`H_out` image; ragged last block of a mover is the same block operation with fewer atoms |
| Blocks per mover | `ceil(atom_count / block_atoms)`; `block_atoms = min(max atoms over movers, L1 cap)` (CT) → one block per mover for every golden INPUTS shape |

Atoms per production bf16 case (`atom_rows = 2`): `(1,16,6,256)→(1,1,16,1536)` 384; `(16,1,64)→(1,1,16,64)` 16; `(16,1,1536)→(1,1,16,1536)` 384; `(1,1,16,4608)→(16,1,4608)` 2304; `(1,16,64)→(16,1,64)` 32; `(16,1,5120)→(1,1,16,5120)` 1280; `(2,1,5120)→(1,1,2,5120)` 160; `(1,8,6,256)→(1,1,8,1536)` 192; `(8,1,64)→(1,1,8,64)` 8; `(8,1,1536)→(1,1,8,1536)` 192; `(1,1,8,4608)→(8,1,4608)` 1152; `(1,8,64)→(8,1,64)` 16; `(8,1,5120)→(1,1,8,5120)` 640; `(1,1,6,256)→(1,1,1536)` 48. Element-run bf16: `(1,185)→(1,1,185,1)` 93; `(1,512)→(1,1,512,1)` 256.

Host contract (all in Python, one dispatch):

| Item | Decision |
|------|----------|
| Dispatch | exactly one `ttnn.generic_op([input_tensor, output_tensor], descriptor)` |
| Kernels | ONE dataflow source (e.g. `kernels/g6_reshape_mover.cpp` + a geometry header) instantiated twice on the same core ranges: mover 0 `ReaderConfigDescriptor()` (RISCV_1, NoC0), mover 1 `WriterConfigDescriptor()` (RISCV_0, NoC1); CT args carry the mover's three CB ids. No compute kernel. Default optimisation level (BB-18: no size-optimised build) |
| CT args | geometry (`e`, `W/H/B/Wt/Ht` of both tensors, `R_out`, `trs` facts), host-precomputed incremental steps, `kFaceRowAligned`, `atom_rows`, `block_atoms`, capacities, `A`, `TensorAccessorArgs(in)`, `TensorAccessorArgs(out)` |
| RT args | common: `[in_addr, out_addr]`; per core per mover: `[atom_begin, atom_count]` |
| Descriptor cache (BB-07, BB-16) | module-level dict keyed by `(input logical+padded shape, resolved e, dtype, layout, memory_config, device id, grid)`; on a hit only the common RT buffer addresses are patched — buffer addresses must never be CT (generic_op's hash excludes RT values, `generic_op_device_operation.cpp:78-99`, so CT addresses would rebuild every call and break trace replay) |
| Plan report (BB-08, BB-21) | `describe_plan(input_tensor, shape) -> dict` exported from `ttnn.operations.g6_reshape`; runs `validate()`, never dispatches; keys: `regime`, `num_cores`, `core_ranges`, `atoms_total`, `atom_rows`, `block_atoms`, `src_face_rows_bound` (`S`), `l1_bytes_per_core`, `roles` (per core/mover: core xy, mover id, RISC, NoC, `atom_begin`, `atom_count`) |
| Profiling (BB-21) | `DeviceZoneScopedN` around each block operation (`g6.gather`, `g6.await`, `g6.permute`, `g6.store`, `g6.drain`); compiled out in the plain build |
| Completion (BB-03) | every mover ends with `noc_async_read_barrier()` (already per block) and `noc_async_write_barrier()` |

## Circular Buffers

All six are private scratch of one mover: the same kernel instance is the only producer and only consumer, so no `push_back`/`wait_front` handshakes exist (push = wait = 0). Each is one page of `total_size` bytes; kernels take the base with `get_write_ptr(cb)`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_src_stage_m0` | 0 | `stage_bytes` | 1 | `stage_depth · 2T·S·(F + A)`: spans the block's input face-rows (≤ `S` per output face-row) + ≤ `A` phase pad per run | input dtype | mover 0 (NoC reads) | mover 0 (RISC-V loads) | gather → permute of each block |
| `cb_dst_image_m0` | 1 | `image_bytes` | 1 | `image_depth · 2T·F`: spans the block's output face-rows, compacted | input dtype | mover 0 (RISC-V stores) | mover 0 (NoC writes) | permute → write flush of each block |
| `cb_move_plan_m0` | 2 | `plan_bytes` | 1 | `16·S·T + 16·T`: ≤ `2T·S` copy entries × 8 B + ≤ `T` tile entries × 16 B | UInt32 | mover 0 | mover 0 | gather → store of each block |
| `cb_src_stage_m1` | 3 | `stage_bytes` | 1 | as m0 | input dtype | mover 1 | mover 1 | as m0 |
| `cb_dst_image_m1` | 4 | `image_bytes` | 1 | as m0 | input dtype | mover 1 | mover 1 | as m0 |
| `cb_move_plan_m1` | 5 | `plan_bytes` | 1 | as m0 | UInt32 | mover 1 | mover 1 | as m0 |

Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `gather_block` | `block_atoms` atoms → column segments × `trs_halves` × rows | no (TensorAccessor + raw NoC read; walk is op-specific) | — (DRAM input) | `cb_src_stage_mX` (reads in flight), `cb_move_plan_mX` (plan + tile table complete) | staging landing; plan valid |
| 2 | `await_block` | block | no | `cb_src_stage_mX` | — | staging valid |
| 3 | `permute_block` | all plan entries of the block | no (raw RISC-V copies; see API Mapping) | `cb_src_stage_mX` (valid), `cb_move_plan_mX` | `cb_dst_image_mX` | image valid and drained; staging free for the next block |
| 4 | `store_block` | all tile entries of the block | no (TensorAccessor + raw NoC write) | `cb_dst_image_mX`, tile table in `cb_move_plan_mX` | — (DRAM output) | image in flight until the next `permute_block`'s `writes_flushed` |
| 5 | `drain` | — | no | — | — | all writes acknowledged |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| `gather_block` | raw_api | `TensorAccessor::get_noc_addr(page_id, offset, noc)` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165` | input accessor (`TensorAccessorArgs(in)` CT), mover NoC | — | — | — |
| `gather_block` | raw_api | `noc_async_read(src_noc_addr, dst_l1, bytes, noc)` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | one call per read run | — | `cb_src_stage_mX` | run length grows with `block_atoms` (face-column coalescing) |
| `await_block` | raw_api | `noc_async_read_barrier(noc)` | `dataflow_api.h:1921` | once per block | `cb_src_stage_mX` | — | frequency = `num_blocks` |
| `permute_block` | raw_api | `noc_async_writes_flushed(noc)` | `dataflow_api.h:1980` | once per block before overwriting the image | — | `cb_dst_image_mX` | — |
| `permute_block` | raw_api | `invalidate_l1_cache()` | `tt_metal/hw/inc/internal/tt-1xx/cache.h:12` | once per block after `await_block` | `cb_src_stage_mX` | — | — |
| `permute_block` | raw_api | volatile L1 word/halfword copies (`dataflow_kernel_lib::addr_to_l1_ptr`) | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:30` | one copy per plan entry | `cb_src_stage_mX` | `cb_dst_image_mX` | — |
| `permute_block` | raw_api | `ckernel::load_blocking(last written word)` | `tt_metal/tt-llk/tt_llk_blackhole/common/inc/ckernel.h:144` (usage: `tt_metal/programming_examples/add_2_integers_in_riscv/kernels/reader_writer_add_in_riscv.cpp:48-49`) | once per block before `store_block` | — | `cb_dst_image_mX` | — |
| `store_block` | raw_api | `TensorAccessor::get_noc_addr` + `noc_async_write(src_l1, dst_noc_addr, bytes, noc)` | `tensor_accessor.h:165`, `dataflow_api.h:876` | one call per merged output run | `cb_dst_image_mX` | — | runs merge more as blocks cover more rows of a tile |
| `drain` | raw_api | `noc_async_write_barrier(noc)` | `dataflow_api.h:1951` | once per kernel | — | — | — |

Helpers considered and rejected:

| Block operation | Candidate | File:line of the mismatch | Why it cannot express the shape |
|-----------------|-----------|---------------------------|---------------------------------|
| `permute_block` | `tt::data_movement::common::copy_via_memmove<copy_async>` | `ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp:136-148` | it runs `invalidate_l1_cache()` and (sync) a drain on EVERY call; the block contract pays both once per block over hundreds of 32-B entries. Acceptable for element-run pieces with `copy_async = true` inside the block's single invalidate/drain pair |
| `permute_block` | `tt_memmove` (NoC self-copy) | `common.hpp:218-283` | needs `dst ≡ src (mod 16)`; element-run pieces violate it, and a NoC transaction per 32-B face-row costs more issue than 8 word copies (kept as the copy-engine lamp) |
| `permute_block` | `copy_tile_column` | `ttnn/cpp/ttnn/kernel_lib/tile_column_dataflow.hpp:92` | moves a tile COLUMN into a tile column; the element-run move is a row segment into a column or an arbitrary column shift |
| all | compute helpers `tilize` / `untilize` / `eltwise copy` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:6,101` (pack_untilize via unpack→math→pack) | values would pass unpacker/FPU/packer formats — the spec forbids any conversion path; the op does no arithmetic |
| `gather_block` | `set_read_trid` / `async_read_barrier_with_trid` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:147,158` | not needed at `stage_depth = 1` (one block per mover in production); it is the overlap lamp's mechanism |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| 64-B DRAM read congruence on Blackhole | face-rows sit at `ρ·32` offsets, i.e. ≡ 32 (mod 64) for odd rows; reading one into an arbitrary L1 slot is illegal | staging address of each run ≡ source NoC address (mod `A`); never read DRAM straight into the output image |
| CPU copies between NoC phases | Blackhole RISC L1 cache is write-through but holds stale lines across reused staging; CPU stores may retire before landing in L1, the NoC is a separate L1 client | `invalidate_l1_cache()` after `await_block`; one `ckernel::load_blocking` drain of the last written word before the first NoC write of `store_block` |
| Poisoned input padding (Rule 2) | `fill_implicit_tile_padding(NaN)` fills padding rows and columns | plan entries cover logical elements only; `tile_face_row` face-rows are fully logical; element-run pieces stop at `W_in`; padding rows are never addressed |
| Ownership atoms | bf16 rows `2k, 2k+1` share a 64-B DRAM word per face | all core/mover/block boundaries on atoms; atoms restart at every output image (odd `H_out` → 1-row tail atom) |
| Per-face-row CPU cost is the critical path | ~2–20 face-rows per mover, each costing index math + a plan entry + a copy | incremental counters only; 64-bit div once per (segment, half); `kFaceRowAligned` CT specialisation (one piece, fixed 32-bit copy) |
| Half 1 of the last output tile column | `W_out % 32 ∈ (0,16]` → half 1 is pure padding | `trs_halves = 1` there: no copy, no write, no image space |
| Rank 1/2 geometry and padding | rank-1 TILE tensors and rank-2 shapes fold to `(B=1, H=1 or d_0, W)` | page geometry from `padded_shape`; host asserts `buffer_num_pages() == B·Ht·Wt` for both tensors |
| Element-run staging bound | `W_in < 16` (e.g. the round trip `(1,1,185,1)→(1,185)`) makes one output face-row touch up to 16 input face-rows | `S` from `W_in` sizes stage and plan; `--dev` `ASSERT` on capacity |
| Cache key completeness (BB-07) | the same input reshaped to two targets with different page maps (loose `sequence` case) | key includes resolved `e` and every geometry field; addresses only in RT args |
| Trace replay (BB-16) | a cached descriptor reused under capture | no allocation besides the output tensor; descriptor cache built before capture by the eager call; no kernel build on replay |
| Two movers, one core | both movers issue reads and writes; each must stay on its own NoC | mover NoC = its config's NoC (`noc_index`); every `get_noc_addr` / barrier passes that NoC |
| Device-side misalignment hides in release builds | wrong alignment corrupts silently without watcher | implementer runs the acceptance test once with `--dev` (NoC sanitizer) |
