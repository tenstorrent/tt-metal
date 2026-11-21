# Operation Design: rms_norm_head_rope

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (per-head row reduction + broadcast eltwise + tile matmul + eltwise), single native dispatch |
| Goal | Per-head RMSNorm over head_dim=128, optional per-column weight, optional interleaved-pair RoPE (cos/sin shared by all heads), for FLUX.2-dev per-chip q/k. Fill the 12x10 Blackhole grid by splitting **(row-tile, head-group)**, not row-tile alone. |
| Math | `ms[r,h] = (1/128)·Σ_{d<128} x[r,h·128+d]²` ; `y = x·rsqrt(ms+eps)` ; `y *= w[h·128+d]` (if weight) ; `out = y·cos[r,d] + rot(y)[d]·sin[r,d]` with `rot(y)[2i] = -y[2i+1]`, `rot(y)[2i+1] = y[2i]` (if rope), else `out = y` |
| References | `eval/golden_tests/rms_norm_head_rope/{feature_spec.py,helpers.py}` (torch golden `pytorch_rms_norm_head_rope`, `rot_transformation_mat`); `.claude/references/{blocking-model.md,l1-footprint-discipline.md,precision_convention.md,ttnn-cb-memory-fundamentals.md}`; `.claude/references/generic_op_template/`; `ttnn/cpp/ttnn/kernel_lib/` (reduce, eltwise chain, broadcast); perf catalog `ttnn/ttnn/operations/examples/master.md` (absent on this branch; entries cited by name: `double_buffer`, `noc_placement`, `width_split`, `shared_input_reuse`, `compute_fusion`, `row_reduce_accumulate`, `compute_block_size`) |

Companion artifact: `l1_ledger.md` (CB ledger + data-movement budget), same directory.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | ttnn.Tensor | yes | (1,1,R,W), bf16, TILE, DRAM interleaved, R%32==0, W%128==0 | — | buffer address RT; TensorAccessorArgs CT |
| `weight` | ttnn.Tensor | no | (1,1,1,W) bf16 TILE (DRAM interleaved) | None | presence → CT define `HAS_WEIGHT`; address RT |
| `rope_cos` | ttnn.Tensor | no (with `rope_sin`) | (1,1,R,128) bf16 TILE | None | presence → CT define `HAS_ROPE`; address RT |
| `rope_sin` | ttnn.Tensor | no (with `rope_cos`) | (1,1,R,128) bf16 TILE | None | address RT |
| `transformation_mat` | ttnn.Tensor | no | if given: (1,1,32,32) bf16 TILE — validated, then **ignored** | None | — (the rotation tile is generated in-kernel, see Dataflow) |
| `epsilon` | float | no | > 0 | 1e-6 | compute RT arg (fp32 bit pattern) |
| `compute_kernel_config` | ttnn.ComputeConfigDescriptor | no | any fidelity / approx; `fp32_dest_acc_en ∈ {True, False}` | `default_compute_kernel_config()` = HiFi4, fp32_dest_acc_en=True, math_approx_mode=False | kernel config (fidelity, `FP32_DEST_ACC_EN`) |

`default_compute_kernel_config()` is a public **factory** exported from `ttnn.operations.rms_norm_head_rope` (fresh descriptor per call; `precision_convention.md`). The entry point resolves `None` through it; `golden_tests/.../axes.py` already imports it. `math_fidelity` / `math_approx_mode` are never gated.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | (1, 1, R, W); R = 32·Rt, W = 128·H (H heads, `HEAD_DIM = 128` = `HEAD_TILES = 4` tiles) |
| Dtype | bfloat16 (input, weight, cos, sin) |
| Layout | TILE |
| Memory | DRAM interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | DRAM interleaved (`ttnn.DRAM_MEMORY_CONFIG`), allocated by the entry point (`ttnn.allocate_tensor_on_device`) |

### Validation (entry point, raises `ValueError`/`RuntimeError`)

| Check | Error |
|-------|-------|
| input dtype ≠ bf16, layout ≠ TILE, memory ≠ DRAM INTERLEAVED | ValueError |
| rank ≠ 4, shape[0..1] ≠ (1,1), R % 32 ≠ 0, W % 128 ≠ 0 | ValueError |
| exactly one of rope_cos / rope_sin given | ValueError |
| weight shape ≠ (1,1,1,W), or dtype ≠ bf16, or layout ≠ TILE, or not on device | ValueError |
| cos/sin shape ≠ (1,1,R,128), dtype ≠ bf16, layout ≠ TILE | ValueError |
| transformation_mat given and shape ≠ (1,1,32,32) / dtype ≠ bf16 / layout ≠ TILE | ValueError |
| fp32_dest_acc_en=False | **accepted** (bf16 accumulate) |

No host-side tensor transformation (no to_layout / pad / slice / tilize / composition of ttnn ops). Exactly **one** `ttnn.generic_op` dispatch per call. Nothing derived from weight / cos / sin is cached across calls; the program descriptor may be rebuilt per call (addresses are RT args).

---

## Blocking Model

The op's tile index space is 3-D: **row-tile r ∈ [0,Rt)** × **head h ∈ [0,H)** × **head-dim tile d ∈ [0,4)**. (Leading dims are 1 and fold away.) A **block** is the rectangle `block_row_tiles × block_heads × HEAD_TILES` = `1 × block_heads × 4` tiles, owned by one core, driven through one pass of the phase sequence.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| head-dim tile `d` (4 tiles = 128 cols) | **dependent** — the RMS sum spans all 128 columns of a head; RoPE pairs (2i,2i+1) stay inside one 32-col tile | `HEAD_TILES` | 4 (whole extent, never split) | host constant `HEAD_TILES = HEAD_DIM // 32` → compute CT arg | not split across cores — every core holds whole heads | scheme-change (cross-core combine) — `rejected`, see Regimes |
| head `h` | **independent** for x/out (per-head norm, no cross-head term); **reuse-shared** for cos/sin (cos/sin do not vary with h — every head of a row-tile reads the same 4 cos + 4 sin tiles) | `block_heads` | `min(heads_per_core, MAX_BLOCK_HEADS)` — the core's whole head group (= 2 at R=256, 8 at R=2048/2304) | host `block_heads` → compute/reader/writer CT arg | split into `G` head-groups (`heads_per_core = ceil(H/G)`), G chosen by the selection function (Work Distribution) | knob-turn (G, block_heads) |
| row-tile `r` | **independent** for x/out/cos/sin (rows never interact); **reuse-shared** for weight (w does not vary with r — a core loads its head-group's weight once and keeps it resident for all its rows) | `block_row_tiles` | **1** — justified: cos/sin are indexed per row-tile and the chain's `InputTileMapping::Row` maps them by column only (`eltwise/api/chain.hpp` InputTileMapping doc); a 1-row block already spans the core's whole head group (up to 32 tiles), so the block is coarse along the axis that carries the volume; rows_per_core is 1–2 at the target shapes | host `block_row_tiles` CT arg (asserted == 1 in Phase 0) | split into `Rg` row-groups (`rows_per_core = ceil(Rt/Rg)`), contiguous row ranges | knob-turn (raising `block_row_tiles` needs Strided cos/sin addressing) |

Every axis is decided; no axis is left without a core-assignment. The stage-produced data (sumsq, rstd, normed y, rot(y)) inherit the (r, h, d) axes and are computed by the core that owns the block — there is no coordinator core and no stage left on one core.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `input_depth` | 2 blocks | reader fetches block b+1 while compute runs block b (`double_buffer`: 2× block, 4–8+ reads per barrier) |
| `cb_rope_cos`, `cb_rope_sin` | `rope_depth` | 2 row-tiles | next row-tile's cos/sin prefetched during current block |
| `cb_output_tiles` | `output_depth` | 2 blocks | writer drains block b while compute fills b+1 |
| all compute-internal CBs | — | 1 block | sequential helper intermediates (producer and consumer are both compute; no pipelining possible) |
| resident CBs (weight, trans_mat, scaler) | — | 1 (whole lifetime) | loaded once per kernel |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| L1 residency of the per-core weight (whole head group resident for all rows) | `heads_per_core` | selection function only accepts G with `ceil(H/G) ≤ MAX_RESIDENT_HEADS = 24` | weight CBs exceed L1 → allocation failure |
| L1 working set per block (`l1_ledger.md` total) | `block_heads` | `block_heads = min(heads_per_core, MAX_BLOCK_HEADS = 8)`; blocks per row-tile = `ceil(heads_per_core / block_heads)`, tail block = remainder heads | CB allocation exceeds L1 |
| RoPE pair rotation as a 32×32 tile matmul | `HEAD_TILES` tile boundary | pairs (2i,2i+1) never straddle a tile because 32 is even — no clamp needed; stated so nobody "optimizes" to a non-32-aligned head layout | wrong rotation |
| `BinaryFpu` DEST accumulation (`DestAccumulation::PerRow`) | grid row width = `HEAD_TILES` | requires `Dst::D0` (`chain.inl:1273`), output `(PerOuter, PerOuter)` (`chain.inl:1142-1148`) | static_assert |
| DEST capacity in the rope chain (two live operand slots D0, D1) | chain block (tiles per acquire) | chain clamps `block_size·lane_width ≤ DEST_AUTO_LIMIT` itself (`chain.hpp` §3); lane width 2 → ≤ 2 tiles/acquire with fp32 DEST | — (helper-clamped) |
| BH DRAM read alignment 64 B | weight row-0 read size | each weight tile read as two 64 B reads at page offsets 0 and 512 (face0 rows 0–1, face1 rows 0–1) | misaligned DRAM read → corrupt data |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1. 2-D (row-group × head-group) split, interleaved streaming, per-core resident weight, per-block resident x** | **built** | always (all valid inputs); (G, Rg) chosen by `select_grid_split` | `1 × block_heads × 4` | x: 1 DRAM read; out: 1 DRAM write (= minimum). cos/sin: G reads each (minimum 1) — bounded by the selection objective, which prices it (+12% total bytes at R=2304, G=3). weight: 1 row-0 read per core (Rg × `heads_per_core·4·128 B` ≈ 0.5 MB at R=2304; useful-byte minimum 6 KB once). trans_mat: 0 (generated in L1) | Per block: one reader barrier over `4·block_heads` tile reads, one writer barrier, one set of chain/matmul/reduce inits per phase, one cos/sin wait. Larger `block_heads` = fewer inits/barriers per tile; intended frequency: once per block = once per (row-tile, head-group) at Phase 0 |
| R2. cos/sin multicast within a row-group (cos/sin reuse-shared over the G head-group cores) | **deferred** | G > 1 | same as R1 | saves (G−1)·(cos+sin) DRAM bytes (2.4 MB at R=2304, G=3; 1.4 MB at R=256, G=12); adds one mcast of 8 tiles per row-tile per row-group + semaphore round trip | — |
| R3. Split head-dim `d` across cores + cross-core sumsq combine | **rejected** — superseded by R1: independent axes give 192–1728 (row-tile, head) units ≥ 120 cores at every target shape, so no parallelism is missing; `width_split` measured 0.69–0.95× at Wt=4; the combine adds a NoC round per head for no residency benefit (a head is only 4 tiles, already resident) | — | — | adds partial-sum traffic + semaphore per head; saves nothing | — |
| R4. Streaming (non-resident) reduction over head_dim | **rejected** — the reduced extent is fixed at 4 tiles, resident by construction; there is no size at which residency fails, so no fits-in-L1 dual path is needed | — | — | would re-read x per head | — |

R2 reason (positive): the catalog's `shared_input_reuse` measured a one-shot broadcast paying only when the shared window is ≥ 0.5× the core's own input bytes; here the window is 32 KB of cos/sin per row-tile against 128 KB of x per row-tile per core (0.25×), and `mcast_topology` measured lines of ≤ 4 cores losing (0.65–0.91×) — G = 3 at the dominant shapes. The selection function already caps the re-read. **Reachability:** cos/sin live in their own reader-fed CBs (`cb_rope_cos/sin`), consumed only by compute; R2 replaces the non-injector cores' DRAM read with a `ttnn.Mcast` receive into the same CB — compute and writer are untouched, and cores of one row-group are already enumerated adjacently (core index `k = rg·G + g`), so the receiver set is a contiguous run.

R1 single-regime proof obligations: domain = every valid input (R%32==0, W%128==0). No other regime applies because the reduced axis is a fixed 4-tile extent (no residency predicate exists) and the independent axes alone fill the grid. Named boundary = DRAM: minimum is x once + out once + cos/sin once + weight useful bytes once; actual crossings stated in the row and in `l1_ledger.md`.

### Traffic ranking

Bytes per (row-tile, head) unit: x 8 KB + out 8 KB = 16 KB. cos+sin per row-tile: 16 KB. Weight row-0 per head: 512 B. Shapes: R=2304 → Rt=72, H=24.

| Candidate split | DRAM bytes (R=2304) | Cross-core | Per-core max (16 KB units) | Rank |
|-----------------|---------------------|------------|----------------------------|------|
| rows only (G=1, stock scheme) | 28.3 MB + cos/sin 1.2 MB + weight 72×48 KB row-0 = 0.9 MB → **30.4 MB** (cheapest) | none | 25 (72 of 120 cores busy; at R=256 only 8 cores) | 2 |
| **rows × head-groups (G=3, Rg=40)** — chosen by `select_grid_split` | 28.3 + cos/sin 3×1.2 = 3.5 + weight 0.5 → **32.3 MB** (+6%) | none | **18** (120 cores) | **1** |
| rows × head-groups (G=8, Rg=15) | 28.3 + 9.4 + 0.2 = 37.9 MB | none | 20 | 4 |
| any G + cos/sin mcast (R2) | as G=1 for cos/sin: ≈30.9 MB | (G−1)·cos/sin over NoC, semaphores | as chosen G | 3 (deferred) |
| split head-dim + combine (R3) | ≥ 30.4 MB + partials | partial tiles per head | — | 5 (rejected) |

Chosen: rows × head-groups. Rows-only moves 6% fewer DRAM bytes but leaves 40% of the grid idle at R≥2048 and 93% idle at R=256, and per-core work is 25 vs 18 units. The op is near-co-bound on compute (≈6 FPU/SFPU tile ops per output tile at HiFi4), so per-core tile count, not only DRAM bytes, sets the floor. The selection objective prices **both** (per-core bytes include the cos/sin re-read), so it lands on the cheap-traffic end of the full-grid options. Whether the rows-only point (G=1) wins at R=2304 is a measurable knob-turn (`G`), recorded as a perf lamp, not a different algorithm.

Weight: full-tile weight reads would cost Rg·`heads_per_core`·4·2 KB = 7.8 MB at R=2304 (31/32 padding). Phase 0 reads only the row-0 bytes (2 × 64 B per tile), cutting that to 0.5 MB; weight multicast is therefore not worth a regime row (it would save < 0.5 MB).

Stall-shadow check: the only waits are compute on reader (x, cos/sin) and writer on compute — covered by depth-2 CBs. No stage waits on a peer core. Algebraic note: rsqrt(ms+eps) is a per-row scalar and every later stage (weight, rot, rope) is linear in y, so `out = rstd ⊙ rope(rot(x∘w))` is legal (rstd applied last). Phase 0 applies rstd **first**, because there is no cross-core wait to hide and rstd-first lets the weight multiply fold into the same pass via DEST reuse (see Block schedule). The reorder is recorded, not built; it would change fp rounding order (gate on the precision baseline if ever used).

### Block schedule

Per core: `rows_this_core` row-tiles starting at `row_start`, heads `[head_start, head_start + heads_this_core)`. Blocks per row-tile: `blocks_per_row = ceil(heads_this_core / block_heads)`; `num_blocks_this_core = rows_this_core · blocks_per_row`.

```cpp
prologue_load_constants();              // once per kernel: scaler(1/128), trans_mat (generated), weight row-0 tiles
expand_weight_rows();                   // once per kernel [HAS_WEIGHT]: w row-bcast -> full tiles, resident
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_block(block_idx);              // x: bh x 4 tiles; cos/sin: 4+4 tiles when the row-tile changes [HAS_ROPE]
    sum_squares_block(block_idx);       // bh x 4 -> bh tiles: DEST-accumulated x*x per head
    rsqrt_mean_block(block_idx);        // bh -> bh tiles: row-reduce (scale 1/128) + eps + rsqrt
    normalize_block(block_idx);         // bh x 4: y = x * rstd(col-bcast) [* w_full via DEST reuse]
    rotate_pairs_block(block_idx);      // bh x 4 [HAS_ROPE]: rot = y @ T
    apply_rope_block(block_idx);        // bh x 4 [HAS_ROPE]: out = y*cos + rot*sin
    store_block(block_idx);             // bh x 4 tiles -> DRAM
}
```

| Operation | Block shape | Resident across it | Intended init / barrier frequency |
|-----------|-------------|--------------------|-----------------------------------|
| `prologue_load_constants` | scaler 1 tile, T 1 tile, weight `heads_this_core × 4` tiles | all three for the whole kernel | once per kernel; weight: 2 reads/tile, one barrier for all |
| `expand_weight_rows` | `heads_this_core × 4` tiles | `cb_weight` (full tiles) for the whole kernel | once per kernel, one bcast init |
| `load_block` | x `block_heads × 4`; cos/sin `4` each | cos/sin across all blocks of the same row-tile | one read barrier per block (all `4·block_heads` reads issued first); cos/sin once per row-tile |
| `sum_squares_block` | grid(block_heads, 4) → block_heads tiles | x stays in `cb_input_tiles` (not popped) | one mul init per block |
| `rsqrt_mean_block` | block_heads × 1 → block_heads | x | one reduce init per block; eps+rsqrt in the post-op per tile |
| `normalize_block` | grid(block_heads, 4) | rstd, w_full; x popped at end | one chain per block (inits per element as the chain decides) |
| `rotate_pairs_block` | block_heads·4 tiles | y (not popped), T | one matmul init per block |
| `apply_rope_block` | grid(block_heads, 4) | cos/sin (Row mapping, popped after the last block of the row-tile) | one chain per block |
| `store_block` | block_heads·4 tiles | — | one write barrier per block |

Without rope, `normalize_block` packs straight into `cb_output_tiles` (pack into destination — `cb_normed` / `cb_rotated` are not allocated). Without weight, `normalize_block` omits the DEST-reuse element and the weight CBs are not allocated.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| **Overlap** | Coarse `block_heads` = whole head group gives only 1–2 blocks per core at the target shapes (R=256: 1 block of 8 tiles; R=2304: 2 blocks of 32 tiles), so load of the first block and store of the last block are exposed | `block_heads ∈ {1, 2, 4}` (knob-turn); or keep the block and let `sum_squares_block` consume x per head as it arrives (`WaitPolicy::Cumulative`) |
| **Grid synchronization / launch floor** | R=256 at G=12 gives 3 units (48 KB) per core; launch + prologue (~1 µs) may rival the work | cap occupancy (e.g. G=6, 48 cores) via the selection objective |
| Grid-split objective | Objective prices per-core bytes with cos/sin = one unit; if DRAM-aggregate-bound, G=1 (−6% bytes, 72 cores) may win at R≥2048; if compute-bound, a compute-weighted objective prefers G=5/8 (15 vs 16 compute units) | sweep G ∈ {1,3,5,8} at R=2304 |
| normalize fusion | `normalize_block` fuses FPU→FPU via DEST reuse (`compute_fusion` measured FPU-consumer fusion at 0.72× vs packing to L1) | two passes: `x*rstd → cb_normed`, then `cb_normed * w_full` (bcast Row) → second CB |
| RoPE formulation | rope chain uses two FPU muls + SFPU add + a separate matmul pass | `D0 = y*cos`, then matmul-accumulate `(y∘sin_p) @ T` into D0 (sin_p = pair-swapped sin, computed once per row-tile) — removes the SFPU add and the `cb_rotated` pack of rot |
| Intermediate page format | `cb_normed` / `cb_rotated` are Float32 when fp32 DEST is on (ledger audit 2); doubles their pack/unpack bytes | bf16 pages for `cb_normed` / `cb_rotated` (y is rounded to bf16 at output anyway) — gate on golden PCC/RMS |
| NoC placement | DRAM-saturating, ≥16 reads in flight (`noc_placement`) | row-wise core order (Phase 0) vs column order; reads NoC0 / writes NoC1 (Phase 0) |

---

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x DRAM → `cb_input_tiles` | bf16 tiles | reader, `TensorAccessor` page reads, NoC0; tile id = `r·Wt + h·4 + d` (Wt = W/32) | per block: reserve `4·bh`, issue all reads, one barrier, push `4·bh` |
| cos/sin DRAM → `cb_rope_cos` / `cb_rope_sin` | bf16 tiles | reader, `TensorAccessor`; tile id = `r·4 + d` | once per row-tile (4 tiles each), pushed before that row-tile's first x block |
| weight DRAM → `cb_weight_row` | bf16 tile pages, **only row 0 valid** | reader: per weight tile `t = head·4 + d`, two 64 B reads `page_addr(t)+0` → L1 offset 0 and `page_addr(t)+512` → L1 offset 512 | rows 1–31 of the L1 page are don't-care (row broadcast reads row 0 only); once per kernel |
| trans_mat → `cb_trans_mat` | bf16 tile | reader **generates** T in L1: zero the page, then for i<16: `T[2i][2i+1] = +1.0 (0x3F80)`, `T[2i+1][2i] = -1.0 (0xBF80)` placed in tile-face order (element (r,c) → face `(r/16)·2 + c/16`, offset `(r%16)·16 + c%16`; all nonzeros land in faces 0 and 3) | matches `helpers.rot_transformation_mat`; caller's `transformation_mat` is ignored (also covers `None`) |
| scaler → `cb_reduce_scaler` | bf16 | reader: `dataflow_kernel_lib::prepare_reduce_scaler<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1/128)` | non-standard scaler (SUM × 1/HEAD_DIM) → `prepare_reduce_scaler`, pool-type-aware |
| compute | fp32 DEST (or bf16 DEST if disabled); intermediates in DEST-width pages | helpers + raw matmul (API Mapping) | |
| `cb_output_tiles` → DRAM | bf16 tiles | writer, `TensorAccessor`, NoC1; same tile ids as x | per block: wait `4·bh`, issue writes, one barrier, pop |

**Deferred R2 contract (cos/sin mcast):** row-group `rg` = cores `k ∈ [rg·G, rg·G+G)`; the core with `g = 0` reads cos/sin from DRAM into its `cb_rope_cos/sin` and multicasts the 8 tiles to the other G−1 cores' same-address CB slots via `ttnn.Mcast` (`McastArgs` sender/receiver, one semaphore per row-tile); receivers replace their DRAM read with `receive()`. Compute is unchanged.

**Placement (TARGET `memory_layout`):** TARGET is INTERLEAVED only. For the record: HEIGHT_SHARDED (cut rows) and head-aligned WIDTH/BLOCK_SHARDED (cut at multiples of 128 cols) are **knob-turns** — rows and heads are independent, the shard simply fixes (Rg, G) and each core consumes its shard in place through a CB backed on the sharded buffer (`ttnn.cb_descriptor_from_sharded_tensor`). A width shard that cuts a head would cut the dependent `d` axis → scheme-change (R3).

**Structural impossibilities:** none beyond `feature_spec.INVALID`.

**Index axes:** none (the op has no `dim`).

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | block = (1 row-tile) × (`block_heads` heads) × 4 tiles |
| Grid | `device.compute_with_storage_grid_size()` (12×10 on BH), runtime value — never inlined; `num_cores_avail = x·y` |
| Per-core work | core `k = rg·G + g` (k < G·Rg), enumerated **row-wise** over the grid: rows `[row_start(rg), +rows(rg))`, heads `[head_start(g), +heads(g))` |
| Remainder | rows: first `Rt mod Rg` row-groups get `ceil(Rt/Rg)`, rest `floor` (`ttnn.split_work_to_cores` semantics, or the same arithmetic inline); heads: same with H over G. Tile counts: `Rt = ceil(R/32)` (R is validated %32), `Wt = ceil(W/32)`, `H = W/128`. Cores `k ≥ G·Rg` get no kernels (core range excludes them) |

**Selection function `select_grid_split(Rt, H, num_cores, has_rope)`** (host, deterministic, pinned):

```
best = None
for G in 1 .. min(H, num_cores):
    Rg  = min(Rt, num_cores // G)
    hpc = ceil(H / G);  rpc = ceil(Rt / Rg)
    if hpc > MAX_RESIDENT_HEADS (24): continue
    per_core = rpc * (hpc + (1 if has_rope else 0))      # 16 KB units: x+out per (row,head) + cos+sin per row
    total    = Rt * H + (Rt * G if has_rope else 0)       # aggregate DRAM units
    key = (per_core, total, G * Rg)
    best = min(best, (key, G, Rg))
return G, Rg
```

Expected picks (12×10 grid, rope on): (Rt=8, H=24) → G=12, Rg=8 (96 cores, 1 row × 2 heads); (Rt=64, H=24) → G=3, Rg=40 (120 cores, ≤2 rows × 8 heads); (Rt=72, H=24) → G=3, Rg=40; (Rt=1, H=24) → G=24, Rg=1; (Rt=1, H=1) → G=1, Rg=1. Only one regime exists, so no regime-pinned tests are needed; the acceptance test covers G=1, G>1 with Rg>1, and uneven row/head remainders.

## Circular Buffers

`bh = block_heads`, `BT = bh·HEAD_TILES` (block tiles), `HC = heads_this_core` max over cores = `ceil(H/G)`. `DF` = DEST-width format: Float32 if `fp32_dest_acc_en` else Float16_b. Tile sizes: bf16 2048 B, Float32 4096 B.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_input_tiles` | 0 | bf16 tile | `input_depth · BT` | spans h,d of one block; streams r | Float16_b | reader | compute | per block (pop at end of `normalize_block`) |
| `cb_rope_cos` | 1 | bf16 tile | `rope_depth · HEAD_TILES` | spans d; streams r; reused over h | Float16_b | reader | compute | per row-tile [HAS_ROPE] |
| `cb_rope_sin` | 2 | bf16 tile | `rope_depth · HEAD_TILES` | as cos | Float16_b | reader | compute | per row-tile [HAS_ROPE] |
| `cb_weight_row` | 3 | bf16 tile | `HC · HEAD_TILES` | spans the core's h,d; resident over r | Float16_b | reader | compute | prologue only [HAS_WEIGHT] |
| `cb_trans_mat` | 4 | bf16 tile | 1 | constant | Float16_b | reader | compute | whole kernel [HAS_ROPE] |
| `cb_reduce_scaler` | 5 | bf16 tile | 1 | constant | Float16_b | reader | compute | whole kernel |
| `cb_weight` | 6 | bf16 tile | `HC · HEAD_TILES` | spans core's h,d; resident over r | Float16_b (exact copy of bf16 w) | compute | compute | whole kernel [HAS_WEIGHT] |
| `cb_sumsq` | 7 | DF tile | `bh` | one tile per head of the block | DF | compute | compute | sum_squares → rsqrt_mean |
| `cb_rstd` | 8 | DF tile | `bh` | one tile per head (col 0 valid) | DF | compute | compute | rsqrt_mean → normalize |
| `cb_normed` | 9 | DF tile | `BT` | full block (y read by matmul and by rope) | DF | compute | compute | normalize → apply_rope [HAS_ROPE] |
| `cb_rotated` | 10 | DF tile | `BT` | full block | DF | compute | compute | rotate_pairs → apply_rope [HAS_ROPE] |
| `cb_output_tiles` | 16 | bf16 tile | `output_depth · BT` | spans h,d of a block; streams r | Float16_b | compute | writer | per block |

CB sync (push = wait): reader pushes `BT` per block to `cb_input_tiles`, compute waits `BT` once (sum_squares, no pop), reduce/normalize index into the same window, normalize pops `BT`. cos/sin: reader pushes 4 per row-tile; compute waits 4 at the row-tile's first block and pops 4 after its last block. `cb_weight_row`: push `HC·4` once; compute waits once, pops after expansion. `cb_weight`: compute pushes `HC·4` once, waits once, never pops until kernel end (then pops `HC·4` to balance). Scaler / trans_mat: push 1, wait 1, pop 1 at kernel end. `cb_sumsq`: push `bh`, reduce waits/pops `bh`. `cb_rstd`: push `bh`, normalize waits `bh` and pops `bh` at block end. `cb_normed`: push `BT`; matmul waits `BT` (no pop); rope chain pops `BT`. `cb_rotated`: push `BT`, rope pops `BT`. `cb_output_tiles`: compute pushes `BT` per block, writer waits/pops `BT`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `prologue_load_constants` | 1 + 1 + HC·4 | reader + `prepare_reduce_scaler` | DRAM | `cb_reduce_scaler` 1, `cb_trans_mat` 1, `cb_weight_row` HC·4 | resident |
| 1 | `expand_weight_rows` | HC·4 tiles | yes — `UnaryBcast<Row>` chain | `cb_weight_row` HC·4 (wait upfront, pop at end) | `cb_weight` HC·4 | `cb_weight_row` empty; `cb_weight` resident for kernel |
| 2 | `load_block` | x BT; cos/sin 4+4 at row change | reader | DRAM | `cb_input_tiles` BT; cos/sin 4 each | — |
| 3 | `sum_squares_block` | grid(bh, 4) | yes — `BinaryFpu<Mul, x, x, D0, PerRow>` + `PackTile` (PerOuter) | `cb_input_tiles` BT (wait Upfront, pop None) | `cb_sumsq` bh | x still resident |
| 4 | `rsqrt_mean_block` | bh × 1 | yes — `reduce<SUM, REDUCE_ROW>` + post-op | `cb_sumsq` bh (BulkWaitBulkPop), `cb_reduce_scaler` (never popped) | `cb_rstd` bh | — |
| 5 | `normalize_block` | grid(bh, 4) | yes — chain: `BinaryFpu<Mul, x Block (wait None, pop AtEnd), rstd bcast Col / mapping Col (wait Upfront, pop AtEnd)>` [+ `DestReuseBinary<Mul, cb_weight (Offset base = block's first head·4, Block mapping, wait/pop None), DEST_TO_SRCA>`] + `PackTile` | `cb_input_tiles` BT, `cb_rstd` bh, `cb_weight` | `cb_normed` BT (rope) **or** `cb_output_tiles` BT (no rope) | x, rstd popped |
| 6 | `rotate_pairs_block` | BT tiles × 1 T | raw matmul (see API Mapping) | `cb_normed` BT (wait, no pop), `cb_trans_mat` | `cb_rotated` BT | y still resident |
| 7 | `apply_rope_block` | grid(bh, 4) | yes — chain: `BinaryFpu<Mul, normed Block, cos Row-mapping, D0>`, `BinaryFpu<Mul, rotated Block, sin Row-mapping, D1>`, `AddBinary<D0,D1,D0>`, `PackTile` | `cb_normed` BT, `cb_rotated` BT (pop AtEnd), `cb_rope_cos/sin` 4 (wait None/pop None — caller-managed per row-tile) | `cb_output_tiles` BT | normed/rotated popped |
| 8 | `store_block` | BT | writer | `cb_output_tiles` BT | DRAM | — |

The exact wait/pop spellings above are the intended lifecycle; the implementer picks the legal policy pair per `is_legal_input_policy_for_mapping` (chain.inl) and may move a wait/pop to explicit caller-managed calls where the chain's legality table requires it, as long as the CB sync counts in the Circular Buffers section hold.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot | raw_api (mandatory) | `compute_kernel_hw_startup(icb0, icb1, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60` | `(cb_input_tiles, cb_input_tiles, cb_sumsq)` first statement of MAIN | — | — | — |
| prologue (scaler) | helper | `dataflow_kernel_lib::prepare_reduce_scaler<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(dfb, 1.0f/HEAD_DIM)` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:70` | scaler = 1/128 (exact in bf16) | — | `cb_reduce_scaler` | — |
| expand_weight_rows | helper | `compute_kernel_lib::eltwise_chain(IterationShape::tiles(HC·4), UnaryBcast<BroadcastDim::Row, input(cb_weight_row, Upfront, AtEnd, Block)>{…}, PackTile<output(cb_weight, Upfront, AtEnd)>{…})` (or `unary_bcast<Row,…>` convenience) | `ttnn/cpp/ttnn/kernel_lib/eltwise/broadcast/bcast.hpp:26,34` | — | `cb_weight_row` | `cb_weight` | `HC` |
| sum_squares_block | helper | `eltwise_chain(IterationShape::grid(bh, HEAD_TILES), BinaryFpu<BinaryFpuOp::Mul, input(cb_input_tiles,…Block), input(cb_input_tiles,…Block), Dst::D0, DestAccumulation::PerRow>{x, x}, PackTile<output(cb_sumsq, ReservePolicy::PerOuter, PushPolicy::PerOuter, …, DestAccumulation::PerRow)>{sumsq})` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp` (BinaryFpu alias, DestAccumulation enum); `core/chain.inl:1142-1148,1273,1394-1397` (PerRow contract, D0, `mul_init(..., acc_to_dest)`) | same CB both sides → same mapping & policy (`chain.inl` same_dfb asserts) | `cb_input_tiles` | `cb_sumsq` | `bh` (grid rows) |
| rsqrt_mean_block | helper | `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_sumsq, cb_reduce_scaler, cb_rstd, ReduceInputPolicy::BulkWaitBulkPop>(sumsq, scaler, rstd, ReduceInputBlockShape::of(bh, 1), ReduceInputMemoryLayout::contiguous(), NoAccumulation{}, post_op)`; `post_op(dst) = binop_with_scalar_tile_init(); add_unary_tile(dst, eps_bits); rsqrt_tile_init(); rsqrt_tile<Default>(dst);` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:420-442` (signature), `:396-408` (post-op example); `tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:36,157`; `tt_metal/hw/inc/api/compute/eltwise_unary/rsqrt.h:21,45` | `ReduceFp32Mode::Fast` (FPU; fp32 page → TF32 srcA, error ≪ bf16 ulp) | `cb_sumsq` | `cb_rstd` | `bh` (rows) |
| normalize_block | helper | `eltwise_chain(IterationShape::grid(bh, HEAD_TILES), BinaryFpu<Mul, input(cb_input_tiles,…, Block), input(cb_rstd, BroadcastDim::Col, …, InputTileMapping::Col)>{x, rstd}, [DestReuseBinary<Mul, input(cb_weight, None, None, Block, …, TileAddressing::Offset), DestReuseType::DEST_TO_SRCA>{w, head_off·4},] PackTile<output(cb_normed or cb_output_tiles)>{…})` | `eltwise/api/chain.hpp` (BinaryFpu / DestReuseBinary / BroadcastDim / InputTileMapping::Col docs) | — | `cb_input_tiles`, `cb_rstd`, `cb_weight` | `cb_normed` / `cb_output_tiles` | `bh` |
| rotate_pairs_block | **raw_api** | `matmul_init(cb_normed, cb_trans_mat, cb_rotated)` once per block, then per tile `matmul_tiles(cb_normed, cb_trans_mat, i, 0, dst)` + `pack_tile` under `tile_regs_*`, packing up to `DEST_AUTO_LIMIT` tiles per acquire | `tt_metal/hw/inc/api/compute/matmul.h:113,172`; `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:103` | in1 = single T tile, index 0 | `cb_normed`, `cb_trans_mat` | `cb_rotated` | `BT` |
| apply_rope_block | helper | `eltwise_chain(IterationShape::grid(bh, HEAD_TILES), BinaryFpu<Mul, input(cb_normed,…Block), input(cb_rope_cos,…, InputTileMapping::Row), Dst::D0>{y, cos}, BinaryFpu<Mul, input(cb_rotated,…Block), input(cb_rope_sin,…, InputTileMapping::Row), Dst::D1>{rot, sin}, AddBinary<Dst::D0, Dst::D1, Dst::D0>{}, PackTile<output(cb_output_tiles)>{out})` | `eltwise/api/chain.hpp` (InputTileMapping::Row: "same tile-row re-read for every output row"); `eltwise/binary/sfpu/basic.hpp:15-20` (AddBinary) | — | normed, rotated, cos, sin | `cb_output_tiles` | `bh` |

**Helpers considered and rejected for `rotate_pairs_block` (raw matmul):**

| Candidate | Mismatch | Citation |
|-----------|----------|----------|
| `matmul_block_helpers.hpp` (`matmul_block`) | Does not exist on this branch — `ttnn/cpp/ttnn/kernel_lib/` contains no matmul helper (`grep -rl matmul ttnn/cpp/ttnn/kernel_lib` hits only tilize/reduce internals) | `ls ttnn/cpp/ttnn/kernel_lib/` |
| `eltwise_chain` elements | No chain element performs a cross-column (intra-tile) permutation; BinaryFpu/SFPU binaries are element-wise and `UnaryBcast` only replicates row/col 0 | `eltwise/api/chain.hpp` §5 element list; `broadcast/bcast.hpp:26` |

Gap to close: a `matmul_block` helper with `MatmulBlockShape::of(Mt=BT, Kt=1, Nt=1)` would express this directly (in0 = BT tiles stacked along M, in1 = T). Realize as one block operation: init once per block, not per tile.

Post-op raw calls inside `reduce` are the helper's designed extension point (its own `@example` uses raw `recip_tile` there), not a bypass.

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| expand_weight_rows | copy-bcast | `cb_weight_row`: Row0 (rows 1–31 garbage by design) | — | Row |
| sum_squares | mul (acc) | `cb_input_tiles`: All | `cb_input_tiles`: All | None |
| rsqrt_mean | reduce row | `cb_sumsq`: All | `cb_reduce_scaler`: per reduce-scaler layout | — (output `cb_rstd`: Col0) |
| normalize | mul | `cb_input_tiles`: All | `cb_rstd`: Col0 | Col |
| normalize (weight) | DEST-reuse mul | DEST: All | `cb_weight`: All (expanded) | None |
| rotate_pairs | matmul | `cb_normed`: All | `cb_trans_mat`: All | — |
| apply_rope | mul ×2, SFPU add | `cb_normed` / `cb_rotated`: All | `cb_rope_cos` / `cb_rope_sin`: All (tile index by column via Row mapping) | None |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| ELWMUL with `acc_to_dest` | `sum_squares_block` relies on FPU mul accumulating into D0 across the 4 tiles of a head | Chain supports it (`chain.inl:1394-1397`); if it misbehaves, fallback: square into a `BT`-tile scratch (`BinaryFpu<Mul,x,x>`) and `reduce` over `of(bh, 4)` — same CB budget as `cb_normed` |
| Row-0-only weight reads | L1 rows 1–31 of `cb_weight_row` pages hold stale bytes | Only `BroadcastDim::Row` consumes them (row 0 only). Fallback define: full-tile read (cost: +7 MB DRAM at R=2304) |
| `cb_weight` indexing per block | DestReuse operand must be the block's heads, not tile 0 | `TileAddressing::Offset` base = `(first head of block − head_start)·4`; `cb_weight` stays waited, never popped mid-kernel |
| cos/sin lifetime across blocks of one row-tile | multiple blocks per row-tile when `heads_this_core > block_heads` | compute waits cos/sin at first block of the row-tile and pops after the last; chain uses caller-managed (None/None) policy for them |
| x reused across three phases | sum_squares and reduce must not pop x | sum_squares: pop None; normalize pops `BT` at end |
| Same CB as both FPU operands (x·x) | chain dedups B-side wait/pop | identical mapping/policy on both sides (static_assert in `chain.inl`) |
| Reduce scaler | SUM with factor 1/128 is non-standard | `prepare_reduce_scaler<…, SUM, REDUCE_ROW>(1/128)`, bf16 CB; never popped by reduce — compute pops once at kernel end |
| rsqrt accuracy | must be accurate to bf16 round-off | `rsqrt_tile<Default>` (not Fast), math_approx_mode honored from config but default False; ms computed in fp32 DEST |
| Precision with `fp32_dest_acc_en=False` | intermediates become bf16 | `DF` follows config (ledger audit 2); accepted per spec |
| Variant explosion | 4 (weight × rope) combos | compile-time defines `HAS_WEIGHT`, `HAS_ROPE`; the no-rope path packs `normalize_block` straight into `cb_output_tiles` |
| Uneven split | Rt not divisible by Rg; H not by G | per-core RT args `row_start, num_rows, head_start, num_heads`; `blocks_per_row` derived per core; tail block has `num_heads mod bh` heads (all phases take bh as a runtime extent) |
| Small shapes | R=32, W=128 → 1 core, 1 head | selection returns G=Rg=1; covered by acceptance test |
