# Operation Design: mhc_pre

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (DRAM-bound streaming reduction + cross-core combine + per-token coefficient math + weighted stream mix) |
| Goal | One device program computes, per token row x (length n·C): RMS statistic, 24-wide projection, pre / post / Sinkhorn(comb) coefficients, and the y-mix `y = Σ_i pre_i · x_i`. Every element of X crosses DRAM exactly once. |
| Math | `r = rsqrt(Σ x² / (nC) + norm_eps)`; `mix = (x @ W) · r`; `pre = σ(a_pre·mix[0:n] + b[0:n]) + eps`; `post = 2σ(a_post·mix[n:2n] + b[n:2n])`; `L = a_res·mix[2n:] + b[2n:]` (n×n, `L[i][j] = [i·n+j]`); `comb = Sinkhorn(L)`; `y = Σ_i pre[i] · x[i·C:(i+1)·C]` |
| Sinkhorn | `m = softmax_j(L) + eps` (row-max subtracted); `m /= (colsum(m) + eps)`; `(iters-1)×{ m /= (rowsum(m)+eps); m /= (colsum(m)+eps) }` — this exact order |
| Mode | Hybrid: helpers (`matmul_block`, `eltwise_chain`, `reduce`, `mcast_pipe`) plus one custom SFPU block operation (coefficients + Sinkhorn) and one custom fp32-exact combine block operation |
| References | `models/demos/deepseek_v3_d_p/reference/mhc/mhc_reference.py` (torch truth); `eval/golden_tests/mhc_pre/helpers.py` (`pytorch_mhc_pre`, `sinkhorn_knopp`, tolerances, doubly-stochastic gate); `tt_metal/tt-llk/tt_llk_blackhole/common/inc/sfpu/experimental/ckernel_sfpu_sinkhorn.h` (SFPU lane layout, mechanics only); `ttnn/ttnn/operations/examples/master.md` (perf catalog); `.claude/references/blocking-model.md`, `l1-footprint-discipline.md` |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` (X) | ttnn.Tensor | yes | rank 2/3/4, last dim n·C, C % 32 == 0 | — | buffer address RT |
| `proj_weight` (W) | ttnn.Tensor | yes | shape (n·C, n(n+2)); n(n+2)=24 → n=4 | — | buffer address RT |
| `proj_bias` (b) | ttnn.Tensor | yes | shape (1, n(n+2)), float32 TILE | — | buffer address RT |
| `scale` | tuple[float,float,float] | yes (kw) | finite | — | RT (fp32 bits) to compute |
| `sinkhorn_iters` | int | no | ≥ 1 | 20 | RT to compute (loop count) |
| `eps` | float | no | > 0 | 1e-6 | RT (fp32 bits) to compute |
| `norm_eps` | float | no | > 0 | 1e-6 | RT (fp32 bits) to compute |
| `compute_kernel_config` | ttnn.ComputeConfigDescriptor | no | fp32_dest_acc_en must be True | `default_compute_kernel_config()` (HiFi4, fp32 DEST, approx off) | compute kernel config; math_fidelity / math_approx_mode honoured |

Derived (host): `n` from `W.shape[-1]` (solve n(n+2)=mix; ValueError if none); `C = X.shape[-1] // n`; `inv_nc = 1/(n·C)` (fp32 bits, RT).

## Tensors

### Input

| Property | X | W | b |
|----------|---|---|---|
| Shape | (..., T, n·C), rank 2–4 | (n·C, n(n+2)) | (1, n(n+2)) |
| Dtype | float32 (Phase 0); bfloat16 (TARGET) | float32 (Phase 0, tf32-representable by contract); bfloat16 (TARGET) | float32 |
| Layout | TILE | TILE | TILE |
| Memory | DRAM interleaved | DRAM interleaved | DRAM interleaved |

### Output

| Property | y | post | comb |
|----------|---|------|------|
| Shape | (..., T, C) | (..., T, n) | (..., T, n·n), `comb[..., i·n+j] = comb[i][j]` |
| Dtype | X's dtype | float32 always | float32 always |
| Layout | TILE | TILE | TILE |
| Memory | DRAM interleaved | DRAM interleaved | DRAM interleaved |

Tile-space geometry (alignment-aware from day 1, per-image padding):

| Symbol | Definition |
|--------|-----------|
| `tensor_token_tiles` (Mt) | `prod(X.shape[:-2]) · ceil(X.shape[-2] / 32)` (rank 2: `ceil(T/32)`) — each image is tile-padded independently, never `floor(total_rows/32)` |
| `tensor_c_tiles` (Ct) | `C / 32` (exact by contract) |
| `tensor_k_tiles` (Kt) | `n · Ct` |
| X tile (m, i, c) | page `m·Kt + i·Ct + c` |
| W tile (i, c) | page `(i·Ct + c) · 1` (W is 1 tile wide: 24 of 32 columns valid) |
| y tile (m, c) | page `m·Ct + c` |
| post / comb tile m | page `m` (1 tile wide each; cols ≥ n resp. ≥ n² written as 0) |

## Blocking Model

### Axes

| Axis | Character (+ reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|---------------------|-------------|---------------|-------------|-----------------|--------------|
| token tile-row `m` | **independent** — every statistic, coefficient and y value is per token row | `block_token_tiles` | `min(core_token_tiles_max, floor((l1_budget − fixed_bytes) / per_token_tile_bytes))` (≥1; see Mechanism caps / selection function) | host constant, passed as CT arg to all kernels | split across **groups** with `split_work_to_cores`-style balance (`core_token_tiles` = ceil/floor classes); every group loops `num_blocks_this_core` blocks | knob-turn |
| stream-column `c` (C tiles of one stream) | **dependent** for the projection and Σx² (results span all nC columns); **independent** for y (y[c] needs only column c of each stream) | `core_c_tiles` (whole per-rank slice, not sub-blocked) | `ceil/floor(Ct / group_cores)` per rank | host, RT arg per core | split across the **ranks of a group** (`group_cores` cores); partial sums combined on device (gather → fp32 sum → mcast) | built (this is the Phase-0 scheme-change the rules require) |
| stream `i` (n) | **dependent** — both the projection K and the y-mix sum span it | `n` (whole) | n (=4) | derived from W shape | **not split**: every rank holds all n streams of its C slice → the y-mix stays core-local (no second combine) | splitting it would add a y combine: rejected (no traffic benefit) |
| mix column (n(n+2)=24 → 1 tile) | **independent**, 1 tile wide | whole (1 tile) | 1 | constant of W geometry | not split (one output tile per token row) | none |
| per-token n×n matrix (i, j) of the Sinkhorn | **dependent within a token** (row / column sums) | whole n×n, held lane-wise | n² slots | derived | never crosses a lane: token = SFPU lane in the coefficient-major tile (below) | none |
| token-within-tile-row (32 lanes) in the coefficient stage | **independent** | 32 (whole) | 32 | tile geometry | 32 tokens processed as one SIMD vector per slot | none |

**Axes of the intermediate stages** (re-running the table on data the scheme creates):

| Stage data | Axes | Assignment decision |
|-----------|------|--------------------|
| per-rank partial pair (mix partial + Σx² partial), per token tile-row | token (independent), rank (dependent: summed) | gathered to the group **root**; the sum is `group_cores−1` fp32 tile adds per token row — the only stage on one core, and it is O(group_cores) tiles vs O(Kc) matmul tiles per rank. The root is otherwise a normal worker (it owns a C slice). |
| combined tile pair S (per token tile-row) | token (independent) | mcast to every rank; **every rank computes r / mixes / pre / post itself** (≈ a dozen SFPU vector ops per slot per 32 tokens — cheaper than a second round trip) |
| Sinkhorn + post/comb output (per token tile-row) | token (independent) | **spread round-robin over the group**: group-local token row `j_local` is owned by rank `j_local mod group_cores`; only the owner runs the Sinkhorn and writes post/comb. Not left on the root. |

### Group geometry (core-assignment knobs, host derivation — single source)

```
grid_x, grid_y        = device.compute_with_storage_grid_size()          # runtime query, never a literal
group_w               = min(grid_x, Ct)                                   # ranks along a physical row
group_h               = 1 if Mt >= grid_y else max(1, min(grid_y // Mt, Ct // group_w, group_cores_cap // group_w))
group_cores           = group_w * group_h                                 # G_k
groups_x, groups_y    = grid_x // group_w, grid_y // group_h
num_groups            = groups_x * groups_y                               # G_t
rank (within group)   = (y - gy0) * group_w + (x - gx0); root = rank 0 (top-left core of the group rectangle)
core_c_tiles[rank]    = Ct split over group_cores (first Ct % G_k ranks get ceil, rest floor)
c_start[rank]         = prefix sum of core_c_tiles
core_token_tiles[g]   = Mt split over num_groups (first Mt % G_t groups get ceil); groups with 0 rows are idle
core_k_tiles_max      = n * ceil(Ct / group_cores)
```

`group_cores_cap` = 32 (host constant; Mechanism caps). Example, Blackhole p150 11×10 grid:

| Shape (T, C) | Mt | group_w × group_h | G_k | G_t | busy cores | core_k_tiles_max | core_token_tiles | block_token_tiles (fp32) | blocks/core |
|---|---|---|---|---|---|---|---|---|---|
| 640, 7168 | 20 | 11×1 | 11 | 10 | 110 | 84 | 2 | 1 | 2 |
| 640, 1792 | 20 | 11×1 | 11 | 10 | 110 | 24 | 2 | 2 | 1 |
| 1280, 4096 | 40 | 11×1 | 11 | 10 | 110 | 48 | 4 | 2 | 2 |
| 4096, 7168 | 128 | 11×1 | 11 | 10 | 110 | 84 | 13/12 | 1 | 13/12 |
| 256, 6144 | 8 | 11×1 | 11 | 10 | 88 | 72 | 1 | 1 | 1 |
| 64, 256 (C=256) | 2 | 8×1 (`Ct // group_w = 1` caps group_h: every rank ≥ 1 C tile) | 8 | 10 | 16 | 4 | 1 | 1 | 1 |
| 128, 1024 (rank 3) | 4 | 11×2 | 22 | 5 | 88 | 8 | 1 | 1 | 1 |
| 1, 7168 (decode) | 1 | 11×2 | 22 | 5 | 22 | 44 | 1 | 1 | 1 |
| 32, 32 (C=32) | 1 | 1×1 | 1 | 110 | 1 | 4 | 1 | 1 | 1 |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x_resident` | `x_block_depth` | 2 (falls back to 1 only if the L1 predicate fails at `block_token_tiles = 1`) | Reader prefetches X block b+1 from DRAM while block b sits in the combine round trip (the stall shadow) and in the y-mix. DRAM never idles on the combine. |
| `cb_y_out` | `y_depth` over a `y_chunk_tiles` window | `y_depth = 2`, `y_chunk_tiles = min(core_c_tiles_max, 8)` | Writer drains y while compute produces the next chunk. The catalog (`double_buffer`) finds 4–8 reads in flight per barrier saturates. |
| `cb_partial` | none (one block) | 1 block (2·`block_token_tiles` pages) | The writer sends a block's partials in one transfer. A second slot buys nothing: the next partial waits on the combine anyway. |
| `cb_gathered` | none (one block) | 1 block | Flow control is implicit in the writer's order (see Dataflow Strategy: gather-slot reuse invariant). |
| all other CBs | none | 1 block or 1 tile | They are single-shot per block, or resident. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| `matmul_block` in0 retention: `InputPolicy::WaitAndRetainOnLastBlock` keeps in0 only on the last K-block. NoWaitNoPop is illegal for in0 (matmul_block_helpers.inl:115-120, :537) | K-block of the projection must be the **whole** `core_k_tiles` (`in0_block_k = core_k_tiles`, `num_k_blocks = 1`) | fixed: `num_k_blocks = 1` | Earlier K-blocks get popped, so the y-mix reads freed or overwritten X tiles (silent wrong y). |
| `matmul_block` output subblock `sb_h·sb_w ≤ DEST_AUTO_LIMIT` (matmul_block_helpers.inl:158; 4 tiles in fp32 half-sync, dest_helpers.hpp:89-103) | `out_subblock_h ≤ DEST_AUTO_LIMIT`, `out_subblock_w = 1` | `out_subblock_h` = the largest divisor of `block_token_tiles` ≤ `DEST_AUTO_LIMIT`. This is the tuner's choice, not a design knob. | ASSERT / DEST overflow |
| Flat-root gather: `cb_gathered` holds `group_cores·2·block_token_tiles` tiles | `group_cores` | `group_cores ≤ group_cores_cap = 32` (host) | L1 on every core grows linearly with group size (e.g. 880 KB at 110 ranks). The root sum becomes the serial stage. |
| L1 residency: `x_block_depth·block_token_tiles·core_k_tiles_max·x_tile_bytes + core_k_tiles_max·w_tile_bytes + …` must fit (see `l1_ledger.md` total) | `block_token_tiles`, `x_block_depth` | Selection function: `block_token_tiles ← min(core_token_tiles_max, floor(avail / per_token_tile_bytes))`. If that is < 1, set `x_block_depth = 1` and retry. If still < 1, double `group_h` (while `group_cores ≤ cap`, `group_h ≤ grid_y`). If still < 1, `RuntimeError` (unreachable for TARGET: worst case C=7168 fp32, 11-wide group → 1.30 MB). | CB allocation failure / L1 overlap with the kernel stack |
| Coefficient-major tile: 32 SFPU slots × 32 lanes per tile | slots used: n(n+2)+1 = 25 ≤ 32; lanes = 32 tokens | n fixed by W (n=4 → 25 slots). Host asserts `n*(n+2)+1 <= 32`, i.e. n ≤ 4. Larger n is a `ValueError`: outside TARGET. | Slot overflow corrupts other coefficients |
| Monotonic gather semaphore (32-bit counter, waits `≥ group_cores·(block_idx+1)`) | blocks per call | `num_blocks_this_core · group_cores < 2³²` — always true | wrap → early release |
| `sinkhorn_iters` loop inside one SFPU call | iters | RT arg, ≥ 1 (validated host-side) | iters = 0 would skip the mandatory first column normalisation |

### Regimes

| Regime | Status | Predicate | Block | Data movement (vs. minimum: each input crosses DRAM once, each output once) | What a bigger block buys |
|--------|--------|-----------|-------|--------------------------------------------------------------------------|--------------------------|
| **R1 `group_ksplit_resident`**: 2-D grid. Token tile-rows go to groups; each group splits nC by stream-column slice across ranks; a rank's X block stays resident from projection to y-mix; partials go to the group root (fp32 SFPU sum) and are mcast back; every rank computes pre itself; the Sinkhorn is owned round-robin. | **built** | every shape where the L1 predicate holds with `block_token_tiles ≥ 1`. That is all of TARGET / INPUTS on BH and WH. | `block_token_tiles × core_k_tiles` (X, resident) → `block_token_tiles × 1` (partials) → `block_token_tiles × core_c_tiles` (y) | X: **once** (minimum). y / post / comb: once. W: **once per group** (G_t× — above minimum, see R2). b: once per core (1 tile). NoC: `Mt·G_k·2` partial tiles gathered, `Mt·2` tiles mcast to `G_k−1` receivers each. | Per block: one gather + one mcast round trip (≈ one semaphore wait + NoC latency), one `matmul_block` init/reconfig, one reduce init, one coefficient-op SFPU init, one pipeline fill of the combine. Intended frequency: all of these **once per block**. `mm`/`reduce` hw_startup once per kernel. |
| **R2 `group_ksplit_resident + W column broadcast`**: the same, but W slice `(rank r)` is read from DRAM once by the rank-r core of group 0 and multicast to rank r of every other group (the same physical column when `group_w = grid_x`) | **built (Refinement 3)**, gated to `group_h == 1` with ≥ 2 full active group rows (a physical column then holds one rank); otherwise R1. `Mcast1D(PerColumn)` on the reader NoC, Counter data-ready, no handshake (the `cb_weight` landing is write-once). Originally deferred: it is a stepping-stone successor (blocking-model §5): it only changes who fills `cb_weight` (reader: DRAM read → `ReceiverPipe`/`SenderPipe` over `Mcast1D(PerColumn)`), and compute, CB layout and the combine are untouched. Phase 0 keeps a single cross-core mechanism (the group combine) so its handshake is validated in isolation first. | same as R1 | same | W DRAM crossings G_t → 1. At T=640, C=7168: W fp32 3.67 MB × 10 = 36.7 MB → 3.67 MB (−33 MB DRAM, ≈ −30 % of total bytes for fp32 X, ≈ −45 % for bf16 X). Adds a one-shot 3.67 MB column mcast at kernel start. | n/a (one-shot) |
| **R3 `token_row_only`**: one core per token tile-row, whole nC per core | **rejected** — superseded by R1 everywhere. The rules forbid it: at T=640 only 20 cores are busy. `core_k_tiles = 896` tiles (3.5 MB fp32) cannot be resident, so it forces R4's re-read. | — | `1 × Kt` | X twice (or K-chunked re-read), ≤ Mt cores | — |
| **R4 `two_pass_reread`**: pass 1 streams X for Σx² and the projection; pass 2 re-reads X for the y-mix after `pre` exists | **rejected** (dead end) — superseded by R1. R1 reaches read-once for every TARGET shape because the resident unit is one token block of one rank's K slice, not a whole row. R4 is built *instead of* R1, not on top of it. | — | streaming window | X crosses DRAM **2×** (+73 MB at T=640 C=7168 fp32; +587 MB at T=5120) | — |
| **R5 `allgather_no_root`**: every rank mcasts its partial pair to the whole group; every rank sums all partials in rank order | **rejected** — superseded by R1. It replaces one gather + one mcast with G_k mcasts and G_k² tile deliveries per token row (121 vs 32 at G_k=11), and needs a G_k-slot landing on every core. The root sum it saves is ~G_k SFPU tile adds. Reachable by swapping only the combine block op if a measurement ever says otherwise. | — | — | NoC ×≈G_k/2 | — |
| **Sharded placements** | not in TARGET (`memory_layout` is absent from feature_spec TARGET) | — | — | — | — |

Selection: single built regime. Knob selection has branches (`group_h > 1`, `group_cores = 1`, `block_token_tiles > 1`, `x_block_depth = 1` fallback), so the acceptance test pins shapes that hit each reachable branch on an 11×10 grid (see Work Distribution).

### Traffic ranking

Candidate splits, bytes per tier at T=640, C=7168, fp32 X/W (X = 73.4 MB, W = 3.67 MB, y = 18.4 MB):

| Rank | Split | DRAM | Cross-core NoC | Occupancy | Verdict |
|------|-------|------|----------------|-----------|---------|
| 1 | tokens × stream-column (R1) + W column broadcast (R2) | X + W + y ≈ 95.5 MB (minimum) | partials 20·11·2 tiles = 1.8 MB gathered; S 20·2 tiles × 10 receivers = 1.6 MB mcast; W mcast 3.67 MB × 9 receivers | 110 | cheapest; R2 deferred |
| 2 | tokens × stream-column (R1, built) | X + 10·W + y ≈ 128.5 MB | 3.4 MB | 110 | **implemented** |
| 3 | stream-column only (one group = whole grid, loop all Mt) | X + W + y ≈ 95.5 MB | partials 20·110·2 tiles = 17.6 MB, **all into one root** (serialized ingress); S × 109 receivers | 110 | rejected on the root's NoC ingress and serial sum (concentration, not total bytes) |
| 4 | tokens only (R3) | X ≥ 2× (does not fit) | 0 | 20 | rejected |
| 5 | tokens × stream `i` (split the n streams) | as rank 2 | adds a y combine of `C`-wide partials (≈ y bytes again on NoC) | 110 | rejected: a second, larger combine for no DRAM saving |

The dependent axis split (rank 2) was chosen over the independent-only split (rank 4) on DRAM bytes: splitting nC across ranks shrinks each core's slice to `core_k_tiles`. That makes the slice resident and is what turns a re-read of X into a combine of 2 tiles per token row per rank. Traffic for the combine is ~2 % of X.

Operand-reuse check (mechanical):

| Operand | Varies along the token (group) split? | Varies along the stream-column (rank) split? | Consequence |
|---------|---------------------------------------|---------------------------------------------|-------------|
| X | yes | yes | none |
| W | **no** | yes | reuse-shared by construction of the token split → broadcast row R2 (deferred) |
| b (1 tile) | no | no | reuse-shared. Accepted as direct reads: 110 × 4 KB = 0.44 MB, 0.6 % of X. It could ride R2's mcast. |

Stall-shadow check:

| Waiting stage | What it waits for | Work scheduled into the stall | Legality |
|---------------|------------------|------------------------------|----------|
| rank's writer / compute after sending partial(b) | root's S(b) mcast | **reader** keeps streaming X block b+1 (`x_block_depth = 2`), so DRAM stays busy through the round trip | independent data; no FP reorder |
| rank's y-mix(b) | pre(b) | nothing else on compute depends only on data present. proj(b+1) could run here (perf lamp L3). | — |
| owner's Sinkhorn(b) | nothing (it is the waiter's successor) | **moved after `ymix_block(b)`**, so the y-mix (the next block's critical path) is not delayed by 40 row/col normalisations. comb is independent of y. | no FP-order change |

### Block schedule

Logical schedule, per core (reader / compute / writer realize their parts asynchronously):

```cpp
load_resident_constants();              // once: W slice -> cb_weight, bias -> cb_bias_coef (coefficient-major), scaler
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    load_x_block(block_idx);            // reader, prefetches block_idx+1 while block_idx is in flight
    project_block(block_idx);           // mixes partial   = X_blk @ W_slice
    sumsq_block(block_idx);             // Σx² partial     = rowsum(Σ_k x_k ⊙ x_k)
    send_partial_block(block_idx);      // writer: partial pair -> root gather slot [rank]
    combine_block(block_idx);           // root only: S = Σ_rank partial (fp32, rank order)
    broadcast_combined_block(block_idx);// root writer: mcast S to group; every writer scatters S -> coefficient-major tile
    coefficients_block(block_idx);      // r, mixes, pre, post, logits (lane-wise fp32 SFPU)
    expand_pre_block(block_idx);        // writer: pre slots -> n column tiles; post -> DRAM (owner rows)
    ymix_block(block_idx);              // y = Σ_i x_i ⊙ bcast_col(pre_i); pops the resident X block
    store_y_block(block_idx);           // writer
    sinkhorn_block(block_idx);          // owner rows only; after ymix (stall-shadow reorder)
    store_comb_block(block_idx);        // writer, owner rows only
}
```

**Perf 2 — group-width selection and Sinkhorn.**
- **Group width (bf16 X, `Mt >= grid_y`).** The width is no longer "fewest blocks, widest among those".
  - Every `group_w` that fits L1 is a candidate. A plan at depth < 2 with more than one block goes last, and the
    rest are ranked by `_block_schedule_cost`, which models the critical path, pipeline included.
  - The model counts three things: the rank's X blocks streaming back to back, one exposed round trip + y-mix /
    y write for the last block, and each middle step's round trip minus what `pipe_at` hides under that block's K.
  - Example (BH p150): 640×7168 bf16 now runs 11×1 groups with 2 blocks at depth 2 (it was 5×1, 1 block,
    depth 1).
- **Sinkhorn.** `sinkhorn_block`'s iterations are hand-scheduled `SFPLOADMACRO` passes. They are bit-identical to
  the plain formulation with fused Newton MADs, and `c_owned` drops from 6.7 to 4.5 µs.
- Measurements are in changelog Perf 2.

**Perf 1 — cross-block pipeline (implemented schedule).** The loop above is the logical per-block order. The
kernels now software-pipeline the step at block b when `pipe_at(b)` holds: `b + 1 < num_blocks` and
(`x_block_depth ≥ 3` or `b + x_block_depth ≥ num_blocks`). At the default depth 2 only the last step is pipelined;
holding X(b) longer on earlier steps delays the reader's prefetch of X(b+2), and that measured slower. Depth-1 plans
are always serial (X(b+1) cannot be resident next to X(b)).
- Compute, pipelined step: proj + Σx²(b+1), reading X(b+1) behind X(b) at a wrap-aware page offset; root: the fold
  of b if not done yet; coefficients + Sinkhorn(b); root: the fold of b+1; y-mix(b).
- Writer, pipelined step: [S(b)]; send P(b+1); root: gather + multicast S(b+1); y(b), post/comb(b); non-root:
  receive S(b+1).
- A rank sends P(b+1) only after S(b) landed, so the root has already folded gather slot b (slot-reuse invariant).
- `cb_coef_in` holds 2 blocks, and the group multicast has no consumer-ready handshake (Flag data-ready; a Counter
  hangs in the send's atomic barrier on the looped-back root copy).
- Measured (BH p150): 1280×4096 bf16 X / fp32 W 147.5 → 138.4 µs (see changelog Perf 1).

| Operation | Block shape it acts on | Resident across it | Intended frequency of fixed costs |
|-----------|-----------------------|--------------------|-----------------------------------|
| `load_resident_constants` | `core_k_tiles × 1` (W), 1 tile (bias), 1 tile (scaler) | — | once per kernel |
| `load_x_block` | `block_token_tiles × core_k_tiles` tiles, K order `(c major, i minor)` | `cb_weight` | one NoC read burst per block, barrier per ≤ 8 tiles in flight |
| `project_block` | in0 `block_token_tiles × core_k_tiles`, in1 `core_k_tiles × 1`, out `block_token_tiles × 1` | X block **retained** (in0 `WaitAndRetainOnLastBlock`, num_k_blocks=1), W retained forever | one matmul init/reconfig per block |
| `sumsq_block` | per block: `block_token_tiles` rows × `core_k_tiles` tiles → Q (one tile per row, DEST-accumulated), then one row-reduce per row | X block retained | one FPU-mul init + one reduce init per block |
| `send_partial_block` | `2·block_token_tiles` tiles | — | one NoC write + one semaphore inc per block |
| `combine_block` (root) | `group_cores × 2·block_token_tiles` → `2·block_token_tiles` | — | one copy/add-binary init per block |
| `broadcast_combined_block` | `2·block_token_tiles` tiles | — | one mcast (with handshake) per block |
| `coefficients_block` | `block_token_tiles` coefficient-major tiles (32 tokens × 25 slots each) | `cb_bias_coef` | one SFPU init per block |
| `expand_pre_block` | `block_token_tiles` coefficient tiles → `n·block_token_tiles` pre-column tiles | — | once per block |
| `ymix_block` | `block_token_tiles × core_c_tiles` output tiles, each an n-deep DEST accumulation | pre-column tiles | one FPU bcast-mul init per block. DEST accumulates the n streams without packing intermediates. |
| `store_y_block` | `block_token_tiles × core_c_tiles` tiles in `y_chunk_tiles` windows | — | one barrier per window |
| `sinkhorn_block` | owned rows of the block (≤ `block_token_tiles` coefficient tiles), all `sinkhorn_iters` inside one SFPU call | — | one SFPU init per block. Iterations stay in DEST/LREGs with no pack in between. |
| `store_comb_block` | owned rows | — | once per owned row |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 **overlap** | Coarsest `block_token_tiles` can make `num_blocks_this_core = 1` (e.g. T=640 C=1792: block 2 = whole core share). Then nothing overlaps: the whole X read precedes the matmul (in0 waits the full block), and the combine, y-mix and y-write follow serially. | `block_token_tiles = ceil(core_token_tiles / x_block_depth)`. Also `x_block_depth = 3` for small C, where one block's read (~13 µs at C=1792 bf16) is close to the round trip. |
| L2 **grid synchronization** | Flat root gather + mcast per block. For small C, per-rank work is tiny (`core_k_tiles` = 24 at C=1792) and the round trip may dominate. On a busy 2-D grid the catalog's `tensix_all_reduce` measured tree-reduce + mcast 1.45–1.60× over flat root. | `tree_reduce_mcast` combine (swap only `combine_block` / `send_partial_block`). Alternatively a narrower group (`group_w < grid_x`: fewer ranks, more groups). |
| L3 compute skew | Compute idles during the round trip. Harmless while DRAM-bound (the reader keeps streaming), but not if a shape turns compute- or latency-bound. | Issue `project_block(b+1)` + `sumsq_block(b+1)` before `coefficients_block(b)`: `cb_x_resident` becomes two alternating one-block CBs so both blocks are addressable from the front. |
| L4 NoC assignment | Reader on NoC0 streams X; the writer carries the gather, mcast and y writes on NoC1 (catalog `noc_placement`: reads NoC0 / writes NoC1 is 2.5–4.8× faster). Refinement 5: the top `round(0.4·grid_y)` rows swap the two NoCs (NoC0 starves them), and with the W column all-gather the W share rides the reader's NoC; path-gated off where it measured slower (bf16 X / bf16 W; bf16 W without the broadcast). | The mcast on NoC0 vs NoC1 (mcast rectangle routing direction; catalog `tensix_all_reduce_ring_transport`: 6× by direction). |
| L5 y-mix unit | FPU bcast-col mul with an n-deep DEST accumulation (catalog `compute_fusion`: never use SFPU for FPU work) | Only if a fp32-y accuracy refinement asks for an exact y: SFPU mul-add from `UnpackToDestFp32` copies. |

Catalog entries used: `double_buffer` (`x_block_depth`, `y_chunk_tiles` 4–8 in flight), `noc_placement` (reader NoC0 / writer NoC1), `width_split` (split the wide axis across cores), `tensix_all_reduce` (flat root at ≤ 32 ranks, pull/push, uneven-split slot stride), `tensix_all_reduce_compute` / `row_reduce_accumulate` (DEST-accumulate then reduce once for Σx²), `compute_block_size` (one pass per block, no redundant reconfig), `sfpu_tile_scope` (the coefficient op touches only the 25 used slots of 32), `shared_input_reuse` / `mcast_topology` (R2's mechanism).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| W slice DRAM → `cb_weight` | W dtype tiles | reader, TensorAccessor, once | K order p = `c·n + i` ↔ W page `i·Ct + c_start + c` |
| bias DRAM → `cb_bias_coef` | fp32 | reader reads the bias tile into its own scratch page, then writes b_k into every lane of coefficient slot k (k < 24), 0 elsewhere | one-time scalar scatter (768 stores) |
| scaler | bf16 | `prepare_reduce_scaler<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f)` | once |
| X DRAM → `cb_x_resident` | X dtype tiles | reader, TensorAccessor, `block_token_tiles × core_k_tiles` per block | L1 slot for (row t, c, i) = `(t·core_k_tiles + c·n + i)`. This one K order serves both the matmul (matches `cb_weight`) and the y-mix (`Block` mapping on `grid(core_c_tiles, n)`). |
| compute → `cb_partial` → root gather slot | fp32 | writer: `noc_async_write` of `2·block_token_tiles` tiles to `root_noc(cb_gathered_base + rank·2·block_token_tiles·4096)`, `noc_async_write_barrier`, `noc_semaphore_inc(root, sem_gather_arrived, 1)`. The root sends to itself the same way (local NoC address). | Push model. `cb_gathered` is allocated on every core so its L1 address is identical on all cores: senders use their own base address. |
| root: gather slots → `cb_gathered` (CB credit) | fp32 | root writer: `cb_reserve_back`, `noc_semaphore_wait_min(sem_gather_arrived, group_cores·(block_idx+1))`, `cb_push_back(2·group_cores·block_token_tiles)` | monotonic counter, never reset (no reset race) |
| root: `cb_gathered` → `cb_combined` | fp32 | compute `combine_block` (rank order, fp32 SFPU adds) | fixed order → bitwise-deterministic |
| root → group: `cb_combined` | fp32 | root writer: `SenderPipe::send(src = cb_combined rd ptr, dst = same address, 2·block_token_tiles·4096)` over the group rectangle, root excluded (`Mcast2D`, `sender_in_rect`), `PRE_HANDSHAKE=true`. Non-roots: writer `ReceiverPipe::receive(block_idx)`. | `cb_combined` is compute→writer on the root and an mcast landing on non-roots. It is one allocation with two disjoint per-core roles. |
| S → coefficient-major tile → `cb_coef_in` | fp32 | writer scalar scatter (below) | 25 slots × 32 lanes = 800 moves per token tile-row |
| `cb_coef_out` → `cb_pre_cols` | fp32 | writer: slot i (i < n) → column 0 of pre-column tile i (32 values) | 128 moves per token tile-row |
| `cb_coef_out` → post DRAM (owner rows) | fp32 | writer: slots n..2n−1 → `cb_out_stage` post tile (rows = tokens, cols 0..n−1, other cols stay 0) → DRAM page m | |
| compute → `cb_y_out` → y DRAM | y dtype | writer, `y_chunk_tiles` window, page `m·Ct + c_start + c` | |
| `cb_comb_coef` → comb DRAM (owner rows) | fp32 | writer: slots 2n + i·n + j → comb tile col i·n+j → DRAM page m | |

**Coefficient-major ("SoA") tile.** A 32×32 fp32 tile viewed as 32 SFPU vectors (`slot` k = the 32 datums `dst_reg[k]` / `SFPLOAD` address 2k touches) × 32 lanes. Lane l = token row l of the token tile-row. Every per-token computation is then purely lane-wise: no cross-lane shuffles, full fp32. From the SFPLOAD lane layout (`ckernel_sfpu_sinkhorn.h:14-18`: Row = (Addr & ~3) + Lane/8, Col = (Lane & 7)·2 + ((Addr & 2) ? 1 : 0), face base address 16·f), the L1 datum index of (slot k, lane l) is:

```
face       = k >> 3
face_row   = 4 * ((k & 7) >> 1) + (l >> 3)
face_col   = 2 * (l & 7) + (k & 1)
datum_idx  = face * 256 + face_row * 16 + face_col        // fp32 words from the tile base
```

The implementer defines this once (a constexpr function in a header shared by the writer's scatter / unscatter and the reader's bias build). Its first on-device test is a probe that fills slot k with value k plus lane/64, applies the coefficient op's identity path, and checks the round trip.

| Slot | Input meaning (`cb_coef_in`) | Output meaning (`cb_coef_out` / `cb_comb_coef`) |
|------|------------------------------|-------------------------------------------------|
| 0..n−1 | raw mix sum (x@W)[k] | pre_k, RNE-rounded to tf32 (exact through the FPU's tf32 srcB) |
| n..2n−1 | raw mix sum | post_k |
| 2n..2n+n²−1 | raw mix sum | logits L (to `cb_logits_coef`); after `sinkhorn_block`: comb (to `cb_comb_coef`) |
| n(n+2) = 24 | Σx² total | r (debug only) |
| 25..31 | unused (0) | unused |

**Gather-slot reuse invariant (why `cb_gathered` needs one block, not `x_block_depth`).** A rank's writer runs `send_partial(b)` only after its own `receive S(b−1)` (sequential writer loop). S(b−1) is mcast only after the root's compute has unpacked every gathered tile of block b−1. So the slot writes of block b never overtake the reads of block b−1. The root's `cb_reserve_back` on `cb_gathered` supplies the matching CB credit. The implementer must keep the writer order `send_partial(b)` → `receive S(b)` → … → `send_partial(b+1)`.

**Placement axis vs TARGET.** TARGET has no `memory_layout` axis (DRAM interleaved only). The logical shards the work split defines are: a height shard (token groups), a knob-turn; and width slices of nC per rank, a scheme-change that is already built. They would surface as physical placements only if a later TARGET adds sharded X.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block: `block_token_tiles` token tile-rows × one rank's `core_k_tiles` (X in), `core_c_tiles` (y out) |
| Grid | `device.compute_with_storage_grid_size()` (runtime), tiled into `num_groups` rectangles of `group_w × group_h` |
| Per-core work | group g's token rows `[t_start[g], t_start[g] + core_token_tiles[g])` × rank's columns `[c_start[r], c_start[r] + core_c_tiles[r])` of every stream |
| Remainder | tokens: `Mt = prod(lead)·ceil(T/32)`, split with ceil/floor classes across groups (idle groups get 0 blocks and are not launched). Columns: ceil/floor classes across ranks, so `core_k_tiles` differs by ≤ n between ranks. CB capacities use `core_k_tiles_max` and push/wait counts use the rank's own value (same on producer and consumer). Last block: `last_block_extent = core_token_tiles − (num_blocks−1)·block_token_tiles`. That extent goes to the same block operations; the partial/gather pages stay at the nominal `2·block_token_tiles`, and the unused rows are simply not written. |
| Ragged tokens (T % 32 ≠ 0) | Padded rows of the last tile-row of each image are computed like real rows (every statistic is per row / per lane) and land only in the padded region of y / post / comb. No masking is needed. Padded X rows may be anything: all math is row-, or lane-, local. |
| Owner of post/comb row | group-local row `j_local`: rank `j_local mod group_cores` |
| Kernel RT args (derived by the implementer) | per core: rank, group origin, root NoC coords, `t_start`, `core_token_tiles`, `c_start`, `core_c_tiles`, mcast args (`Mcast2D.runtime_args`), buffer addresses; compute: scale / eps / norm_eps / inv_nc bits, iters |

Regime-pinned shapes the acceptance test covers on an 11×10 grid: `group_cores = 1` (C=32), `group_h > 1` (Mt < grid_y with Ct ≥ 22), multi-block per core (T=640, C=7168: 2 blocks), `block_token_tiles > 1` (T=640, C=1792), ragged T, rank-4 batch > 1 (per-image padding).

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_x_resident` | 0 | X tile | `x_block_depth · block_token_tiles · core_k_tiles_max` | spans token (block) × K (whole rank slice); depth for the prefetch | X dtype | reader | compute | per block (proj → y-mix) |
| `cb_weight` | 1 | W tile | `core_k_tiles_max` | spans K; streams tokens (reused by all blocks) | W dtype | reader | compute | whole kernel |
| `cb_bias_coef` | 2 | 4096 | 1 | constant (24 used slots) | Float32, UnpackToDestFp32 | reader | compute | whole kernel |
| `cb_reduce_scaler` | 3 | 2048 | 1 | constant | Float16_b | reader | compute | whole kernel |
| `cb_sq_acc` | 4 | 4096 | 1 | one Q tile per token row (DEST-accumulated Σ_k x_k⊙x_k), reduced straight away | Float32 | compute | compute | per token row within `sumsq_block` |
| `cb_partial` | 5 | 4096 | `2 · block_token_tiles` | spans the block's token rows × {mix, Σx²} | Float32 | compute | writer | per block |
| `cb_gathered` | 6 | 4096 | `group_cores · 2 · block_token_tiles` | spans ranks × token rows × {mix, Σx²}; used on the root only, allocated grid-wide for a uniform address | Float32, UnpackToDestFp32 | writer (root; remote NoC fill + CB credit) | compute (root) | per block |
| `cb_combined` | 7 | 4096 | `2 · block_token_tiles` | spans the block's token rows × {mix, Σx²} | Float32 | root: compute; non-root: remote mcast (no local producer) | writer | per block |
| `cb_coef_in` | 8 | 4096 | `block_token_tiles` | one coefficient-major tile per token row | Float32, UnpackToDestFp32 | writer | compute | per block |
| `cb_coef_out` | 9 | 4096 | `block_token_tiles` | same | Float32 | compute | writer | per block |
| `cb_logits_coef` | 10 | 4096 | `block_token_tiles` | owned rows' logits, carried from `coefficients_block` to `sinkhorn_block` across `ymix_block` | Float32, UnpackToDestFp32 | compute | compute | per block |
| `cb_comb_coef` | 11 | 4096 | `block_token_tiles` | owned rows' comb | Float32 | compute | writer | per block |
| `cb_pre_cols` | 12 | 4096 | `n · block_token_tiles` | spans streams × token rows (column-0-valid) | Float32 | writer | compute | per block |
| `cb_y_out` | 13 | y tile | `y_depth · y_chunk_tiles` | streaming window over the block's `block_token_tiles × core_c_tiles` outputs | y dtype (= X dtype) | compute | writer | per chunk |
| `cb_out_stage` | 14 | 4096 | 2 | writer scratch: one post and one comb DRAM staging tile (padding columns zeroed once) | Float32 | writer | writer | per owned row |

Every CB's page format follows fp32 DEST (Float32) except X, W and y, which follow their tensor dtype, and the scaler (bf16 by the reduce-scaler convention). Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (name, pages, state) | Output CB (name, pages) | CB state after |
|---|-----------------|-------------|---------|-------------------------------|--------------------------|----------------|
| 0 | `load_resident_constants` | W `core_k_tiles`, 1, 1 | dataflow + `prepare_reduce_scaler` | DRAM | `cb_weight` (core_k_tiles), `cb_bias_coef` (1), `cb_reduce_scaler` (1) | all three resident, never popped |
| 1 | `load_x_block` | `block_token_tiles·core_k_tiles` | dataflow (TensorAccessor) | DRAM | `cb_x_resident` (+block) | pushed |
| 2 | `project_block` | M=`block_token_tiles`, K=`core_k_tiles`, N=1 | `matmul_block` | `cb_x_resident` (block, **retained**), `cb_weight` (retained) | `cb_partial` (block_token_tiles) | X block still at the front |
| 3 | `sumsq_block` | per row: `tiles(core_k_tiles)` → 1, then reduce 1 → 1 | `eltwise_chain` BinaryFpu Mul DEST-accumulate + `reduce` | `cb_x_resident` (no wait / no pop, Offset base t·core_k_tiles), `cb_sq_acc` | `cb_sq_acc` (1) → `cb_partial` (+block_token_tiles) | `cb_partial` holds `[mix rows…, Σx² rows…]` |
| 4 | `send_partial_block` | `2·block_token_tiles` | dataflow (raw NoC + semaphore) | `cb_partial` | root `cb_gathered` slot | `cb_partial` popped |
| 5 | `combine_block` (root) | `group_cores × 2·block_token_tiles` → `2·block_token_tiles` | custom block op: copy_tile + SFPU `add_binary_tile` (raw) | `cb_gathered` (all) | `cb_combined` (2·block_token_tiles) | `cb_gathered` popped |
| 6 | `broadcast_combined_block` | `2·block_token_tiles` | `mcast_pipe` SenderPipe / ReceiverPipe + writer scatter | `cb_combined` | `cb_coef_in` (block_token_tiles) | root pops `cb_combined` after the send completes |
| 7 | `coefficients_block` | `block_token_tiles` coefficient tiles | custom SFPU block op (raw SFPI) | `cb_coef_in`, `cb_bias_coef` (resident) | `cb_coef_out` (block_token_tiles), `cb_logits_coef` (owned rows) | `cb_coef_in` popped |
| 8 | `expand_pre_block` | `block_token_tiles` → `n·block_token_tiles` | dataflow | `cb_coef_out` | `cb_pre_cols`, `cb_out_stage` → DRAM (post) | `cb_coef_out` popped |
| 9 | `ymix_block` | per row: `grid(core_c_tiles, n)` PerRow-accumulated | `eltwise_chain` BinaryFpu Mul, B bcast Col, DestAccumulation::PerRow | `cb_x_resident` (Offset base, pop the whole block at the end), `cb_pre_cols` (Col mapping, Offset base t·n, pop at the end) | `cb_y_out` (PerOuter) | X block freed → reader may load block+2 |
| 10 | `store_y_block` | `block_token_tiles·core_c_tiles` in windows | dataflow | `cb_y_out` | DRAM | popped |
| 11 | `sinkhorn_block` | owned coefficient tiles, `sinkhorn_iters` inside | custom SFPU block op (raw SFPI) | `cb_logits_coef` | `cb_comb_coef` | popped |
| 12 | `store_comb_block` | owned rows | dataflow | `cb_comb_coef` | `cb_out_stage` → DRAM | popped |

## API Mapping

Paths are relative to `ttnn/cpp/ttnn/kernel_lib/` unless absolute.

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| (boot) | raw_api | `compute_kernel_hw_startup<SrcOrder::Reverse>(cb_x_resident, cb_weight, cb_partial)` once, before any helper | matmul_block_helpers.hpp:100 | — | — | — | — |
| `project_block` | helper | `matmul_block<false, false, LastBlockTarget::Out, OutputCBLayout::SubblockMajor, InitMode::Short, InputPolicy::WaitAndRetainOnLastBlock /*in0*/, InputPolicy::WaitAndRetainOnLastBlock /*in1: num_k_blocks=1, never popped*/>(cb_x_resident, cb_weight, cb_partial, cb_partial /*interm: unused with 1 K-block*/, MatmulBlockShape::of(block_token_tiles/out_subblock_h, 1, out_subblock_h, 1, core_k_tiles, 1))` | matmul_block_helpers.hpp:358 (fn), :160 (`of`), :77 (InputPolicy); .inl:537-550 (retain) | fidelity from the caller's config (HiFi4 default: tf32 W × tf32/bf16 X is exact per product) | `cb_x_resident`, `cb_weight` | `cb_partial` | `block_token_tiles` (M), `core_k_tiles` (K) |
| `sumsq_block` (Q) | helper | `eltwise_chain(IterationShape::tiles(core_k_tiles), BinaryFpu<BinaryFpuOp::Mul, input(cb_x_resident, WaitPolicy::None, PopPolicy::None, InputTileMapping::Block, DataFormatReconfig::Enabled, TileAddressing::Offset), input(cb_x_resident, same…, BroadcastDim::None), Dst::D0, DestAccumulation::WholeShape>{t·core_k_tiles, t·core_k_tiles}, PackTile<output(cb_sq_acc, ReservePolicy::OneUpfront, PushPolicy::OneAtEnd, …, DestAccumulation::WholeShape)>{})` | eltwise/api/chain.hpp:533 (fn), :485-494 (BinaryFpu), :283 (DestAccumulation), :261 (Offset); eltwise/core/chain.inl:961-996 (pack rules) | the (None, None) wait after `project_block` has already waited on the block | `cb_x_resident` | `cb_sq_acc` | `core_k_tiles` (shape) |
| `sumsq_block` (row reduce) | helper | `reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, cb_sq_acc, cb_reduce_scaler, cb_partial, ReduceInputPolicy::WaitAndPopPerTile, INPUT_AND_OUTPUT, ReduceFp32Mode::Accurate>(ReduceInputBlockShape::single())` | reduce_helpers_compute.hpp:616-637 (fn), :624 (fp32_mode); reduce_helpers_common.hpp:18 (Accurate = SFPU full fp32) | Accurate: the 32-column collapse of Q is not truncated to tf32 | `cb_sq_acc` | `cb_partial` (col 0 valid) | — |
| scaler | helper | `dataflow_kernel_lib::prepare_reduce_scaler<cb_reduce_scaler, PoolType::SUM, ReduceDim::REDUCE_ROW>(1.0f)` | reduce_helpers_dataflow.hpp:59 | pool-type-aware overload | — | `cb_reduce_scaler` | — |
| `send_partial_block`, gather credit | raw_api | `noc_async_write`, `noc_async_write_barrier`, `noc_semaphore_inc`, `noc_semaphore_wait_min` | tt_metal dataflow_api | — | `cb_partial` | root `cb_gathered` | `block_token_tiles`, `group_cores` |
| `combine_block` | raw_api | `copy_tile` (UnpackToDestFp32) + `add_binary_tile` (SFPU), rank-ordered fold per output tile | used by `CopyTile` (eltwise/core/chain.inl:921) and `AddBinary` (eltwise/binary/sfpu/basic.hpp:20) | — | `cb_gathered` | `cb_combined` | `group_cores`, `block_token_tiles` |
| `broadcast_combined_block` | helper | `SenderPipe::send` / `ReceiverPipe::receive(round)` via `McastArgs`; host `ttnn.Mcast2D(device, group_rect, root, McastConfig(noc=…, handshake=True))` | mcast_pipe.hpp:139, :155, :205, :218, :243; host/mcast_host.hpp:37, :134 | `PRE_HANDSHAKE=true` | `cb_combined` | `cb_combined` (remote) | payload `2·block_token_tiles` tiles |
| `coefficients_block` | raw_api | custom SFPI function over one tile in DEST (lane-wise over slots) | — | RT: a_pre, a_post, a_res, eps, norm_eps, inv_nc (fp32 bits) | `cb_coef_in`, `cb_bias_coef` | `cb_coef_out`, `cb_logits_coef` | `block_token_tiles` (tiles per call) |
| `ymix_block` | helper | `eltwise_chain(IterationShape::grid(core_c_tiles, n), BinaryFpu<BinaryFpuOp::Mul, input(cb_x_resident, WaitPolicy::None, PopPolicy::None /*pop whole block after last row*/, InputTileMapping::Block, …, TileAddressing::Offset), input(input(cb_pre_cols, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Col, …, TileAddressing::Offset), BroadcastDim::Col), Dst::D0, DestAccumulation::PerRow>{t·core_k_tiles, t·n}, PackTile<output(cb_y_out, ReservePolicy::PerOuter, PushPolicy::PerOuter, …, DestAccumulation::PerRow)>{})`. After the block's last row, explicit `cb_pop_front(cb_x_resident, block_token_tiles·core_k_tiles)` and `cb_pop_front(cb_pre_cols, n·block_token_tiles)`. | eltwise/api/chain.hpp:485-494, :305 (BroadcastDim), :232 (InputTileMapping), :283; chain.inl:961-996 | `grid(H = core_c_tiles, W = n)`: row = output tile c, columns = streams accumulated in DEST | `cb_x_resident`, `cb_pre_cols` | `cb_y_out` | `core_c_tiles`, `block_token_tiles` |
| `sinkhorn_block` | raw_api | custom SFPI function over one tile in DEST | — | RT: eps, iters | `cb_logits_coef` | `cb_comb_coef` | — |

**Helpers considered and rejected (every raw_api entry):**

| Raw entry | Helper considered | Mismatch (file:line) | Concrete reason |
|-----------|-------------------|----------------------|-----------------|
| `combine_block` | `reduce<SUM,…, ReduceAlgorithm::AccumulateViaAdd, …, ReduceWithinTile::Skip>` | reduce_helpers_compute.hpp:167-171 (AccumulateViaAdd step 1 is `add_tiles(acc_to_dest)`, an FPU srcA/srcB read), :240-255 | The FPU reads fp32 partials at tf32 by truncation (precision contract). Every partial would be truncated, which puts tf32-class noise into bf16-stream post/comb, whose gate is ~fp32 accumulation noise. `ReduceFp32Mode::Accurate` applies only to the within-tile collapse (reduce_helpers_common.hpp:12-17); with Skip "this helper never touches the SFPU" (:184). |
| `combine_block` | `eltwise_chain` + `AddBinary` | eltwise/api/chain.hpp:533: the element list is compile-time; `binary_sfpu` (convenience.hpp:79) is exactly 2 inputs | A runtime `group_cores`-deep fold in one DEST window is not expressible. The raw block op is the chain's own two primitives (`copy_tile`, `add_binary_tile`) in a loop. Closing the gap: a chain "N-ary SFPU fold" element. |
| `coefficients_block`, `sinkhorn_block` | `eltwise_chain` SFPU elements (`Sigmoid` activations.hpp:24, `Rsqrt`/`Recip`/`Exp` math.hpp:22-38, `AddBinary`/`MulBinary`/`DivBinary` basic.hpp:20-30) | chain elements are whole-tile elementwise over DEST slots (chain.hpp:400, slots `< DEST_AUTO_LIMIT` = 4 fp32 tiles) | The math is across **slots within one tile** (row sums over j, column sums over i, per-token softmax). A whole-tile formulation needs one tile per coefficient (25 tiles), which exceeds DEST, or a pack/unpack round trip per step (≈ 40 normalisations × several tile ops per token row). The custom op should call the same non-approximate SFPI primitives those elements dispatch to (exp, reciprocal, rsqrt, sigmoid, approx off) on `dst_reg[slot]` vectors. |
| `sinkhorn_block` | the Blackhole LLK `_sinkhorn_4x4_` | ckernel_sfpu_sinkhorn.h:229-330 (each iteration = row-norm then col-norm), :287/:319 (`SFPARECIP` approximate reciprocal), :356-357 (RNE store to bf16), :61 (Blackhole-only) | Wrong iteration order: the required order is softmax+eps, one col-norm, then (iters−1)×(row, col). It also has an approximate reciprocal and bf16 storage (which break the 1e-5 column-sum bias gate) and a different (4×4-block) data layout. Used for mechanics only: the lane layout at :14-18. |
| `send_partial_block` | `mcast_pipe` | mcast_pipe.hpp (one-to-many only; the library has no many-to-one) | A gather is many-to-one. |

**Custom SFPU op contracts** (lane-wise; `v[k]` = slot k vector, 32 tokens):

| Op | Exact sequence (fp32, approx off) |
|----|-----------------------------------|
| `coefficients_block` | `r = rsqrt(v[24]·inv_nc + norm_eps)`; for k<24: `mix = v[k]·r`; k<n: `v[k] = rne_tf32(sigmoid(a_pre·mix + b[k]) + eps)`; n≤k<2n: `v[k] = 2·sigmoid(a_post·mix + b[k])`; 2n≤k<2n+n²: `v[k] = a_res·mix + b[k]` (logits). `b[k]` = slot k of `cb_bias_coef` (copied into a second DEST tile). Pack the tile to `cb_coef_out`, and also to `cb_logits_coef` when the row is owned. |
| `sinkhorn_block` | `L[i][j] = v[2n + i·n + j]`. Row softmax: `mx_i = max_j L_ij`; `e_ij = exp(L_ij − mx_i)`; `m_ij = e_ij · recip(Σ_j e_ij) + eps`. Col: `m_ij ·= recip(Σ_i m_ij + eps)`. Repeat iters−1 times: row `m_ij ·= recip(Σ_j m_ij + eps)`, then col as above. `v[·] = m`. Reciprocal accuracy ≤ 2 ulp (Newton-refined, no bare `SFPARECIP`). Sums add in fixed index order. |

## Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast Dim |
|-------|-----|-------------------|-------------------|---------------|
| `sumsq_block` | Mul (x ⊙ x), DEST-accumulate | `cb_x_resident`: All | `cb_x_resident` (same tile): All | None |
| `ymix_block` | Mul (x_i ⊙ pre_i), DEST-accumulate over i | `cb_x_resident`: All | `cb_pre_cols`: Col0 (32 rows) | Col |
| `sumsq_block` reduce | REDUCE_ROW SUM of Q | `cb_sq_acc`: All | `cb_reduce_scaler`: per reduce-scaler convention | — (output Col0) |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| tf32 truncation of fp32 values on the FPU | The FPU reads fp32 at tf32 (truncation). An FPU normalisation biases every column sum low (~−5e-4), and the depth-122 regression and the 1e-5 mean-bias gate catch it. | All coefficient math and the Sinkhorn run in the SFPU on `UnpackToDestFp32` copies. The combine is an SFPU fold. The Σx² collapse uses `ReduceFp32Mode::Accurate`. pre is RNE-rounded to tf32 before the FPU y-mix reads it, so the FPU read is lossless. The only FPU fp32 reads are the projection and the y-mix of **fp32** X, which the contract allows (tf32-class). |
| X block must survive from projection to y-mix | `matmul_block` pops in0 per K-block | `num_k_blocks = 1`, in0 `WaitAndRetainOnLastBlock`. `sumsq_block` uses (None, None). `ymix_block` pops the block explicitly after its last row. |
| Gather slot overwrite | Push-model gather with one-block slots | Gather-slot reuse invariant (Dataflow Strategy). The writer order is mandatory. |
| Determinism (bitwise) | Arrival order at the root varies | Slots are addressed by rank and folded in rank order. The monotonic counter is a count, not an order. |
| CB address uniformity for remote writes | Senders address the root's `cb_gathered` / `cb_combined` with their own base address | Both CBs are allocated on the whole launched core range with identical descriptors. |
| `cb_combined` has two per-core roles | Root: compute→writer. Non-root: remote mcast landing (no local push). | Non-root writers never `cb_wait_front` it. They read at the CB base after `ReceiverPipe::receive`. The root pops after `send` returns. |
| `group_cores = 1` | No gather peers, empty mcast rectangle | Root sends to itself, the semaphore target is 1, and the mcast is skipped when `num_receivers = 0`. |
| SoA lane map | A wrong (slot, lane) → datum map silently permutes tokens and coefficients | One shared constexpr function plus the probe in Dataflow Strategy. |
| Softmax overflow | Logits up to ~80 | Row-max subtraction lane-wise before exp. Padded lanes (garbage X rows) stay lane-local. |
| Padded rows (T % 32) | Garbage or zero X rows | Every operation is row- or lane-local, and padded rows land only in padded output rows. post/comb padding **columns** are written as 0 (stage tile zeroed once). |
| Uneven `core_k_tiles` across ranks | Ct % group_cores ≠ 0 | CB capacity from `core_k_tiles_max`. Pushes, waits and matmul K use the rank's own value on both sides. |
| bf16 refinements (knob-turns) | Mixed srcA bf16 / srcB fp32 in the matmul (W fp32) and the y-mix (pre fp32) | Helpers reconfigure per operand (`DataFormatReconfig::INPUT_AND_OUTPUT`). `cb_x_resident` / `cb_y_out` take the X dtype. No kernel structure changes. |
| fp32_dest_acc_en=False | Outside TARGET | `validate()` raises `UnsupportedAxisValue`. |
| Wormhole | The custom SFPU ops use only lane-wise SFPI (no SHFT2/TRANSP), so they are arch-neutral. The L1 predicate may select `x_block_depth = 1` for C=7168 fp32 on an 8-wide grid. | Selection function. |

## Registry / Entry Point

| Item | Decision |
|------|----------|
| `INPUT_TAGGERS` | `{"alignment": lambda inputs, axes: "tile_aligned" if inputs[0][-2] % 32 == 0 else "h_non_aligned"}`; inputs = `(X_shape, W_shape)` |
| `SUPPORTED` (Phase 0, implementer writes) | dtype [float32], layout [TILE], weight_dtype [float32], fp32_dest_acc_en [True], alignment [tile_aligned, h_non_aligned] |
| `validate()` | axes = {dtype: X.dtype, layout: X.layout, weight_dtype: W.dtype, fp32_dest_acc_en: cfg.fp32_dest_acc_en (cfg = None → default), alignment: tagger}. Check SUPPORTED per axis (`UnsupportedAxisValue`, message names the axis, e.g. `fp32_dest_acc_en=False not in SUPPORTED [True]`), then EXCLUSIONS (`ExcludedCell`). Structural shape errors (W / b shape vs X, C % 32, n > 4) raise `ValueError`. |
| Entry point | `validate()` first line. Then: allocate y / post / comb (`ttnn.allocate_tensor_on_device`), build one ProgramDescriptor, and make one `ttnn.generic_op([X, W, b, y, post, comb], desc)`. No other device op (no reshape-with-copy: the leading dims are folded in tile-index arithmetic, since outputs are allocated at the final shapes). |
| TARGET → refinements | dtype bf16: knob-turn (CB formats). weight_dtype bf16: knob-turn. R2 W broadcast: deferred regime. |
| Structural impossibilities | none beyond feature_spec's (INVALID = []) |
