# Operation Design: qkv_proj_heads_rope

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (matmul + head split + rotary epilogue), one device program |
| Goal | Replace `linear -> nlp_create_qkv_heads -> 2x rotary_embedding_hf` of the Nomic Embed v2 encoder with one dispatch that writes q, k, v directly in head layout. Primary target 8x512 <= 90 us on Blackhole p100a (stock 226.3 us). |
| Math | `P[r, c] = sum_k x[r, k] * W[k, c] + b[c]` over token row `r = (b, s)`, column `c = t*768 + a*64 + d`; `V[b,a,s,d] = P[(b,s), col(2,a,d)]`; for t in {q,k}: `out[d<32] = P0*cos_lo - P1*sin_lo`, `out[d>=32] = P1*cos_hi + P0*sin_hi` with `P0 = P[.., col(t,a,d mod 32)]`, `P1 = P[.., col(t,a,32 + d mod 32)]` |
| Mode | Hybrid (new kernels built on kernel_lib helpers; no existing op composed) |
| References | `eval/golden_tests/qkv_proj_heads_rope/feature_spec.py`, `.../helpers.py` (fp64 oracle, tolerances), `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp`, `ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast.hpp`, catalog `ttnn/ttnn/operations/examples/master.md`, `l1_ledger.md` (this directory) |

Key structural facts the design rests on:

| Fact | Consequence |
|------|-------------|
| D/2 = 32 = one tile width | `rotate_half` is a **tile swap plus a sign** on the two d-tiles of a head: no intra-tile permutation, no transformation-matrix matmul. Rotary = 2 elementwise products per output tile. |
| A head (64 cols) = 2 tiles; 12 heads per third; 36 head-pairs total | Head = atomic unit of the N axis (rotary couples its two tiles). Heads per core divides 12 so every core writes exactly one of q/k/v. |
| Bias is a row broadcast over all tokens | Folded into the matmul as a 25th K tile: in0 gets a constant "ones-column" tile, in1 gets the bias tile row. No separate bias pass. |
| B*S <= 4096, tile-aligned S >= 32 | Phase 0 (tile_aligned) has `Mt = B*S/32 <= 128` tile-rows: every core's assignment is one block. |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| input_tensor | ttnn.Tensor | yes | (B,1,S,768), bf16, TILE, interleaved | - | RT (address) |
| weight | ttnn.Tensor | yes | (768, 2304), bf16, TILE, interleaved DRAM | - | RT (address) |
| bias | ttnn.Tensor | yes | (2304,) or (1,2304), bf16, TILE, interleaved DRAM | - | RT (address) |
| cos, sin | ttnn.Tensor | yes | (1,1,S_tab,64), S_tab >= S, bf16, TILE, interleaved | - | RT (address) |
| num_heads | int | no | exactly 12 (else UnsupportedAxisValue) | 12 | host constant |
| compute_kernel_config | ttnn.ComputeConfigDescriptor | no | fp32_dest_acc_en must be True (registry axis); any math_fidelity / math_approx_mode honored | `default_compute_kernel_config()` | CT (compute config) |
| memory_config | ttnn.MemoryConfig | no | interleaved L1 or DRAM | `input_tensor.memory_config()` | host (output alloc) |

`default_compute_kernel_config()` is exported from the op module (single source, a factory per `references/precision_convention.md`): `math_fidelity=HiFi3, fp32_dest_acc_en=True, math_approx_mode=False`. Packer L1 accumulation is an **internal invariant of this design** (always on; required by the `caller_owns_pack_target` matmul path, see Mechanism caps). `ttnn.ComputeConfigDescriptor` has no `packer_l1_acc` field (binding `ttnn/cpp/ttnn-nanobind/program_descriptors.cpp:639-660`); the entry point reads `getattr(cfg, "packer_l1_acc", True)` and raises `UnsupportedAxisValue("packer_l1_acc", False)` when a descriptor that carries the field sets it False (reject, do not silently override). See Key Risks for the golden-helper consequence.

Entry-point order: `validate()` first line; then `ValueError` shape/dtype/layout/device checks listed in the prompt (each message names the offending tensor: `weight`, `bias`, `cos`/`sin`, `input_tensor`; the acceptance test matches on it); then output allocation (3 x `ttnn.allocate_tensor_on_device((B,12,S,64), bf16, TILE, device, memory_config or input_tensor.memory_config())`); then one `ttnn.generic_op([x, w, b, cos, sin, q, k, v], program_descriptor)`; return `(q, k, v)`. No host-side tensor transforms.

Registry axes (names from `feature_spec.py` TARGET): `dtype`, `layout`, `alignment` (tagger `tag_alignment` on `inputs[0][-2] % 32`), `fp32_dest_acc_en`, `memory_layout`, `buffer_type` (x), `out_buffer_type` (q/k/v), `table_buffer_type` (cos/sin). Phase 0 SUPPORTED per the prompt (L1 everywhere, tile_aligned). Shape limits (S > 512, B*S > 4096, H != 768, num_heads != 12) raise `UnsupportedAxisValue`.

## Tensors

### Input

| Property | x | weight | bias | cos / sin |
|----------|---|--------|------|-----------|
| Shape | (B,1,S,768) | (768,2304) | (2304,) or (1,2304) | (1,1,S_tab,64) |
| Tile grid | `Mt = B*ceil(S/32)` rows x 24 | 24 x 72 | 1 x 72 (row 0 valid) | `ceil(S_tab/32)` x 2 |
| Page index | `r*24 + kt`, `r = b*St + st` | `kt*72 + nt` | `nt` | `st*2 + dt` |
| Dtype | bf16 | bf16 | bf16 | bf16 |
| Layout | TILE | TILE | TILE | TILE |
| Memory | interleaved L1 (Phase 0) / DRAM (TARGET) | interleaved DRAM | interleaved DRAM | interleaved L1 (Phase 0) / DRAM (TARGET) |

`St = ceil(S/32)` (per-image tile padding; never `B*S//32`).

### Output

| Property | Value |
|----------|-------|
| Shape | q, k, v each (B, 12, S, 64) |
| Page index | `((b*12 + a)*St + st)*2 + dt` |
| Dtype | bf16 |
| Layout | TILE |
| Memory | interleaved, `memory_config` (default: x's) |
| Padding | rows s in [S, 32*St) exactly 0 (TARGET h_non_aligned; writer-side zeroing) |

## Blocking Model

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `token_rows` (tile-rows `r = b*St + st`; batch and seq folded: contiguous in x, identical math) | independent (each output row depends on its own x row only); W and bias are reuse-shared across it | `block_m` (tile-rows per block) | `min(ceil(Mt/grid_m), block_m_cap(N_c))` = whole per-core assignment (11 at Mt=128) | host constant -> CT arg `block_m` | split over `grid_m` M-groups, balanced (ceil/floor), every N-group repeats the same M split | knob-turn (more/fewer M-groups, smaller block) |
| `head_cols` (head-pairs, 36 = 3 thirds x 12 heads) | independent across heads; x and cos/sin are reuse-shared across it | `heads_per_core` (h_c); `N_c = 2*h_c` tiles | selection function (4 at Mt=128) = whole per-core N slice | host constant -> CT arg `heads_per_core` | split over `grid_n = 36/h_c` N-groups; h_c divides 12 | knob-turn |
| `head_dim_tile` (dt in {0,1}) | dependent within a head (rotate_half couples d and d+32 -> both d-tiles in one block) | fixed 2 | 2 (whole) | constant `TILES_PER_HEAD = 2` (= D/32) | not split (atomic) | none (structural) |
| `qkv_third` (t in {q,k,v}) | independent; selects the epilogue (rotary for q,k; convert for v) | folded into `head_cols` (h_c divides 12 keeps one t per core) | 1 per core | derived from N-group index | N-groups `[0, 12/h_c)` q, next k, next v | knob-turn |
| `hidden_k` (K' = 24 x-tiles + 1 bias tile = 25) | dependent (reduction; results span it) | `k_block` (in0_block_k) | 5 (`num_k_blocks = 25/5 = 5`) | host constant -> CT arg `k_block` | not split across cores (every core runs the full K'); Phase 0 whole axis per core | scheme-change (cross-core K split + combine: rejected, see Regimes) |

Re-run on intermediate data (`P` = matmul partials, axes `token_rows x head_cols`): the epilogue (rotary / convert) is independent per (row, head) and runs **on the core that computed the block** - no designated core, no single-core stage, no cross-core hand-off. The table rows (cos/sin) share `token_rows`; they are fetched per valid row by the same core.

### Knob table (one row per axis)

| Axis | Character | Block factor (Phase 0) | Core assignment | Buffer depth | Why |
|------|-----------|------------------------|-----------------|--------------|-----|
| token_rows | independent | `block_m` = whole per-core rows (11 at 8x512) | `grid_m` groups, ceil/floor balanced | in0 2 K-blocks; out 2 rows; table 2 rows | coarsest block = one pipeline pass per core: one matmul init, one K-loop, one epilogue init |
| head_cols | independent | `N_c = 2*h_c` = whole per-core slice (8 at 8x512) | `grid_n = 36/h_c` groups | in1 2 K-blocks | whole slice -> W slice crosses into the core once |
| head_dim_tile | dependent (rotary pair) | 2 (whole) | not split | - | rotary partner tile must be core-local |
| qkv_third | independent | 1 per core | via N-group index | - | one output tensor and one epilogue kind per core |
| hidden_k | dependent (reduction) | `k_block = 5` of K' = 25 | not split (whole K' on every core) | in0/in1 depth 2 | 5 is the only divisor of 25 that is neither 1 (25 L1-acc packs/tile) nor 25 (in0+in1 single K-block = 950 KiB at 8x512, no fit) |

### Core assignment (selection function, host, closed form)

Inputs: `Mt`, `num_cores` (from `device.compute_with_storage_grid_size()`, runtime - never a literal), `block_m_cap(N_c)` (ledger formula).

| Step | Rule |
|------|------|
| 1 | For each `h_c in {1,2,3,4,6,12}`: `grid_n = 36/h_c`, `grid_m = min(floor(num_cores/grid_n), Mt)`; skip if `grid_m == 0`. |
| 2 | `rows_max = ceil(Mt/grid_m)`; `block_m = min(rows_max, block_m_cap(2*h_c))`; `num_blocks = ceil(rows_max/block_m)`. |
| 3 | `tiles_max = rows_max * 2*h_c` (critical-path output tiles); `ingress = 25*(rows_max + num_blocks*2*h_c)` tiles per core. |
| 4 | Keep candidates with `tiles_max <= (1 + occupancy_slack) * min(tiles_max)`, `occupancy_slack = 0.05`; pick min `ingress`; ties -> larger h_c. |

Resulting Phase 0 grid on p100a (110 cores):

| (B,S) | Mt | h_c | grid_m x grid_n | cores | rows/core | out tiles/core (ideal) |
|-------|----|-----|-----------------|-------|-----------|------------------------|
| 8x512, 16x256, 32x128, 64x64, 128x32 | 128 | 4 | 12 x 9 | 108 | 11 / 10 | 88 (83.8) |
| 8x384 | 96 | 4 | 12 x 9 | 108 | 8 | 64 (62.8) |
| 4x512, 8x256 | 64 | 2 | 6 x 18 | 108 | 11 / 10 | 44 (41.9) |
| 2x512, 4x256 | 32 | 1 | 3 x 36 | 108 | 11 / 10 | 22 (20.9) |

Logical core `l = n_group*grid_m + m_group`, laid row-major (x fastest) over the compute grid: an N-group's `grid_m` cores are **consecutive**, which is the receiver-group shape `ttnn.Mcast` partitions (`mcast.hpp:119-131`, "equal consecutive groups") for the deferred W multicast.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| cb_activation_block | `act_depth` | 2 | x K-block j+1 lands while K-block j multiplies |
| cb_weight_block | `weight_depth` | 2 | W K-block j+1 lands while K-block j multiplies |
| cb_matmul_partials | (none: accumulator) | 1 block | in-place L1 accumulation across K-blocks requires exactly one resident block |
| cb_rotary_table | `table_depth` (rows) | 2 | reader fetches row i+1's 4 table tiles during row i's rotary; first 2 rows prefetched during the last K-blocks |
| cb_out_tiles | `out_depth` (rows) | 2 | writer drains row i while compute packs row i+1 |

All five are host constants passed to the program descriptor; every CB page count and every loop bound is derived from them plus `block_m`, `heads_per_core`, `k_block`.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DEST capacity, fp32 DEST half-sync (`dest_helpers.hpp:89-103`, assert `matmul_block_helpers.inl:158`) | matmul subblock `sb_h*sb_w <= 4` | host subblock rule (below) | assert / DEST overrun |
| matmul subblock integrality (`MatmulBlockShape`, `matmul_block_helpers.hpp:139-170`) | `sb_h | block_m`, `sb_w | N_c` | host subblock rule: max `sb_h*sb_w <= 4` with both divisibilities, tie -> wider `sb_w` (11x8 -> 1x4; 8x8 -> 2x2; 11x2 -> 1x2) | rows/cols silently skipped |
| `caller_owns_pack_target` legality (`matmul_block_constraints.hpp:28-31`, `.inl:136-139`) | partials path | requires TileRowMajor + packer_l1_acc + Interm: all three fixed by this design | static_assert / deadlock |
| L1-acc in-place accumulation (`.inl:410-414`, `:421-434`) | `cb_matmul_partials` capacity == one block | capacity = `block_m*N_c`; compute reserves and pushes the **nominal** block every block | K-block j+1 accumulates onto the wrong addresses (wrong results) |
| CB ring wrap (pointer resets only on exact `fifo_limit`) | in0/in1 push quantum | every block of every core is nominal `block_m` rows; a core with fewer valid rows computes the nominal block (stale rows) and the writer drops them | straddling push corrupts the ring |
| Rotary pairing | N-block boundaries | whole heads only (`N_c = 2*h_c`) | partner d-tile on another core: wrong q/k |
| One tensor per core | `h_c | 12` | candidate set `{1,2,3,4,6,12}` | core straddles q|k or k|v: writer / epilogue wrong |
| K' = 25 blocking | `k_block | 25` | `k_block in {1,5,25}`; 5 used | last K-block reads past K' (or needs `KBlockInnerDimFn`, `matmul_block_helpers.hpp:206-212`) |
| PerRow DEST accumulation (`chain.inl:1272-1274`, `:1141-1148`) | rotary chain | accumulating `BinaryFpu` on `Dst::D0`, `PackTile` with `(PerOuter, PerOuter)` + `DestAccumulation::PerRow` | static_assert |
| Table rows | `st < ceil(S_tab/32)` | `S_tab >= S` validated (ValueError) | reads past the table |
| L1 budget | `block_m <= block_m_cap(N_c)` | ledger formula, `cb_budget_bytes = 900 KiB` | CB/buffer clash at program creation |

### Regimes

Minimum (named boundary: each tensor crosses its storage tier once): x once, W+bias once (DRAM), cos/sin rows once, q/k/v once.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| R1 2D output-stationary, unicast reads | **built** | every supported call (selection function always returns a grid) | `block_m x N_c` outputs, K' in `k_block` steps | W+bias cross DRAM `grid_m * num_blocks_this_core` times (12x at 8x512) - above minimum because W is reuse-shared across M-groups; x crosses the NoC `grid_n` times (9x, L1 source, 0 DRAM in Phase 0); tables `grid_n_qk` times per row; outputs once | per block: one `matmul_block_init` + reconfig, one K-loop fill/drain, one L1-acc enable/disable, one epilogue init (eltwise mul-acc or copy), one partials reserve/push; bigger `block_m` also cuts W re-crossings (`num_blocks_this_core`) |
| R2 W multicast within each N-group (R1 + broadcast of the reuse-shared operand) | **deferred** - stepping stone: layers onto R1's reader without touching compute, CBs or the selection function; R1 is the correctness baseline the semaphore protocol is debugged against | `weight` in DRAM (always) | same as R1 | W+bias cross DRAM once (3.5 MiB at every shape) vs 12x (42.5 MiB) in R1; adds one multicast per K-block per N-group (`k_block*N_c` tiles to `grid_m` receivers) | per K-block multicast + semaphore round trip amortized over `k_block*N_c` tiles; bigger `k_block` = fewer rounds |
| R3 x multicast within each M-group | **deferred** - required only for `buffer_type = DRAM` (TARGET), where R1 re-reads x from DRAM `grid_n` times; with x in L1 (Phase 0) the re-read never crosses DRAM | `buffer_type == DRAM` (or measured win at L1) | same as R1 | x crosses its tier once vs `grid_n` (9x = 56.6 MiB at 8x512) | per K-block multicast amortized over `block_m*k_block` tiles |
| R4 W-slice resident, x streamed per M-block | **deferred** - only reachable when `num_blocks_this_core > 1`, i.e. `ceil(Mt/grid_m) > block_m_cap`, which needs `S % 32 != 0` with tiny S and large B (e.g. B*St > 1400); no tile_aligned shape (Mt <= 128) and no INPUTS row (max Mt = 130 -> 11 rows/core) reaches it | `num_blocks_this_core > 1` | `block_m x N_c` with in1 retained (`InputPolicy::WaitAndRetainOnLastBlock`, `matmul_block_helpers.hpp:68-77`, `num_k_blocks = 1`) | R1 re-crosses W per block; R4 holds `25*N_c` W tiles (400 KiB at N_c=8) and crosses W once per core | removes the per-block W crossing term |
| R5 split `hidden_k` across cores + combine | **rejected** - superseded by R1+R2: independent axes give 36*Mt (row, head) units (4608 at 8x512) >> 110 cores, and a 2-way K split moves `Mt*72` fp32 partials (37.7 MB at 8x512) through the NoC to save ~21.6 MB of ingress; the K' slice already streams in 5 K-blocks, so residency gains nothing | - | - | +37.7 MB cross-core per extra K split | - |
| R6 1D token split (all 72 N tiles per core) | **rejected** - superseded by R1: each core must ingest all of W (1728 tiles, 3.5 MiB) vs 200 tiles in R1; with multicast a single injector serializes 3.5 MiB to 108 receivers | - | - | W per core 8.6x R1's | - |
| R7 1D head split | **rejected** - superseded by R1: at most 36 head-granular cores (one third of the grid); each core ingests all of x (6.3 MB) | - | - | x per core = whole x | - |
| R8 separate bias epilogue (`add_bias_bcast_rows`) | **rejected** - superseded by the bias K-tile in R1: the fold costs 1 extra K step per output tile (1/25 of the matmul); the epilogue costs a full Float32 unpack+FPU+pack pass over every output tile and, on q/k, a second Float32 hand-off into rotary | - | - | same bytes; more compute passes | - |

Regime selection: R1 is the only built regime; no predicate is evaluated in Phase 0. When R2/R3/R4 are built, the selector is host-side on `(weight/x buffer_type, num_blocks_this_core)` and each gets a regime-pinned test (R4 needs a shape with `num_blocks_this_core > 1`, e.g. (B,S)=(64,33) at a reduced `cb_budget_bytes`).

### Traffic ranking

Per-core ingress for an `rows x N_c` output block is `25*(rows + N_c)` tiles; the split decides how many times each tensor crosses each tier. Counted at 8x512 (x 6.29 MB, W 3.54 MB, bias 0.15 MB, outputs 3 x 6.29 MB):

| Rank | Split | DRAM crossings | Cross-core (NoC) | Notes |
|------|-------|----------------|------------------|-------|
| 1 | 2D (M x heads) + W mcast + x mcast (R1+R2+R3) | W 1x, bias 1x | x 1x (mcast), W 1x mcast, outputs 1x | minimum; two multicast channels |
| 2 | 2D + W mcast (R1+R2) | W 1x | x 9x unicast from L1 banks (56.6 MB) | Phase 0 x is in L1, so 0 DRAM for x |
| 3 | **2D unicast (R1, built)** | W 12x (42.5 MB), bias 12x | x 9x (56.6 MB) | stepping stone for 2 and 1 |
| 4 | 2D + K split (R5) | as 3 | +37.7 MB fp32 partials | rejected |
| 5 | 1D tokens (R6) | W 108x unicast / 1x mcast | W 3.5 MiB into every core | rejected |
| 6 | 1D heads (R7) | W 1x | x 36x (226 MB) | rejected, under-fills grid |

Chosen: 2D, built as R1 with the two broadcasts as deferred rows. Within 2D, the head-group count is the extent ranking's job (selection function: occupancy first with 5% slack, then min ingress).

Operand-reuse check against the chosen split (M-groups x N-groups):

| Operand | Varies along token_rows? | Varies along head_cols? | Reuse-shared by the split across | Broadcast row |
|---------|--------------------------|-------------------------|----------------------------------|---------------|
| x | yes | no | N-groups (`grid_n` cores per M-group) | R3 |
| W | no | yes | M-groups (`grid_m` cores per N-group) | R2 |
| bias | no | yes | M-groups | rides in R2 (it is K tile 24 of the W stream) |
| cos/sin | yes (st only, not b) | no | q/k N-groups, and images | none: 4 tiles/row, ~0.1% of ingress; covered by R3's channel if built |
| ones tile | no | no | constant | core-local synthesis, never crosses |

Stall-shadow check:

| Stage that waits | Waits on | Work scheduled into the stall |
|------------------|----------|-------------------------------|
| compute, first K-block | x + W K-block 0 (fill) | nothing independent in compute (every output needs K); reader starts with x K-block 0, W reader with W K-block 0, so the fill is one K-block (`(block_m + N_c)*k_block` tiles), not the whole slice - the reason K is blocked at all |
| epilogue of a block | the full K' reduction of that block (dependent axis) | reader prefetches the first `table_depth` table rows during the last K-blocks; writer has nothing to write before the first epilogue row (no algebraic permission to rotate partial sums early: rotary is linear, so it *could* be applied per K-block, but that multiplies the rotary work by `num_k_blocks` - not taken) |
| writer (BRISC) during the K-loop | first epilogue row | it streams the W K-blocks (in1 producer), so BRISC is busy during the K-loop |
| v cores after their convert pass | nothing (finish early) | load imbalance, not a stall: perf lamp L3 |
| bias fold | changes FP summation order (bias accumulated as the 25th K term in fp32 DEST/L1) | precision-neutral at fp32 accumulation; validated by the golden tolerance (PCC 0.9999, rel RMS 4e-3) |

### Block schedule

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {  // Phase 0: 1
    load_k_blocks(block_idx);        // reader: x rows (+ ones tile); W-reader: W rows (+ bias row)
    project_block(block_idx);        // matmul over K' = 25 into Float32 partials (bias included)
    epilogue_block(block_idx);       // q/k cores: rotate_block; v cores: convert_block
    store_block(block_idx);          // writer: scatter rows into q|k|v head layout
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| load_k_blocks | per K-block: in0 `block_m x k_block`, in1 `k_block x N_c`; last K-block carries the ones tile (in0 col 4) and bias tiles (in1 row 4); table rows (`4` tiles per valid row) after the last x K-block | nothing (streams through depth-2 CBs) | one read barrier per K-block per kernel; ones-tile synthesis and bias-row zeroing once per row/tile of the last K-block; sin_lo sign flip once per table row |
| project_block | `block_m x N_c` outputs, `num_k_blocks = 5` K-blocks of `k_block = 5` | partials block (Float32) for the whole block | one `matmul_block_init` + data-format reconfig per block; L1-acc off for K-block 0, on for 1..4, explicitly off after the block; one partials reserve before, one push after |
| epilogue_block (rotate_block, q/k cores) | `valid_rows x h_c` heads, each head = grid(2 output tiles, 2 terms) | partials block (popped at block end), one table row (4 tiles) per row | eltwise mul-acc init + reconfig once per block (hoisted, `InitReconfigOwner::Caller`); table wait/pop once per row; out push per row |
| epilogue_block (convert_block, v cores) | `valid_rows x N_c` | partials block | copy init + reconfig once per block |
| store_block | per valid row: `N_c` tiles to `N_c` distinct head pages | nothing | one write barrier per row; h_non_aligned (TARGET): pad-row zeroing on the last tile-row of each image |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 Overlap | whole per-core assignment as one block: the epilogue and all output writes start only after the last K-block | `block_m = ceil(rows/2)` (2 blocks per core): block 1's matmul overlaps block 0's writes; costs one extra W slice crossing per core (free under R2) |
| L2 Grid synchronization (R2/R3) | 108 cores in groups of 12 (W) / 9 (x) multicast every K-block; per-round semaphore cost vs `k_block*N_c` = 40-tile payload | fewer, larger N-groups (h_c = 6, grid 18x6) or larger `k_block` via a ragged last K-block |
| L3 q/k vs v imbalance | v cores run a convert pass (~1 unpack+pack per tile), q/k cores run 2 FPU mul-acc per tile: q/k cores are the critical path | cost-weighted M split: fewer cores (more rows) for v N-groups, more for q/k |
| L4 Head-group choice under R1 | selection minimizes ingress (R2 world); under unicast, W DRAM crossings = grid_m (12x for h_c=4 vs 3x for h_c=1 at Mt=128) dominate | h_c = 1 (grid 3x36) at Mt=128 while R2 is unbuilt |
| L5 K-block size | `k_block = 5` gives 5 Float32 L1-acc packs per output tile | ragged `k_block = 8` or `13` with `KBlockInnerDimFn` (`matmul_block_helpers.hpp:206-212`) on the last block, L1 permitting |
| L6 Subblock shape | 11 rows force 1x4 at 8x512; catalog `matmul_output_subblock` measured 1.40x (2x2) vs 1.46x (8-tile) | `block_m` 12/10 split (2x2 everywhere) or `dst_full_sync_en` (8-tile fp32 subblocks) |
| L7 Rotary realization | FPU elementwise mul with DEST accumulation (2 FPU passes/tile) is the default | two `BinaryFpu` muls into D0/D1 + SFPU add (catalog `compute_fusion`: SFPU combine 0.82x) |
| L8 NoC/RISC placement | x on NCRISC/NoC0, W + writes on BRISC/NoC1 (catalog `noc_placement`: NoC0 reads 2.5-4.8x) | W reads on NoC0 too; `bank_stagger` start offset per core for W (1.13-1.32x) |
| L9 Table depth | 2 rows; rotary may wait on table reads | prefetch the whole block (`block_m*4` tiles, +80 KiB) |

Catalog entries that informed knobs: `compute_block_size` (one pass per core, keep reconfigs off same-format boundaries -> block_m = whole assignment), `double_buffer` / `trid_double_issue` (depth 2, block >= 4 tiles per transfer -> K-block transfers of 40-55 tiles), `matmul_output_subblock` (4-tile subblocks under fp32 DEST), `eltwise_l1_vs_dest_accumulate` (packer L1 acc 5x over read-modify-write -> L1-acc partials), `mcast_topology` / `shared_input_reuse` (R2/R3 deferred rows; P100a 2.34x / 1.82x), `noc_placement`, `bank_stagger`, `compute_fusion` (L7).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x -> cb_activation_block | bf16 tiles | reader (NCRISC, NoC0), `TensorAccessor`, `noc_async_read_tile` per tile, barrier per K-block | rows of this core's block; K tiles `[j*5, j*5+5)`; for K-block 4 the 5th column is the **ones tile** (column 0 = 1.0 in all 32 rows, rest 0) synthesized in place |
| W, bias -> cb_weight_block | bf16 tiles | W-reader/writer kernel (BRISC, NoC1), `TensorAccessor` | W pages `kt*72 + n0 + n`; K-block 4 row 4 = bias pages `n0 + n`, with rows 1..31 of each bias tile zeroed in L1 after the barrier (padding may be non-zero / non-finite) |
| cos, sin -> cb_rotary_table | bf16 tiles | reader, after the last x K-block | per valid row: `[cos(st,0), -sin(st,0), sin(st,1), cos(st,1)]`; `-sin` = XOR 0x8000 on every bf16 of the tile in L1 after the barrier (q/k cores only) |
| compute matmul -> cb_matmul_partials | Float32 | `matmul_block` Interm + packer L1 acc + TileRowMajor + caller-owned target | tile (r, n) at `r*N_c + n` |
| partials -> cb_out_tiles | Float32 -> bf16 | q/k: rotary mul-acc chain; v: copy chain | per valid row, `N_c` tiles in (head, dt) order |
| cb_out_tiles -> q / k / v | bf16 tiles | writer (BRISC, NoC1), `TensorAccessor` on the core's one output tensor | page `((b*12 + a0 + n/2)*St + st)*2 + n%2`; TARGET h_non_aligned: for `st == St-1` zero rows `[S%32, 32)` of every tile in L1 before the write (faces `2*(row/16)+{0,1}`, row `row%16`, 32 B each) |

Deferred inter-Tensix contracts (kept reachable by R1's structure):

| Row | Contract |
|-----|----------|
| R2 W multicast | `ttnn.Mcast` (`mcast.hpp:119-131`) over the `grid_n` consecutive N-groups (`receiver_group_size = grid_m`), `McastRotatingSenderConfig` so the K-blocks' DRAM reads spread across the group's cores (catalog `shared_input_reuse`: single injector serializes); kernel side `McastArgs` `sender(noc).send()` / `receiver(noc).receive(round)` (`mcast_args.hpp:41-47`); round = K-block index; payload = one `cb_weight_block` K-block (`k_block*N_c` tiles) written to the same CB offset on every receiver; the CB contract (depth 2, K-block quantum) is unchanged |
| R3 x multicast | second `Mcast` on disjoint semaphores (`next_semaphore_id()`), receivers = the `grid_n` cores of an M-group (stride `grid_m` in logical order: irregular set -> `McastConfig.irregular_receiver_set_mode`, or reorder logical cores to make M-groups consecutive and N-groups strided once R2's grouping is measured); payload = one `cb_activation_block` K-block |
| Sharded placement | out of TARGET (memory_layout is INTERLEAVED only) |

TARGET axis mapping:

| TARGET value | Cost class | What changes |
|--------------|-----------|--------------|
| alignment = h_non_aligned | knob-turn | writer pad-row zeroing; all geometry already `ceil`-based; rows are independent so non-finite x padding cannot leak into valid rows |
| buffer_type = DRAM (x) | knob-turn for correctness (TensorAccessor is placement-agnostic); perf needs R3 | none in kernels |
| out_buffer_type = DRAM | knob-turn | output allocation only |
| table_buffer_type = DRAM | knob-turn | none in kernels |

Structural impossibilities: none beyond `feature_spec.py`'s INVALID.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: `block_m` tile-rows x one head-group (`h_c` heads, `N_c = 2*h_c` tiles) x full K' |
| Grid | `grid_m x grid_n` logical cores from the selection function, row-major over `device.compute_with_storage_grid_size()`; 108 of 110 on p100a for every perf row |
| Per-core work | M-group `m`: rows `[row_start(m), row_start(m) + core_m_tiles(m))`; N-group `g`: third `t = g // (12/h_c)`, first head `a0 = (g % (12/h_c))*h_c`, first column tile `n0 = t*24 + 2*a0` |
| Remainder | `Mt = B*ceil(S/32)`; first `Mt % grid_m` M-groups get `ceil(Mt/grid_m)` rows, the rest `floor` (same split as `ttnn.split_work_to_cores` on Mt over grid_m); every block is computed at nominal `block_m`, only valid rows are read (x), tabled, epilogued and written |
| Per-row geometry | `b = r // St`, `st = r % St` (per image) |

Single built regime: no runtime regime selection in Phase 0. Tests must still pin grid-dependent behavior: the acceptance test covers Mt = 1, 4, 9, 32, 64, 128, which the selection function maps to h_c = 1, 1, 1, 1, 2, 4 (grids 1x36, 3x36 with 2/1 rows, 3x36, 3x36, 6x18, 12x9), so 1-row cores, ragged M-groups and every h_c class the perf rows use execute.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| cb_activation_block | 0 | 2048 | `act_depth * block_m * k_block` (110) | spans token_rows (block_m) and one K-block; streams over K' | Float16_b | reader | compute | K-loop |
| cb_weight_block | 1 | 2048 | `weight_depth * k_block * N_c` (80) | spans head_cols (N_c) and one K-block; streams over K' | Float16_b | writer | compute | K-loop |
| cb_matmul_partials | 2 | 4096 | `block_m * N_c` (88) | spans token_rows x head_cols; accumulator over K' | Float32 | compute | compute | K-loop through epilogue |
| cb_rotary_table | 3 | 2048 | `table_depth * 4` (8) | 4 tiles per row, streams over token_rows | Float16_b | reader | compute | epilogue (q/k cores; allocated on all) |
| cb_out_tiles | 4 | 2048 | `out_depth * N_c` (16) | spans head_cols, streams over token_rows | Float16_b | compute | writer | epilogue |

Values in parentheses: 8x512 Phase 0 (`block_m=11, N_c=8, k_block=5`). Full ledger, sharing decisions and footprint: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | boot | - | `compute_kernel_hw_startup<SrcOrder::Reverse>(cb_activation_block, cb_weight_block, cb_matmul_partials)` once | - | - | - |
| 1 | load_k_blocks | in0 `block_m x k_block`, in1 `k_block x N_c` per K-block | dataflow (TensorAccessor) | - | cb_activation_block (`block_m*k_block` per push), cb_weight_block (`k_block*N_c` per push), cb_rotary_table (4 per valid row) | pushed |
| 2 | project_block | `block_m x N_c`, 5 K-blocks of 5 | yes: `matmul_block` | cb_activation_block (`block_m*k_block`, waited+popped per K-block), cb_weight_block (`k_block*N_c`, waited+popped per K-block) | cb_matmul_partials (`block_m*N_c`, caller reserves before / pushes after) | partials pushed (nominal), L1 acc must be switched off by the caller |
| 3a | rotate_block (q/k) | per valid row, per head: grid(2 out tiles, 2 terms) | yes: `eltwise_chain` | cb_matmul_partials (waited once for the nominal block, read with Offset base `row*N_c + 2*head`, popped at block end), cb_rotary_table (waited/popped 4 per row, Offset base 0) | cb_out_tiles (1 tile per walk row, `N_c` per row) | partials popped (nominal) |
| 3b | convert_block (v) | per valid row: `N_c` tiles | yes: `eltwise_chain` (CopyTile -> PackTile) | cb_matmul_partials (as 3a) | cb_out_tiles (`N_c` per row) | partials popped (nominal) |
| 4 | store_block | per valid row: `N_c` tiles | dataflow | cb_out_tiles (`N_c` per row) | q / k / v DRAM or L1 pages | popped |

Rotary chain contract (per row r, head h; partials tiles `P0 = r*N_c + 2h`, `P1 = P0 + 1`; table row `[C0, -S0, S1, C1]`):

| Walk row (output) | term 0 | term 1 | Result |
|-------------------|--------|--------|--------|
| 0 -> out tile (r, h, dt=0) | `P0 * C0` | `P1 * (-S0)` | `P0*cos - P1*sin` |
| 1 -> out tile (r, h, dt=1) | `P0 * S1` | `P1 * C1` | `P1*cos + P0*sin` |

A = partials with `InputTileMapping::Row` (index = base + term), B = table with `InputTileMapping::Block` (index = 2*walk_row + term); `BinaryFpu<BinaryFpuOp::Mul, A, B, Dst::D0, DestAccumulation::PerRow>` + `PackTile<output(cb_out_tiles, PerOuter, PerOuter, Enabled, Direct, DestAccumulation::PerRow)>` over `IterationShape::grid(2, 2)`.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot | helper prerequisite | `compute_kernel_hw_startup<SrcOrder::Reverse>` | `matmul_block_helpers.hpp:100-106`, `:235-239` | (cb_activation_block, cb_weight_block, cb_matmul_partials) | - | - | - |
| project_block | helper | `compute_kernel_lib::matmul_block` | `matmul_block_helpers.hpp:339-371` | `<transpose=false, packer_l1_acc=true, LastBlockTarget::Interm, OutputCBLayout::TileRowMajor, InitMode::Short, WaitAndPopPerKBlock, WaitAndPopPerKBlock, NoPostCompute, NoPreKBlock, NoPostKBlock, 0, NoKBlockInnerDimFn, NoIn0Source, NoIn1BaseOffset, caller_owns_pack_target=true>`; args `(in0=cb_activation_block, in1=cb_weight_block, out=cb_matmul_partials, interm=cb_matmul_partials, MatmulBlockShape::of(block_m/sb_h, N_c/sb_w, sb_h, sb_w, k_block, 5), {}, {}, N_c, N_c)` | cb_activation_block, cb_weight_block | cb_matmul_partials | `block_m`, `N_c` (via subblock counts), `k_block`, `num_k_blocks` |
| project_block | helper (shape) | `MatmulBlockShape::of` | `matmul_block_helpers.hpp:139-170` | see above | - | - | as above |
| project_block (L1-acc off) | raw_api | `PACK((llk_pack_reconfig_l1_acc(0)))` after the block | used by helper at `matmul_block_helpers.inl:413` | - | - | - | - |
| rotate_block | helper | `compute_kernel_lib::eltwise_chain<InitReconfigOwner::Caller>` with `BinaryFpu` + `PackTile` | `eltwise/api/chain.hpp:619-620`, `:562-575`, `:587-591`, `IterationShape :146-171`, `DestAccumulation :308-312`, `TileAddressing :286`, `InputTileMapping :257-262` | see Rotary chain contract; A `input(cb_matmul_partials, None, None, Row, Enabled, Offset)`, B `input(cb_rotary_table, None, None, Block, Enabled, Offset)` | cb_matmul_partials, cb_rotary_table | cb_out_tiles | `h_c` (heads walked per row) |
| convert_block | helper | `eltwise_chain` with `CopyTile` + `PackTile` | `eltwise/api/chain.hpp:553-557`, `:587-591` | `CopyTile<input(cb_matmul_partials, None, None, Block, Enabled, Offset)>{partials, r*N_c}`, `PackTile<output(cb_out_tiles, Upfront, AtEnd)>`, `IterationShape::tiles(N_c)` | cb_matmul_partials | cb_out_tiles | `N_c` |
| load / store | raw_api (dataflow) | `TensorAccessor`, `noc_async_read_tile`, `noc_async_write_tile`, barriers | `tech_reports/tensor_accessor/tensor_accessor.md` | - | - | - | per-K-block transfer quantum |

Helpers considered and rejected:

| Raw / alternative | Candidate | File:Line of mismatch | Reason |
|-------------------|-----------|-----------------------|--------|
| bias via K tile instead of a bias helper | `add_bias_bcast_rows` | `bias_add_helpers.hpp:94-99`, `:162-168` | The helper is a separate partials -> out pass; on q/k cores its `out_buf` must feed rotary, i.e. a second Float32 block hand-off, and on every core it adds one Float32 unpack + FPU add + pack per output tile. The K-tile fold keeps the phase inside `matmul_block` (1 extra K step of 25). Regime R8. |
| v-core final pack | `matmul_block` `LastBlockTarget::Out` | `matmul_block_helpers.inl:415-418` (Out reloads the last partial), `:516-529` (Out drain + reload cadence) | The Out + L1-acc path spills through a FIFO interm whose in-place accumulation needs capacity == the core's actual block; with the uniform nominal-block contract that holds, but it forks the matmul path between v and q/k cores; `Interm` + caller-owned target + `convert_block` keeps one matmul path and a ragged-safe ring. |
| pack L1-acc disable | (no helper exposes it) | `bias_add_helpers.hpp:101-105` states the caller owns L1_ACC disable | raw `llk_pack_reconfig_l1_acc(0)` once per block |
| dataflow sin negation / ones tile / bias padding zero | no dataflow helper transforms tile contents | `tilize_helpers_dataflow.hpp`, `local_copy_helpers_dataflow.hpp` copy only | RISC word loop over the CB page (or MEM_ZEROS_BASE NoC read for the zero fill); once per row / tile |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| project_block (bias K tile) | matmul in0 x in1 | cb_activation_block ones tile: column 0 of all 32 rows (rest exactly 0) | cb_weight_block bias tile: Row0 (rows 1..31 zeroed by the W-reader) | none (matmul produces the row broadcast: out[i,j] += 1*b[j]) |
| rotate_block | mul (DEST-acc) | cb_matmul_partials: All | cb_rotary_table: All (row s of the table = token row s of the block row) | None |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Golden helper passes `packer_l1_acc=True` to `ttnn.ComputeConfigDescriptor` | the binding has no such kwarg (`program_descriptors.cpp:639-660`; `TypeError` reproduced); every golden row with an explicit config errors before the op runs, and the prompt's factory spec lists the field | entry point never depends on the field (`getattr` default True); report to the user: fix in `/golden-tests` (drop the kwarg) or extend `ComputeConfigDescriptor` + nanobind (C++ rebuild). The acceptance test does not pass the kwarg. Do not edit the golden suite. |
| Packer L1 acc left enabled after `matmul_block` | the helper leaves `l1_acc = 1` after the last K-block; the epilogue's pack would accumulate onto stale cb_out_tiles memory | `PACK((llk_pack_reconfig_l1_acc(0)))` right after each block's matmul |
| Partials ring | caller-owned target packs at absolute offsets from the reserve base; a non-nominal push shifts the next block's base | always reserve/push/pop `block_m*N_c` (nominal) |
| Reverse SrcOrder boot vs eltwise epilogue | matmul maps in0->SrcB, in1->SrcA; rotary needs SrcA = Float32 partials, SrcB = bf16 table | epilogue chain keeps `DataFormatReconfig::Enabled` on both inputs; next block's `matmul_block` re-issues its own reconfig + init (`INPUT_AND_OUTPUT`) |
| DEST-accumulating ELWMUL under fp32 DEST | correctness of `mul_init(..., acc_to_dest=1)` (`chain.inl:1400`) on Blackhole is the rotary's only non-matmul FPU dependency | first TDD stage tests rotary on a single head against torch; fallback L7 (two muls + SFPU add) |
| Ones tile / bias tile zero pattern | `0 * inf = NaN` if bias padding rows are non-finite; a ones tile with nonzeros outside column 0 multiplies W-side garbage | ones tile = column 0 only; W-reader zeroes bias rows 1..31 after the read barrier, before push |
| sin_lo sign flip in L1 | must happen after `noc_async_read_barrier` and before `cb_push_back` | reader owns the page between barrier and push |
| Nominal rows on short cores | stale in0 rows (possibly inf/NaN from a previous block) produce garbage partial rows | rows are independent; epilogue and writer touch only valid rows |
| Table row indexing | `st = r % St` uses the **x** tile-row count; tables may have `S_tab > S` | page `st*2 + dt` into the table's own tile grid; S_tab >= S validated |
| Writer pad zeroing (TARGET) | x padding can hold inf; q/k/v pad rows must be exactly 0 and finite | writer overwrites rows `[S%32, 32)` of the last tile-row per image in L1 before the NoC write (CB consumer owns the page) |
| L1 footprint with production L1 tensors | 8x512 keeps x + q/k/v (~230 KiB/core) in L1 beside 780 KiB of CBs | `cb_budget_bytes = 900 KiB` caps `block_m`; lamp to re-measure if the model's resident set grows |
| Three outputs through `generic_op` | the descriptor writes q, k, v; `generic_op` returns one tensor | allocate q, k, v before dispatch, pass all eight tensors in the io list, return the three allocated tensors |
