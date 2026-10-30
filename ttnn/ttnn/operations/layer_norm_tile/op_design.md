# Operation Design: layer_norm_tile

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (row reduction + broadcast eltwise), compute-bound at the target shapes |
| Goal | LayerNorm over the last dim of a TILE-layout, L1-interleaved bf16 tensor `(B, 1, S, 768)`, optionally on `x + residual`, with required affine `gamma`/`beta`; one `ttnn.generic_op` dispatch per call. |
| Math | `x' = x + residual` (if given); `mean = sum_j(x'_ij)/W`; `c = x' - mean`; `var = sum_j(c_ij^2)/W`; `y = c * rsqrt(var + eps) * gamma_j + beta_j` |
| Mode | Derivative (helper-library composition; no new LLK) |
| Public API | `from ttnn.operations.layer_norm_tile import layer_norm` (package = op dir, function = `layer_norm`, same convention as `ttnn.operations.layer_norm_rm`); the package also exports `default_compute_kernel_config` and `INPUT_TAGGERS` |
| Device measured at plan time | Blackhole, `compute_with_storage_grid_size() = 11x10` (110 cores), `get_l1_alignment() = 16`, `get_dram_alignment() = 64` (probe `tests/ttnn/unit_tests/operations/layer_norm_tile/probes/probe_002.py`). The grid is a runtime parameter, never a constant. |
| References | `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `.claude/references/ttnn-cb-memory-fundamentals.md`, `.claude/references/precision_convention.md`, `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp`, `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp`, `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp`, catalog `ttnn/ttnn/operations/examples/master.md` (entries `compute_block_size`, `row_reduce_accumulate`, `reduce_block`, `compute_fusion`, `eltwise_l1_vs_dest_accumulate`, `double_buffer`, `noc_placement`, `width_split`, `sfpu_tile_scope`) |
| feature_spec.py | `eval/golden_tests/layer_norm_tile/feature_spec.py` does **not exist** at planning time. This design names its axes with the vocabulary below so `/golden-tests` can author it against this document (see "Expected TARGET axes"). |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | bf16, TILE, interleaved (L1 target; DRAM accepted), rank 4 `(N, C, S, W)`, `W % 32 == 0`, `W == gamma volume` | - | addresses RT |
| `gamma` | `ttnn.Tensor` | yes | bf16, volume `W`; TILE with logical last dim `W` (e.g. `(768,)`, padded `[32, 768]` = 24 tile pages) **or** ROW_MAJOR with `shape[-1] % 32 == 0` (e.g. `(1,1,24,32)` or `(1,1,1,768)`); interleaved, DRAM or L1 | - | address RT, layout + stick width CT |
| `beta` | `ttnn.Tensor` | yes | same rules as `gamma`, independently (each has its own layout CT flag) | - | same |
| `residual_input_tensor` | `ttnn.Tensor` or `None` | no | same shape, dtype, layout, memory layout as `input_tensor` | `None` | presence CT (`HAS_RESIDUAL`), address RT |
| `epsilon` | `float` (kw-only) | no | `> 0` | `1e-5` | RT (fp32 bit pattern) so a new epsilon never recompiles |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` (kw-only) | no | any `math_fidelity`, any `math_approx_mode`; `fp32_dest_acc_en` in `{True, False}` | `default_compute_kernel_config()` = HiFi4, `fp32_dest_acc_en=True`, `math_approx_mode=False` | compute config (CT) |

Validation (entry point, before any device work):

| Check | Failure |
|-------|---------|
| `input_tensor.dtype != bfloat16` (also gamma/beta/residual dtype) | `ValueError` (requirement) |
| `input_tensor.layout != TILE_LAYOUT` (also residual) | `ValueError` (TILE is the only accepted input layout; no host-side conversion is ever performed) |
| rank != 4, `W % 32 != 0`, gamma/beta volume != `W`, residual shape mismatch, RM gamma with `shape[-1] % 32 != 0` | `ValueError` |
| sharded memory layout on any tensor | `NotImplementedError` (`UnsupportedAxisValue`) |
| resident predicate fails (`block_height_cap(INPUT_DEPTH_MAX) < 1`, see Regimes) | `NotImplementedError` (cannot happen for `W = 768`) |
| `compute_kernel_config` | never rejected on fidelity / approx mode; `fp32_dest_acc_en` read and honored (both values supported for bf16) |

The entry point never calls `to_layout` / `tilize` / `untilize` / `pad` / `slice`. Output is allocated with `ttnn.allocate_tensor_on_device` (shape = input logical shape, bf16, TILE, `input_tensor.memory_config()`), then exactly one `ttnn.generic_op` runs.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(N, C, S, W)`; target domain `N=B, C=1, W=768, S<=512, 1024 <= B*S <= 4096`, `S` any value (non-tile-aligned accepted) |
| Dtype | bfloat16 |
| Layout | TILE (each `(n, c)` image tile-padded independently: `ceil(S/32)` tile-rows, `W/32` tile-cols) |
| Memory | interleaved; L1 is the target placement |

### Output

| Property | Value |
|----------|-------|
| Shape | identical to input (logical); padded identically |
| Dtype | bfloat16 (= input dtype) |
| Layout | TILE |
| Memory | `input_tensor.memory_config()` (L1 interleaved for the target) |

### Expected TARGET axes (for `/golden-tests`; feature_spec.py absent)

| Axis | Values | How this design treats it |
|------|--------|---------------------------|
| `dtype` | `[ttnn.bfloat16]` | only dtype; others are a `ValueError` (outside the universe, not a refinement) |
| `fp32_dest_acc_en` | `[True, False]` | both in Phase 0; intermediate CB formats follow it (Float32 / Float16_b) |
| `layout` | `[ttnn.TILE_LAYOUT]` | only layout |
| `alignment` | `["tile_aligned", "h_non_aligned"]` | `W` is always tile-aligned (768); `S % 32 != 0` is handled natively (see Key Risks: padded rows) |
| `memory_layout` | `[INTERLEAVED]` (buffer type L1) | sharded values are regime rows below, not Phase 0 |
| `residual` | `["none", "present"]` | compile-time `HAS_RESIDUAL` |
| `affine_layout` | `[ttnn.TILE_LAYOUT, ttnn.ROW_MAJOR_LAYOUT]` | gamma/beta layouts; per-tensor CT flag in the writer |

Structural impossibilities: none beyond the universe itself (one dtype, one layout).

## Blocking Model

The work unit is a **tile-row**: 32 rows x the full `W` (= `tensor_w_tiles` = 24 tiles). Every statistic is per row and spans the whole of `W`, so `W` is the dependent axis; everything above a tile-row is independent.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `N` (batch) | independent - no statistic crosses images | folded into `tile_rows` (images are contiguous in page space: global tile-row `g = (n*C + c)*ceil(S/32) + s_tile`) | folded | host: `tensor_rows_tiles = N*C*ceil(S/32)` | via `tile_rows` | knob-turn |
| `C` (=1) | independent - same as `N` | folded into `tile_rows` | folded | same | via `tile_rows` | knob-turn |
| `tile_rows` (S tiles, all images) | independent - each row normalizes alone | `block_height` (tile-rows per block) | `min(max_core_rows_tiles, block_height_cap(depth))` - the whole per-core assignment when it fits (see derivation below) | host function `derive_blocking()`; CT arg to all three kernels | `split_work_to_cores(grid, tensor_rows_tiles, row_wise=True)`; each core owns a contiguous range `[start_row, start_row + core_rows_tiles)` | knob-turn (raise/lower `block_height`, grow grid) |
| `intra_tile_rows` (32 rows inside a tile) | independent - rows never mix | tile height | 32 (whole tile) | tile shape of every CB (32x32) | not split | scheme-change: `half_tile_rows` regime (deferred) |
| `W` (hidden, `tensor_w_tiles` = 24) | dependent - mean and variance span all of `W` | `block_width` | `tensor_w_tiles` (whole extent, resident) | CT `tensor_w_tiles`, derived from `input.padded_shape[-1] // 32` | not split across cores | scheme-change: `w_split_combine` regime (deferred) |
| `W` of `gamma`/`beta` | reuse-shared - does not vary along `tile_rows`, the split axis, so every active core needs all of it | `tensor_w_tiles` (one row-0 tile per tile-column) | whole extent, fetched once per core, resident for the kernel lifetime | same CT | every active core fetches its own copy (1.5-3 KiB) | scheme-change: `affine_mcast` regime (deferred) |

Stage data (re-running the table on what each phase produces):

| Stage data | Axes | Blocking / assignment | Why |
|------------|------|------------------------|-----|
| `x + residual` (`cb_xsum`) | `tile_rows` x `W` | same block as input, same core | elementwise of input; independent rows stay on the owning core |
| `mean`, `rstd`, `sumsq` partial (`cb_mean`, `cb_rstd`, `cb_sumsq`) | `tile_rows` only (`W` collapsed to one col-0-valid tile) | `block_height` tiles per block, same core | per-row results; nothing to spread beyond the row split |
| `c`, `c*rstd`, `c*rstd*gamma` (`cb_centered` / `cb_normed` / `cb_scaled`) | `tile_rows` x `W` | same block, same core | elementwise |

No core collects, coordinates or hands out work in Phase 0: every core runs the whole phase sequence on its own rows, so no stage computes on one core while others wait.

Blocking derivation (host, single source of truth; every CB size and loop extent below is derived from these):

```
T_in  = ttnn.tile_size(bfloat16)                        # 2048
T_acc = 4096 if fp32_dest_acc_en else 2048              # Float32 / Float16_b page
row_bytes(depth) = tensor_w_tiles * (depth * T_in * (1 + HAS_RESIDUAL) + T_acc) + T_acc
fixed_bytes      = tensor_w_tiles * T_in * (2 + OUTPUT_DEPTH)
block_height_cap(depth) = (CB_L1_BUDGET_BYTES - fixed_bytes) // row_bytes(depth)

max_core_rows_tiles = rows of core group 1 from split_work_to_cores
if max_core_rows_tiles <= block_height_cap(1):
    input_depth = 1; block_height = max_core_rows_tiles          # one block per core
else:
    input_depth = INPUT_DEPTH_MAX; block_height = min(max_core_rows_tiles, block_height_cap(INPUT_DEPTH_MAX))
num_blocks_this_core = ceil(core_rows_tiles / block_height)      # per core, RT
last_block_extent    = core_rows_tiles - (num_blocks_this_core - 1) * block_height   # per core, RT
```

Host constants (each defined once): `CB_L1_BUDGET_BYTES = 768 * 1024`, `INPUT_DEPTH_MAX = 2`, `OUTPUT_DEPTH = 2`. Phase 0 values on the 110-core grid: every target shape with `B*ceil(S/32) <= 220` (all power-of-two `S`: at most 128 tile-rows) has `max_core_rows_tiles <= 2` and `block_height_cap(1) >= 2` (worst case fp32 + residual: `row_bytes(1) = 200704`, `fixed_bytes = 196608`, cap = 2), so those shapes run **one block per core containing the core's whole assignment** with `input_depth = 1`; small odd `S` (e.g. `S = 33`, `B = 124`: 248 tile-rows) can exceed the cap and take the multi-block path.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_input_tiles` | `input_depth` (derived; max `INPUT_DEPTH_MAX`) | 1 when one block per core, else 2 | depth 2 lets block `b+1`'s reads overlap block `b`'s compute; depth 1 when there is no next block (no L1 spent on an overlap that cannot happen) |
| `cb_residual_tiles` | `input_depth` (same knob) | same | same |
| `cb_output_tiles` | `OUTPUT_DEPTH` (tile-rows) | 2 | compute packs tile-row `r+1` while the writer drains tile-row `r` |
| work / stats alias groups, gamma/beta | none (block- or kernel-resident) | 1 | - |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| `reduce_mean` / `reduce` `AccumulateViaAdd` aligned path (no partial mask) | `W % 32 == 0` | validate rejects other `W`; no mask CB exists | padded W lanes would be summed into mean/var (wrong results) |
| `reduce_mean` with `WaitUpfrontNoPop` bulk-reserves all outputs (`reduce_helpers_compute.inl:342`) | `cb_mean` capacity >= rows in the block | stats group capacity = `block_height` | reserve blocks forever (hang) |
| eltwise chain `Upfront` + `Block` wait of `Ht*Wt` tiles (`chain.inl:59-63`) | `cb_centered` capacity >= `block_height * tensor_w_tiles` | work group capacity = `block_height * tensor_w_tiles` | hang |
| BinaryFpu DEST accumulation requires `Dst::D0` (`chain.inl:1273-1275`) and PackTile `(PerOuter, PerOuter)` for `PerRow` (`chain.inl:1142-1148`) | one live accumulator per tile-row | `sumsq_block` uses `PerRow`, one tile out per tile-row | compile error |
| Gamma/beta row extraction: DRAM reads need 64 B alignment (BH), L1 16 B | gamma chunk transfer size | every gamma/beta NoC read is 64 B at a 64 B-aligned source offset (TILE: page offsets 0 and 512; RM: `((32k) % stick_w) * 2`, which is a multiple of 64) | misaligned NoC read (wrong data / NoC error) |
| RM gamma: a 32-element chunk must lie in one stick | `stick_w % 32 == 0` | validate | chunk straddles two pages |
| L1 budget | `block_height` | `block_height_cap(depth)` above; validate requires `block_height_cap(INPUT_DEPTH_MAX) >= 1` | CB/L1-buffer collision at program creation |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `row_resident` | **built** | `block_height_cap(INPUT_DEPTH_MAX) >= 1` (always true for `W = 768`: worst case `block_height = 1`, `input_depth = 2`, fp32 + residual = 495,616 B < 786,432 B budget) | `block_height x tensor_w_tiles`, whole `W` resident across all phases | Minimum against the remote-L1-bank (NoC) boundary: input, residual and output tiles each cross once. Above minimum only by the reuse-shared affine rows: 2 x (1.5 KiB RM or 3 KiB TILE) per active core instead of once per op (deferred `affine_mcast`). | The 8 compute phases each pay one init + format reconfig + pipeline fill/drain per block (catalog `compute_block_size`: ~1.6 us per extra pass on a 5-phase chain); a block of the whole assignment pays them once per core. Reader/writer barriers are per tile-row regardless. |
| `w_split_combine` | **deferred** - it is a scheme-change whose two cross-core combines (row-sum partials, then centered sum-of-squares partials) reorder the floating-point summation of both statistics; the precision baseline those reorders are gated on is established by `row_resident` first. It layers on `row_resident` (stepping stone): the per-core phase sequence and CB inventory survive, with `block_width = tensor_w_tiles / k`. | `tensor_rows_tiles < num_cores` (target end: 32 tile-rows on 110 cores = 29% occupancy) or `ceil(tensor_rows_tiles / num_cores) > 1` with a ragged tail (128 tile-rows: makespan 2 tile-rows, `k = 3` gives 1.33) | `block_height x (tensor_w_tiles / k)`, `k` in divisors of 24 | Input/output same as minimum; adds per core per tile-row 2 rounds x (k-1) x `T_acc` (unicast all-gather of col-0-valid partials) - `k = 3`, fp32: 16 KiB vs 32 KiB of the core's own input+output (+50% NoC bytes); affine rows shrink by `1/k`. | Same per-phase amortization; plus each combine round's rendezvous is paid once per block, so larger `block_height` amortizes the two semaphore round trips over more tile-rows. |
| `half_tile_rows` | **deferred** - the compute helpers have no 2-face (16x32) tile handling (no `num_faces` / tile-shape handling in `eltwise/core/chain.inl` or `reduce_helpers_compute.inl`), so it needs a new helper capability, not a knob. | `tensor_rows_tiles < num_cores` | 16-row half tiles (faces 0-1 / 2-3 are contiguous 1 KiB halves of a bf16 tile) | identical bytes to `row_resident`; transfer count x2 (1 KiB reads) | doubles row parallelism with no combine |
| `row_streaming_reread` | **rejected** - superseded by `row_resident` on the whole domain (W = 768 always fits resident). Re-reading the input once per statistic pass is the dead end. | `block_height_cap(INPUT_DEPTH_MAX) < 1` (unreachable) | 1 tile-row, `W` streamed | input crosses the NoC 3x (mean pass, variance pass, normalize pass) | - |
| `one_pass_moments` (`E[x^2] - E[x]^2`) | **rejected** - catastrophic cancellation when `|mean| >> std`; the requirement defines the two-pass centered variance. | - | - | same bytes as `row_resident` | - |
| `affine_mcast` | **deferred** - the gamma/beta fetch is already off the critical path: the writer RISC fetches it at kernel start while the reader fetches block 0 and compute runs phases `add_residual..rstd` that do not need it (stall-shadow); it is 3-6 KiB per core vs >= 96 KiB of the core's own input+output. Reachable: an injector core reads the rows once and multicasts into the same `cb_gamma_rows`/`cb_beta_rows` (`ttnn.Mcast` + `McastArgs`); nothing downstream changes. | all shapes | Wt row-0 tiles | saves `(active_cores - 1) x 2 x (1.5 or 3) KiB` of NoC reads; adds one semaphore rendezvous per kernel | - |
| `height_sharded` (physical shard of `tile_rows`) | **deferred** - requirement scope is L1 interleaved. Knob-turn: the shard is the per-core assignment; `cb_input_tiles` / `cb_residual_tiles` / `cb_output_tiles` become `cb_descriptor_from_sharded_tensor` (zero-copy), reader/writer NoC loops disappear for those tensors, block = whole shard (clamped by `block_height_cap`). | input `memory_layout == HEIGHT_SHARDED` with shard height a multiple of 32 and shard width = W | shard | input/output cross 0x (resident) | - |
| `width_sharded` / `block_sharded` | **deferred** - cuts `W` (dependent): it is `w_split_combine` with the caller pinning `k` and the placement; same reason as `w_split_combine`. | input `memory_layout in {WIDTH_SHARDED, BLOCK_SHARDED}` | shard | as `w_split_combine`, input/output 0x | - |

Regime selection: Phase 0 has one built regime; `validate()` refuses everything outside it. The block-size selection inside `row_resident` (single block, `input_depth = 1` vs. multi-block, `input_depth = 2`) is grid-dependent and must be **pinned by test**: `(256, 1, 16, 768)` with residual (256 tile-rows) is multi-block on every grid with fewer than 128 cores (`block_height_cap(1) = 2` at fp32 + residual); every other acceptance shape is single-block on the 110-core grid.

### Traffic ranking

Tier: input / residual / output live in remote L1 banks (interleaved over all cores), so the expensive tier is the NoC crossing to the computing core; TILE gamma/beta live in DRAM.

| Rank | Candidate split | Bytes crossing NoC (per op) | Fan-out / rendezvous | Verdict |
|------|-----------------|-----------------------------|----------------------|---------|
| 1 | `tile_rows` across cores (`row_resident`) | `3 x tensor_bytes` with residual (in + res + out), `2 x` without; + affine rows 3-6 KiB per active core | none | **chosen** - minimum bytes, no combine |
| 2 | `intra_tile_rows` (half tiles) | same bytes, 2x transfers | none | deferred (helper gap) |
| 3 | `W` across `k` cores + combine | rank 1 + `2 (k-1) T_acc` per core per tile-row (+50% at k=3 fp32) | 2 rendezvous per block | deferred (precision-baseline gate); it is the available parallelism where rank 1 under-fills the grid |
| 4 | `tile_rows` x `W` (2-D) | rank 3's combine on a subset | 2 rendezvous per block | covered by `w_split_combine` with `k` chosen per shape |
| 5 | streaming re-read | `3 x` input | none | rejected (dead end) |

The chosen split is also the cheapest-traffic split. Occupancy is accounted separately (it motivates `w_split_combine`, it does not re-rank traffic).

### Block schedule

```cpp
fetch_affine_rows();                         // writer, once per kernel (stall-shadowed)
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {
    const uint32_t block_rows = (block_idx + 1 == num_blocks_this_core) ? last_block_extent : block_height;
    load_block(block_idx);                   // reader
    add_residual_block(block_rows);          // compute, HAS_RESIDUAL only
    mean_block(block_rows);                  // compute
    center_block(block_rows);                // compute
    sumsq_block(block_rows);                 // compute
    rstd_block(block_rows);                  // compute
    normalize_block(block_rows);             // compute
    scale_block(block_rows);                 // compute
    shift_block(block_rows);                 // compute
    store_block(block_idx);                  // writer
}
```

| Block operation | Kernel | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|--------|-------------|--------------------|-----------------------------------|
| `fetch_affine_rows` | writer | `tensor_w_tiles` row-0 tiles each of gamma and beta | - | once per kernel: 2 x 24 chunk reads (64 B; RM adds a core-local 32 B copy per chunk), one barrier per tensor, one push of `tensor_w_tiles` per tensor |
| `load_block` | reader | `block_rows x tensor_w_tiles` input tiles (+ same residual tiles) | - | one reserve/push of `tensor_w_tiles` pages per tile-row per tensor; one read barrier per tile-row (24 or 48 reads in flight) |
| `add_residual_block` | compute | `block_rows * tensor_w_tiles` tiles | - | one init + reconfig per block; streams both inputs per tile |
| `mean_block` | compute | `block_rows x tensor_w_tiles` -> `block_rows` col-0 tiles | x (`cb_input_tiles` or `cb_xsum`) stays resident (no pop) | one helper call per block; SFPU finalize init hoisted per call (`reduce_helpers_compute.inl` "one-time SFPU-macro load ... hoisted OUT of the per-output loop") |
| `center_block` | compute | `grid(block_rows, tensor_w_tiles)` | `cb_mean` consumed per tile-row | one init per block; pops x per tile |
| `sumsq_block` | compute | `grid(block_rows, tensor_w_tiles)` -> `block_rows` tiles | `cb_centered` resident (no pop) | one init per block; one DEST acquire + one pack per tile-row |
| `rstd_block` | compute | `block_rows x 1` -> `block_rows` col-0 tiles | `cb_centered` resident | one helper call per block; post-op SFPU inits per output |
| `normalize_block` | compute | `grid(block_rows, tensor_w_tiles)` | `cb_rstd` consumed per tile-row | one init per block |
| `scale_block` | compute | `grid(block_rows, tensor_w_tiles)` | `cb_gamma_rows` resident (kernel lifetime) | one init per block |
| `shift_block` | compute | `grid(block_rows, tensor_w_tiles)` | `cb_beta_rows` resident | one init per block; pushes output per tile |
| `store_block` | writer | `block_rows x tensor_w_tiles` | - | one wait/pop of `tensor_w_tiles` pages per tile-row; one write barrier per tile-row |

No multicast, no semaphores, no cross-core barriers in Phase 0.

**Stall-shadow decisions.**

| Waiting stage | What it waits for | Work scheduled into the wait | Why it is legal |
|---------------|-------------------|------------------------------|-----------------|
| compute at kernel start | block 0 input from the reader | nothing on compute; the **writer** fetches gamma/beta rows during this window instead of the reader, so the reader issues block 0 first | gamma/beta are consumed only by `scale_block`/`shift_block`; the five phases before them do not depend on the affine fetch |
| `scale_block` of block 0 | `cb_gamma_rows` push | `add_residual..normalize` of block 0 | same dependence argument; no floating-point reorder |
| reader during compute of block `b` | free CB space | prefetch block `b+1` (only when `num_blocks_this_core > 1`, `input_depth = 2`) | rows are independent |
| (deferred `w_split_combine`) each combine rendezvous | peer partials | the next tile-row's local partial (`mean_block` of row `r+1` while row `r`'s mean combine is in flight) | rows are independent; summation order within a row unchanged |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap | `block_height` = whole assignment means a 2-row core reads both rows before `mean_block` starts (no-residual path waits upfront) and writes nothing until `shift_block` | `block_height = 1`, `input_depth = 2` on shapes with 2 rows/core, e.g. `(16,1,256,768)` |
| Phase E fusion | default is three FPU passes with L1 round trips (`normalize`, `scale`, `shift`) per catalog `compute_fusion` (FPU dest-reuse lost 0.82x vs the L1 round trip) | one chain: `BinaryFpu Mul (rstd Col bcast)` -> `DestReuseBinary Mul gamma` -> `DestReuseBinary Add beta` -> pack, with gamma/beta replicated to all 32 rows once per kernel (`UnaryBcast<Row>`) |
| Data-format reconfig | reduce helpers default to `INPUT_AND_OUTPUT` reconfig (catalog `compute_block_size` second lever: up to 1.19x when formats are constant) | `ReduceDataFormatReconfigMode::NONE`/`INPUT` where the previous phase already programmed the same formats |
| rsqrt scope | `rstd_block` post-op runs `rsqrt` on all 32 SFPU vectors of a col-0-valid tile (catalog `sfpu_tile_scope`: col-0 scoping ~3.8x on the SFPU part) | col-0-scoped rsqrt (raw sfpi stride) in the post-op |
| Grid synchronization | does not apply in Phase 0 (no collective); applies to `w_split_combine` when built (compare `k` = max vs. smaller `k`) | - |

Catalog entries that informed the knobs: `compute_block_size` (block = whole assignment; one helper call per phase per block), `row_reduce_accumulate` / `reduce_block` (`AccumulateViaAdd` for the 24-tile row mean: 2.9-5.4x over the matmul reduce at 32 tiles, more accurate in bf16 DEST), `eltwise_l1_vs_dest_accumulate` (sum of squares accumulated in DEST, one pack per tile-row), `compute_fusion` (FPU consumers get L1 round trips, SFPU epilogue fused as reduce post-op), `double_buffer` (batch reads per tile-row then one barrier; double-buffer only when a next block exists), `noc_placement` (`row_wise=True`), `width_split` (evidence for `w_split_combine` when rows are few), `sfpu_tile_scope` (rsqrt lamp).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| input / residual L1 banks -> core | bf16 tiles | reader (NCRISC, NoC0), `TensorAccessor` page reads, page id `g * tensor_w_tiles + w` for global tile-row `g` | contiguous page range per core; one barrier per tile-row |
| gamma / beta -> core | bf16 row-0 chunks into `cb_gamma_rows` / `cb_beta_rows` tile `k` | writer (BRISC), once per kernel | TILE source: 64 B from page `k` offset 0 -> tile offset 0 (face 0 rows 0-1), 64 B from page `k` offset 512 -> tile offset 512 (face 1 rows 0-1). RM source: 64 B chunk `k` (elements `32k..32k+31`, page `32k // stick_w`, byte offset `((32k) % stick_w) * 2`) -> tile offset 0, then a core-local copy of tile bytes `[32, 64)` to tile offset 512 (self-aimed read via `local_addr()`). Only face 0 row 0 and face 1 row 0 are meaningful; Row broadcast reads nothing else. |
| reader -> compute | bf16 | `cb_input_tiles`, `cb_residual_tiles` | |
| compute internal | `T_acc` format (Float32 if `fp32_dest_acc_en` else Float16_b) | alias groups `work` (`cb_xsum`, `cb_centered`, `cb_normed`, `cb_scaled`) and `stats` (`cb_mean`, `cb_sumsq`, `cb_rstd`) | each alias id gives a real push/wait edge between phases over the same bytes; all ids of a group move identical quanta (lockstep) |
| compute -> writer | bf16 | `cb_output_tiles` | |
| core -> output L1 banks | bf16 tiles | writer (BRISC, NoC1), `TensorAccessor` page writes | whole tiles including padded rows |
| (deferred `w_split_combine`) core <-> core | `T_acc` col-0-valid partial tiles | each core of a `k`-group unicasts its partial into a dedicated receive CB slot on each peer (`k-1` slots), increments the peer's semaphore; each core waits for `k-1`, then sums the `k` partials with `reduce<SUM, REDUCE_ROW, ..., AccumulateViaAdd, ReduceWithinTile::Skip>` (+ `1/W` post-op) - every core computes the combine itself (no coordinator). The receive CB is a dedicated handoff CB, never the accumulator. | two rounds per block: `mean_block` partial, then `sumsq_block` partial |
| (deferred `height_sharded`) | shard in place | `cb_descriptor_from_sharded_tensor` for input / residual / output | no NoC re-read of a resident shard |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | one block = `block_height x tensor_w_tiles` tiles (the block is the unit that moves and computes) |
| Grid | `device.compute_with_storage_grid_size()` (runtime parameter; 11x10 here) |
| Split | `ttnn.split_work_to_cores(grid, tensor_rows_tiles, row_wise=True)` -> `all_cores`, group 1 (`rows_g1` each), group 2 (`rows_g2 = rows_g1 - 1` each); cores are enumerated in the split's order and each gets `start_row = sum of earlier cores' rows` |
| Per-core work | `core_rows_tiles` contiguous global tile-rows; `num_blocks_this_core = ceil(core_rows_tiles / block_height)`; `last_block_extent` as above |
| Tile geometry | `tensor_rows_tiles = N * C * ceil(S / 32)` (per-image `ceil`, never `floor(N*C*S/32)`); `tensor_w_tiles = W / 32` (validated aligned; derived from `padded_shape[-1] // 32`) |
| Remainder | ragged last block per core runs the same block operations with `block_rows = last_block_extent`; it is always the core's final block, so a partial CB lap at the end of the kernel is harmless and identical across every id of an alias group |
| Inactive cores | not in `all_cores`; no kernels placed there |

## Circular Buffers

`T_in` = bf16 tile (2048 B). `T_acc` = 4096 B (Float32) if `fp32_dest_acc_en` else 2048 B (Float16_b). `Wt` = `tensor_w_tiles`.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|----------|
| `cb_input_tiles` | 0 | `T_in` | `input_depth * block_height * Wt` | spans `tile_rows` (block) and `W` (whole): no-residual `mean_block` reads the whole block resident; depth for next-block prefetch | bf16 | reader | compute | per block (`load_block` .. `center_block`, or `add_residual_block` with residual) |
| `cb_residual_tiles` | 1 | `T_in` | `input_depth * block_height * Wt` (allocated only if `HAS_RESIDUAL`) | streamed by `add_residual_block`; sized like the input so the reader fills both in one pass per tile-row | bf16 | reader | compute | per block |
| `cb_gamma_rows` | 2 | `T_in` | `Wt` | one row-0 tile per tile-column, resident | bf16 | writer | compute | kernel |
| `cb_beta_rows` | 3 | `T_in` | `Wt` | same | bf16 | writer | compute | kernel |
| `cb_xsum` | 4 | `T_acc` | `block_height * Wt` (alias group `work`) | x+res, resident through `mean_block` | `T_acc` | compute | compute | `add_residual_block` .. `center_block` |
| `cb_centered` | 5 | `T_acc` | alias group `work` | `c`, resident through `sumsq_block` | `T_acc` | compute | compute | `center_block` .. `normalize_block` |
| `cb_normed` | 6 | `T_acc` | alias group `work` | `c * rstd` | `T_acc` | compute | compute | `normalize_block` .. `scale_block` |
| `cb_scaled` | 7 | `T_acc` | alias group `work` | `c * rstd * gamma` | `T_acc` | compute | compute | `scale_block` .. `shift_block` |
| `cb_mean` | 8 | `T_acc` | `block_height` (alias group `stats`) | one col-0 tile per tile-row; helper bulk-reserves all | `T_acc` | compute | compute | `mean_block` .. `center_block` |
| `cb_sumsq` | 9 | `T_acc` | alias group `stats` | one DEST-accumulated tile per tile-row | `T_acc` | compute | compute | `sumsq_block` .. `rstd_block` |
| `cb_rstd` | 10 | `T_acc` | alias group `stats` | one col-0 tile per tile-row | `T_acc` | compute | compute | `rstd_block` .. `normalize_block` |
| `cb_output_tiles` | 16 | `T_in` | `OUTPUT_DEPTH * Wt` | streaming window over `tile_rows`, spans `W` (writer pops one tile-row) | bf16 | compute | writer | per tile-row |

Alias groups are one `ttnn.CBDescriptor` each with several `CBFormatDescriptor`s (identical `page_size = T_acc`, `total_size = block_height * Wt * T_acc` for `work`, `block_height * T_acc` for `stats`). Without residual, `cb_xsum` is still declared in the group and simply carries no traffic.

Ring closure: every id pushes and pops exactly `block_rows * Wt` (work) / `block_rows` (stats) pages per block, i.e. one full lap per full block; reader pushes and writer pops in quanta of `Wt`, and `cb_input_tiles` / `cb_output_tiles` capacities are multiples of `Wt`.

Full ledger: `l1_ledger.md`.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `fetch_affine_rows` | `Wt` tiles x 2 | dataflow (raw NoC) | gamma / beta tensors | `cb_gamma_rows` (`Wt`), `cb_beta_rows` (`Wt`) | resident until kernel end; compute pops both once after the last block |
| 1 | `load_block` | `block_rows x Wt` | dataflow (raw NoC + `TensorAccessor`) | input / residual tensors | `cb_input_tiles` (`block_rows*Wt`), `cb_residual_tiles` (`block_rows*Wt`) | pushed per tile-row |
| 2 | `add_residual_block` | `tiles(block_rows*Wt)` | helper `add` | `cb_input_tiles` + `cb_residual_tiles`, streamed, popped per tile | `cb_xsum` (`block_rows*Wt`) | inputs freed; `cb_xsum` full |
| 3 | `mean_block` | `of(block_rows, Wt)` | helper `reduce_mean` | `x_src` = `cb_xsum` (residual) or `cb_input_tiles`; `WaitUpfrontNoPop` | `cb_mean` (`block_rows`) | `x_src` still full |
| 4 | `center_block` | `grid(block_rows, Wt)` | helper `sub` | `x_src` streamed + popped per tile; `cb_mean` Col bcast, popped per tile-row | `cb_centered` (`block_rows*Wt`) | `x_src` empty, `cb_mean` empty |
| 5 | `sumsq_block` | `grid(block_rows, Wt)` | helper `eltwise_chain` (`BinaryFpu Mul` same CB both operands, `DestAccumulation::PerRow`) | `cb_centered`, `Upfront`/`None` (stays full) | `cb_sumsq` (`block_rows`) | `cb_centered` still full |
| 6 | `rstd_block` | `of(block_rows, 1)` | helper `reduce` (`AccumulateViaAdd`, post-op `x * (1/W) + eps -> rsqrt`) | `cb_sumsq`, `BulkWaitBulkPop` | `cb_rstd` (`block_rows`) | `cb_sumsq` empty |
| 7 | `normalize_block` | `grid(block_rows, Wt)` | helper `mul` | `cb_centered` streamed + popped; `cb_rstd` Col bcast, popped per tile-row | `cb_normed` | `cb_centered`, `cb_rstd` empty |
| 8 | `scale_block` | `grid(block_rows, Wt)` | helper `mul` | `cb_normed` streamed + popped; `cb_gamma_rows` Row bcast, `Upfront`/`None`, `Row` mapping | `cb_scaled` | gamma still resident |
| 9 | `shift_block` | `grid(block_rows, Wt)` | helper `add` | `cb_scaled` streamed + popped; `cb_beta_rows` Row bcast, `Upfront`/`None`, `Row` mapping | `cb_output_tiles` (per tile) | beta still resident |
| 10 | `store_block` | `block_rows x Wt` | dataflow | `cb_output_tiles`, wait/pop `Wt` per tile-row | output tensor | - |

Compute kernel boot: `compute_kernel_hw_startup(cb_input_tiles, cb_input_tiles, cb_centered)` once, first statement; no re-call.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot | raw_api | `compute_kernel_hw_startup` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:31-35` (contract) | `(cb_input_tiles, cb_input_tiles, cb_centered)` | - | - | - |
| `add_residual_block` | helper | `compute_kernel_lib::add` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:52` | `<input(cb_input_tiles), input(cb_residual_tiles), output(cb_xsum)>(IterationShape::tiles(block_rows * Wt), ...)` | `cb_input_tiles`, `cb_residual_tiles` | `cb_xsum` | `block_rows` |
| `mean_block` | helper | `compute_kernel_lib::reduce_mean` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:691` | `<REDUCE_ROW, x_src, cb_gamma_rows /*scaler id, never read*/, cb_mean, ReduceInputPolicy::WaitUpfrontNoPop>(ReduceInputBlockShape::of(block_rows, Wt), W)`; algorithm defaults to `AccumulateViaAdd` (`:688`); all four policies legal for it (`reduce_helpers_compute.inl:829`); the scaler CB is waited only for a partial mask or `CopySeedZeroPair` (`reduce_helpers_compute.inl:334-337`), neither used, so no scaler CB is allocated | `x_src` | `cb_mean` | `block_rows` (rows of the shape) |
| `center_block` | helper | `compute_kernel_lib::sub` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:61` | `<input(x_src), input(cb_mean, BroadcastDim::Col, WaitPolicy::PerTile, PopPolicy::PerTile, InputTileMapping::Col), output(cb_centered)>(IterationShape::grid(block_rows, Wt), ...)` (`PerTile+Col` streams one tile per grid row: `chain.hpp:221-228`, legal per `chain.inl:67-68`) | `x_src`, `cb_mean` | `cb_centered` | `block_rows` |
| `sumsq_block` | helper | `compute_kernel_lib::eltwise_chain` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp:620` | `grid(block_rows, Wt)`, `BinaryFpu<BinaryFpuOp::Mul, input(cb_centered, Upfront, None, Block), input(cb_centered, Upfront, None, Block), Dst::D0, DestAccumulation::PerRow>{c, c}`, `PackTile<output(cb_sumsq, ReservePolicy::PerOuter, PushPolicy::PerOuter, Enabled, Direct, DestAccumulation::PerRow)>{sumsq}`; `mul_init(..., acc_to_dest)` at `chain.inl:1400`; same-CB dedup at `chain.inl:1285-1292` | `cb_centered` | `cb_sumsq` | `block_rows` |
| `rstd_block` | helper | `compute_kernel_lib::reduce` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:637` | `<SUM, REDUCE_ROW, cb_sumsq, cb_gamma_rows /*never read*/, cb_rstd, ReduceInputPolicy::BulkWaitBulkPop, INPUT_AND_OUTPUT, ReduceFp32Mode::Fast, ReduceAlgorithm::AccumulateViaAdd, ReduceWithinTile::Collapse>(ReduceInputBlockShape::of(block_rows, 1), contiguous(), NoAccumulation{}, post_op)`; `post_op(dst)`: `binop_with_scalar_tile_init(); mul_unary_tile(dst, inv_w_bits); add_unary_tile(dst, eps_bits); rsqrt_tile_init(); rsqrt_tile(dst);` (`tt_metal/hw/inc/api/compute/eltwise_unary/binop_with_scalar.h:156,60,35`, `.../rsqrt.h:21,44`); post-op contract `reduce_helpers_compute.hpp:519-523` | `cb_sumsq` | `cb_rstd` | `block_rows` |
| `normalize_block` | helper | `compute_kernel_lib::mul` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:70` | `<input(cb_centered), input(cb_rstd, BroadcastDim::Col, PerTile, PerTile, InputTileMapping::Col), output(cb_normed)>(grid(block_rows, Wt), ...)` | `cb_centered`, `cb_rstd` | `cb_normed` | `block_rows` |
| `scale_block` | helper | `compute_kernel_lib::mul` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:70` | `<input(cb_normed), input(cb_gamma_rows, BroadcastDim::Row, WaitPolicy::Upfront, PopPolicy::None, InputTileMapping::Row), output(cb_scaled)>(grid(block_rows, Wt), ...)` (`Row` mapping legal with `Upfront/None`: `chain.inl:64-66`) | `cb_normed`, `cb_gamma_rows` | `cb_scaled` | `block_rows` |
| `shift_block` | helper | `compute_kernel_lib::add` | `ttnn/cpp/ttnn/kernel_lib/eltwise/api/convenience.hpp:52` | `<input(cb_scaled), input(cb_beta_rows, BroadcastDim::Row, Upfront, None, InputTileMapping::Row), output(cb_output_tiles)>(grid(block_rows, Wt), ...)` | `cb_scaled`, `cb_beta_rows` | `cb_output_tiles` | `block_rows` |
| `fetch_affine_rows`, `load_block`, `store_block` | raw_api | `noc_async_read` / `noc_async_write` / barriers + `TensorAccessor` | `tt_metal/hw/inc/api/dataflow/dataflow_api.h`; `tech_reports/tensor_accessor/tensor_accessor.md` | page ids and sub-page offsets as in Dataflow Strategy | - | - | `block_rows` (reader/writer loop extents) |

Helpers considered and rejected:

| Raw-API entry | Candidate helper | Mismatch (file:line) | Why it cannot express the shape |
|---------------|------------------|----------------------|----------------------------------|
| `fetch_affine_rows` (remote chunk reads) | `dataflow_kernel_lib` tilize dataflow helpers (`tilize_helpers_dataflow.hpp`) | they stage whole sticks for a compute tilize, not sub-page chunks at face offsets | the transfer is a 64 B sub-page chunk of a remote page into face 0 / face 1 row 0 of a CB tile. The RM path's core-local 32 B copy (tile bytes `[32,64)` -> offset 512) **does** use the helper: a self-aimed read built with `dataflow_kernel_lib::local_addr()` (`local_copy_helpers_dataflow.hpp:57`). |
| `load_block` / `store_block` | none in `kernel_lib` for interleaved whole-tile page streams | - | plain `TensorAccessor` page loop with one barrier per tile-row is the pattern (`double_buffer` catalog) |
| `rstd_block` post-op raw SFPU calls | `eltwise_chain` `MulUnary` / `AddUnary` / `Rsqrt` (`eltwise/unary/scalar.inl:54-56`, `eltwise/unary/math.inl:60-68`) | the reduce post-op hook is a callable on a DEST index (`reduce_helpers_compute.hpp:519-523`), not a chain | fusing into the reduce's DEST window (catalog `compute_fusion`: SFPU consumers fuse) avoids a pack/unpack of `block_rows` tiles; the same functions the chain elements wrap are called directly |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `add_residual_block` | add | `cb_input_tiles` All | `cb_residual_tiles` All | None |
| `center_block` | sub | `x_src` All | `cb_mean` Col0 (REDUCE_ROW out) | Col |
| `sumsq_block` | mul (DEST acc) | `cb_centered` All | `cb_centered` All | None |
| `normalize_block` | mul | `cb_centered` All | `cb_rstd` Col0 (REDUCE_ROW out) | Col |
| `scale_block` | mul | `cb_normed` All | `cb_gamma_rows` Row0 (face 0 / face 1 row 0 written; 1D `[W]`) | Row |
| `shift_block` | add | `cb_scaled` All | `cb_beta_rows` Row0 | Row |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| In-place CB transforms race | Chain phases have no in-place CB support (`center`/`normalize`/`scale` over the same bytes) and a reader-fed CB transformed by compute would have two producers | alias groups: distinct ids over one allocation, real push/wait per id, identical quanta; never pass the input CB as the output CB |
| Alias lockstep drift | the `work` ids share bytes; a phase that pushes or pops a different count than the others breaks entry `i` == entry `i` | every `work` id moves `block_rows * Wt` per block, every `stats` id `block_rows`; ragged blocks use the same `block_rows` on all ids |
| Padded rows (`S % 32 != 0`) | the last tile-row of each image holds `32 - S % 32` padding rows | no masking needed: every reduction is along `W` (aligned), rows never mix; padding rows compute finite values (zero padding -> `y = beta`) and are written as part of whole tiles; logical rows are unaffected |
| gamma/beta tile rows 1-31 are garbage | only row 0 of faces 0/1 is written | Row broadcast reads row 0 only; never use these tiles with `BroadcastDim::None` or as srcA |
| Scaler placeholder | `reduce_mean` / `reduce` require a scaler CB id even when unused | pass `cb_gamma_rows` (configured on the core); the aligned `AccumulateViaAdd` path never waits it (`reduce_helpers_compute.inl:334-337`); never introduce a partial mask without adding a real mask CB |
| `fp32_dest_acc_en=False` from a caller-built descriptor | `ttnn.ComputeConfigDescriptor(math_fidelity=LoFi)` defaults `fp32_dest_acc_en=False` (probed) | honor it: `T_acc` becomes Float16_b; only `None` maps to the default factory |
| Page format vs DEST width | a Float32 intermediate under 16-bit DEST doubles bytes for no precision | `T_acc` is derived from `fp32_dest_acc_en` for every compute-internal CB; IO CBs are bf16 |
| epsilon / 1/W encoding | SFPU scalar ops take fp32 bit patterns | host packs `struct.unpack('I', struct.pack('f', v))`; epsilon RT, `1/W` CT |
| Gamma TILE 1-D tensor `(768,)` | padded to `[32, 768]`, 24 pages (probed) | page `k` = tile-column `k`; read offsets 0 and 512 only |
| L1 budget vs L1-interleaved tensors | input/residual/output banks share L1 with CBs (6 MiB per tensor at `B*S=4096` = ~56 KiB per core each) | `CB_L1_BUDGET_BYTES = 768 KiB` leaves room; worst Phase 0 footprint 598,016 B (fp32 + residual, `block_height = 2`) |
| Grid-dependent block selection | multi-block path only appears when `max_core_rows_tiles > block_height_cap(1)` | regime-pinned acceptance shape `(256, 1, 16, 768)` + residual |
| Reader/writer roles | the writer also produces `cb_gamma_rows`/`cb_beta_rows` | each CB still has one producer (writer) and one consumer (compute); the writer must push gamma/beta before waiting on `cb_output_tiles` |
