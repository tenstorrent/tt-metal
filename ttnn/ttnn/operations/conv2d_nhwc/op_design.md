# Operation Design: conv2d_nhwc

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (im2col gather + tilize + matmul + bias + untilize, one dispatch) |
| Goal | 2D convolution on channels-last activations, as one `ttnn.generic_op` program per call. It is built as an im2col-gathered, K-blocked matmul spread over a 2-D (output-pixel × output-channel) core split. Phase 0: fp32 ROW_MAJOR DRAM-interleaved, dense, unit stride and dilation, tile-aligned channels. |
| Math | `out[n,ho,wo,co] = Σ_{kh,kw,ci} in[n, ho·s − p + d·kh, wo·s − p + d·kw, g(co)·C_in_g + ci] · W[kh,kw,ci,co] + bias[co]`, zero outside the image; optional `relu` after the bias |
| Matrix view | `out[M, C_out] = A[M, K'] · B[K', C_out] + bias`, with `M = N·H_out·W_out` output pixels (flattened n,ho,wo order) and `K' = kH · band_tiles · 32` (each kernel row's packed taps padded to whole tiles — see Prepared weight layout) |
| Mode | Derivative (helper-library kernels, generic_op program descriptor) |
| References | `.claude/references/generic_op_template/`, `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `ttnn/cpp/ttnn/kernel_lib/{matmul_block,bias_add,tilize,untilize}_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/mcast/` (deferred regimes), `ttnn/ttnn/operations/examples/master.md`, `eval/golden_tests/conv2d_nhwc/{feature_spec,helpers}.py`, `eval/op_template.py` |

`docs/NOC_ALIGNMENT_IN_TT_METAL.md` is not in this working tree. The alignment facts used here come from first principles and the HAL headers. L1 alignment is 16 B on both WH and BH (`tt_metal/hw/inc/internal/tt-1xx/{wormhole,blackhole}/core_config.h`, `LOG_BASE_2_OF_L1_ALIGNMENT 4`). DRAM alignment is 32 B on WH (`LOG_BASE_2_OF_DRAM_ALIGNMENT 5`) and 64 B on BH (`6`).

### Derived symbols (host computes each once; every kernel arg derives from these)

| Symbol | Definition | Phase 0 value |
|--------|-----------|---------------|
| `elem_bytes` | 4 fp32, 2 bf16 (bf8b untilizes to bf16 for the gather) | 4 |
| `l1_granule_elems` | `16 / elem_bytes` (16 B L1 NoC alignment) | 4 |
| `C_in_g` | `C_in / groups` | `C_in` |
| `tap_width` | `align_up(C_in_g, l1_granule_elems)`: elements one tap occupies in the window and in the K order | `C_in` |
| `band_tiles` | `ceil(kW · tap_width / 32)`: tiles per kernel-row band of K | `kW · C_in / 32` |
| `K_t` | `kH · band_tiles` | `kH·kW·C_in/32` |
| `N_t` | `ceil(C_out / 32)` | `C_out/32` |
| `H_out, W_out` | `(H + 2p − d(k−1) − 1)//s + 1` (same for W) | `H+2p−k+1` |
| `M`, `M_t` | `N·H_out·W_out`, `ceil(M/32)`. Tile-rows run over the flattened pixel order and may straddle image rows and images. Each row's 32 pixels are gathered individually. | |
| `T` | tile bytes of the compute format (`32·32·elem_bytes`; fp32 4096) | 4096 |
| `num_cores` | `grid.x · grid.y` of `device.compute_with_storage_grid_size()`, read at runtime and never inlined | 110 on p150b |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | ttnn.Tensor | yes | `[N,H,W,C_in]`, or `[1,1,N·H·W,C_in]` with `input_spatial` | — | buffer address RT |
| `weight_tensor` | ttnn.Tensor | yes | output of `prepare_conv2d_weights` | — | buffer address RT |
| `kernel_size` | int | yes | ≥1, `H+2p ≥ d(k−1)+1` and same for W | — | CT (`kH`,`kW`, `band_tiles`, `K_t`) |
| `bias` | ttnn.Tensor \| None | no | output of `prepare_conv2d_bias` | None | CT flag `has_bias` + address RT |
| `padding` | int | no | ≥0 | 0 | RT (gather geometry) |
| `stride` | int | no | ≥1 (Phase 0: 1) | 1 | RT |
| `groups` | int | no | divides C_in and C_out (Phase 0: 1) | 1 | CT |
| `dilation` | int | no | ≥1 (Phase 0: 1) | 1 | RT |
| `input_spatial` | (N,H,W) \| None | no | `N·H·W == input.shape[-2]` | None | host only (selects the output shape) |
| `activation` | str \| None | no | None / "none" / "relu" (Phase 0: none). `None` canonicalizes to "none". | None | CT flag (refinement) |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `[N,H,W,C_in]`, or flattened `[1,1,N·H·W,C_in]` + `input_spatial=(N,H,W)`. A TILE activation is accepted 4-D only when `W % 32 == 0` and `W_out % 32 == 0`, and is otherwise refused. |
| Dtype | Phase 0: float32. TARGET adds bfloat16 and bfloat8_b (TILE only). |
| Layout | Phase 0: ROW_MAJOR (one page = one `C_in` stick = one pixel). TARGET adds TILE. |
| Memory | Phase 0: DRAM interleaved. TARGET adds HEIGHT / WIDTH / BLOCK sharded in L1. |

### Weight (output of `prepare_conv2d_weights`)

| Property | Phase 0 value |
|----------|---------------|
| Logical shape | `[1, 1, K_t·32, C_out]`. The tile padding of `C_out` is implicit in TILE, so `C_out = weight.shape[-1]` is how the op recovers C_out. |
| K order | `k_row = kh·band_tiles·32 + kw·tap_width + ci`. Rows `≥ kW·tap_width` inside a band and lanes `ci ≥ C_in_g` are zero. With Phase 0's aligned C_in this is plain `[kh][kw][ci]`. |
| Dtype / layout / memory | `weight_dtype`, TILE, DRAM interleaved. Tile `(kt, nt)` is page `kt·N_t + nt`. |

### Bias (output of `prepare_conv2d_bias`)

| Property | Phase 0 value |
|----------|---------------|
| Logical shape | `[1, 1, 32, C_out]`. Row 0 holds the bias and rows 1..31 are zero (the RowBroadcast add reads row 0). |
| Dtype / layout / memory | `weight_dtype`, TILE, DRAM interleaved. Tile `nt` is page `nt`. |

### Output

| Property | Value |
|----------|-------|
| Shape | `[N, H_out, W_out, C_out]`. With `input_spatial`, it is `[1, 1, N·H_out·W_out, C_out]`. |
| Dtype | the input dtype |
| Layout | the input layout |
| Memory | follows the input. Interleaved in gives DRAM interleaved out, and X_SHARDED in gives X_SHARDED out in L1 (the op picks the geometry). Phase 0: DRAM interleaved. |

## Blocking Model

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| `m`: output-pixel tile-rows (flattened n·ho·wo, 32 pixels per tile-row) | **independent**: each output pixel's dot products never read another pixel's result. Its input *footprint* overlaps neighbours (the halo of `d(k−1)` input rows), so the window is affine in `block_m_tiles`. | `block_m_tiles` (bm) | `core_m_tiles` (the core's whole m range) unless L1 forces a smaller value (selection rule below). Every Phase 0 golden shape except one ends at `bm = core_m_tiles`. | host `select_blocking()` → CT arg `block_m_tiles` | split over `g_m = ceil(M_t / core_m_tiles)` core groups, `core_m_tiles = ceil(M_t / min(M_t, num_cores))` | knob-turn |
| `n`: output-channel tiles | **independent**: each output channel has its own weight column. The activation does not vary along n, so the activation is **reuse-shared across n-cores**. | `block_n_tiles` (bn) | `core_n_tiles` (the core's whole n slice) | `select_blocking()` → CT arg `block_n_tiles` | split over `g_n` cores **only when the m split leaves cores idle**: `core_n_tiles` = the smallest divisor of `N_t` with `N_t/core_n_tiles ≤ max(1, num_cores // g_m)`, and `g_n = N_t / core_n_tiles` | knob-turn (the split). Activation multicast across the n-cores is a scheme-change (regime R4). |
| `k`: reduction (kh, kw, ci in prepared K order) | **dependent**: every output element sums over all of K. | `block_k_tiles` (bk), a divisor of `K_t` | the largest divisor of `K_t` that fits. In the Phase 0 golden set `num_k_blocks` ranges from 1 to 18. | `select_blocking()` → CT arg `block_k_tiles`. `num_k_blocks = K_t / bk` is derived. | **not split across cores**: accumulated sequentially in place (packer L1 accumulation) inside one core | scheme-change (regime R6) |
| `g`: groups | **independent**: block-diagonal weight, so a group's outputs read only that group's input channels | `block_groups` | 1 (Phase 0 SUPPORTED is dense only, `groups == 1`) | CT arg | not assigned. With groups == 1 there is nothing to split. | knob-turn for tile-aligned per-group channels (each n-block's K range is offset to its group). Depthwise is its own regime (R11). |
| `h_in × w_in`: the per-block input window | not an independent axis. It is the **footprint** of an m-block (rows `(R_out−1)·s + d(k−1) + 1` per image crossed, all W columns). | derived from `block_m_tiles`, never set separately | `window_sticks_max` = the host-computed maximum over every block of every core | `select_blocking()` → CT arg `window_sticks_max` | follows `m` | knob-turn: a column-clipped window (R10) |

**Stage-data re-check (intermediates have axes too).** Every stage keeps the (m, n) assignment of its producing core:

| Stage | m | n | k |
|---|---|---|---|
| im2col tile-rows | m-local | not present | streamed per K-block |
| tilized in0 block | bm | not present | bk |
| partials / biased output | bm | bn | reduced |
| untilized output rows | streamed per tile-row | bn | — |

No stage is collected on one core. No core plays a coordinator role, so no stage leaves a group idle.

**Operand-reuse check (every operand × the chosen split).**

| Operand | Varies along m? | Varies along n? | Consequence |
|---|---|---|---|
| activation window / im2col | yes | **no** | Reuse-shared across the `g_n` cores that share an m range. Phase 0 re-reads it from DRAM per n-core. Broadcast successor: R4 (one reader per m range plus row multicast). |
| weight | **no** | yes | Reuse-shared across the `g_m` cores that share an n slice. Phase 0 re-reads it from DRAM per m-core (stepping stone). Broadcast successor: R1 (weight multicast). |
| bias | **no** | yes | `bn` tiles per core, read once per core (≤ 8 KB). The broadcast is not worth a channel, so no row. |

**Stall-shadow check.**

| Waiting stage | Waits on | Work independent of the wait, scheduled into it |
|---|---|---|
| compute at the start of m-block b+1 | the reader's window fetch for b+1 | Block b's bias add + untilize + the writer's output drain. The reader starts fetching window b+1 right after its last im2col push for b (window b is dead once gathered), so the fetch runs under compute's last K-blocks and the output phase of b. No reorder of floating-point operations. |
| matmul K-block j | weight K-block j (writer) | `cb_weight_tiles` has depth 2: the writer prefetches K-block j+1 while j multiplies. |
| writer after the last weight K-block of (m,n) block | `cb_out_rm` tile-rows | Nothing: the next block's weights cannot be pushed until compute has consumed them. Recorded as perf lamp L4. |

### Block-selection rule (`select_blocking()`, the single source of every knob)

1. `core_m_tiles`, `g_m`, `core_n_tiles`, `g_n` per the Axes table. Core list = the first `g_m·g_n` cores of the compute grid, row-major. Core `c` owns m range `[(c // g_n)·core_m_tiles, …)` (clamped to `M_t`) and n slice `[(c % g_n)·core_n_tiles, …)`. Cores sharing an m range are adjacent in this order, which is the layout the R4 row multicast needs.
2. `bn = core_n_tiles`, `bm = core_m_tiles`. `bk` = the largest divisor of `K_t` that is `≥ min(K_t, 4)` with `footprint(bm,bk,bn) ≤ CB_BUDGET_BYTES`.
3. If none fits: walk `bm` downward from `core_m_tiles` to 1, with `bk` over all divisors of `K_t` (largest first). Take the first `(bm, bk)` that fits.
4. If still none fits: walk `bn` over the divisors of `core_n_tiles` downward and repeat step 3.
5. If still none fits: repeat steps 2–4 against the hard cap `l1_cap_bytes() = get_max_worker_l1_unreserved_size() − 128 KiB` (verifier addition: the 768 KiB budget is a footprint *preference*, not a limit). Only if that also fails, raise `NotImplementedError("conv2d_nhwc: input window exceeds L1 — DRAM-slicing regime R10 not built")`. No golden shape reaches this after the verifier's column clip (R10 row).
6. `footprint(bm,bk,bn) = window_sticks_max(bm)·tap_width·elem_bytes + 2·bk·T + bm·bk·T + 2·bk·bn·T + bm·bn·T_acc + has_bias·bn·T + 2·bn·T`. This is the closed form from `l1_ledger.md`.
7. `CB_BUDGET_BYTES = 768 KiB`, a module constant (perf lamp L5).

Checked against all 62 Phase 0-eligible INPUTS (fp32, 110 cores):

| Check | Result |
|---|---|
| Fits the budget | all 62 |
| Peak footprint | 676 KiB |
| `bm = core_m_tiles` | all but one case |
| `K_t` divisor walk | settles at `bk ∈ {4…72}` |

### Subblocking (derived, not a knob)

| Field | Rule |
|---|---|
| `out_subblock_w` | the largest divisor of `bn` that is ≤ `dst_tiles` |
| `out_subblock_h` | the largest divisor of `bm` with `h·w ≤ dst_tiles` |
| `dst_tiles` | 4 when `fp32_dest_acc_en`, else 8 (half-sync) |

The matmul helper and the bias add use the same four values.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_act_rm` | `act_rm_depth` (in units of one tile-row × bk) | 2 | The reader gathers tile-row t+1 while compute tilizes t. |
| `cb_weight_tiles` | `weight_depth` (in units of bk×bn K-blocks) | 2 | The writer prefetches K-block j+1 during matmul j. |
| `cb_out_rm` | `out_rm_depth` (in units of one tile-row × bn) | 2 | Untilize of tile-row t+1 overlaps the writer's DRAM write of t. |
| `cb_act_window` | 1 (single) | 1 | No overlap needed: window b is dead before the fetch of b+1, and the fetch overlaps compute (stall-shadow table). |
| `cb_act_tiles`, `cb_partials`/`cb_biased`, `cb_bias_tiles` | none (whole block, single) | 1 | Producer and consumer are the same compute thread, run sequentially, so depth buys nothing. |

All depths are host constants in the program-descriptor module (one source) and feed both CB sizes and the footprint.

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| `matmul_block` K-blocking (`MatmulBlockShape.in0_block_k` is uniform per call) | bk must divide `K_t` | bk walks the divisors of `K_t` only | The last K-block would read past the weight / im2col rows, which is wrong results. |
| DEST capacity: one output subblock lives in DEST | `out_subblock_h·out_subblock_w ≤ dst_tiles` (4 fp32-acc / 8) | subblocking rule above | DEST overflow, which is corruption |
| NoC L1↔L1 copy alignment (16 B; source and destination offsets must agree mod 16) | window stick stride and every gather run offset | stride = `tap_width·elem_bytes`. The im2col row offsets are `kh·band_tiles·32 + kw·tap_width` elements, both multiples of `l1_granule_elems`. | Misaligned NoC reads silently shift data. |
| `MEM_ZEROS_BASE` region (512 B, `dev_mem_map.h`) | size of one zero-fill transfer | issue zero runs in ≤ 512 B pieces | reads past the zero region (garbage) |
| `tilize` / `untilize` `block_width_tiles` is a compile-time template param (`tilize_helpers.hpp:187-197`, `untilize_helpers.hpp:145-154`) | bk, bn must be compile-time | bk, bn are CT args, per-op compiled | n/a (a compile error, not silent) |
| `bn` uniform across cores (single CT arg) | `core_n_tiles` must divide `N_t` in Phase 0 | `core_n_tiles` chosen as a divisor | A ragged n slice would read weight columns past C_out. R4/R5 add zero-filled pad tiles (technique 5). |
| Output DRAM write of a partial stick (n slice of a pixel) | the byte offset `n_start·32·elem_bytes` must be DRAM-aligned (32 B WH / 64 B BH) | always a multiple of 128 B for fp32 / 64 B for bf16 | n/a (holds structurally) |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum (minimum = each input, weight and output crosses DRAM once) | What a bigger block buys |
|--------|--------|-----------|-------|------------------------------------------|--------------------------|
| **R0: 2-D (m×n) split, DRAM window gather** | **built (Phase 0)** | `memory_layout == INTERLEAVED ∧ layout == ROW_MAJOR` ∧ a blocking fits (rule step 5) | `bm × bn` output tiles per pass. K is streamed in bk blocks. The input window is `window_sticks_max(bm)` sticks. | **Input**: `g_n · ρ_halo` crossings, where `ρ_halo = Σ_blocks window_sticks / (N·H·W)` (1.0–6.4 over the Phase 0 set). It is above the minimum because of halo duplication between blocks/cores and the per-n-core re-read. **Weights**: `g_m · ceil(core_m_tiles/bm)` crossings of each n slice (one per m-block), above the minimum by the per-m-core re-read. **Output**: 1 (minimum). No cross-core traffic. | Larger `bm` pays fewer weight re-reads (one per m-block), fewer halo re-fetches (`d(k−1)` rows per block boundary), and fewer tilize/untilize/bias inits. Larger `bk` pays fewer per-K-block matmul re-inits (`ShortAfterPreKBlock`), tilize init/uninit pairs, L1-acc toggles and CB handshakes. |
| R1: R0 + weight multicast over the `g_m` cores sharing an n slice | **deferred**. R1 layers onto R0 without restructuring: same `cb_weight_tiles`, same K-block cadence on every receiver, and only the writer's fill source changes. It is built in the sharded-HEIGHT refinement (R3), where the shard fixes the receiver group. R0's interleaved path inherits it through R7. | same as R0, plus `g_m > 1` | as R0 | Weights cross DRAM `ceil(core_m_tiles/bm)` times (once when bm = core_m_tiles), plus a NoC multicast of `|W_slice|` to `g_m` receivers per K-block | as R0, plus fewer multicast handshakes per K-block (larger bk) |
| R2: per-tap DRAM im2col (each tap stick read from DRAM per output pixel) | **rejected**: a dead end superseded by R0's window | — | — | Input crosses DRAM `kH·kW/s²` times as `C_in`-byte transactions, which is per-transaction-bound. | — |
| R3: HEIGHT_SHARDED: shard-local window + neighbour halo over the NoC + weight multicast. TILE inputs untilize the shard and halo once at kernel start (technique 1). | **deferred**: the TARGET `memory_layout` value HEIGHT_SHARDED (refinement). The im2col gather code from R0 is reused unchanged and only the window fill changes (shard + remote halo runs instead of DRAM). | `memory_layout == HEIGHT_SHARDED` | m is pinned by the shard (ragged, may be shorter than one image row). Default one block = the whole shard. | Input: shard once (resident) plus halo runs over the NoC. Weights: once from DRAM plus multicast. Output: written into the L1 output shard (CB backed on the sharded buffer, zero-copy). | as R1 |
| R4: BLOCK_SHARDED: K is sliced by channel over grid x. Each core multicasts its shard's tilized im2col slice along its grid row; weights are per column. | **deferred**: the TARGET BLOCK_SHARDED value (refinement, scheme-change). K order becomes `[slice][kh][kw×ci_slice padded per band]` (technique 6). C_out tiles are padded to `S·ceil(N_t/S)` with zero tiles (technique 5). | `memory_layout == BLOCK_SHARDED` | bm × ceil(N_t/S) per core, K-block = one slice-band | Input: once (shard) plus a NoC multicast of the slice im2col to S cores. Weights: each column's slice read by its column (or multicast down the column). | fewer multicast handshakes per K-block (technique 8: 120 vs 24 K-blocks measured 940 vs 810 µs) |
| R5: WIDTH_SHARDED: R4 with one grid row | **deferred**: TARGET WIDTH_SHARDED (refinement). It is the same kernels as R4 with `g_m = 1`. | `memory_layout == WIDTH_SHARDED` | as R4 | as R4 | as R4 |
| R6: K split across cores + cross-core partial combine | **deferred**. The independent axes fill ≥ 80 % of the grid for every perf-focus shape. The only Phase 0 shapes with `g_m·g_n < num_cores/2` have `M·C_out ≤ 50K` outputs, where one combine round (semaphore plus a gather of `bm·bn` partial tiles per K-slice) costs about as much as the work it parallelises. Reachable: the per-core K range would become an RT arg on the same packer-L1 accumulation, plus a combine step on the partials CB. | `g_m·g_n < num_cores/2 ∧ num_k_blocks ≥ 2` | bm × bn × (K/g_k) | adds `(g_k − 1)·bm·bn·T` NoC bytes per output block. Saves nothing in DRAM. | fewer combine rounds |
| R7: interleaved → in-op HEIGHT/BLOCK placement (technique 7: runtime HEIGHT when `N·H·W ≥ 361` with a granule-aligned dense C_in; auto block-shard for weights ≥ 1 MB with ≥ 4 grid columns) | **deferred**: R7 is R0's DRAM window fill swapped for "own stick range from DRAM once + neighbour halo over the NoC" plus R1 or R4. It is built after R3/R4 exist, because those supply its compute path. | `INTERLEAVED ∧` the technique-7 predicates (to be pinned when built) | as R3 / R4 | Input once plus halo over the NoC. Weights once plus multicast. | as R3 / R4 |
| R8: kernel-stride fold (`k == s > 1`, d = 1, dense) | **rejected as a separate regime**. A separate `ttnn.fold` dispatch violates the one-dispatch rule. Inside R0 a `k == s` window has no overlap and every kernel-row band is one contiguous `kW·tap_width` run, so R0's gather already has fold's traffic. Weights need no reorder because the K order already matches. | — | — | — | — |
| R9: 1×1 stride-1 TILE: in0 = the shard's tiles in place (technique 3) | **deferred**: comes with the TILE layout refinement. No gather, tilize or untilize: HEIGHT matmuls the shard tiles, BLOCK/WIDTH multicast them. | `layout == TILE ∧ k == 1 ∧ s == 1 ∧ p == 0 ∧ sharded` | bm = whole shard, bk = the shard's channel tiles | Input never leaves L1 (HEIGHT) or is multicast once (BLOCK). | — |
| R10: window exceeds L1 (DRAM slicing / column-clipped window) | **column clip built by the verifier** (9 translated tt-forge fp32 shapes, e.g. 128×128×512 and 512×512×256, failed with a full-row window): a block inside one output row gets a 2-D window `rows × [col_lo, col_hi]`; a multi-row block gets a contiguous flattened range whose first / last input rows are trimmed to the columns read (unless padding-clamped). Host `_window_bounds` mirrors the reader. DRAM output-row slicing stays **deferred** (no golden shape reaches it). It is reachable because the window is defined as row-runs: clipping each run to the columns the block reads changes only the reader's run bounds, and output-row slices change only the m-range RT args. | rule step 5 fires | bm small, window clipped to `(bm·32·s + d(k−1))` columns per row | Input re-read per column slice (halo columns) | — |
| R11: depthwise (`groups == C_in`) | **deferred**: TARGET groups_mode=depthwise (refinement, scheme-change in compute). Instead of a block-diagonal matmul, the output tile `(pixels × channels)` is `Σ_taps act_tap ⊙ w_tap` (eltwise mul with a row-broadcast of the tap weights, accumulated in DEST, via `eltwise_chain`). The gather emits per-tap tiles. | `groups == C_in` | bm × channel tiles, taps streamed | Input as R0/R3. Weights: `kH·kW` rows per channel. | — |

**Regime selection (Phase 0).** One built row (R0). The predicate is the SUPPORTED gate plus rule step 5. Every row added later carries the predicate above, pinned by a regime-pinned test.

### Traffic ranking

For each candidate split, the bytes per tier on the representative heavy case (14×14×256 → 512, k=3, fp32: input 196 KiB, weights 4.5 MiB, 5 m tile-rows):

| Rank | Split | DRAM crossings | Cross-core | Grid fill | Verdict |
|------|-------|----------------|------------|-----------|---------|
| 1 | m-split, weights multicast (R1), resident window with neighbour halo (R7) | input ≈1, weights 1, output 1 | weights multicast to g_m, halo over the NoC | g_m only (5 cores here) | Cheapest traffic, but it under-fills when `M_t < num_cores` and needs R3's machinery. Deferred (R1/R7). |
| 2 | **m×n 2-D split (R0, chosen)** | input `g_n·ρ` (16 × 1.8 × 196 KiB ≈ 5.6 MiB), weights `g_m` × (each n slice once) = 5 × 4.5 MiB, output 1 | none | `g_m·g_n` = 80 cores | Chosen. Same weight bytes as pure m, plus input re-reads `∝ g_n`, and in exchange the per-core weight stream drops by `g_n` (288 KiB vs 4.5 MiB per core). |
| 3 | pure m split, weights from DRAM per core | input `ρ`, weights `g_m` | none | g_m = 5 | Cheaper than #2 by `(g_n−1)·ρ·|input|`, but 5 cores each stream 4.5 MiB serially. When `g_m == num_cores` this collapses into #2 with `g_n = 1`, which R0 then is. |
| 4 | m × k split + combine (R6) | input `≈ρ` (each K-core reads only its band's rows), weights `g_m` | `(g_k−1)·M·C_out·4 B` partials plus a rendezvous per block | fills | Adds NoC traffic and a rendezvous with no DRAM saving over #2. Deferred. |
| 5 | per-tap DRAM gather (R2) | input `kH·kW` | none | fills | Rejected. |

Decision: R0 takes split #2 only when the m split leaves cores idle (`g_n > 1` iff `g_m < num_cores`). When `M_t ≥ num_cores`, `g_n = 1` and R0 is split #3, so it adds no re-read. The traffic-cheapest split (#1) is deferred with its delta in `l1_ledger.md`.

### Block schedule

Per core, over its `num_m_blocks_this_core = ceil(core_m_tiles_this_core / bm)` m-blocks and `num_n_blocks = core_n_tiles / bn` n-blocks:

```cpp
load_bias_slice();                                    // once per core: core_n_tiles bias tiles
for (uint32_t m_block_idx = 0; m_block_idx < num_m_blocks_this_core; ++m_block_idx) {
    fetch_window_block(m_block_idx);                  // reader: input sticks of this m-block's footprint -> L1 window
    for (uint32_t n_block_idx = 0; n_block_idx < num_n_blocks; ++n_block_idx) {
        conv_matmul_block(m_block_idx, n_block_idx);  // K-loop: gather_im2col_kblock -> tilize_kblock -> matmul accumulate
        bias_block(n_block_idx);                      // has_bias only, in place on the accumulator
        untilize_store_block(m_block_idx, n_block_idx);
    }
}
```

| Operation | Block shape | Resident across it | Intended fixed-cost frequency |
|---|---|---|---|
| `load_bias_slice` | `1 × core_n_tiles` tiles | stays fronted in `cb_bias_tiles` for the kernel's life | one DRAM read burst and one wait per core |
| `fetch_window_block` | `window_sticks(m_block)` sticks × `tap_width` elems, as row-runs of consecutive input sticks (one NoC read per stick: interleaved RM pages). Pixels outside the image are not fetched. | the window, until the block's last n-block gather | one read barrier per m-block |
| `gather_im2col_kblock` (inside `conv_matmul_block`) | one tile-row (32 pixels) × `bk` tiles of im2col, pushed per tile-row | the window | Per pixel and per kernel-row band intersecting the K-block: one contiguous L1→L1 copy of the in-image part of `kW·tap_width` (dilation 1), plus ≤ 2 zero runs for padding columns, or a whole zero run for padding rows. Pixels `≥ M` (last tile-row) are zero-filled. One read barrier per tile-row. |
| `tilize_kblock` (inside `conv_matmul_block`, as `PreKBlockFn`) | `bm × bk` tiles | — | one tilize init/uninit per K-block, with L1-acc disabled around it (Key Risks) |
| `conv_matmul_block` | `bm × bn` output, `num_k_blocks` K-blocks of bk | `cb_partials` accumulates in place across all K-blocks (packer L1 accumulation) | one `matmul_block` call per (m,n) block. Matmul re-init once per K-block (`ShortAfterPreKBlock`), one reserve and one push of the accumulator per block. |
| `bias_block` | `bm × bn` | the bias slice (`bias_offset = n_block_idx·bn`) | one bias-add init per block |
| `untilize_store_block` | `bm × bn`, streamed one tile-row at a time to the writer | — | one untilize init per block. The writer issues one barrier per tile-row and writes only rows `< M`, each `bn·32·elem_bytes` at byte offset `n_start·32·elem_bytes` of the output stick. |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 grid-synchronization | When `M_t·N_t ≈ num_cores`, every core gets `bm = bn = 1`. A whole input window (up to 224 KiB) is fetched for 32 pixels, so the halo factor ρ reaches 6. | half the cores (`g_m/2`) with `bm = 2` |
| L2 n-split | `g_n > 1` re-reads the window per n-core (traffic) to cut the per-core weight stream (occupancy). | `g_n = 1` vs the chosen `g_n` on the 14×14×256→512 and 56×56×256→256 cases |
| L3 overlap | With `num_k_blocks == 1`, `cb_weight_tiles` depth 2 overlaps nothing within an m-block, and tilize of the whole bm×K block precedes the first FMA. | `bk = K_t/2` |
| L4 writer serialization | The writer streams weights for block (m,n) and then drains its output before prefetching the next block's weights. | Interleave: push next-block K-block 0 before draining `cb_out_rm`. |
| L5 CB budget | 768 KiB is a guess. The perf-focus cases weigh latency against `achievable_l1_peak_bytes` (161 KB–1.19 MB). | 384 KiB vs 768 KiB, recording both latency and peak L1 |
| L6 core order | Row-major assignment. The catalog's `noc_placement` says rows beat columns (2.9×). Window reads from DRAM all start at bank `lo % num_banks`. | a staggered window start (`bank_stagger`, 1.01–1.3×) |
| L7 bk floor | `bk ≥ min(K_t, 4)` comes from the catalog's `double_buffer` flat region (block 4..64). Prime `K_t` forces bk = 1 or K_t. | allowing the band to pad K to a multiple of a preferred bk in prepare |
| L8 zero-fill source | Padding zero runs as NoC reads from `MEM_ZEROS_BASE` cost one transaction per run. | a RISC-V store loop for runs < 64 B |

Catalog entries used: `compute_block_size` (tilize→matmul→untilize per tile-row is 1.65× slower than blocks of 8 → coarse bm and bk), `double_buffer` (depth 2 with a block ≥ 4 → the depth knobs and the bk floor), `eltwise_l1_vs_dest_accumulate` (packer L1 accumulation 5× over read-modify-write → `packer_l1_acc`), `matmul_output_subblock` (subblock tuning is the helper's), `mcast_topology` (two 1-D multicasts for the 2-D split → R1/R4 shape), `noc_placement` / `bank_stagger` (L6), `compute_fusion` (FPU bias add stays a separate packed phase, not fused in DEST).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| DRAM input → `cb_act_window` | RM sticks, `C_in·elem_bytes` each, stride `tap_width·elem_bytes` | reader (NCRISC, NoC0): `TensorAccessor` page read per stick over the m-block's contiguous flattened stick range `[lo, hi]` | `lo = flat(n0, max(0, ho0·s − p), 0)`, `hi = flat(n1, min(H−1, ho1·s − p + d(k−1)), W−1)`, with (n0,ho0) / (n1,ho1) the first / last valid pixel of the block. The window is reader-private scratch. |
| `cb_act_window` → `cb_act_rm` | RM im2col rows: 32 rows × `bk·32` elems per tile-row (bk tile-pages) | reader: local NoC reads (`noc_async_read` from its own L1) per (pixel, band) run; zero runs from `MEM_ZEROS_BASE` | im2col column `kh·band_tiles·32 + kw·tap_width + ci` ↔ window stick `flat(n, ho·s − p + d·kh, wo·s − p + d·kw) − lo`. For d = 1 a band is one run. |
| DRAM weight → `cb_weight_tiles` | TILE, `weight_dtype` | writer (BRISC, NoC1): `TensorAccessor` tile reads, K-block = `bk` rows × `bn` cols at `kt·N_t + n_start + j` | double-buffered |
| DRAM bias → `cb_bias_tiles` | TILE | writer, once per core: `core_n_tiles` tiles | |
| `cb_act_rm` → `cb_act_tiles` | TILE | compute: `tilize<bk>` (bm tile-rows) inside `matmul_block`'s `PreKBlockFn` | |
| matmul → `cb_partials` | TILE, Float32 (fp32 DEST) | `matmul_block<packer_l1_acc, Interm, TileRowMajor, caller_owns_pack_target>` | in-place K accumulation |
| `cb_partials` → `cb_biased` (alias, same L1) | TILE | `add_bias_bcast_rows<RowBroadcast, TileRowMajor>` | in place via the alias pair, with real credit edges |
| `cb_biased` (or `cb_partials` when no bias) → `cb_out_rm` | RM, 32 rows × `bn·32` per tile-row | `untilize<bn>` (bm tile-rows) | |
| `cb_out_rm` → DRAM output | RM sticks | writer: `TensorAccessor` page write per pixel row `< M`, `bn·32·elem_bytes` at offset `n_start·32·elem_bytes` | |

### Placement axis = the blocking scheme seen as a TARGET axis

| TARGET `memory_layout` | Which logical split it pins | Axis character cut | Cost class | Regime |
|---|---|---|---|---|
| INTERLEAVED | none (R0 chooses m×n) | — | built | R0 → R7 |
| HEIGHT_SHARDED | the m split (stick ranges, ragged) | independent | **knob-turn in compute, placement change in data movement**: shard-backed input CB (`ttnn.cb_descriptor_from_sharded_tensor`, zero-copy, never re-read through a `TensorAccessor`), halo runs from neighbour shards, output shard-backed CB. Weight multicast (R1) comes with it. | R3 |
| WIDTH_SHARDED | a channel split of the **input**, i.e. the K axis | **dependent** (K) | **scheme-change**: K slices are gathered by row multicast of each core's im2col slice (reuse) rather than combined. The output is an n split. | R5 |
| BLOCK_SHARDED | m over grid y, K (input channels) over grid x | m independent, K dependent | **scheme-change** (as WIDTH, per row) | R4 |

The output layout follows the input. Sharded output geometry: HEIGHT keeps the input core set with `bm·32` rows per core rounded to whole tiles for TILE output. BLOCK/WIDTH use `ceil(N_t/S)` tiles per slice (the shards together are wider than C_out, technique 5).

### Prepared-weight K order per placement (prepare sees `input_memory_config`, never the layout)

| Placement | K order | Columns |
|---|---|---|
| INTERLEAVED, HEIGHT | `[kh][kw × tap_width]`, each kh band padded to `band_tiles·32` rows | `N_t` tiles (C_out tile-padded, zero) |
| BLOCK, WIDTH (S channel slices) | `[slice][kh][kw × ci_slice_width]`, each (slice, kh) band padded to whole tiles (technique 6) | `S·ceil(N_t/S)` tiles, the trailing ones zero (technique 5) |

### Target-axis notes for refinements

| Axis | Change it requires (not Phase 0) |
|---|---|
| dtype bf16 / bf8b | Page formats follow `fp32_dest_acc_en`. `cb_partials` stays Float32 when the accumulation is fp32 (audit 2). bf8b TILE untilizes to bf16 before the gather (technique 1). |
| channel alignment | `tap_width`/`band_tiles` already encode the padding. The reader zero-fills each window stick's pad lanes once at kernel start (a 0 weight × a stale NaN gives NaN). The writer writes `C_out` columns only. |
| stride / dilation | Only the window bounds and the gather run decomposition change. With d > 1 a band becomes `kW` runs of `tap_width`. |
| relu | `pack_relu_config(ReluConfig::zero())` around the final output pack only (the bias pack, or the untilize pack when there is no bias), cleared before the next block's tilize and partial packs (technique 4) |
| TILE | R3/R4/R5 untilize the shard and halo once per kernel (technique 1). R9 is the 1×1 path (`TILE_DIRECT_IN0`). As built (Refinement 7), a DRAM TILE input runs on R0: per m-block, the window's whole tile-rows are read and untilized (technique 2, with no in-op resharding). The output is packed as tiles. |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a `bm × bn` output block (with all of K) |
| Grid | `device.compute_with_storage_grid_size()` (runtime). The first `g_m·g_n` cores in row-major order are active. |
| Per-core work | m range `[mi·core_m_tiles, min(M_t, (mi+1)·core_m_tiles))` with `mi = c // g_n`, and n slice `[ni·core_n_tiles, (ni+1)·core_n_tiles)` with `ni = c % g_n` (RT args: `m_tile_start`, `m_tiles_this_core`, `n_tile_start`) |
| Remainder | m: the last m-core may own fewer tile-rows. `num_m_blocks_this_core = ceil(m_tiles_this_core / bm)`, and every block pushes and pops nominal `bm`-sized counts (the gather zero-fills and the writer drops rows `≥ M` or past the core range). The last tile-row has `M − 32·(M_t−1)` valid pixels. n: no remainder (`core_n_tiles` divides `N_t`). All tile counts use `ceil`. `M` is the flattened pixel count, so tile-rows do not pad per image. |
| Regime selection | single built regime (R0). `select_blocking()` is pure and deterministic in (shapes, dtype, num_cores), and is unit-testable on host. |

`ttnn.split_work_to_cores` is **not** used for m: the 2-D split with equal `core_m_tiles` (last core ragged) is what makes `bm` a single CT arg.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_act_window` | 0 | `tap_width·elem_bytes` (one stick) | `window_sticks_max` | spans the m-block footprint (affine in bm). Reader-private scratch, no push/pop. | Float32 (input dtype) | reader | reader (scratch; no CB handshake) | per m-block, reused |
| `cb_act_rm` | 1 | `T` (32 RM rows of a tile-width slice) | `act_rm_depth · bk` | spans k (bk), streams m (one tile-row) | Float32 | reader | compute | per K-block tile-row |
| `cb_weight_tiles` | 2 | `T_w` (weight tile) | `weight_depth · bk · bn` | spans k, n. Streams K-blocks. | Float32 (weight_dtype) | writer | compute | per K-block |
| `cb_bias_tiles` | 3 | `T_w` | `core_n_tiles` (= bn·num_n_blocks) | spans n (whole core slice) | Float32 (weight_dtype) | writer | compute | whole kernel. Allocated only if `has_bias`. |
| `cb_act_tiles` | 4 | `T` | `bm · bk` | spans m, k (matmul in0 block) | Float32 | compute (tilize) | compute (matmul) | per K-block |
| `cb_partials` | 5 | `T_acc` | `bm · bn` | spans m, n. Accumulates over k. | Float32 | compute (matmul) | compute (bias or untilize) | per (m,n) block |
| `cb_biased` | 6 | `T_acc` | `bm · bn` (**alias of `cb_partials`**: one `CBDescriptor`, two format descriptors) | same bytes as `cb_partials` | Float32 | compute (bias add) | compute (untilize) | per (m,n) block. Exists only if `has_bias`. |
| `cb_out_rm` | 7 | `T` (32 RM rows × one tile-width) | `out_rm_depth · bn` | spans n, streams m (one tile-row) | Float32 (output dtype) | compute (untilize) | writer | per tile-row |

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_bias_slice` | 1 × core_n_tiles | dataflow (writer) | DRAM bias | `cb_bias_tiles`, core_n_tiles pushed once | Compute `wait_front(core_n_tiles)` once at start and `pop_front` once at kernel end. |
| 2 | `fetch_window_block` | window_sticks(m_block) sticks | dataflow (reader) | DRAM input | `cb_act_window` (scratch, base address) | valid until the block's last gather |
| 3 | `gather_im2col_kblock` | 1 tile-row × bk | dataflow (reader) | `cb_act_window` | `cb_act_rm`, bk pages per tile-row (reserve bk, push bk) | — |
| 4 | `tilize_kblock` | bm × bk | helper `tilize<bk, cb_act_rm, cb_act_tiles>(bm)` in `PreKBlockFn` | `cb_act_rm`, bk per tile-row (waited and popped by the helper) | `cb_act_tiles`, bm·bk | fronted for the matmul |
| 5 | `conv_matmul_block` | bm × bn, num_k_blocks × bk | helper `matmul_block` | `cb_act_tiles` bm·bk per K-block (wait/pop), `cb_weight_tiles` bk·bn per K-block (wait/pop) | `cb_partials`, bm·bn (caller reserves before, pushes after) | `cb_partials` holds the full sum, pushed |
| 6 | `bias_block` | bm × bn | helper `add_bias_bcast_rows` | `cb_partials` (wait/pop per row-group), `cb_bias_tiles` (caller-fronted, `bias_offset = n_block_idx·bn`) | `cb_biased`, per row-group (same bytes) | `cb_partials` empty, `cb_biased` full |
| 7 | `untilize_store_block` | bm × bn | helper `untilize<bn, cb_biased|cb_partials, cb_out_rm>(bm)` + dataflow writer | `cb_biased` (or `cb_partials`), bn per tile-row | `cb_out_rm`, bn per tile-row | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot | raw_api | `compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act_tiles, cb_weight_tiles, cb_partials)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60`. Reverse is mandated by `matmul_block_helpers.hpp:100-106`. | once at the top of `kernel_main` | — | — | — |
| `tilize_kblock` | helper | `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:187-197` | `<block_k_tiles, cb_act_rm, cb_act_tiles>` defaults (InitAndUninit, WaitBlock, UnpackAndPackReconfigure, Fp32Mode::Fast). Arg `num_blocks = block_m_tiles`. Symmetric pages. | `cb_act_rm` | `cb_act_tiles` | `block_width_tiles = bk`, `num_blocks = bm` |
| `conv_matmul_block` | helper | `compute_kernel_lib::matmul_block` | `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp:338-371` | `<transpose=false, packer_l1_acc=true, LastBlockTarget::Interm, OutputCBLayout::TileRowMajor, InitMode::ShortAfterPreKBlock, WaitAndPopPerKBlock, WaitAndPopPerKBlock, NoPostCompute, TilizePreKBlock, NoPostKBlock, 0, NoKBlockInnerDimFn, NoIn0Source, NoIn1BaseOffset, caller_owns_pack_target=true>`. Args `(act_tiles, weight_tiles, partials, partials, MatmulBlockShape::of(bm/sb_h, bn/sb_w, sb_h, sb_w, bk, num_k_blocks), {}, tilize_pre_k_block, bn, bn)`. caller_owns legality: `matmul_block_constraints.hpp:25-28`. | `cb_act_tiles`, `cb_weight_tiles` | `cb_partials` | `in0_block_k = bk`, `num_k_blocks`, the subblock counts (derived from bm, bn) |
| `bias_block` | helper | `compute_kernel_lib::add_bias_bcast_rows` | `ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp:156-168` | `<BiasBroadcast::RowBroadcast, OutputCBLayout::TileRowMajor>`. Args `(partials, bias, biased, BiasAddShape::of(bm/sb_h, bn/sb_w, sb_h, sb_w, bn), {}, n_block_idx·bn)`. | `cb_partials`, `cb_bias_tiles` | `cb_biased` | the subblock grid (from bm, bn) |
| `untilize_store_block` | helper | `compute_kernel_lib::untilize` | `ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp:145-154` | `<block_n_tiles, cb_biased (or cb_partials), cb_out_rm>`. Arg `num_blocks = bm`. | `cb_biased` / `cb_partials` | `cb_out_rm` | `block_width_tiles = bn`, `num_blocks = bm` |
| L1-acc reset around tilize / before bias | raw_api | `PACK((llk_pack_reconfig_l1_acc(0)))` | the helper leaves L1 accumulation on after K-block ≥ 1: `matmul_block_helpers.inl:410-414` | at the start of `TilizePreKBlock::operator()` and after `matmul_block` returns | — | — | — |
| `fetch_window_block`, `gather_im2col_kblock`, weight / bias / output movement | raw_api (dataflow) | `TensorAccessor` + `noc_async_read` / `noc_async_write` / `get_noc_addr` | `tech_reports/tensor_accessor/tensor_accessor.md`. `MEM_ZEROS_BASE`: `tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h:115`, `MEM_ZEROS_SIZE 512` at `:88`. | — | — | — | the run bounds derive from bm, bk |

**Helpers considered and rejected**

| Raw use | Helper considered | Mismatch (file:line) | Why it cannot express the shape |
|---|---|---|---|
| L1-acc reset | `matmul_block` `PreKBlockFn` contract | `matmul_block_helpers.hpp:182-193`: under `ShortAfterPreKBlock` the helper restores the unpack/math state and the pack format after `pre_k_block()`, but `matmul_block_helpers.inl:410-414` leaves `llk_pack_reconfig_l1_acc(1)` set from the previous K-block | The tilize inside the functor would pack with L1 accumulation on and sum into stale `cb_act_tiles` bytes. One packer-register write closes it. Nothing to wrap. |
| window fetch / gather | `tilize_helpers_dataflow.hpp`, `local_copy_helpers_dataflow.hpp` | Neither expresses a sliding-window gather with per-pixel padding runs (`local_copy_helpers_dataflow.hpp` copies one contiguous region per call) | The gather is the block operation the library lacks. It is built as a reader-side block function over one tile-row. |
| reblock_and_untilize | `reblock_untilize_helpers.hpp:70-73` | static_assert: SubblockMajor only | The output is TileRowMajor, so plain `untilize` applies. |
| `matmul_block` `OutWithRelu` / `OutWithUntilize` | `matmul_block_helpers.hpp:53-61` | OutWithUntilize is SubblockMajor-only, and a bias phase sits between the matmul and the untilize | — |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| bias_block | add (bcast rows) | `cb_partials`: All | `cb_bias_tiles`: Row0 (prepared bias row 0 = bias, rows 1..31 zero) | Row (`add_tiles_bcast_rows`) |

## Op-surface contract (registry model)

| Item | Phase 0 decision |
|---|---|
| INPUT_TAGGERS (7, in order) | `channel_alignment`, `stride_mode`, `groups_mode`, `dilation_mode`, `memory_layout`, `layout`, `activation`, exactly as specified in the task. Any code unpacking INPUTS slices `inputs[:7]` first. |
| SUPPORTED | `dtype [float32]`, `weight_dtype [float32]`, `layout [ROW_MAJOR]`, `bias_mode [with_bias, no_bias]`, `activation ["none"]`, `channel_alignment [tile_aligned]`, `stride_mode [unit]` (verifier: `+ strided` — the gather and window bounds were already stride-parametric; probed and golden-verified), `groups_mode [dense]`, `dilation_mode [unit]`, `memory_layout [INTERLEAVED]` |
| EXCLUSIONS | empty. Mixed precision (`dtype != weight_dtype`) cannot occur inside a one-dtype SUPPORTED rectangle. It becomes an EXCLUSIONS entry the moment a refinement grows either list without piping mixed formats. |
| `validate(input, weight, kernel_size, bias, padding, stride, groups, dilation, input_spatial, activation)` | Reconstruct `inputs = ((N,H,W,C_in), C_out = weight.shape[-1], k, p, s, g, d)` (N,H,W from `input_spatial` when flattened). Run the taggers. Override `memory_layout` from `input.memory_config().memory_layout`, `layout` from `input.layout` and `activation` from the kwarg (None → "none"). Add `dtype` / `weight_dtype` / `bias_mode`. Check SUPPORTED, then EXCLUSIONS, raising `UnsupportedAxisValue` / `ExcludedCell` (`ttnn.operations._op_contract`). Structural refusals (a 4-D TILE input with W or W_out not a tile multiple, or `input_spatial` inconsistent with the tensor) raise `ValueError`. |
| `prepare_conv2d_weights` / `prepare_conv2d_bias` | Call a shared `_validate_prepare(...)` that tags `((1,1,1,C_in), C_out, k, p, s, g, d)` (C_in = `torch_weight.shape[1]·groups`) and checks dtype, weight_dtype, the channel/stride/groups/dilation axes and `input_memory_config.memory_layout` against SUPPORTED. Weights then build the K order above (band padding explicit in torch) → `ttnn.from_torch(dtype=weight_dtype, layout=TILE, device, DRAM_MEMORY_CONFIG)`. Bias builds a `[1,1,32,C_out]` tensor with row 0 = bias, same placement. |
| Entry point | `conv2d_nhwc` calls `validate()` as its first line, then `select_blocking()`, allocates the output, builds the descriptor and makes **one** `ttnn.generic_op` call. |

### Structural impossibilities

None beyond feature_spec's `{bf8b, ROW_MAJOR}`.

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| L1 accumulation left on across the tilize in `PreKBlockFn` | `matmul_block_helpers.inl:410-414` sets `llk_pack_reconfig_l1_acc(1)` for K-blocks ≥ 1 with an Interm target. The next K-block's tilize would accumulate into `cb_act_tiles`. | `PACK((llk_pack_reconfig_l1_acc(0)))` first thing in the functor, and again after `matmul_block` returns (before the bias/untilize packs). The first-block tests pass even when this is wrong, which is why the acceptance test includes multi-K-block shapes (14×14×256→512: 3 K-blocks). |
| Alias `cb_partials` / `cb_biased` drift | Two FIFOs over one region | Identical page size and capacity `bm·bn`. The matmul pushes `bm·bn` and the bias pops and pushes per row-group (`sb_h·bn`), summing to `bm·bn`. The untilize pops per tile-row (`bn`), summing to `bm·bn`. Every cycle closes the ring, so both pointers return to base every block. Compute runs single-threaded in phase order, so the next block's partial packs follow the untilize's unpacks. |
| A ragged last m-block or core | `M` is not a multiple of 32, and `m_tiles_this_core < core_m_tiles` on the last core | Nominal `bm` push/pop everywhere. The gather zero-fills pixels `≥ M` and tile-rows past the core range. The writer writes only valid rows. |
| Halo overlap across image boundaries | A tile-row can straddle images. The window `[lo, hi]` then spans whole intermediate images. | Exact `window_sticks_max` is computed on host over every block of every core (the CB size). Bounds come from (n, ho) of the first and last valid pixel. Padding rows inside the range are never read: the gather maps out-of-image (h, w) to zero runs. |
| CB ring wrap on `cb_act_rm` / `cb_out_rm` | per-tile-row quanta | Capacity = depth × quantum, so every push is one quantum and the ring closes. |
| Precision (fp32) | Tilize Fast truncates to tf32. The matmul reads tf32 through SrcA/B anyway (`tilize_helpers.hpp:47-62`). | `ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True)` (`matmul_block_helpers.hpp:258-264`). `cb_partials` uses Float32 pages. |
| `bk` must divide `K_t` | A ragged K-block breaks the uniform `in0_block_k` | the divisor-only walk (mechanism caps) |
| Window pad lanes (refinement) | With `tap_width > C_in` the stick pad lanes are never written by DRAM reads | Zero the window once at kernel start in the alignment refinement. Phase 0 has no pad lanes. |
| Tests that only hit one regime or core count | `g_n > 1` only occurs when `M_t < num_cores` | The acceptance test includes both small-M (`g_n > 1`) and large-M (`g_n = 1`, `bm > 1`) shapes. |
