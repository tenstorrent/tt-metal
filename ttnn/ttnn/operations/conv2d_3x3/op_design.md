# Operation Design: conv2d_3x3

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (im2col gather + tilize + matmul with K-accumulation + bias/ReLU epilogue), one device program per call |
| Goal | Dense 3x3 convolution of ONE image (batch 1, groups 1) on the flattened channels-last contract `[1, 1, H*W, C_in] -> [1, 1, H_out*W_out, C_out]`, multi-core from Phase 0, structured so the Panoptic-DeepLab sharded TILE bf8_b layers (LOOSE_CASES) are reached by placement/format refinements, plus one scheme-change (BLOCK_SHARDED K-combine). |
| Math | `out[o, n] = act( bias[n] + sum_{kh,kw in 0..2, c < C_in} in[h_in, w_in, c] * w[n, c, kh, kw] )`, `o = h_out*W_out + w_out`, `h_in = h_out*s + kh*d - p`, `w_in = w_out*s + kw*d - p`, `in = 0` outside `[0,H) x [0,W)`, `act in {identity, relu}`. As a matmul: `OUT[M, N] = A[M, K_p] x Wp[K_p, N]`, `M = H_out*W_out`, `N = C_out`, `K_p = 3*roundup32(3*C_in_p)`. |
| Output geometry | `H_out = (H + 2p - 2d - 1)//s + 1`, same for `W_out`. |
| References | `eval/golden_tests/conv2d_3x3/feature_spec.py` (TARGET/INPUTS/LOOSE_CASES), `.../helpers.py` (oracle, tolerances, harness order), `docs/NOC_ALIGNMENT_IN_TT_METAL.md`, `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `ttnn/cpp/ttnn/kernel_lib/{matmul_block,tilize,bias_add}_helpers.hpp`, `ttnn/ttnn/operations/examples/master.md` (entries: compute_block_size, matmul_output_subblock, double_buffer, read_coalescing, mcast_topology, shared_input_reuse, split_reader, bank_stagger, noc_placement), `l1_ledger.md` (this directory) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| input_tensor | ttnn.Tensor | yes | `[1, 1, H*W, C_in]`; see Tensors | — | buffer address RT |
| weight_tensor | ttnn.Tensor | yes | output of `prepare_conv2d_weights` for the same input config | — | buffer address RT |
| input_height (H) | int | yes | `>= 1` | — | CT |
| input_width (W) | int | yes | `>= 1` | — | CT |
| bias | ttnn.Tensor or None | no | output of `prepare_conv2d_bias` | None | address RT; presence CT (`has_bias`) |
| padding (p) | int | no | `>= 0` (TARGET: `0` or `== dilation`) | 0 | CT |
| stride (s) | int | no | `{1, 2}` | 1 | CT |
| dilation (d) | int | no | `{1, 2, 4, 6, 8, 12, 18}` | 1 | CT |
| activation | str or None | no | `None`, `"relu"` | None | CT (`relu`) |
| compute_kernel_config | ttnn.ComputeConfigDescriptor | no | any fidelity / approx; `fp32_dest_acc_en` gated (only True supported) | `default_compute_kernel_config()` | compute-kernel config |

Entry points (one module `ttnn.operations.conv2d_3x3`): `conv2d_3x3`, `prepare_conv2d_weights`, `prepare_conv2d_bias`, `default_compute_kernel_config`, plus registry declarations `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS` (implementer writes SUPPORTED). `default_compute_kernel_config()` is a factory returning a fresh `ComputeConfigDescriptor(math_fidelity=HiFi4, fp32_dest_acc_en=True, math_approx_mode=False)`; `conv2d_3x3` resolves `None` through it. Fidelity / approx mode are honored, never gated.

`validate()` (first line of `conv2d_3x3`) runs in this order: (1) contract checks raising `ValueError` with a message prefixed `"conv2d_3x3: "` — input not rank 4 / not `[1, 1, input_height*input_width, C_in]`; `H_out < 1` or `W_out < 1`; `stride` not in {1, 2}, `padding < 0`, `dilation < 1`, `activation` not in {None, "relu"}; weight not `[1, 1, K_p, C_out]` for this C_in and input dtype/layout/memory config; bias not `[1, 1, 1, C_out]`; (2) SUPPORTED per axis (`UnsupportedAxisValue`); (3) EXCLUSIONS (`ExcludedCell`). Axis values come from the tensors and the resolved compute config; shape-derived axes from `INPUT_TAGGERS` over `((H, W, C_in), C_out, padding, stride, dilation)` exactly as `feature_spec.py` documents them. INVALID is never checked.

### Per-pixel channel width `C_in_p` — one host function, shared by prepare_* and validate()

| input_layout | input memory_layout | `C_in_p` | Why |
|--------------|---------------------|----------|-----|
| ROW_MAJOR | INTERLEAVED (DRAM or L1) | `roundup(C_in, ttnn.get_dram_alignment() // 2)` (BH 32, WH 16) | Each slab row is the DRAM page stride, so every DRAM->L1 page read lands on a DRAM-aligned L1 address (NOC_ALIGNMENT §1, §5 Strategy C). Tile-aligned C_in (Phase 0) gives `C_in_p == C_in` on both archs. |
| ROW_MAJOR | HEIGHT_SHARDED | `shard_spec.shape[1]` (= `roundup(C_in, l1_align // 2)`; stem: 8) | L1->L1 reads need 16 B only; the shard width is already L1-aligned. |
| TILE | INTERLEAVED, HEIGHT_SHARDED | `roundup(C_in, 32)` | The in-kernel untilize produces rows of the tile-padded width. |
| ROW_MAJOR / TILE | BLOCK_SHARDED | per core-column slice width `C_slice_p = shard_spec.shape[1]` | K is laid out per column slice (R4). |

`K_p = 3 * roundup32(3 * C_in_p)` (non-BLOCK); `K_p = n_cols * 3 * roundup32(3 * C_slice_p)` (BLOCK, R4). `tensor_kkh_tiles = roundup32(3*C_in_p)/32` (tiles per kh block), `tensor_k_tiles = 3 * tensor_kkh_tiles`.

### Prepared weight / bias contract (Phase 0; refinements may change the per-memory-config layout, never logical C_out)

| Property | Weight | Bias |
|----------|--------|------|
| Logical shape | `[1, 1, K_p, C_out]` (padded last dim `C_out_p = roundup32(C_out)`) | `[1, 1, 1, C_out]` |
| Row order | row `kh*roundup32(3*C_in_p) + kw*C_in_p + c`, c fastest; zero rows for `c >= C_in` and for each kh block's tail `[3*C_in_p, roundup32(3*C_in_p))` | — |
| Build | torch `[C_out, C_in, 3, 3]` -> permute `[kh, kw, c, C_out]` -> zero-pad c to `C_in_p` -> reshape `[3, 3*C_in_p, C_out]` -> zero-pad dim 1 to `roundup32` (PER kh block, never once at the end) -> reshape `[1, 1, K_p, C_out]` -> `ttnn.from_torch(dtype=weight_dtype, layout=TILE, device, DRAM interleaved)` | `ttnn.from_torch(bias.reshape(1,1,1,C_out), weight_dtype, TILE, device, DRAM)` — tile row 0 holds the bias |
| Refusals (raise before any upload) | dtype(input_dtype), layout(input_layout), weight_dtype, memory_layout(input_memory_config), channel_alignment (C_in, C_out from the torch weight), stride_mode, dilation_mode, padding_mode; EXCLUSIONS entries whose keys are all derivable from these | same |
| Hot path | `conv2d_3x3` does no host work on weights: no torch, no from_torch/to_torch, no re-preparation | same |

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `[1, 1, H*W, C_in]`, row `h*W + w` = one pixel (contract `ValueError` otherwise) |
| Dtype | bfloat16 (Phase 0); bfloat8_b (TILE only; R3) |
| Layout | ROW_MAJOR (Phase 0); TILE (R3) |
| Memory | DRAM INTERLEAVED (Phase 0); L1 HEIGHT_SHARDED (R2/R3); L1 BLOCK_SHARDED (R4). Shard geometry is read from `input_tensor.shard_spec()`; no grid is assumed; shards are not aligned to image rows. |

### Output

| Property | Value |
|----------|-------|
| Shape | `[1, 1, H_out*W_out, C_out]` |
| Dtype | = input dtype |
| Layout | TILE, always |
| Memory | Interleaved input -> DRAM INTERLEAVED. Sharded input -> same scheme and same grid: HEIGHT shard `[roundup32(ceil(M / n_cores)), C_out_p]`; BLOCK shard `[roundup32(ceil(M / n_rows)), roundup32(ceil(C_out / n_cols))]` (cores past the last needed shard hold empty shards). |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Every extent below has ONE source: the host function
`select_blocking(geometry, grid, l1_budget) -> Conv3x3Blocking`. CB page counts, loop trip counts, matmul
shapes and per-core runtime args are all derived from its fields; no kernel or descriptor restates a literal.

Batch is fixed at 1 by contract (not an axis). Tensor-scope names: `tensor_m_tiles = ceil(M/32)`,
`tensor_n_tiles = ceil(C_out/32)`, `tensor_kkh_tiles`, `tensor_k_tiles`.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| M — output pixel `o = h_out*W_out + w_out` (tiles of 32 pixels) | **independent** — each output pixel's sum is self-contained; its input footprint overlaps its neighbours' (halo), which is a data-movement cost, not a dependency | `block_m_tiles` | `core_m_tiles` (the whole per-core assignment = one M-block per core); shrunk only by the L1 solve (Mechanism caps #1) | `Conv3x3Blocking.block_m_tiles` (CT) | split across the grid: `core_m_tiles = ceil(tensor_m_tiles / grid_cores)`, `num_m_groups = ceil(tensor_m_tiles / core_m_tiles)`, contiguous M ranges, grid = `device.compute_with_storage_grid_size()` at call time. Sharded: pinned by the shard (`core_m_tiles` = output shard height / 32). | knob-turn (bm), placement (sharded) |
| N — output channel `C_out` | **independent**, but the im2col activation is **reuse-shared** across it (every N-block needs the same A rows) | `block_n_tiles` | `tensor_n_tiles` (whole C_out) | `Conv3x3Blocking.block_n_tiles` (CT) + `n_tile_start` (RT) | not split (`num_n_groups = 1`): splitting re-gathers and re-tilizes A once per N-group (traffic ranking). HEIGHT output needs full C_out per core anyway. BLOCK output splits N across core columns (R4). | knob-turn (`num_n_groups`; perf lamp P4) |
| K — reduction `(kh, kw, c)`, `K_p` rows, kh-block aligned | **dependent** — every output sums over it; within a core a sequential accumulate (matmul K-blocks into fp32 L1), across cores a NoC combine | `block_k_tiles` | `tensor_kkh_tiles` (one K-block = one kernel row kh = 3 taps x `C_in_p`), i.e. `num_k_blocks = 3`; reasons in "K-block default" below | `Conv3x3Blocking.block_k_tiles` (CT); `num_k_blocks = 3 * tensor_kkh_tiles / block_k_tiles` derived | not split across cores (`num_k_groups = 1`). BLOCK sharding cuts its channel component c across core columns (R4). | scheme-change (R4 combine; R6 reuses it) |
| c — input channel (K's channel component) | **dependent** (part of K) | `C_slice_p` (channels a core holds per pixel) | `C_in_p` (all channels on every core) | `Conv3x3Blocking.c_in_p` (CT) | not split; BLOCK_SHARDED (caller-pinned) splits it over core columns | scheme-change (R4) |
| Input footprint (slab) — input pixels the M-block reads, as 3 kh-bands | **derived from M** (affine in `block_m_tiles`, dilated by s, d, W) — not an independent knob | `band_len_px`, `band_stride_px` | `band_len_px = max over the core's M-blocks of [(ho1-ho0)*s*W + (wo1-wo0)*s + 2d + 1]` (block covers output pixels `[o0, o1]`); `band_stride_px = min(d*W, band_len_px)`; slab = `2*band_stride_px + band_len_px` pixels | `Conv3x3Blocking` (CT), computed from `block_m_tiles` | follows M | — |

**K-block default (`block_k_tiles = tensor_kkh_tiles`, not `tensor_k_tiles`).** One kh block is (1) the unit the gather produces natively — for d=1 one kernel row of one output pixel is ONE contiguous 3-pixel read; (2) the only pipelining granularity available when every core runs one M-block (all Phase 0 INPUTS do): weights and im2col rows for kh+1 stream while compute tilizes+multiplies kh, and the slab fill for band kh+1 overlaps the gather of band kh (stall-shadow below); (3) the prompt's streaming contract (weights stream one kh block at a time); (4) a fixed small multiplier (3) on per-K-block fixed cost. Coarsening to one K-block per M-block is perf lamp P1. Sub-kh K-blocks (divisors of `tensor_kkh_tiles`) are taken only under L1 pressure (Mechanism caps #1).

**Phase 0 is multi-core.** M is spread over the whole live grid on every call: `num_m_groups = min(tensor_m_tiles, grid_cores)` cores for the INPUTS (2..64 cores), up to the full grid for large M. Grid size is a call-time parameter, never a constant.

**Ragged tail.** Only the last M-group can hold fewer than `core_m_tiles` tiles (ceil split). It runs the **nominal** block (nominal push/pop/wait counts and matmul shape everywhere); the reader gathers only its valid rows and the writer writes only valid output tiles. Rows past the tail carry stale L1 data that never reaches an output (matmul, tilize, bias add, and bf8_b packing are all row-independent). Chosen over `split_work_to_cores`'s two-group split: identical makespan (`ceil(tensor_m_tiles / grid_cores)` tiles on the busiest core), one CB spec for all cores, and it is exactly the HEIGHT output-shard geometry the sharded regimes must produce.

**Operand-reuse check (chosen split = M).** Weights and bias do not vary with M -> reuse-shared by construction of the split -> broadcast regime R1 (built in Refinement 3). Activation varies with M -> not shared. Output varies with M -> not shared.

**Stall-shadow check.**

| Stage that waits | Waits on | Work scheduled into the window | Legal because |
|------------------|----------|--------------------------------|---------------|
| reader: slab fill barrier | DRAM reads of the M-block's footprint | Bands are filled in kh order and fenced per band (NoC transaction ids, `local_copy_helpers_dataflow.hpp:147,158`); the gather of K-block kh waits only for band kh while bands kh+1, kh+2 are in flight. For d=1 the bands overlap and the merged range is issued band-0-first. | K-block kh reads only band kh (data independence); K-block order is unchanged, so no floating-point reorder and no precision gate. |
| compute: matmul K-block 0 | `cb_weights` K-block 0 and bias | Writer issues bias and weight K-blocks 0..`weight_depth-1` at kernel start, concurrently with the reader's slab fill (separate RISC, separate NoC). | Weights and bias do not depend on the activation. |
| writer: output drain of M-block j | compute finalize of block j | Writer prefetches weight K-blocks of block j+1 between output row groups; each such `reserve_back` cannot block, because compute pops every K-block of block j before finalize(j) starts (deadlock-freedom argument). | Weights are M-independent. |
| compute: tilize of K-block kh | reader gather of K-block kh | Nothing independent (matmul of kh needs the tilized block; matmul of kh-1 already ran). Stated, not filled. | — |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_im2col_rm` | `im2col_rm_depth_rows` (tile-rows of 32 gathered pixels) | 2 | Reader gathers tile-row t+1 while compute tilizes t. Raising it to `block_m_tiles` lets the reader run a whole K-block ahead during the matmul (lamp P2). |
| `cb_weights` | `weight_depth` (K-blocks) | 2 (= 1 only if `num_k_blocks * num_m_blocks_this_core == 1`) | Writer streams K-block k+1 during matmul of k. |
| `cb_output_tiles` | `out_depth_row_groups` (row groups of `out_subblock_h x block_n_tiles` tiles) | 2 | Writer drains row group g while compute packs g+1. |
| `cb_im2col_tiles` | — (full block, depth 1) | 1 | Sequential helper intermediate (tilize then matmul on the same TRISCs): no overlap is possible, so a second slot buys nothing. |
| `cb_matmul_partials` | — (exactly one output block) | 1 | Fixed by mechanism cap #2 (L1-acc in place). |
| `cb_halo_slab`, `cb_bias` | — | 1 | Slab: one M-block's footprint (reader-private). Bias: persistent per core. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| 1. L1 residency (the weight K-block alone can be 7x L1: K=18432 x N=256 bf16 = 9.4 MB) | `block_k_tiles`, `block_m_tiles` | `select_blocking` solve, in this order: start `bm = core_m_tiles`, `bk = tensor_kkh_tiles`, `bn = tensor_n_tiles`; while footprint (l1_ledger.md total) > budget: (a) shrink `bk` to the next smaller divisor of `tensor_kkh_tiles`, not below the tap floor `C_in_p/32` (one pixel's channels; only when `C_in_p % 32 == 0`, else floor = `tensor_kkh_tiles`); (b) then shrink `bm` (`bm = ceil(core_m_tiles / num_m_blocks)`, `num_m_blocks = 2, 3, ...`; sharded output: only divisors of `core_m_tiles`, since the blocks pack consecutively into the shard), re-trying the coarsest `bk` above the tap floor at each `bm` (measured: stem.conv3 bm 19, bk 2 -> 6, 669 -> 331 us); (c) then `bk` below the tap floor. `bn` is never shrunk (HEIGHT needs full C_out; N re-gather costs bytes). Order rationale: shrinking bk costs no bytes (only per-K-block fixed cost); shrinking bm re-streams weights and re-reads the halo (bytes). Budget: `L1_CB_BUDGET_BYTES` (host constant per arch, derived from device L1 size minus the unreserved base) minus the bytes of L1-resident tensors on the core: R2 subtracts 2 x (input shard + output shard), this call's pair plus the one a chained model or a cache-hit re-dispatch keeps alive (every enqueue re-checks CBs against the lowest live L1 buffer). The shard-backed `cb_output_tiles` is not CB-region bytes. No solution -> host error naming the shape (never reached by TARGET shapes; see l1_ledger.md). | CB allocation clashes with L1 buffers at program creation, or the op silently uses overlapping L1. |
| 2. Packer L1 accumulation (`packer_l1_acc=true`, TileRowMajor, Interm) | `cb_matmul_partials` capacity | Capacity = exactly `block_m_tiles * block_n_tiles` fp32 tiles — never rounded up. Each K-block's row-group reserves must land on the SAME addresses as the previous K-block (the helper drains the FIFO back to base after every non-last K-block, `matmul_block_helpers.inl:501-518`). | K-block k accumulates onto a different region -> silently wrong sums. |
| 3. K-block / kh alignment | `block_k_tiles` | `block_k_tiles` divides `tensor_kkh_tiles` (or equals `tensor_k_tiles` in lamp P1). A K-block never straddles a kh boundary. | The gather's (kh, kw, c) slot map and the per-kh-padded weight rows disagree -> silently wrong output. |
| 4. DRAM->L1 read alignment (64 B BH / 32 B WH) | slab row stride, `C_in_p` (RM interleaved) | `C_in_p` rule above: slab row bytes = DRAM-aligned page stride. | Misaligned DRAM->L1 transfer corrupts data or faults. |
| 5. L1->L1 gather alignment (16 B) | tap slot offsets `kw*C_in_p*2`, im2col row stride `block_k_tiles*64` B | `C_in_p % 8 == 0` (bf16); row stride is a multiple of 64 B by construction. | Misaligned NoC transaction. |
| 6. Zero source `MEM_ZEROS_BASE` (512 B, `tt_metal/hw/inc/internal/tt-1xx/blackhole/dev_mem_map.h:88,115`; WH :80,:94) | zero-fill segment size | Zero segments (out-of-image taps, kh-block K padding, `C_in..C_in_p` lanes) are issued in chunks of <= 512 B. | Reads past the zero region copy non-zero L1 into the im2col row (0 x garbage can be NaN). |
| 7. DEST capacity (fp32 DEST, half-sync: `DEST_AUTO_LIMIT` = 4 tiles, `dest_helpers.hpp:103`) | matmul subblock — NOT a block bound | Host derives `out_subblock_w` = largest divisor of `block_n_tiles` <= 4, then `out_subblock_h` = largest divisor of `block_m_tiles` with `h*w <= 4` (width first: catalog matmul_output_subblock — size matters, shape does not; width pays). Helper-owned decomposition; not a knob. | Helper ASSERT (`matmul_block_helpers.inl:158`) / DEST overflow. |
| 8. Tilize width is a compile-time template arg | `block_k_tiles` | `block_k_tiles` is a CT arg (part of the program hash; it is a function of C_in, which is already in the hash). | — |
| 9. CB ring-wrap (quanta close the cycle) | all FIFO CBs | Nominal quanta everywhere: `cb_im2col_rm` pushes `bk` per tile-row (capacity `2*bk`); `cb_im2col_tiles` pushes `bk` per tile-row, pops `bm*bk` per K-block (capacity `bm*bk`); `cb_weights` `bk*bn` per K-block (capacity `weight_depth*bk*bn`); `cb_output_tiles` `out_subblock_h*bn` per row group (capacity `2*out_subblock_h*bn`). Ragged tail never changes a push/pop count. | Pointer walks past `fifo_limit` -> neighbouring L1 corrupted, no error in release. |
| 10. Index width | pixel indices (stem input: 524288 pixels) | `uint32_t` throughout; no 16-bit index or packed transport unit. | Wraparound -> wrong addresses. |

### Regimes

Minimum (named boundary = DRAM for interleaved, L1/NoC for sharded): the input crosses once, the weights once, the output once.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R0 interleaved-slab** | **built (Phase 0)** | `memory_layout == INTERLEAVED and layout == ROW_MAJOR` | `(block_m_tiles = core_m_tiles, block_k_tiles = tensor_kkh_tiles, block_n_tiles = tensor_n_tiles)`; slab `2*band_stride_px + band_len_px` pixels x `C_in_p` | Input: each core reads only its M-block's 3 kh-bands from DRAM, one page read per pixel (interleaved pages cannot coalesce); crossings = sum of slab pixels / (H*W) — above minimum by the halo, e.g. 3.2x at `bm=1, W=64` (bands disjoint), 1.06x for one core. Weights: every core streams all of `K_p x C_out_p` -> `num_m_groups * num_m_blocks_this_core` crossings (above minimum; R1 removes it). Output: once (minimum). No cross-core traffic. | Per M-block: slab fill fences (<= 3), halo re-read (`2*band_stride_px` pixels), full weight re-stream (if > 1 M-block per core), 3 x (tilize init/uninit + matmul short init + reconfig + one L1-acc pass over `bm*bn` fp32 tiles), one bias-add init + reconfig, `bm/out_subblock_h` output handshakes. Bigger `bm` amortizes the halo and weight re-stream; bigger `bk` amortizes the per-K-block inits and L1-acc passes. |
| R-direct direct-dram-im2col | **rejected** — dead end, superseded by R0 | — | — | Each im2col row read straight from DRAM pages: 9 page reads per output pixel; input crosses DRAM 9x (d=1, s=1); 9x the transaction count of R0's slab fill + L1 gather. Nothing of its per-tap DRAM addressing survives into any sharded regime (all gather from L1). | — |
| R1 weight-mcast | **built (Refinement 3, every regime)** — overlay on R0/R2/R3: the cores that run the most M-blocks (all but a ragged last core) form one `ttnn.Mcast` channel `weight_mcast` on the writer's NoC 1, fixed sender = work core 0 (`McastExplicitFixedSenderConfig`). Per K-block round the sender reads `bk*bn` tiles from DRAM into its own `cb_weights` slot and multicasts them to the same slot everywhere; a receiver reserves its slot, then `receive()` (ack + wait), then `push_back(bk*bn)`. A core outside the channel (ragged last core, or > 3 rectangles, or fewer than `WEIGHT_MCAST_MIN_CORES` participants) keeps R0's own DRAM stream. No CB, block extent or compute phase changed. | `num_m_groups > 1` (any regime) | same as host regime | Weights: `num_m_groups` DRAM crossings -> 1; adds `(num_m_groups - 1) * K_p * C_out_p * elem` bytes of multicast payload (one transfer per K-block, sender-serialized). Delta, INPUTS `32x64x128->128`: 18.4 MB -> 288 KB DRAM; LOOSE `semantic res2.fuse_conv.0` (103 cores, 705 KB bf8_b weights): 72 MB -> 705 KB. Largest single perf lever for the HEIGHT-sharded LOOSE layers. | Fewer, larger mcast transfers (one per K-block): bigger `bk` amortizes the per-K-block semaphore round trip. |
| R2 height-sharded-RM | **built (Refinement 2, both layouts)** — unlocks `memory_layout=HEIGHT_SHARDED` with `layout=ROW_MAJOR` (stem.conv1); with R3 on top it is the TILE HEIGHT path. Work index i runs on input shard core i (shard order from the shard spec's grid and orientation) and owns output shard i; `bm` is restricted to divisors of the output shard's tile-rows, because the M-blocks pack consecutively into the shard-backed `cb_output_tiles`; the writer only streams weights and bias. Placement change on R0: same block pipeline; the slab fill reads page runs through a TensorAccessor over the sharded input (`num_contiguous_pages`, `tensor_accessor.h:192`, stops at shard edges) from ANY owner core (owner = pixel // shard_h); `cb_output_tiles` backed on the HEIGHT output shard (`ttnn.cb_descriptor_from_sharded_tensor`), so packs land in place and the writer moves no output. | `memory_layout == HEIGHT_SHARDED and layout == ROW_MAJOR` | `block_m_tiles = out shard tiles`, rest as R0 | Input: NoC/L1 only, ~1x + halo (stem: slab 9217 px x 16 B per core vs 4767-row shard). The local shard is copied into the slab by a self-aimed read: a deliberate layout transform (contiguous across shard edges), not a re-consumption of the resident block (lamp P8 measures the zero-copy alternative). Output: in place (minimum). Weights as R0/R1. ROW_MAJOR (Refinement 6, `SLAB_SPLIT_NOCS`): the fill is split by NoC direction: the reader (NoC0) reads runs of its own and lower-index shards, the writer (NoC1) those of higher-index shards into the same slab (a NoC0 read response from a higher-index core wraps the ring; stem.conv1 band waits 20-25 us on the high cores). Core-local `slab_fwd_done` / `slab_free` semaphores hand the slab over per M-block. | Same as R0 plus fewer halo bytes per resident shard (one block = the whole shard). |
| R3 tile-input-untilize | **built (Refinement 1, interleaved; HEIGHT is Refinement 2)** — unlocks `layout=TILE` and `dtype=bfloat8_b` (22 of 23 LOOSE cases). Adds one compute phase before the gather: the M-block's footprint tile-rows (bands rounded out to 32-pixel tile rows; local shard via a CB backed on the shard, remote/DRAM tiles via TensorAccessor) are untilized (`untilize_helpers.hpp`) into `cb_halo_slab` (bf16 RM). `cb_halo_slab` becomes a compute->reader CB (push per band, reader pops after the last K-block). Gather/tilize/matmul unchanged; `cb_im2col_tiles` takes the activation dtype (bf8_b re-quantization is exact when tap boundaries are 16-channel aligned: a bf8_b exponent group = 16 consecutive channels of one pixel in both layouts). | `layout == TILE` (interleaved or HEIGHT) | as R0/R2 | Input: footprint tiles cross once + halo (bands rounded to tile rows). Extra compute: untilize of `slab_px * C_in_p` elements per M-block. | Halo untilize amortized over a bigger `bm`. |
| R4 block-sharded-k-split | **built (Refinement 4)** — unlocks `memory_layout=BLOCK_SHARDED` (9 LOOSE cases). Scheme-change: the caller cut c across `k_cols` core columns (`k_cols` = input shard columns; one column is R2 on that column, no split). Core (r, c) owns output tile-rows of grid row r (`m_split` over the grid rows) and channel slice c: it builds the slab of its slice (halo along its column from any owner shard: the slice's rows are page `row * row_page_stride + c * pages_per_slice_row`, one read per owner-shard run) and gathers its K-slice (`3*roundup32(3*C_slice_p)` per kh; a short last slice keeps the lane rules) against its column's weight K-slice (weight rows ordered `(col, kh, kw, c_slice)`, `prepare_conv2d_weights` keyed on `input_memory_config`; the writer offsets by `col * tensor_k_tiles`). The matmul packs fp32 partials `[bm, tensor_n_tiles]` into `cb_matmul_partials` with `caller_owns_pack_target` (compute reserves/pushes once per M-block, the WRITER consumes). Row-local reduce-scatter per M-block: owning column y reserves its `cb_combine_slots` (`k_cols` slots of `[bm, out_n_tiles]` fp32) and credits every row peer; each core writes N-slice y into slot c of column y (own and lower columns from the writer on NoC1, higher columns from the reader on NoC0, `COMBINE_SPLIT_NOCS`), then bumps y's arrival count; after `k_cols` arrivals compute sums the slots (`reduce<SUM, REDUCE_COL, AccumulateViaAdd, Skip>`: pairwise FPU adds into fp32 DEST, odd counts seeded by a copy), adds the bias N-slice (`add_bias_bcast_rows` from `cb_combined`), applies ReLU, and packs into its BLOCK output shard `[core_m_tiles*32, out_n_tiles*32]`, backed on the shard. Columns past the last C_out tile own no output and only send. Weights (Refinement 5): one R1 channel with one group per grid column (`McastFixedSenderConfig`, `ColumnMajor` groups on a ROW_MAJOR grid, `RowMajor` on COL_MAJOR); the column's first participating row reads the column's K-slice and multicasts it down the column. | `memory_layout == BLOCK_SHARDED` | `(bm = out shard rows / 32 or a divisor, bk <= kh block of the slice, bn = tensor_n_tiles for partials, out N-slice = ceil(tensor_n_tiles / k_cols))` | Input: own shard + halo bands of the slice (no activation exchange). Combine: each core sends `bm*32 * C_out_p * 4 B` (its own slice included; res4.x: 256 KB) and receives `k_cols * bm * out_n_tiles` fp32 tiles. Weights: one DRAM read of each column's K-slice x all N per M-block pass, multicast down the column (res4.x: 5 MB -> 625 KB DRAM; it became the bound once the tap-major gather landed: 41.7 -> 36.0 us). | One combine round per M-block: one block = whole shard means one combine and one semaphore round per call. |
| R4b block-sharded-act-allgather | **deferred** — perf alternative to R4 where the full-C_in slab fits L1 (res4.x, res5.x): each core gathers its row peers' channel slices (row multicast) and computes its N-slice with full K, no partial combine. R4 is built first because it is the only one that fits every BLOCK LOOSE case (aspp needs a 3.3 MB full-C_in RM slab). | `BLOCK_SHARDED and slab_px * C_in_p * 2 + rest <= budget` | `bn = out N-slice` | Activation exchange `slab_px * C_in * (n_cols-1)/n_cols` per core (res4.x: 92 KB vs R4's 229 KB) instead of the partial combine. | — |
| R5 weights-resident | **deferred** — reachable fast path: when `K_p x block_n_tiles` weight tiles fit next to the block working set AND `num_m_blocks_this_core > 1`, the writer reads the weights once per core and compute re-reads them per M-block (`InputPolicy::NoWaitNoPop` + `In1BaseOffsetFn`, `matmul_block_helpers.hpp:72-77,223-229`). No Phase 0 INPUTS shape runs > 1 M-block per core (all take `bm = core_m_tiles`), so streaming and residency move identical bytes there; first shape that benefits is LOOSE stem.conv3 (bm forced below 38 tiles, 78 KB of weights). | `num_m_blocks_this_core > 1 and K_p*bn*T_w + footprint(bm, bk) <= budget` | as host regime, `cb_weights` = whole K | Weights: 1 crossing per core instead of one per M-block. | — |
| R6 interleaved-k-split | **deferred** — knob-turn once R4's combine exists: for grids that M under-fills (INPUTS `8x16x2048->256`: 4 cores each streaming 9.2 MB of weights), split K by kh band across 3 cores per M-group with R4's combine. Reachable: K-blocks are already kh-aligned and the combine is R4's. | `tensor_m_tiles * 3 <= grid_cores` and weights dominate | `bk` per core = its kh block | Same DRAM bytes as R0, per-core weight stream /3, + combine `M * C_out_p * 4 B * 2/3`. | — |

Regime selection function (host, exact, one place): `select_regime(input) = {INTERLEAVED+ROW_MAJOR: R0, INTERLEAVED+TILE: R3, HEIGHT+ROW_MAJOR: R2, HEIGHT+TILE: R2+R3, BLOCK: R4}`; R1 and R5 are orthogonal overlays chosen by their predicates. Selection depends only on the tensor's memory layout and layout — never on the grid — except R5's L1 predicate, which needs its own pinned test when built. Phase 0 built R0; Refinement 1 added R3 (interleaved), Refinement 2 R2 (HEIGHT, either layout) and Refinement 4 R4 (BLOCK, either layout). R1 overlays every regime; under a K split (R4 with `k_cols > 1`) it is one group per grid column, because the columns multiply different weight slices.

### Traffic ranking

Candidate splits, counted as crossings per tier (no time estimates). `W = K_p * C_out_p * elem` (weights), `A = H*W*C_in_p*elem` (input), `P_m, P_n, P_k` = groups along M, N, K.

| Rank | Split | DRAM crossings | Cross-core | Notes |
|------|-------|----------------|------------|-------|
| 1 | M + weight broadcast (R0 + R1) | input ~1 + halo, weights 1, output 1 | weight mcast payload `(P_m - 1) * W` | Minimum DRAM. Built as the R1 overlay (Refinement 3). |
| 2 | **M only (chosen, R0)** | input ~1 + halo, weights `P_m`, output 1 | 0 | No combine, no duplicated compute. |
| 3 | M x K (dependent split + combine, R6/R4) | input ~1 + halo (bands partitioned by kh), weights `P_m` total, output 1 | partials `M * C_out_p * 4 B * (P_k - 1)` | Same DRAM as rank 2; cuts each core's weight stream and slab by `P_k` (a residency mechanism as much as parallelism). Built for BLOCK (R4) where the caller already cut c. |
| 4 | M x N | input `P_n` (+ gather and tilize compute duplicated `P_n` times), weights `P_m` total, output 1 | 0 | Strictly more bytes than rank 2. Occupancy alternative only (lamp P4). |

Chosen: rank 2 (M only) — cheapest built split; independent axis, no combine. The cheapest split (rank 1) is the R1 overlay on this exact structure.

### Block schedule

```cpp
// per core; logical schedule, realized across reader (NCRISC/NoC0), compute, writer (BRISC/NoC1)
prefetch_bias();                                                     // writer, once per core (has_bias)
for (uint32_t m_block_idx = 0; m_block_idx < num_m_blocks_this_core; ++m_block_idx) {
    fill_halo_slab(m_block_idx);           // reader: 3 kh-bands of the block's input footprint -> cb_halo_slab
    stream_weight_blocks(m_block_idx);     // writer: num_k_blocks x (bk x bn) weight tiles -> cb_weights
    gather_im2col_blocks(m_block_idx);     // reader: num_k_blocks x (bm*32 rows x bk*32 cols) RM -> cb_im2col_rm
    tilize_matmul_block(m_block_idx);      // compute: K-walk of num_k_blocks: tilize K-block -> matmul-accumulate (fp32, L1 acc)
    finalize_block(m_block_idx);           // compute: + bias (row broadcast) [+ ReLU] -> cb_output_tiles
    store_block(m_block_idx);              // writer: output tiles -> DRAM (R0) / nothing (sharded: packed in place)
}
```

| Block operation | Block shape it acts on | Resident across it | Intended frequency of fixed-cost work |
|-----------------|------------------------|--------------------|---------------------------------------|
| `prefetch_bias` | `1 x block_n_tiles` tiles | bias stays fronted in `cb_bias` for the whole kernel | one read barrier per core |
| `fill_halo_slab` | 3 kh-bands x `band_len_px` pixels x `C_in_p` (overlapping bands = one merged range of `2*band_stride_px + band_len_px` pixels) | slab persists for all K-blocks of the M-block | one transaction-id fence per band (<= 3) per M-block; page reads all issued before the fence |
| `stream_weight_blocks` | per K-block `block_k_tiles x block_n_tiles` tiles, tile order k-major/n-fastest (matmul in1 layout) | — (streamed) | one read barrier per K-block |
| `gather_im2col_blocks` | per K-block: `block_m_tiles` tile-rows x 32 im2col rows x `block_k_tiles*32` columns; row = `[tap kw=0 | kw=1 | kw=2 | kh pad]` of kh | slab | one barrier + one `push_back(bk)` per tile-row; per output row: d=1 & all taps in image -> ONE combined slab read of `3*C_in_p*2` B, otherwise one read per in-image tap + zero reads from `MEM_ZEROS` for out-of-image taps, kh-block K pad and `C_in..C_in_p` lanes. Ragged-tail rows are skipped (nominal push). **Tap-major** (Refinement 5; `TAP_MAJOR_IM2COL`, when stride 1, `C_in_p % 32 == 0`, no padded lanes and `bk` is whole taps): a tile-row is laid out as `bk / Ct` blocks of `[32 pixels, C_in_p]`, one per tap; the pixels of one output image row read consecutive slab pixels, so a tap costs one read per image row it spans plus zero reads for padding (res4.x: 768 -> ~100 reads per core). Compute tilizes each block `Ct` wide, which yields the same tiles in the same `[m][k]` order (bit-identical output). **Face-direct** (Refinement 6; `FACE_DIRECT_IM2COL`, when `stride * C_in_p == 16`, bf16 RM activation, no padded lanes, and `d == 1` if a face row holds two taps; `bk` whole kernel rows): one output pixel steps the slab by one 16-element face row, so a 16x16 face of an in0 tile is one contiguous 512 B slab run; the reader (and, with `GATHER_SPLIT_RISCS`, the writer for odd row groups) writes in0 tiles straight into `cb_im2col_tiles`, 4 reads per tile, and compute does not tilize. The face holding tap 2 also reads the next pixel into its K-padding lanes (zero weight rows; read only inside the image, zero-filled outside; the band reaches tap `band_last_tap` = 3); K-padding-only faces are zeroed once. With one K-block, in0 is pushed per row group of `in0_stream_rows` and compute runs one matmul per group (weights retained) and the bias add per `BIAS_STREAM_GROUPS` groups, so the matmul overlaps the gather. stem.conv1: 214.5 -> 24.7 us. |
| `tilize_matmul_block` | output `block_m_tiles x block_n_tiles`, K walk of `num_k_blocks` blocks of `block_k_tiles` | `cb_matmul_partials` (fp32) accumulates in place across all K-blocks | per K-block: `pack_reconfig_l1_acc(0)` (+ `pack_relu_config(none)`), one tilize init/uninit (`bm` tile-rows of width `bk`), one matmul short init + data-format reconfig (`InitMode::ShortAfterPreKBlock`), one L1-acc pass over `bm*bn` tiles |
| `finalize_block` | `block_m_tiles x block_n_tiles` | bias | once per M-block: `pack_reconfig_l1_acc(0)`, [relu config on/off], bias-add init + reconfig; handshakes per row group. no-bias variant: the matmul's last K-block packs straight to `cb_output_tiles` (`LastBlockTarget::Out` / `OutWithRelu`) and this op is empty. |
| `store_block` | `block_m_tiles x block_n_tiles` output tiles in row groups | — | one write barrier per row group; weight prefetch for block j+1 interleaved |

No multicast, no semaphore, no cross-core barrier exists in Phase 0. Refinement 3 adds one: the R1 `weight_mcast` channel (two semaphores per participant, one handshake round per weight K-block) inside `stream_weight_blocks`.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| P1 overlap — K-block granularity | `bk = tensor_kkh_tiles` pays 3 tilize inits + 3 matmul inits + 3 L1-acc passes over `bm*bn` fp32 tiles per M-block; for tiny kh blocks (stem: `tensor_kkh_tiles = 1`) that fixed cost dominates the useful work. | `bk = tensor_k_tiles` (one K-block per M-block: all 3 kh blocks together, allowed by mechanism cap 3) on stem-like shapes. **Built (Refinement 6)** on face-direct shapes (`FACE_DIRECT_WHOLE_K_BLOCK`): first L1-solve candidate there. |
| P2 overlap — `im2col_rm_depth_rows` (moot on face-direct shapes: no `cb_im2col_rm`) | With `bm > 2` the reader stalls after 2 tile-rows of K-block k+1 while compute runs matmul k, serializing the gather against the matmul. | `im2col_rm_depth_rows = block_m_tiles` (reader runs a whole K-block ahead). |
| P3 grid synchronization — M occupancy | Every core streams all weights (R0), so `bytes ∝ num_m_groups`; at `core_m_tiles = 1` (every INPUT) each core reads `K_p*C_out_p` weights for 32 output pixels and re-reads the halo 3x. | `core_m_tiles in {2, 4}` (fewer cores, more work each) on weight-dominated shapes; disappears once R1 lands. |
| P4 occupancy — N split | INPUTS `8x16x2048->256` runs on 4 cores, each streaming 9.2 MB of weights; one core's read rate, not total bytes, bounds it. | `num_n_groups = min(tensor_n_tiles, grid_cores // num_m_groups)` (knob-turn: `n_tile_start` and `block_n_tiles` are already per-core args), accepting the duplicated gather/tilize. |
| P5 packer L1 accumulation | `packer_l1_acc=true` trades the software reload (unpack of fp32 partials per K-block) for in-place packer accumulation; with 3 K-blocks either may win. | `packer_l1_acc=false` (spill/reload). |
| P6 slab-fill issue depth | Interleaved RM pages cannot coalesce (catalog read_coalescing: not valid across interleaved pages); 64-256 B page reads are issue-bound (catalog split_reader: 64 B 3.4x slower than 1 KB). | Issue all slab pages then fence (default) vs trid batches of 8 (catalog double_buffer: 4-8 in flight is flat) vs moving half the band reads to the writer RISC (catalog split_reader: 1.2-1.7x when issue-bound). |
| P7 gather NoC state | The combined per-kh read has a fixed size (`3*C_in_p*2` B) for every in-image row. | `set_read_state` + `read_with_state` (`local_copy_helpers_dataflow.hpp:74-95`) when the size is <= `NOC_MAX_BURST_SIZE`. **Built (Refinement 6)** for the 512 B interior face reads of the face-direct gather (60.6 vs 63.7 us on stem.conv1). |
| P8 sharded local copy (R2/R3) | Copying the resident shard into the slab costs one local copy of the shard per call. | Gather in-shard taps directly from the zero-copy shard CB and only halo taps from the slab (two sources, split combined reads at shard edges). Not built: on stem.conv1 the slow cores' footprints lie 1-3 shards AHEAD of their own shard, so the own-shard copy is not what they wait on; the NoC-direction split of the fill (R2 row) was. |
| P9 bank stagger | Every core's first slab page is at a 32-pixel multiple, so all cores start on the same DRAM bank (catalog bank_stagger: 1.01-1.07x on interleaved sources). | Rotate each core's band read start by its core index. |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| Input DRAM -> `cb_halo_slab` | RM bf16, one pixel row (`C_in_p*2` B) per slab page | reader: TensorAccessor page reads (`noc_async_read_page`, `dataflow_api.h:1241`), runs from `num_contiguous_pages` (= 1 on interleaved) | kh-band ordered; band kh stored at slab pixel offset `kh*band_stride_px`; pixel q of band kh at `kh*band_stride_px + q - (f + kh*d*W)`, `f = (ho0*s - p)*W + wo0*s - p` (can be negative; only in-image pixels are read). Overlapping bands therefore coincide in the slab. |
| `cb_halo_slab` -> `cb_im2col_rm` | RM bf16, tile-sized pages (32 rows x `bk*32` cols = `bk` bf16 tiles per tile-row) | reader: self-aimed L1->L1 NoC reads (`local_copy_helpers_dataflow.hpp:57`) + zero reads from `MEM_ZEROS_BASE` | row r of tile-row t at byte `(r * bk*32*2)` within the tile-row's `bk` pages; slot of tap kw at `kw*C_in_p*2`. |
| `cb_im2col_rm` -> `cb_im2col_tiles` | TILE (in0 format: bf16 Phase 0) | compute: `tilize<bk>` symmetric, `bm` tile-rows | inside the matmul's PreKBlockFn |
| weights DRAM -> `cb_weights` | TILE weight_dtype | writer: TensorAccessor tile reads, page `(kt0 + k)*C_out_p/32 + n_tile_start + n` | k-major, n-fastest = matmul in1 layout (`in1_per_core_w = block_n_tiles`) |
| bias DRAM -> `cb_bias` | TILE weight_dtype, row 0 valid | writer, once per core | page `n_tile_start + n` |
| matmul -> `cb_matmul_partials` | TILE Float32 | compute: `matmul_block` Interm + packer L1 acc, TileRowMajor | fp32 across all K-blocks |
| partials + bias -> `cb_output_tiles` | TILE output dtype | compute: `add_bias_bcast_rows` RowBroadcast, TileRowMajor | ReLU (TARGET) via PACK_RELU around this call only |
| `cb_output_tiles` -> output DRAM | TILE output dtype | writer: tile writes, page `(m_tile_start + mb*bm + r) * C_out_p/32 + n_tile_start + c`, valid rows only | sharded regimes: CB backed on the output shard, no writes |

Deferred inter-Tensix contracts (kept reachable by the structure above):

| Regime | Tensix-to-Tensix contract |
|--------|---------------------------|
| R1 | One injector per weight K-block: reads `bk*bn` tiles once, multicasts into every receiver's `cb_weights` slot (same address on all cores — same CB index/size everywhere), credit via the `McastArgs` sender/receiver semaphore pair; receivers `push_back(bk*bn)` exactly as the Phase 0 writer does. |
| R2/R3 | No synchronization: input shards are immutable and resident before launch, so any core reads any owner's shard at any time (halo from ANY core, several shards away: aspp d=18 at W=64 reaches 1170 rows). |
| R4 | Row-local reduce-scatter, one round per M-block (built): an output-owning column reserves its whole `cb_combine_slots` and increments `combine_credit` on every row peer; a sender waits for its partials and for one credit per owning column, writes N-slice y into slot `my_col` of column y (writer NoC1: own and lower columns; reader NoC0: higher columns, handed over by the core-local `combine_send_go` / `combine_send_done` counters), fences, and increments `combine_arrival` on every owning column; the owner pushes the slots after `k_cols` arrivals. Counters are monotonic (round j waits for (j+1) x the per-round count). The slot and partials CBs are always pushed/popped whole, so their base addresses hold on every core. ReLU never touches a partial. |

### TARGET growth map (each `TARGET[axis] - SUPPORTED[axis]` value; ordered by what blocks the LOOSE acceptance set)

| Order | Axis value(s) | What it needs | Cost class | Blocked by | LOOSE cases that need it |
|-------|---------------|---------------|------------|------------|--------------------------|
| 1 | `weight_dtype=bfloat8_b` (incl. mixed bf16 x bf8_b) | `cb_weights`/`cb_bias` in bf8_b; matmul srcA (in1) bf8_b vs srcB (in0) bf16 — the helper's INPUT reconfig already sets both sources; mixed cells leave EXCLUSIONS | knob-turn (format) | — | all 23 |
| 2 | `activation=relu` | PACK_RELU zero around `finalize_block` only (no-bias: `OutWithRelu`, then explicit clear) | knob-turn (CT flag) | — | 21 |
| 3 | `stride_mode=strided` | values: `band_len_px` with s, gather row mapping `h_in = ho*s + kh*d - p` | knob-turn | — | stem.conv1, res3.0, res4.0 |
| 4 | `channel_alignment=c_in_unaligned` | zero-fill of kh-block K pad and `C_in..C_in_p` lanes in the gather/slab; `C_in_p` rule | knob-turn | — | stem.conv1 |
| 5 | `memory_layout=HEIGHT_SHARDED` (RM) | R2 | placement | 1-4 for stem.conv1 | stem.conv1 |
| 6 | `layout=TILE`, `dtype=bfloat8_b` | R3 (untilize phase) | new compute phase on the same blocking | — | 22 |
| 7 | `memory_layout=HEIGHT_SHARDED` (TILE) | R2 + R3 | placement | 5, 6 | 14 |
| 8 | `dilation_mode=dilated` | values: `band_stride_px = min(d*W, band_len_px)`, three per-tap reads per kh instead of one combined read | knob-turn | — | res5.x, aspp.x |
| 9 | `memory_layout=BLOCK_SHARDED` | R4 (+ per-column weight layout) | scheme-change | 6, 8 | res4.0, res4.x (x2 incl. LoFi), res5.x, aspp.x |
| perf | — | R1, R5, lamps | — | — | — |

`padding_mode`, `bias_mode`, `fp32_dest_acc_en`, `hw_alignment` are fully covered at Phase 0. Index axes: none. Structural impossibilities beyond feature_spec's INVALID: none found.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | an M-block: `block_m_tiles` output tile-rows (32 output pixels each) x all `block_n_tiles` output channels, with its full K walk |
| Grid | `device.compute_with_storage_grid_size()` at call time (R0); the input shard grid (R2-R4). Cores filled row-major (catalog noc_placement: row/diagonal 2.9x faster than the column-major default of `split_work_to_cores`). |
| Per-core work | `core_m_tiles = ceil(tensor_m_tiles / grid_cores)` tiles starting at `m_tile_start = core_idx * core_m_tiles`; `num_m_blocks_this_core = ceil(actual_m_tiles / block_m_tiles)`; `n_tile_start = 0`, `block_n_tiles = tensor_n_tiles` |
| Remainder | only the last core is ragged: `actual_m_tiles = tensor_m_tiles - (num_m_groups - 1) * core_m_tiles`; nominal counts, valid-row writes only. Tile geometry is `ceil`-based on the flattened per-image M (`tensor_m_tiles = ceil(H_out*W_out / 32)`; batch is 1, the flat layout pads once at the end). |
| Per-core RT args | all buffer addresses (input, weight, bias, output) + `m_tile_start`, `actual_m_tiles`, `num_m_blocks_this_core`, `n_tile_start`. Everything shape-derived is CT. No address in CT args, defines, or descriptor-hashed values (generic_op hashes CT args, CB layout and RT-arg counts only: `generic_op_device_operation.cpp:78-125`), so the harness's second dispatch on fresh buffers is a cache hit. |
| Regime pins | Phase 0 has one regime. Each later regime is selected by memory_layout/layout only, so the golden cells pin it; R5 (L1-predicate) needs an explicit regime-pinned test when built. |

## Circular Buffers

Full ledger (capacity vs live set, axis accounting, sharing): `l1_ledger.md`. `T_x` = tile bytes of format x.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_halo_slab` | 0 | `C_in_p * 2` B (one pixel) | `2*band_stride_px + band_len_px` | spans M (affine footprint of `bm`), spans c; streams nothing (filled once per M-block). Scratch, no FIFO handshake in R0/R2 (R3: compute untilize -> reader). HEIGHT RM: the writer fills the higher-shard runs (semaphore handshake) and, with the split gather, reads it too. | bfloat16 (RM) | reader (+ writer, HEIGHT RM) | reader (+ writer, split gather) | one M-block |
| `cb_im2col_rm` | 1 | `T_bf16` (32 RM rows x 32 cols) | `im2col_rm_depth_rows * block_k_tiles` | spans K-block; streams M tile-rows. Not allocated on face-direct shapes. | bfloat16 (RM content) | reader | compute (tilize) | one tile-row |
| `cb_im2col_tiles` | 2 | `T_in0` | `block_m_tiles * block_k_tiles` | spans M-block and K-block (matmul waits the whole in0 block, or one row group per call when streamed) | in0 = activation dtype (bf16 Phase 0) | compute (tilize); face-direct: the reader (reserve/push; the writer writes odd row groups into the reserved block between `gather_go` and `gather_done`) | compute (matmul) | one K-block |
| `cb_weights` | 3 | `T_w` | `weight_depth * block_k_tiles * block_n_tiles` | spans K-block x N; streams K-blocks | weight_dtype | writer | compute (matmul) | one K-block per slot |
| `cb_bias` | 4 | `T_w` | `out_n_tiles` (= `block_n_tiles`; the core's N-slice under R4; allocated only when `has_bias`) | spans the core's output N | weight_dtype | writer | compute (bias add) | whole kernel |
| `cb_matmul_partials` | 5 | `T_fp32` = 4096 B | exactly `block_m_tiles * block_n_tiles` (cap #2) | spans M-block x N; accumulated over K | Float32 | compute (matmul; under R4 one caller-owned reserve/push per M-block) | compute (matmul drain / bias add); under R4 the writer (scatter, one wait/pop per M-block) | one M-block |
| `cb_output_tiles` | 6 | `T_out` | `out_depth_row_groups * out_subblock_h * block_n_tiles` | spans N; streams M row groups | output dtype | compute | writer | one row group |
| `cb_combine_slots` (R4, `k_cols > 1`) | 8 | `T_fp32` | `k_cols * block_m_tiles * out_n_tiles` | spans the row's k_cols channel slices x the core's output block | Float32 | writer (reserve + push; the bytes are NoC-written by every row peer's writer and reader) | compute (`reduce` AccumulateViaAdd, BulkWaitBulkPop) | one M-block |
| `cb_combined` (R4 with bias) | 9 | `T_fp32` | `block_m_tiles * out_n_tiles` | the row-summed output block before the bias | Float32 | compute (reduce) | compute (bias add) | one M-block |

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `prefetch_bias` | `1 x bn` | no (dataflow) | DRAM bias | `cb_bias`, push `bn` once | fronted until kernel end (compute `wait_front(bn)` at start, `pop_front(bn)` at end) |
| 2 | `fill_halo_slab` | 3 bands | no (dataflow; per-page reads with page runs) | DRAM input | `cb_halo_slab` (scratch) | valid for all K-blocks of the M-block |
| 3 | `stream_weight_blocks` | `bk x bn` per K-block | no (dataflow) | DRAM weights | `cb_weights`, push `bk*bn` per K-block | — |
| 4 | `gather_im2col_blocks` | `bm` tile-rows x `bk` tiles per K-block | no (dataflow) | `cb_halo_slab` | `cb_im2col_rm`, push `bk` per tile-row | — |
| 5 | `tilize_matmul_block` | out `bm x bn`, `num_k_blocks` K-blocks of `bk` | yes: `matmul_block` with PreKBlockFn = `tilize` | `cb_im2col_rm` (wait `bk` per tile-row, via tilize), `cb_weights` (wait/pop `bk*bn` per K-block) | `cb_matmul_partials` (row groups; has_bias) or `cb_output_tiles` (no bias, `LastBlockTarget::Out`) | partials: `bm*bn` tiles pushed (row groups) after the last K-block |
| 6 | `finalize_block` | `bm x bn` | yes: `add_bias_bcast_rows` | `cb_matmul_partials` (wait/pop per row group), `cb_bias` (fronted) | `cb_output_tiles`, push per row group | partials drained to base |
| 7 | `store_block` | `bm x bn` | no (dataflow) | `cb_output_tiles` (wait/pop per row group) | DRAM output | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot | raw_api | `compute_kernel_hw_startup(icb0, ocb)` (Regular) | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:113` | `(cb_im2col_rm, cb_im2col_tiles)` — must match the FIRST op (tilize), `:28-31`; the matmul's "Reverse is mandatory" (`matmul_block_helpers.hpp:100-106`) applies when matmul is the first op; here every later phase restores its own state (ShortAfterPreKBlock + INPUT_AND_OUTPUT reconfig; bias helper reconfig). Same boot as the measured catalog chain `examples/compute_block_size/program_descriptor_with_inline_kernels.py:127,146,159`. | — | — | — |
| `tilize_matmul_block` | helper | `compute_kernel_lib::matmul_block` | `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp:339-371` | `transpose=false, packer_l1_acc=true, last_block_target = has_bias ? Interm : (relu ? OutWithRelu : Out), tile_order=TileRowMajor, init_mode=ShortAfterPreKBlock (:108, restore contract :182-196), in0_policy=in1_policy=WaitAndPopPerKBlock, PreKBlockFn=<tilize functor>, reconfig=INPUT_AND_OUTPUT`; shape `MatmulBlockShape::of(bm/sbh, bn/sbw, sbh, sbw, bk, num_k_blocks)` (`:160`); `interm_buf = cb_matmul_partials`; `in1_per_core_w = bn`, `out_row_width = bn` | `cb_im2col_tiles`, `cb_weights` | `cb_matmul_partials` / `cb_output_tiles` | `in0_block_k = block_k_tiles`, `num_k_blocks`, `in0_num_subblocks*out_subblock_h = block_m_tiles`, `in1_num_subblocks*out_subblock_w = block_n_tiles` (subblock split is derived, not a knob) |
| PreKBlockFn body | helper | `compute_kernel_lib::tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp:194-207` (symmetric mode `:133-138`) | `<block_k_tiles, cb_im2col_rm, cb_im2col_tiles, InitAndUninit, WaitBlock, UnpackAndPackReconfigure, Fast>(rm, tiles, num_blocks = bm)`; preceded by `pack_reconfig_l1_acc(0)` (`tt_metal/hw/inc/api/compute/pack.h:190`) and, once relu exists, `pack_relu_config(ReluConfig::none())` (`pack.h:201`) | `cb_im2col_rm` | `cb_im2col_tiles` | `block_width_tiles = block_k_tiles`, `num_blocks = block_m_tiles` |
| `finalize_block` | helper | `compute_kernel_lib::add_bias_bcast_rows` | `ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp:156-168` (prerequisites `:101-105`; own reconfig/init `bias_add_helpers.inl:49-57`) | `<BiasBroadcast::RowBroadcast, OutputCBLayout::TileRowMajor>`, `BiasAddShape::of(bm/sbh, bn/sbw, sbh, sbw, bn)` (`:45`); caller first issues `pack_reconfig_l1_acc(0)` and, for relu, `pack_relu_config(ReluConfig::zero())` before / `none()` after | `cb_matmul_partials`, `cb_bias` | `cb_output_tiles` | same subblock grid as the matmul |
| no-bias ReLU clear | raw_api | `pack_relu_config(ReluConfig::none())` | `pack.h:201`; `ReluConfig` `tt_metal/tt-llk/tt_llk_blackhole/llk_lib/llk_defs.h:93` | after every `matmul_block<OutWithRelu>` call | — | — | — |
| `fill_halo_slab` | raw_api | `TensorAccessor::get_noc_addr` + `num_contiguous_pages`; `noc_async_read` | `tt_metal/hw/inc/api/tensor/tensor_accessor.h:165,192`; `tt_metal/hw/inc/api/dataflow/dataflow_api.h:547` | page runs over `[band_lo, band_hi]` | DRAM / shard | `cb_halo_slab` | `band_len_px`, `band_stride_px` |
| `gather_im2col_blocks` | raw_api | self-aimed read `local_addr` + `Noc::async_read`; trid fencing `set_read_trid` / `async_read_barrier_with_trid` | `ttnn/cpp/ttnn/kernel_lib/local_copy_helpers_dataflow.hpp:57,147,158` | combined 3-tap read (d=1, in-image) or per-tap; zero reads from `MEM_ZEROS_BASE` | `cb_halo_slab` | `cb_im2col_rm` | `block_k_tiles`, `block_m_tiles` |
| `stream_weight_blocks`, `prefetch_bias`, `store_block` | raw_api | `noc_async_read_page` / `noc_async_write_page` via TensorAccessor | `dataflow_api.h:1241`; `tensor_accessor.h:165` | per-tile pages | DRAM / `cb_output_tiles` | `cb_weights` / `cb_bias` / DRAM | `bk`, `bn`, `bm` |

Helpers considered and rejected:

| Raw entry | Candidate helper | Mismatch (file:line) | Reason |
|-----------|------------------|----------------------|--------|
| `gather_im2col_blocks`, `fill_halo_slab` | `dataflow_kernel_lib::read_sticks_for_tilize` | `ttnn/cpp/ttnn/kernel_lib/tilize_helpers_dataflow.hpp:88` (signature `(accessor, total_num_rows, row_bytes, start_page, byte_offset_within_page)`, doc `:63-85`) | Reads `total_num_rows` CONSECUTIVE sticks `start_page..` at one byte offset into consecutive tile-row slots. An im2col row is 3 taps from 3 different pixels (non-consecutive for d>1, out-of-image -> zero), and every pixel feeds 9 rows; neither the tap map nor zero-fill is expressible. The gather is a custom block operation (gap recorded: a "gather rows by index list with zero-fill into tilize-layout pages" dataflow helper would close it). |
| `stream_weight_blocks`, `store_block` | none in `kernel_lib` for tile-page block reads/writes | — | No dataflow helper exists for a rectangular tile-block read; TensorAccessor page reads are the primitive. |
| boot | (matmul's documented `compute_kernel_hw_startup<SrcOrder::Reverse>`) | `matmul_block_helpers.hpp:100-106` vs `compute_kernel_hw_startup.h:28-31` | The first compute op is tilize; startup must match the next op's CBs. |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| finalize_block | add (bias) | `cb_matmul_partials`: All | `cb_bias`: Row0 (prepared bias `[1,1,1,C_out]`, tile row 0) | Row (`add_tiles_bcast_rows`) |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| L1-ACC left enabled after the matmul | `matmul_block` with `packer_l1_acc` sets `llk_pack_reconfig_l1_acc(block == 0 ? 0 : 1)` before each pack and never clears it (`matmul_block_helpers.inl:414,467`). The next pack in the kernel — the next K-block's tilize (PreKBlockFn) or the bias add — would ACCUMULATE onto stale CB contents. | `pack_reconfig_l1_acc(0)` at the top of the PreKBlockFn and before `add_bias_bcast_rows`. |
| PACK_RELU never cleared | `OutWithRelu` sets ZERO_RELU at the last K-block and never resets it (`matmul_block_helpers.inl:217-221`); the bias helper leaves relu to the caller (`bias_add_helpers.hpp:101-105`). Left on, the next M-block's tilize clamps negative ACTIVATIONS and K-partials get relu'd. | Clear with `pack_relu_config(ReluConfig::none())` after every relu'd pack; never enable it before the final value exists (after the last K-block, after bias; after the R4 combine). |
| Prepared-weight K padding per kh block | Padding K once at the end gives the same `K_p` (C_in_p = 8: 96) with zero rows in the wrong places; the output is silently wrong. | Prepare pads each kh block; `validate()` checks `weight.shape[2] == K_p(C_in, input layout/memory config)` (ttnn's `is_valid_device_conv_weights` never checks K). Golden `test_regression.py` single-tap shifts pin the ordering exactly. |
| Garbage x zero weight = NaN | Out-of-image taps, kh-block K pad and `C_in..C_in_p` lanes are read by the matmul against zero weight rows; stale L1 can hold NaN/Inf. | Gather zero-fills them from `MEM_ZEROS_BASE` every time (CB slots are reused). Phase 0 (tile-aligned C_in) has no K pad or lane pad — only image-edge taps. |
| HiFi4 + fp32 DEST with bf16 inputs on Wormhole B0 | `matmul_block_helpers.hpp:263-264` documents silent K-accumulator corruption (#38306); the golden grid runs HiFi4 + fp32 DEST. | Honor the caller's fidelity (contract). If the device is WH B0 and HiFi4 accuracy fails while HiFi2 passes, it is this LLK issue, not the design. |
| fp32 partials through srcA in the bias add | FPU binary unpacks fp32 partials as tf32 (10-bit mantissa). | Below the bf16/bf8_b output rounding; accepted. Partials stay Float32 in L1 (audit 2). |
| DRAM alignment on Blackhole (64 B) | An RM pixel row of C_in bf16 channels is < 64 B for C_in < 32. | `C_in_p` rule pads slab rows to the DRAM page stride (interleaved); sharded RM uses L1 alignment (16 B). Phase 0 (C_in % 32 == 0) is aligned by construction. |
| Halo from any core (R2-R4) | 320-row shards at W=256 need 257 rows from the next core; aspp d=18 at W=64 needs 1170 rows, several shards away; shards are not row-aligned. | The slab fill addresses pixels through a TensorAccessor over the sharded tensor (owner by pixel index), page runs stop at shard edges — no neighbour assumption anywhere. |
| Program cache | The second dispatch (fresh buffers, same shapes) must add no cache entry and be correct. | Addresses only in RT args (per-core vectors); CB sizes and CT args depend on shapes only; sharded CB backing hashes presence only. |
| Ragged last core | Fewer valid tiles than nominal. | Nominal push/pop/wait counts and matmul shape; reader skips invalid rows; writer writes valid tiles only (mechanism cap 9). |
| Deadlock: writer weight prefetch vs output drain (> 1 M-block per core) | Writer blocked on `cb_weights` reserve for block j while compute is blocked in finalize(j-1) on `cb_output_tiles`. | Writer order per M-block: weights of j, then drain output j; prefetch of j+1 weights only between drained row groups (their reserves cannot block — see stall-shadow table). |
