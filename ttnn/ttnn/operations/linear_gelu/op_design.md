# Operation Design: linear_gelu

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (matmul + bias + SFPU activation), one `generic_op` dispatch |
| Goal | `y = gelu_tanh(x @ W + b)` for the Nomic Embed v2 dense FFN fc1 (K = 768, N = 3072) in one device program, at or under `max(T_compute/0.70, T_DRAM/0.60)` (90.8 us at 8x512). |
| Math | `z[b,0,s,n] = sum_k x[b,0,s,k] * W[k,n] + bias[n]`; `y = 0.5 * z * (1 + tanh(sqrt(2/pi) * (z + 0.044715 * z^3)))` |
| Mode | Hybrid: helper-built (`matmul_block` + packer-thread activation + `ttnn.Mcast`) with one additive helper extension (GELU_TANH in the packer-activation dispatch) |
| References | `eval/golden_tests/linear_gelu/feature_spec.py`, `eval/golden_tests/linear_gelu/helpers.py`, `.claude/references/precision_convention.md`, `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/sfpu_activation_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/mcast/host/mcast.hpp`, `tt_metal/hw/inc/api/compute/eltwise_unary/gelu.h`, catalog entries listed under "Catalog evidence", sibling op evidence `ttnn/ttnn/operations/qkv_proj_heads_rope/changelog.md` (same model, same K, same board) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | (B, 1, S, 768), bf16, TILE, interleaved; S in 1..512; B*S <= 4096 | - | RT (buffer address) |
| `weight` | `ttnn.Tensor` | yes | (768, 3072), bf16, TILE, interleaved | - | RT (buffer address) |
| `bias` | `ttnn.Tensor` | yes | (3072,) or (1, 3072), bf16, TILE, interleaved | - | RT (buffer address) |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` | no | any fidelity / approx; `fp32_dest_acc_en` must be True (gated) | `default_compute_kernel_config()` = HiFi2, fp32_dest_acc_en=True, math_approx_mode=False | descriptor (fidelity, DEST width) |
| `memory_config` | `ttnn.MemoryConfig` | no | interleaved L1 (Phase 0); interleaved DRAM (refinement) | `input_tensor.memory_config()` | output allocation |

Entry-point rules:

| Rule | Decision |
|------|----------|
| First line | `validate(input_tensor, weight, bias, compute_kernel_config=..., memory_config=...)` |
| Default config | `default_compute_kernel_config()` is an exported factory (fresh descriptor per call). The entry resolves `cfg = compute_kernel_config if compute_kernel_config is not None else default_compute_kernel_config()`. A caller config is honored as given (fidelity, approx flag); `fp32_dest_acc_en=False` is refused by `validate()` with `UnsupportedAxisValue`, never silently overridden. |
| Check order in `validate()` | structural `ValueError` checks first, then model-constant refusals, then SUPPORTED per axis, then EXCLUSIONS |
| Malformed calls -> `ValueError` | x not rank 4 or `x.shape[1] != 1`; weight not rank 2 or `weight.shape[0] != x.shape[-1]`; bias not one row (`bias.shape` not `(N,)` / `(1, N)` with `N == weight.shape[1]`); tensors on different devices |
| Model constants -> `UnsupportedAxisValue` | `K != 768`, `N != 3072`, `S > 512`, `B*S > 4096`; weight / bias not bf16 TILE interleaved |
| Forbidden on host | `ttnn.to_layout`, `ttnn.tilize`, `ttnn.pad`, `ttnn.slice`, `ttnn.reshape`, or any composed ttnn op. Host work = validation, parameter derivation, `ttnn.allocate_tensor_on_device` for the output, one `ttnn.generic_op`. |
| Output | new tensor (B, 1, S, 3072), bf16, TILE, interleaved, on `memory_config` |

### Registry declarations (op file)

| Item | Value |
|------|-------|
| `INPUT_TAGGERS` | `{"alignment": tag_alignment}`; `tag_alignment(inputs, axes)` returns `"tile_aligned"` if `inputs[0][-2] % 32 == 0` else `"h_non_aligned"` (K, N are tile multiples) |
| Phase 0 `SUPPORTED` (implementer writes it) | dtype [bfloat16], layout [TILE], alignment [tile_aligned], fp32_dest_acc_en [True], memory_layout [INTERLEAVED], buffer_type [L1], out_buffer_type [L1] |
| `EXCLUSIONS` | `[]` |
| Axis sources in `validate()` | dtype / layout / memory_layout / buffer_type from `input_tensor`; out_buffer_type from `memory_config or input_tensor.memory_config()`; fp32_dest_acc_en from the resolved config; alignment from the tagger |

### TARGET axes vs. the Blocking Model

| TARGET value not in Phase 0 | Unlock class | What changes |
|-----------------------------|--------------|--------------|
| alignment = h_non_aligned | knob-turn | Tile geometry is already per-image `ceil` in Phase 0. Only the writer's padded-row zero fill (see Dataflow, stage W3) is added. No loop-nest change. |
| buffer_type = DRAM (x) | knob-turn for correctness, regime R2 for perf | The reader's `TensorAccessor` is placement-agnostic, so correctness is a gate flip. x in DRAM is read Gc = 11 times under R1 (69 MB, about 154 us at 448 GB/s), so the refinement that adds DRAM x also builds R2 (x row multicast). |
| out_buffer_type = DRAM | knob-turn | The writer's `TensorAccessor` is placement-agnostic. 25.2 MB of output at 448 GB/s is 56 us, streamed over the whole op (no end burst), below the 90.8 us target. |

The op has no sharded TARGET values. Every placement value is interleaved, so no physical shard pins the blocking.

### Structural impossibilities

None beyond `feature_spec.py` INVALID.

## Tensors

### Input

| Property | input_tensor (x) | weight (W) | bias (b) |
|----------|------------------|------------|----------|
| Shape | (B, 1, S, 768) | (768, 3072) | (3072,) or (1, 3072) |
| Dtype | bfloat16 | bfloat16 | bfloat16 |
| Layout | TILE (each image tile-padded independently to ceil32(S) rows) | TILE | TILE (one tile row; only row 0 of each tile is data) |
| Memory | interleaved; L1 (Phase 0), DRAM (refinement) | interleaved (DRAM in the model) | interleaved (DRAM in the model) |

### Output

| Property | Value |
|----------|-------|
| Shape | (B, 1, S, 3072); padded storage (B, 1, ceil32(S), 3072) |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | interleaved, `memory_config` (default: x's) |
| Padded rows | s in [S, ceil32(S)) are exactly +0.0 (refinement h_non_aligned; Phase 0 has none) |

Tile geometry (single source, host):

| Symbol | Formula |
|--------|---------|
| `seq_tiles` | `ceil(S / 32)` (per image) |
| `tensor_m_tiles` | `B * seq_tiles` (never `floor(B*S/32)`) |
| `tensor_k_tiles` | `768 / 32 = 24` |
| `tensor_n_tiles` | `3072 / 32 = 96` |
| `k_ext_tiles` | `tensor_k_tiles + 1 = 25` (24 x/W K tiles plus the bias K tile) |
| tile row `m` | `m = b * seq_tiles + st`; tail row iff `st == seq_tiles - 1 and S % 32 != 0`; `valid_rows(m) = S - 32*(seq_tiles-1)` for a tail row, else 32 |
| x page | `m * tensor_k_tiles + k` |
| W page | `k * tensor_n_tiles + n` |
| bias page | `n` |
| out page | `m * tensor_n_tiles + n` |

## Blocking Model

The first design section. Everything below it is downstream of these decisions.
Semantics: `.claude/references/blocking-model.md`.

### Op profile (what the scheme must respect)

| Quantity (8x512) | Value | Consequence |
|------------------|-------|-------------|
| Tile products | 128 x 96 x 25 = 307,200 (incl. bias K tile) | FPU-heavy: about 2,800 products per core |
| W + bias | 4.72 MB, DRAM | Must cross DRAM once. 10 crossings would be 47 MB, at least 105 us, more than the whole target. |
| x | 6.29 MB, L1 interleaved | Crosses no DRAM in production. Shared by every core that owns the same tokens. |
| y | 25.2 MB, L1 interleaved | The largest stream. Must be written progressively, not in a tail burst. The sibling op measured a 12-18 us end-of-K store burst. |
| GELU | about 38 SFPU ops x 32 iterations per tile (accurate form, `ckernel_sfpu_gelu.h:351`) | Per output tile this is comparable to or above the 25-product FPU work, so the activation must overlap the matmul (packer thread). See lamp L1. |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| token rows `m` (B x seq_tiles, folded: rows are treated identically and are contiguous tile rows) | independent: each output row depends only on its own x row | `block_h` (tile rows per activation block) | `min(core_m_tiles, floor((CB_BUDGET_BYTES - fixed_bytes) / per_row_bytes))` = 4 at every 4096-token shape (13 rows per core -> blocks 4,4,4,1) | host `_block_h()` from the ledger formula; passed as compute / reader / writer RT arg `block_h` and the tail extent `last_block_h` | grid rows (y): contiguous `core_m_tiles = ceil(tensor_m_tiles / grid_y)` rows per grid row; `active_rows = ceil(tensor_m_tiles / core_m_tiles)` | knob-turn |
| output features `n` | independent; W and bias are reuse-shared along `m` (their slice does not vary with the token rows) | `block_w` (N tiles per weight block = one W delivery chunk = one compute call) | 3 (32 n-blocks; each grid column owns 3, the last owns 2) | host constant `BLOCK_W`; CT arg `block_w` | grid columns (x): `core_n_blocks = ceil(ceil(96 / block_w) / grid_x)`, `core_n_tiles = block_w * core_n_blocks`, `active_cols = ceil(ceil(96/block_w) / core_n_blocks)` | knob-turn |
| contraction `k` (+ bias K tile) | dependent: every output element sums over all of it | `k_ext_tiles` (K tiles per accumulation = whole extent) | 25 (whole K plus the bias tile, one K-block, accumulated in DEST) | host constant `K_EXT_TILES = 768/32 + 1`; CT arg | not assigned across cores: whole extent per core | scheme-change (R6 K-split + combine) |

Why `block_w = 3` and not the coarsest 9: the W slice arrives over DRAM and multicast before any compute on it (stall-shadow check, below). At 3 the first compute call needs one 150 KB chunk per column, not the whole 450 KB slice, and row-block 0 then consumes chunk j while chunk j+1 arrives. 3 divides 96, so every core's n extent is a whole number of blocks and there is no ragged n-block. With `block_h = 4` the subblock stays at the 4-tile DEST budget (4x1), so the narrower n-block costs no subblock area.

Why `k` is whole: one K-block means the full 25-tile accumulation stays in fp32 DEST. That removes the fp32 partials CB (117 tiles x 4 KB = 468 KB), the per-K-block spill/reload and the separate bias/activation epilogue pass. It also lets each finished subblock be activated and stored at once. Nothing bounds K except L1, and the per-row cost of whole K is `k_ext_tiles` pages in the activation CB, already inside the budget.

Stage-data re-check. The scheme's only internal datum is the fp32 `z` subblock in DEST. Its axes are (`m`, `n`) of one subblock. It is activated in place on the packer and packed to `cb_output_tiles`, on the core that computed it. There is no coordinating core and no stage that runs on one core while the rest wait. The W column sender only issues reads and multicasts. It computes the same work as every other core, so the sender role is a communication role and not a work assignment.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_activation_block` | `ACT_DEPTH` | 2 | The reader fills row-block i+1 (192 KB at block_h=4) during row-block i's compute (about 24 us), so the x read never stalls the FPU. |
| `cb_weight_resident` | none (resident) | 1 copy of the core's whole W slice | Resident: W crosses DRAM once and is reused by every row-block. Depth is not a knob, because the buffer is never recycled. |
| `cb_output_tiles` | `OUT_RING_PAGES` (Refinement 4; was `OUT_DEPTH`) | 24 pages minimum, depth `max(2, ceil(24 / (block_h * block_w)))` whole blocks | The writer drains call j's subblocks while compute packs later calls; about 24 pages ride out DRAM write latency. |
| `cb_bias_ones_tile` | none | 1 page, constant | - |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| fp32 DEST, half-sync (`fp32_dest_acc_en=True`) | output subblock `sb_h * sb_w <= 4` tiles | host derives `(sb_h, sb_w)` = largest-area pair with `sb_h \| rows`, `sb_w \| block_w`, `sb_h*sb_w <= 4`, preferring the larger `sb_h` on ties; re-derived for the tail row-block (`last_block_h`) | DEST overrun: the packer reads tiles from the other half, giving silently wrong output |
| `matmul_block` shape legality (`matmul_block_helpers.hpp:139-170`) | `in0_num_subblocks = rows / sb_h`, `in1_num_subblocks = block_w / sb_w` must be exact | `sb_h` divides the call's rows (nominal and tail separately); `block_w` divides `core_n_tiles` by construction | partial subblocks unpacked from the wrong tiles |
| `cb_weight_resident` must hold the core's full W slice | `core_n_tiles` (via `grid_x`, `block_w`) | host checks `footprint(block_h=1) <= CB_BUDGET_BYTES`; raises `RuntimeError` naming the grid otherwise (unreachable on p100a 11x10 and WH 8x8) | CB allocation over L1 / clobbers tensors |
| Mcast channel: one receiver group per grid column, consecutive in the receiver order | group = the `active_rows` cores of one column | `ttnn.Mcast(..., receiver_group_size=active_rows, McastFixedSenderConfig(sender_index=0), receiver_order=McastCoreOrder.ColumnMajor)` (`mcast.hpp:131-137`) over the `active_cols x active_rows` rectangle | row-major grouping would multicast a column's W to the wrong cores |
| Mcast destination address | W chunk lands at the same L1 offset on every column core | `cb_weight_resident` is one CB with identical index and size on all cores of the rectangle (same allocation address), and chunk j's offset is `j * k_ext_tiles * block_w * tile_bytes` | data lands outside the CB |
| `In1BaseOffsetFn` tile offset (`matmul_block_helpers.hpp:227`, used at `.inl:266`) | chunk j's base = `j * k_ext_tiles * block_w` tiles | layout of `cb_weight_resident` is chunk-major `[j][k][w]`, `in1_per_core_w = block_w` | the unpacker reads another chunk's columns |
| Active rows when `tensor_m_tiles < grid_y` | `active_rows` may be 1 | a one-core column group is a degenerate zero-fan-out sender; `send()` still runs (local copy semantics, `mcast_args.hpp:21-27`) | none if honored; regime-pinned test 1x32 / 1x8 |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R1 weight-stationary 2D grid** (m over grid rows, n over grid columns; W slice resident per core via one column multicast; x row-blocks unicast; full K + bias tile in DEST; GELU on the packer) | **built** | every supported call (K = 768, N = 3072, 1 <= tensor_m_tiles <= 4096) | activation block `block_h x k_ext_tiles`; weight block `k_ext_tiles x block_w`; output block `block_h x block_w` | Minimum (DRAM boundary): W + b once, x once, y once. R1: W + b once (multicast down the column, `active_rows - 1` NoC receivers per byte); y once; x 0 DRAM (L1) but `active_cols` (11) NoC reads per byte. Above the minimum only by x's NoC replication (R2). | `block_h`: fewer W re-walks per x byte. Every row-block re-unpacks the resident W slice (unpack traffic `core_n_tiles * k_ext_tiles` tiles per row-block), so a taller block amortizes it and the per-call matmul init/reconfig, intended once per kernel per subblock shape. Also fewer `cb_activation_block` handshakes (one per row-block). `block_w`: fewer compute calls and writer flushes per row-block, at the cost of a longer first-chunk W wait. |
| R2 x row multicast (R1 plus one injector per grid row reads x row-block i once and multicasts it to the row's `active_cols` cores) | deferred: in every Phase 0 cell x is L1-resident, so the 11x replication is NoC-only, at about 8 GB/s per core against a single reader's capacity. A per-row-block rendezvous would couple the 11 cores of a row so the slowest gates the row. R2 becomes necessary with `buffer_type = DRAM` (69 MB of DRAM at 11x) and is built with that refinement. | `buffer_type == DRAM` (when built) | same as R1 | x: 1 DRAM/L1 crossing plus a multicast to `active_cols - 1` receivers, instead of 11 unicast reads | same as R1 plus one rendezvous per row-block amortized over `block_h` rows |
| R3 W unicast (every core reads its own W slice from DRAM) | rejected: superseded by R1's multicast, which Phase 0 builds. It is the same structure with group size 1. | - | as R1 | W + b: `active_rows` (10) DRAM crossings = 47 MB, at least 105 us of DRAM time in the prologue | - |
| R4 output-stationary K-streamed (both operands streamed in K-blocks, fp32 L1-accumulated partials, bias seed, separate bias/GELU epilogue pass; the stock matmul_2d shape and the sibling op's scheme) | rejected: superseded by R1 | - | out `core_m x core_n` fp32 partials resident; in0/in1 K-blocks streamed | W + b once (with multicast); x 11x NoC; plus fp32 partials packed and unpacked once per K-block (L1-local), 468 KB of fp32 partials L1 | R1 removes the partials CB, the epilogue unpack pass and the end-of-K store burst (sibling op: 12-18 us), and needs only a 450 KB bf16 W slice. R4 offers nothing at these shapes that R1 lacks. |
| R5 1D splits (m only: every core needs all of W, Mt=128 over 110 cores is 2 vs 1 rows, 50% balance; n only: at most 96 cores and every core reads all of x, 604 MB NoC) | rejected: superseded by R1 (better balance and less traffic) | - | - | m-only: W multicast to 110 cores, streamed (does not fit L1); n-only: x 96x | - |
| R6 K-split + cross-core combine (dependent axis; grid rows split K, fp32 partials combined down the column, then bias + GELU on the combining core's share) | deferred: R1 already fills >= 88 of 110 cores at every flagged shape (tensor_m_tiles >= 32). The shapes that under-fill (tensor_m_tiles < 10, B*S < 320) are correctness-only per the prompt. Reachable: `matmul_block` already takes `num_k_blocks` and the column multicast channel gives the combine its topology. | `tensor_m_tiles < grid_y` (when built) | k slice `k_ext_tiles / grid_rows` per core | adds fp32 partial traffic `(grid_rows-1) * block_h * core_n_tiles * 4 KB` per column; saves nothing on DRAM | - |

R1's domain is every supported call. No other regime applies there: R2 only changes x delivery when x is in DRAM, R3 to R5 are superseded, and R6 only pays when R1 under-fills the grid, which no flagged shape does.

### Traffic ranking

Per 4096-token call. Bytes per tier, structural only.

| Candidate split | DRAM crossings | Cross-core NoC | Occupancy (separate) | Rank |
|-----------------|---------------|----------------|----------------------|------|
| 2D (m rows x n cols) + W column multicast + x row multicast (R1 + R2) | W 4.72 MB x1 (+ x 6.29 MB x1 if x in DRAM) | W 4.72 MB x 9 receivers; x 6.29 MB x 10 receivers; y 25.2 MB | 110 / 110 cores; busiest 117 tiles vs 111.7 mean | 1 (cheapest) |
| 2D + W column multicast, x unicast (R1, built) | W x1 | W x 9 receivers; x 6.29 MB x 11 reads = 69 MB; y 25.2 MB | same | 2 |
| 2D, both unicast (R3) | W x10 = 47 MB | x 69 MB; y 25.2 MB | same | 4 |
| 2D + K-split combine (R6) | W x1 | adds fp32 partials per column | more cores only below Mt=10 | 3 at small Mt only |
| 1D m-only | W x1 with a 110-way multicast, or x110 | W 4.72 MB x 109 | 2-vs-1 row imbalance | 5 |
| 1D n-only | W x1 | x 6.29 MB x 96 = 604 MB | 96 cores | 6 |

Chosen: R1. R1 + R2 is cheaper only by x's NoC replication, which in every Phase 0 cell crosses no DRAM. It is deferred as R2 with that reason.

### Stall-shadow check

| Waiting stage | Waits for | Independent work scheduled into the stall | Legality |
|---------------|-----------|-------------------------------------------|----------|
| compute, row-block 0 | W chunk j (DRAM read by the column sender plus multicast) | Compute runs call (row-block 0, chunk j-1) while chunk j arrives. W is delivered and consumed in n-block order with cumulative `wait_front` on `cb_weight_resident`. Reader order: x row-block 0, then W chunks 0..n-1 with x row-block 1 issued after chunk 0. | Reorders no arithmetic: each output tile's K-sum order is unchanged, so floating-point order is identical. Row-block 0 compute (block_h=4: 900 products, about 24 us) exceeds the whole W arrival (DRAM floor 10.5 us), so only chunk 0's wait (about 150 KB per column, a few us) is exposed. |
| compute, row-block i > 0 | x row-block i | nothing needed: `ACT_DEPTH = 2` prefetch | - |
| packer GELU | the math thread finishing subblock s | math computes subblock s+1 into the other DEST half while the packer runs GELU on subblock s | Packer-thread SFPU (`sfpu_activation_helpers.hpp:71-74`). Elementwise, order-free. |
| writer | packed subblocks | nothing (the writer has no other work) | - |

### Block schedule

```cpp
// per core (r, c); num_row_blocks_this_core = ceil(core_m_tiles_this_core / block_h)
// num_n_blocks_this_core = core_n_tiles_this_core / block_w
for (uint32_t row_block_idx = 0; row_block_idx < num_row_blocks_this_core; ++row_block_idx) {
    load_activation_block(row_block_idx);                 // reader
    for (uint32_t n_block_idx = 0; n_block_idx < num_n_blocks_this_core; ++n_block_idx) {
        if (row_block_idx == 0) {
            deliver_weight_block(n_block_idx);            // reader: sender reads + multicasts, others receive
        }
        project_gelu_block(row_block_idx, n_block_idx);   // compute
        store_output_block(row_block_idx, n_block_idx);   // writer
    }
    release_activation_block(row_block_idx);              // compute pops the activation block
}
```

| Block operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------------|-------------|--------------------|-----------------------------------|
| `load_activation_block` | `rows x k_ext_tiles` tiles (rows = block_h, or last_block_h on the tail): 24 x tiles per row from x plus the bias-ones tile at k = 24 | the other `cb_activation_block` slot (previous row-block, being computed) | one `reserve_back` / `push_back` of the nominal `block_h * k_ext_tiles` pages per row-block; one read barrier per row-block (all `rows * 24` x reads plus `rows` local ones-tile copies issued, then one barrier) |
| `deliver_weight_block` | `k_ext_tiles x block_w` tiles: W rows k=0..23 of the chunk's columns plus the bias tile row | earlier W chunks (never recycled) | once per chunk per column, `core_n_blocks` (<= 3) rounds total per kernel: sender issues 72 W + 3 bias DRAM reads with one barrier, zero-fills bias tile rows 1..31, one multicast `send()`; receivers one `receive()` each; every core one `push_back(k_ext_tiles * block_w)` |
| `project_gelu_block` | in0 `rows x k_ext_tiles`, in1 `k_ext_tiles x block_w`, out `rows x block_w` | activation block (retained across the n-block calls), the whole W slice | matmul short-init plus data-format reconfig intended once per kernel per distinct subblock shape (nominal, tail: at most 2); GELU packer init once per kernel at boot; DEST acquire/release per subblock (helper-internal); one cumulative `wait_front` on `cb_weight_resident` per call (free after row-block 0) |
| `store_output_block` | `rows x block_w` tiles in subblock-major order | - | per subblock: one `wait_front(sb_tiles)`, `sb_tiles` page writes, one `noc_async_writes_flushed`, one `pop_front`; one write barrier at kernel end |
| `release_activation_block` | `block_h x k_ext_tiles` pages | - | one `pop_front` of the nominal page count per row-block |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 GELU form (`fast` template flag of `gelu_tanh_tile_pack`) | Accurate `calculate_gelu_tanh` (`ckernel_sfpu_gelu.h:351`, about 38 SFPU ops per 32-lane iteration) costs about 1.2-1.6k cycles per tile against about 910 FPU cycles per tile (25 products at HiFi2). The packer SFPU, not the FPU, may set the floor (about 113 us at 8x512, above the 90.8 us target). | `calculate_gelu_tanh_fast` (`ckernel_sfpu_gelu.h:409`, about 22 ops): the same function by the exact identity `0.5 z (1 + tanh u) = z / (1 + exp(-2u))`, documented within 1 bf16 ULP of the accurate form and free of its negative-tail cancellation. Adopt only if, on the golden model-distributed inputs, PCC and relative RMS vs fp64 are no worse than the accurate path (stock: 0.99996 / 1.21e-2..1.30e-2), with 8x512 device time measured. Otherwise keep accurate. Host constant `GELU_FAST` is the single source (CT arg). |
| L2 `block_w` (W chunk = call width) | 3 trades a short first-chunk W wait against 3x more compute calls and writer flushes than 9 | `block_w` in {1, 3, 9} at 8x512 and 2x512 (W wait share is largest at small M) |
| L3 W sender spread | one fixed sender per column (row 0, a row line, which `noc_placement` favours for DRAM) reads 450 KB serially on its NCRISC. The sibling op measured the single W sender as the small-M binder. | `McastRotatingSenderConfig` within each column (chunk j sent by row `j mod active_rows`, so chunks are read in parallel); and the W path on the writer RISC / NoC1 (sibling: NoC1 W reads slower, measure) |
| L4 x bank stagger | the 11 cores of a grid row read the same x tiles in the same k order at the same time, hitting one L1 bank at a time (`bank_stagger` catalog: 1.32x on BH) | start each core's k walk at `(c * stride) mod 24` and wrap. Activation CB page positions are unchanged, only the issue order. |
| L5 bias as a DEST seed instead of a K tile | the 25th K tile is 1/25 of the FPU work (about 3 us at 8x512). Free if the SFPU binds (L1), not free if L1 is taken. | seed DEST with the row-broadcast bias (`unary_bcast<ROW>` unpack to DEST) before the K loop. Needs a matmul block operation that accumulates onto a pre-loaded DEST, which `matmul_block` does not expose (it acquires DEST internally per subblock, `.inl` subblock loop). |
| L6 overlap: `block_h` | `block_h = 4` (solve at the budget edge) vs 3: a taller block cuts W re-unpack per row but leaves a 1-row tail (13 = 4+4+4+1) | `block_h` in {3, 4} (3 gives 3,3,3,3,1 and frees about 114 KB L1 for `ACT_DEPTH = 3`). Refinement 4: `BLOCK_H_MAX = 3` adopted (shorter x round-0 fill; 8x512 137.2 -> 135.8 us); `ACT_DEPTH = 3` does not fit at block_h 3 with the 24-page output ring. |
| L7 grid synchronization: small M | at tensor_m_tiles = 32, R1 uses 88 cores (8 rows x 11), 36 tiles each, and the W prologue is a larger share | `grid_y`-limited variants (fewer columns with larger `core_n_tiles`) vs R6. Measure 2x512 / 4x256. |

### Catalog evidence

| Knob | Entry | What it set |
|------|-------|-------------|
| W column multicast (Phase 0) | `mcast_topology`, `shared_input_reuse` | broadcast the operand along the axis it does not vary with; 1-D work-split axes need 1-D multicasts |
| sender line placement | `noc_placement` | fixed senders in grid row 0 (a row line) for the DRAM reads |
| block read then one barrier | `double_buffer` | the reader issues a whole row-block / chunk of reads per barrier |
| subblock = 4 tiles (DEST budget with fp32) | `matmul_output_subblock` | shape-agnostic, size-driven; fp32 DEST caps it at 4. Sibling op: 1x4 / 2x2 / 4x1 within 0.5 us at K=25. |
| packer-thread activation | `compute_fusion` | keep the SFPU stage on the pipeline that can overlap the FPU |
| x bank stagger (lamp L4) | `bank_stagger` | measured 1.32x on BH when all cores walk the same banks |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| R1 x -> `cb_activation_block` | bf16 tiles | reader (NCRISC, NoC0), `TensorAccessor` on x, `rows * 24` page reads per row-block, one barrier | page `m * 24 + k`, k = 0..23 into in0 position `(row, k)` of the slot (row-major, row stride `k_ext_tiles`) |
| R2 ones tile -> `cb_activation_block` | bf16 tile: column 0 = 1.0 in all 32 rows, everything else 0 | reader self-aimed local read (`local_copy_helpers_dataflow.hpp:57`, `:94`) from `cb_bias_ones_tile` into position `(row, 24)` | built once at kernel start by the reader. Always full-valid: padded rows are handled at the writer (W3), so no tail variant. |
| R3 W + bias -> `cb_weight_resident` | bf16 tiles | column sender (core in grid row 0 of each active column): `TensorAccessor` reads of W pages `k * 96 + n` (k=0..23) and bias page `n` for the chunk's `block_w` columns into chunk-major `[j][k][w]` at offset `j * k_ext_tiles * block_w`; zero-fill rows 1..31 of each bias tile (row 0 is the data; padding contents are not trusted); `McastArgs::sender(noc).send(src, dst, chunk_bytes)` to the column; receivers `receive(round = j)`; every core `push_back(k_ext_tiles * block_w)` | `ttnn.Mcast` NOC_0, one group per active column (ColumnMajor order, group size `active_rows`), `McastFixedSenderConfig(sender_index=0)`. Each chunk lands in a fresh region, so no consumer-ready handshake is needed after the region is allocated (`handshake` may stay on for the first round; implementer's call). |
| C1 matmul + bias | DEST fp32 | `matmul_block`, one K-block of 25: z = [x \| e0] @ [W ; b] where e0 is the ones column, so `z[r,n] = sum_k x[r,k] W[k,n] + 1 * b[n]` | bias broadcast over rows is realized as a rank-1 K tile. Added in fp32 DEST, no bf16 rounding of z before GELU. |
| C2 GELU | DEST fp32 -> packed bf16 | `Activation` slot of `matmul_block` -> `apply_activation_from_pack` on TRISC2 (`gelu_tanh_tile_pack<DST_ACCUM_MODE, GELU_FAST>`) | per subblock, in place in DEST, then pack to `cb_output_tiles` |
| W1 `cb_output_tiles` -> y | bf16 tiles | writer (BRISC, NoC1), `TensorAccessor` on y, page `m * 96 + n` from subblock-major order | `noc_async_writes_flushed` before each `pop_front`; one final barrier |
| W3 (refinement h_non_aligned) padded-row zero fill | bf16 tile in `cb_output_tiles` | writer, after `wait_front` and before the NoC write, for tiles of a tail row `m`: zero rows `valid_rows(m)..31` (faces 0/1 rows r < 16, faces 2/3 rows r-16) in the L1 page it consumes | matmul output row r depends only on in0 row r, so whatever x's padded rows hold (including NaN/Inf) stays in those rows and is overwritten with +0.0 here. Phase 0 (tile_aligned) has no tail rows. |
| R2-regime x delivery (deferred) | bf16 | one x injector per grid row (fixed sender column 0, RowMajor groups of `active_cols`), same `cb_activation_block` slot address on every row core, `McastArgs` with consumer-ready handshake (slot recycled every 2 row-blocks) | the R1 reader already writes x into a CB slot whose address is uniform across the rectangle, so R2 replaces only the source of R1's per-row-block fill |
| R6 K-split combine (deferred) | fp32 partial tiles | per-column gather of `block_h x block_w` fp32 partials onto the n-owner core (semaphore-gated unicast or the column multicast channel in reverse), then the combining core packs with the GELU activation | `matmul_block` `num_k_blocks` / `Interm` target already express partial production |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | an (activation row-block, n-block) pair; per core a `core_m_tiles x core_n_tiles` output rectangle |
| Grid | `grid = device.compute_with_storage_grid_size()` (p100a: 11 x 10). Rectangle `active_cols x active_rows` from (0,0). n over x (columns), m over y (rows). Grid is a host parameter, never a literal. |
| Per-core work | core (c, r): rows `[r * core_m_tiles, min((r+1) * core_m_tiles, tensor_m_tiles))`, columns `[c * core_n_tiles, min((c+1) * core_n_tiles, 96))` |
| Remainder | last active row: `tensor_m_tiles - (active_rows-1) * core_m_tiles` rows (>= 1); last active column: `96 - (active_cols-1) * core_n_tiles` tiles (a multiple of `block_w`); tail row-block `last_block_h = rows_this_core - (num_row_blocks_this_core-1) * block_h` |
| Tile geometry | per-image `ceil` (see Tensors); `tensor_m_tiles = B * ceil(S/32)` |

Concrete values at Phase 0 (p100a, block_w = 3, block_h = 4 from the budget solve):

| (B, S) | tensor_m_tiles | core_m_tiles x active_rows | core_n_tiles x active_cols | cores | busiest core tiles | row-blocks (busiest) |
|--------|---------------|----------------------------|----------------------------|-------|--------------------|----------------------|
| 8x512, 16x256, 32x128 | 128 | 13 x 10 (last 11) | 9 x 11 (last 6) | 110 | 117 | 4,4,4,1 |
| 8x384 | 96 | 10 x 10 | 9 x 11 | 110 | 90 | 4,4,2 |
| 4x512, 8x256 | 64 | 7 x 10 (last 1) | 9 x 11 | 110 | 63 | 4,3 |
| 2x512, 4x256 | 32 | 4 x 8 | 9 x 11 | 88 | 36 | 4 |
| 1x32 / 1x8 | 1 | 1 x 1 | 9 x 11 | 11 | 9 | 1 (degenerate column multicast) |

Refinement 3 update (built): `block_w = 1`, and `select_blocking` also scores the transposed orientation (logical rows m on grid columns x, logical columns n on grid rows y) and keeps whichever gives fewer output tiles on the busiest core (ties keep m on y). The W channel is a rotating sender inside each logical column (chunk j owned by row `(j + col + 1) mod active_rows`; the `+ 1` keeps it off the row's x round-0 sender), and chunk-0 owners issue their DRAM read at kernel start, before x row-block 0. Concrete values now:

| (B, S) | orientation | core_m_tiles x active_rows | core_n_tiles x active_cols | cores | busiest core tiles | row-blocks (busiest) |
|--------|-------------|----------------------------|----------------------------|-------|--------------------|----------------------|
| 8x512, 16x256, 32x128 | m on y | 13 x 10 (last 11) | 9 x 11 (last 6) | 110 | 117 | 4,4,4,1 |
| 8x384 | m on y (tie) | 10 x 10 | 9 x 11 | 110 | 90 | 4,4,2 |
| 4x512, 8x256 | m on x | 6 x 11 (last 4) | 10 x 10 (last 6) | 110 | 60 | 3,3 |
| 2x512, 4x256 | m on x | 3 x 11 (last 2) | 10 x 10 (last 6) | 110 | 30 | 3 |
| 1x32 / 1x8 | m on y | 1 x 1 | 9 x 11 | 11 | 9 | 1 |

Single regime. Every derived quantity comes from `(grid, tensor_m_tiles, BLOCK_W, K_EXT_TILES, CB_BUDGET_BYTES, ACT_DEPTH, OUT_DEPTH)` in one host function. The tests must still pin the degenerate shapes: `active_rows = 1` (1x32), `tensor_m_tiles < grid_y` (2x37, 16x8) and a tail row-block of 1 (8x512).

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_activation_block` | 0 | bf16 tile (2048 B) | `ACT_DEPTH * block_h * k_ext_tiles` | spans m (block_h) and k (whole k_ext); streams over n (retained across the n-block calls); depth 2 for prefetch | Float16_b | reader | compute | whole kernel, recycled per row-block |
| `cb_weight_resident` | 1 | bf16 tile | `k_ext_tiles * core_n_tiles_max` | spans k and the core's whole n slice; streams over m (reused by every row-block); resident | Float16_b | reader | compute | whole kernel, never recycled |
| `cb_output_tiles` | 2 | bf16 tile | `OUT_DEPTH * block_h * block_w` | spans one call's m x n block, double-buffered; k does not appear (accumulated in DEST) | Float16_b (output dtype) | compute | writer | whole kernel, recycled per subblock |
| `cb_bias_ones_tile` | 3 | bf16 tile | 1 | constant | Float16_b | reader (fills once at boot) | reader (local-read source; no CB credit moves) | whole kernel |

`core_n_tiles_max` is the per-core n extent of a full column (9 on p100a). The last column uses 6 of those pages. The CB is uniform across cores because the multicast needs the same address everywhere.

`cb_bias_ones_tile` is a reader-private constant: the reader writes it once and then reads it as a raw L1 source address. No `push_back` / `wait_front` is ever issued on it, so there is no cross-kernel handshake and no second kernel touches it.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_activation_block` | `rows x k_ext_tiles` | dataflow: `TensorAccessor` + local read | x tensor; `cb_bias_ones_tile` (1, constant) | `cb_activation_block` (+`block_h * k_ext_tiles`, nominal even on the tail) | one slot fronted |
| 2 | `deliver_weight_block` | `k_ext_tiles x block_w` | `ttnn.Mcast` / `McastArgs` | W, bias tensors (sender); multicast landing (receivers) | `cb_weight_resident` (+`k_ext_tiles * block_w`) | cumulative; never popped |
| 3 | `project_gelu_block` | in0 `rows x 25`, in1 `25 x block_w`, out `rows x block_w` | helper `matmul_block` with `Activation = GeluTanhActivation<GELU_FAST>` | `cb_activation_block` (`rows * 25`, fronted, retained); `cb_weight_resident` (cumulative `wait_front((n_block_idx+1) * 25 * block_w)`, caller-managed, NoWaitNoPop) | `cb_output_tiles` (`rows * block_w`, pushed per subblock) | activation still fronted |
| 4 | `store_output_block` | `rows x block_w` | dataflow `TensorAccessor` | `cb_output_tiles` (per subblock) | y tensor | drained |
| 5 | `release_activation_block` | `block_h x 25` | `cb_pop_front` | `cb_activation_block` (nominal) | - | slot freed |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| compute boot | raw_api | `compute_kernel_hw_startup<SrcOrder::Reverse>(cb_activation_block, cb_weight_resident, cb_output_tiles)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:60`; Reverse requirement `matmul_block_helpers.hpp:100-106` | once at the top of `kernel_main` | - | - | - |
| GELU packer init (boot) | helper | `ActivationInitHelper<KernelActivation::GELU_TANH, GELU_FAST>::init()` -> `gelu_tanh_tile_init_pack<GELU_FAST>()` | `sfpu_activation_helpers.hpp:75-108`; `gelu.h:67-75` | once at boot (caller's job, `matmul_block_helpers.hpp:299-303`) | - | - | `GELU_FAST` (lamp L1) |
| `project_gelu_block` | helper | `matmul_block<false, false, LastBlockTarget::Out, OutputCBLayout::SubblockMajor, InitMode::Short (first call per subblock shape) / None (rest), InputPolicy::WaitAndRetainOnLastBlock, InputPolicy::NoWaitNoPop, NoPostCompute, NoPreKBlock, NoPostKBlock, 0, NoKBlockInnerDimFn, NoIn0Source, WeightChunkOffset, false, false, GeluTanhActivation<GELU_FAST>, INPUT_AND_OUTPUT (first) / NONE (rest)>` | `matmul_block_helpers.hpp:342-375`; `.inl:266` (in1 offset), `.inl:274` (NoWaitNoPop skips the in1 wait), `.inl:402-407` (packer activation at the last-block pack) | `MatmulBlockShape::of(rows / sb_h, block_w / sb_w, sb_h, sb_w, 25, 1)`; `in1_per_core_w = block_w`; `interm_buf = cb_output_tiles` (no spill at `num_k_blocks = 1`, `matmul_block_helpers.hpp:310-317`); `WeightChunkOffset{n_block_idx * 25 * block_w}` returns its member for block 0 | `cb_activation_block`, `cb_weight_resident` | `cb_output_tiles` | `rows` (block_h / last_block_h), `block_w`, `in0_block_k = k_ext_tiles`; `sb_h`, `sb_w` derived on host (not design knobs) |
| packer activation dispatch (extension) | helper (additive extension) | add `KernelActivation::GELU_TANH` to `ActivationInitHelper` / `ActivationApplyHelper` (static_asserts and branches): `gelu_tanh_tile_init_pack<P0 != 0>()` / `gelu_tanh_tile_pack<DST_ACCUM_MODE, P0 != 0>(i)`, plus `template <uint32_t Fast = 0> using GeluTanhActivation = ActivationOp<KernelActivation::GELU_TANH, Fast>;` | `sfpu_activation_helpers.hpp:46-50, 77-82, 113-117`; enum `ttnn/cpp/ttnn/operations/matmul/shared_with_host/activation_type.hpp:14`; device fns `gelu.h:67-75, 107-116` | - | - | - | `Fast` |
| `load_activation_block` | raw_api (dataflow) | `TensorAccessor` page reads + `noc_async_read_barrier`; self-aimed local read of the ones tile | `tech_reports/tensor_accessor/tensor_accessor.md`; `local_copy_helpers_dataflow.hpp:57, 94-95` | - | `cb_bias_ones_tile` | `cb_activation_block` | `block_h` |
| `deliver_weight_block` | helper (dataflow + host) | host `ttnn.Mcast(device, McastConfig(noc=NOC_0), rectangle, active_rows, McastFixedSenderConfig(sender_index=0), McastCoreOrder.ColumnMajor)` + `attach`; kernel `McastArgs<...>().sender(noc).send(src, dst, bytes)` / `.receiver(noc).receive(round)` | `mcast/host/mcast.hpp:83-137`; `mcast/kernel/mcast_args.hpp:18-56` | - | - | `cb_weight_resident` | `block_w` (chunk bytes) |
| `store_output_block` | raw_api (dataflow) | `TensorAccessor` page writes, `noc_async_writes_flushed`, final `noc_async_write_barrier` | `tech_reports/tensor_accessor/tensor_accessor.md` | - | `cb_output_tiles` | - | - |

Helpers considered and rejected:

| Candidate | Phase | File:Line | Why it cannot express the planned shape |
|-----------|-------|-----------|-----------------------------------------|
| `add_bias_bcast_rows` (bias + packer activation epilogue) | bias | `bias_add_helpers.hpp:330-342` | It consumes matmul partials from an L1 `partials_buf` (`:268-271`), so the matmul would pack z to an fp32 interm CB and the epilogue would unpack it again: an extra pack and unpack of every output tile plus an fp32 CB. Folding the bias as the 25th K tile keeps z in DEST and the bias add in fp32 inside the same `matmul_block` call. Not a raw-API fallback: the bias is computed by `matmul_block` itself. |
| `eltwise_chain` / `Gelu<>` | activation | `eltwise_chain.hpp` (SFPU unary structs) | A chain runs on the math thread with its own DEST window (CB to CB), so z would have to be packed out and re-read. The packer-thread `Activation` slot of `matmul_block` applies GELU in place on the DEST subblock while math continues. `KernelActivation::GELU` in the existing dispatch is `gelu_tile_pack` (`sfpu_activation_helpers.hpp:129-131`), the erf/CDF GELU, a different function, hence the GELU_TANH extension. |
| `matmul_block` `packer_l1_acc` / `seeded_pack_target` bias seed | bias | `matmul_block_helpers.hpp:296-298` | Requires K-blocking into an L1 accumulator, and the packer activation would then see only the last K-block's contribution, not z. Incompatible with the full-K-in-DEST scheme. |
| `matmul_block` `last_in1_subblock_w_valid` | ragged n | `matmul_block_helpers.hpp:158` | Not needed: `block_w` divides every core's n extent. |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Erf GELU instead of tanh GELU | `GeluActivation` / `KernelActivation::GELU` routes to `gelu_tile_pack` (erf form), and the name looks right | use only `KernelActivation::GELU_TANH` -> `gelu_tanh_tile_pack` via the extension; acceptance test compares to `gelu(approximate="tanh")` on model-distributed inputs |
| `GELU_FAST = true` silently adopted | it is faster and reads as an approximation; the contract forbids lowering precision | Phase 0 was `GELU_FAST = false`. Refinement 2 adopted `GELU_FAST = true` after the lamp L1 gate held: same function by exact identity, PCC / rel RMS vs fp64 identical to 4 significant digits on golden inputs. |
| Bias tile padding rows are not trusted | the ones tile's zeros multiply bias tile rows 1..31; `0 * NaN = NaN` would poison every row | the W sender zero-fills bias tile rows 1..31 before the multicast |
| Padded output rows (h_non_aligned) | the contract says exactly 0 whatever x's padding holds; GELU(bias) != 0 and garbage x rows give garbage z | the writer overwrites rows `valid_rows..31` of tail-row tiles with +0.0 (W3). Matmul rows are independent, so no NaN leaks into valid rows. Do not "fix" it by masking x alone: z = b in a padded row. |
| Matmul source order | matmul maps in0 -> SrcB, in1 -> SrcA | boot with `compute_kernel_hw_startup<SrcOrder::Reverse>` exactly once |
| Weight CB wait semantics | `NoWaitNoPop` means the helper never waits on `cb_weight_resident` | compute issues `wait_front((n_block_idx + 1) * k_ext_tiles * block_w)` before every call (cumulative from the never-moved read pointer) and never pops it |
| Activation CB pop | `WaitAndRetainOnLastBlock` with `num_k_blocks = 1` never pops | compute pops the nominal `block_h * k_ext_tiles` once per row-block after the last n-block call. The reader pushes the same nominal count on the tail row-block, so the ring closes every `ACT_DEPTH` row-blocks. |
| Tail row-block subblock | `last_block_h` (1 at 8x512) may not admit the nominal `sb_h` | host passes `(sb_h, sb_w)` for nominal and tail separately; the tail call re-inits (`InitMode::Short`) with its own subblock dims |
| Column multicast grouping | default `Mcast` grouping is row-major | `McastCoreOrder.ColumnMajor` with group size `active_rows`; regime-pinned shapes 1x32 (1-core groups) and 2x512 (8-core groups) |
| HiFi4 + fp32 DEST with bf16 | `matmul_block_helpers.hpp:263-264`: K-accumulator corruption on Wormhole B0 | the caller config is honored as given (contract). Blackhole is the target board. The acceptance HiFi4 test runs on the target. |
| L1 budget with x and y resident in L1 | at 8x512 x (57 KB/bank) and y (224 KB/bank) share L1 with the CBs | `CB_BUDGET_BYTES = 900 KiB` (the sibling op's production-validated value). `block_h` is solved from the ledger formula, never hand-set. |
| Single-dispatch rule | composing `ttnn.linear` / `ttnn.gelu`, or a host-side reshape of bias | one `ttnn.generic_op`. Bias pages are read as stored for both `(3072,)` and `(1, 3072)` (same 96-page tile row). |
