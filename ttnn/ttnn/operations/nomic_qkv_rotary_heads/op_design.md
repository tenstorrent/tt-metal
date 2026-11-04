# Operation Design: nomic_qkv_rotary_heads

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (2D-multicast matmul + bias + q scale + rotary + head scatter) |
| Goal | The front of every nomic-embed-text-v2-moe attention layer in one `ttnn.generic_op` dispatch on a Blackhole p300c (11 x 10 grid). x goes in where the norm leaves it. q, k and v come out as three interleaved (B, 12, S, 64) tensors for stock SDPA. The (B, 1, S, 2304) QKV product never exists. |
| Math | `qkv = x @ W^T + b` (fp32). Split into q, k, v (768 columns each); heads of 64; `q *= 1/8`. NeoX rotate-half rotary (base 10000) on q and k: `out0 = t0*C - t1*S`, `out1 = t1*C + t0*S` per head tile pair. v is not rotated. Padding rows (s >= S) of q, k and v are exactly 0. |
| Mode | Hybrid. The dataflow scheme is derived from pirotski_matmul (arrangement A, resident weights, norm-shard input reads). Compute uses the kernel_lib `matmul_block` helper with two hooks. |
| References | `ttnn/cpp/ttnn/operations/experimental/nomic_embed_text_v2_moe/pirotski_matmul/` (`in0_stream.cpp`, `in1_stream.cpp`, `stream_protocol.hpp`); `models/experimental/nomic_embed_text_v2_moe/docs/PIROTSKI_MATMUL.md` §2–4, 7.1, 7.7, 8; `docs/SMALL_M_MATMUL.md`; `docs/archive/ROTARY_EMBEDDING_HF.md` §3, 5–8; `scratch/head_copy/probe_fused_rotary.py`; `nlp_create_qkv_heads` (page math); `ttnn/ttnn/operations/nomic_residual_layer_norm/` (x's layout: `_plan`, `input_memory_config`, `output_memory_config`); `ttnn/cpp/ttnn/kernel_lib/` (`matmul_block_helpers.hpp/.inl`, `sfpu_activation_helpers.hpp`, `mcast/`, `l1_helpers.hpp`, `dest_helpers.hpp`); perf catalog `ttnn/ttnn/operations/examples/master.md` (entries listed under Perf lamps) |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | (B, 1, S, 768) bf16 TILE L1, in the norm's layout (`ln_layout`); 32 <= M = B*ceil(S/32) <= 128 | — | x's address and shard geometry are RT (via the x source table) |
| `weights` | `QkvWeights` (frozen dataclass returned by `prepare_qkv_weights`) | yes | exactly what `prepare_qkv_weights` returns | — | address RT |
| `tables` | `RotaryTables` (frozen dataclass returned by `prepare_rotary_tables`) | yes | `tables.seqlen >= S` | — | address RT |
| `memory_config` | `ttnn.MemoryConfig \| None` | no | `ttnn.L1_MEMORY_CONFIG` or `ttnn.DRAM_MEMORY_CONFIG` (interleaved) | `ttnn.L1_MEMORY_CONFIG` | q/k/v accessor args CT, addresses RT |
| `output_tensors` | `list[ttnn.Tensor] \| None` | no | `[q, k, v]`, each (B, 12, S, 64) bf16 TILE with exactly `memory_config` | `None` (the op allocates v, k, q, in that order) | addresses RT |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor \| None` | no | `fp32_dest_acc_en=True` (otherwise refused); the fidelity is taken as given and not gated | `default_compute_kernel_config()` = HiFi2, fp32_dest_acc_en=True, math_approx_mode=False | CT (compute config). The op forces `dst_full_sync_en=False`, because half-sync DEST is a mechanism cap |

The module also exports `prepare_qkv_weights`, `prepare_rotary_tables`, `head_dim_order`, `input_memory_config`, `default_compute_kernel_config`, `circular_buffer_bytes`, `INPUT_TAGGERS`, `SUPPORTED` and `EXCLUSIONS`. `validate()` is the first line of the entry point.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | (B, 1, S, 768); M = B * ceil(S/32) in [32, 128] |
| Dtype | bfloat16 |
| Layout | TILE |
| Memory | L1, ND-sharded as `nomic_residual_layer_norm.output_memory_config(shape, device, nomic_residual_layer_norm.input_memory_config(shape, device))` (`ln_layout`). The read plan is derived at runtime from x's actual `NdShardSpec`; see Dataflow Strategy, "x source table" |

### Output

| Property | Value |
|----------|-------|
| Shape | q, k, v: (B, 12, S, 64) each |
| Dtype | bfloat16 |
| Layout | TILE. Page `((b*12 + h)*S_t + s_t)*2 + d` holds tile row s_t, half d of head h; S_t = ceil(S/32) |
| Memory | INTERLEAVED, in `memory_config`'s buffer type (L1 or DRAM). With `output_tensors`, those exact buffers are written and returned |

### Device-resident inputs this op defines (host preparation, outside the dispatch)

| Object | Built by | Device tensor | Content |
|--------|----------|---------------|---------|
| `QkvWeights` | `prepare_qkv_weights(weight, bias, device)`, once per layer at model load | ROW_MAJOR bfloat16, shape (1, 1, 225, 8192), DRAM interleaved. Page = one row = 16 KB (layout tag `n_pages_8_v1`; `head_pages_v1` before Refinement 3) | **Lane pages** (`WEIGHT_PAGE_TILES = LANE_N_TILES`; catalog `read_coalescing`). Wt = Wᵀ (768 x 2304), with q's 768 columns and b_q multiplied by 1/8 (exact in bf16). Append the bias as K row 768 and zero rows 769–799, which gives K_TILES = 25 tile rows. Page `k*9 + g` (k in 0..24, lane g in 0..8) holds tiles (k, 8g) .. (k, 8g+7) in N order, each in standard 32x32 face order. `WEIGHT_PAGE_TILES = HEAD_TILES` restores the old 4 KB head pages. Tile (24, n) holds b[32n .. 32n+31] in row 0 and zeros in rows 1–31 (the bias K tile). |
| `RotaryTables` | `prepare_rotary_tables(seqlen, device)`, once per sequence length | TILE float32, shape (1, 1, 32*ceil(seqlen/32), 64), L1 interleaved. Page = one fp32 tile = 4096 B | Computed on the host in float64, then cast to fp32. Column i < 32: cos(s*10000^(-2i/64)). Column 32+i: sin(...) of the same angle. Rows s >= seqlen are 0. Page `2*s_t` = C tile, page `2*s_t + 1` = S tile of position tile s_t. The same C and S serve both halves, all 12 heads, q and k. |
| `head_dim_order()` | — | — | `tuple(range(64))`, the reference order. In it, rotate-half is a tile-pair swap at equal lanes (no lane shuffles). |

Index-axis note: the op has no `dim` argument, so no canonicalization is needed.

## Blocking Model

Semantics: `.claude/references/blocking-model.md`. Every block extent below is a host constant or a value derived from one, in `nomic_qkv_rotary_heads.py` (the "knob source"), and is threaded to the kernels as CT/RT args. No kernel restates a literal.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| M — tile rows `m = b*S_t + s_t` (b and s_t fold: they are contiguous and treated identically, except for the rotary position `s_t = m mod S_t` and the padding flag) | **independent**: output row m reads only x row m (matmul, bias, rotary and pack are all row-local) | `core_m_tiles` (share per grid column), `block_m_tiles` (rows per M block) | `core_m_tiles = ceil(M / grid_x)` (3 to 12). Then `num_m_blocks = ceil(core_m_tiles / BLOCK_M_TILES_MAX)` and `block_m_tiles = ceil(core_m_tiles / num_m_blocks)`, with `BLOCK_M_TILES_MAX = 3` | `BLOCK_M_TILES_MAX` host constant (L1 cap; see Mechanism caps); `grid_x` from the device | grid **columns**: column c owns rows `[c*core_m_tiles, min(M, (c+1)*core_m_tiles))`; `num_columns = ceil(M / core_m_tiles)` (9 to 11) | knob-turn |
| N — output tile columns, 72 = 3 tensors x 12 heads x 2 halves | **independent at head granularity**: output column n reads only W column n, but the rotary couples the two tiles of a head (t0, t1), so a head pair is indivisible | `lane_n_tiles` (N tiles per grid row = one block's N extent: the whole lane) | 8 (4 heads). Lane g = N tiles `[8g, 8g+8)`: lanes 0–2 are q, 3–5 k, 6–8 v | `LANE_N_TILES` host constant; `NUM_LANES = 72 / LANE_N_TILES = 9` derived | grid **rows**: row y = lane y | knob-turn (a different lane table; the head-pair unit is kept) |
| K — 24 hidden tiles + 1 bias tile = `K_TILES = 25` | **dependent**: every output tile is a sum over K | `k_block_tiles` | 5 → `num_k_blocks = 5` | `K_BLOCK_TILES` host constant; `num_k_blocks = K_TILES / K_BLOCK_TILES` derived | **not split across cores**: the whole K on every core. Accumulated in fp32 by packer L1 accumulation across K blocks; the last K block is accumulated in DEST | scheme-change (split-K + cross-core fp32 combine; `rejected` below) |
| x operand along N | **reuse-shared by construction of the N split**: x does not vary along N, so the 9 rows of a column need identical bytes | — (rides the M/K blocks) | multicast once down each column | — | column's top core (y = 0) gathers and multicasts | built |
| W operand along M | **reuse-shared by construction of the M split** (and across a core's own M blocks): W does not vary along M | — (rides the K blocks) | multicast once along each row; resident on every core after M block 0 | — | row's rightmost core reads DRAM and multicasts | built |
| C/S tables along N | **reuse-shared**: a position's C and S serve all 12 heads, q and k | — (rides M) | each q/k core reads its block's 2·block_m_tiles pages once per M block; copied into DEST per head | — | each q/k core for itself | broadcast `deferred` (regime table) |

No axis is left at 1 without a reason. `block_m_tiles` stops at 3 because of the L1 budget (Mechanism caps; `l1_ledger.md`), not by default. The N block is the whole lane. K is the whole K, cut into the largest block that 25 admits below 25 itself (5).

**Stage data, re-run through the table**

| Stage | Its data's axes | Who does the work | Assignment decision |
|-------|-----------------|-------------------|---------------------|
| x gather + multicast (in0 stream) | M rows of the column x K block | **Communication role** on RISCV_0 of the column's top core: reads from shard holders, then one multicast. No compute. | Already spread: 11 independent column senders. The 8 other RISCV_0s of a column wait on flags. Per K block the sender's reads are a fraction of the block's compute (perf lamp L5: rotating sender). |
| W stream (in1 stream, M block 0 only) | K block x lane | **Communication role** on RISCV_1 of the row's rightmost core: DRAM reads, then one multicast | Spread over 9 row senders. Later M blocks re-push resident pages with no reads. |
| matmul + bias + rotary + pack | M block x lane | Every core, on its own block | Spread over all 81–99 cores. Nothing runs on a designated core. |
| rotary table load | M block rows x {C, S} | Each q/k core for itself (RISCV_0) | Local; no designated core |
| head scatter (q/k/v page writes) | M block x lane tiles | Each core's RISCV_1, for its own tiles | Spread; every tile is written by the core that finished it |

**Stall-shadow check**

| Wait | What could run in its shadow | Decision |
|------|------------------------------|----------|
| Every core waits for weight K block 0 from DRAM (first M block) | Compute: nothing (every MAC needs W). Dataflow: the x gather of K block 0 and the rotary-table load. | These are scheduled into the shadow; they run on the other RISC or before the first `wait`. Nothing else is independent of W. |
| Reader blocked reserving `cb_rotary_table` for block m (single slot) | The x stream of block m | The table load for block m is issued **after** in0 K block 0 of block m is pushed. Its slot frees when compute finishes block m−1, so the x stream never stalls behind it. |
| Writer waiting on `cb_heads_out` rows of block m−1 | The resident in1 re-pushes of block m | Order on RISCV_1 per M block m >= 1: push resident K blocks 0..3 → drain block m−1's rows → push K block 4. Compute never waits for weights. |
| Compute's last K block per subblock: the SFPU rotary of head j | The FPU matmul of head j+1 | **The core reorder of this design.** The rotary runs as packer-thread SFPU on DEST half A while the math thread runs FPU on half B (`sfpu_activation_helpers.hpp:71-73`: "SFPU activation is fused into the pack stage so it overlaps with the math thread starting the next K-block"). It is legal because the rotary is a position-wise linear map of the finished fp32 tile and needs nothing but that tile and C/S. |
| End of op: the last M block's drain | Nothing (no work left) | Tail kept small: the last block drains row by row as subblocks are pushed during its last K block. Two-RISC drain is perf lamp L7. |

No reorder changes the floating-point order of the K sum. The bias tile sits in the last K block, added in fp32 like any other K tile. The acceptance test is the precision baseline.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_x_block` | `X_BLOCK_DEPTH` (slots of block_m_tiles x k_block_tiles) | 2 | The gather + multicast of K block b+1 overlaps compute on block b (pirotski §3.2: a third slot bought nothing) |
| `cb_weights` | none: capacity is pinned to `K_TILES * LANE_N_TILES` (the resident ring; Mechanism caps) | all 5 K blocks | The sender streams all K blocks of M block 0 without waiting on compute. Later M blocks re-push the same slots with no reads. |
| `cb_rotary_table` | `ROTARY_TABLE_DEPTH` (M blocks) | 1 | The table is needed only at a block's last K block, a whole K loop after it is requested |
| `cb_partials` | none: one M block (it is an accumulator, not a stream) | 1 block | — |
| `cb_heads_out` | `OUT_DEPTH_BLOCKS` (M blocks) | 1 (a whole M block) | Compute pushes block m's tiles only during its last K block and never waits for the writer; the writer drains them during block m+1's K loop |
| Weight sender read-ahead | `WEIGHT_READ_AHEAD_CHUNKS` (in multicast chunks of `W_MCAST_CHUNK_K_ROWS` K rows; one transaction id each) | one K block | Block b+1's DRAM reads are issued before block b's multicast (pirotski §3.3, §4.3; catalog `trid_double_issue`). Refinement 3 measured deeper read-ahead as slower: the weight stream is DRAM-bound grid-wide, so more blocks in flight delay block 0. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Half-sync fp32 DEST holds 4 tiles per half (`dest_helpers.hpp:89-103`; asserted at `matmul_block_helpers.inl:158`) and the rotary needs t0, t1, C, S live together | **q/k subblock** = exactly 1 x 2 (one head, one tile row): slots 0–1 for the head and slots 2–3 for C and S. The block is not affected. | q/k cores: `out_subblock_h = 1`, `out_subblock_w = 2`, pinned. v cores: the helper's tuner picks the subblock (<= 4 tiles). `dst_full_sync_en` forced False. | A wider q/k subblock leaves no slots for C/S. The hook would overwrite accumulator tiles and produce wrong q/k. |
| Head-pair co-location (the rotary pairs N tiles 2h and 2h+1) | `lane_n_tiles` even, and lane start even | `LANE_N_TILES % 2 == 0`, asserted on the host | A head split across two rows of the grid cannot be rotated locally: wrong q/k |
| Bias as K tile 24 makes K_TILES = 25 | `k_block_tiles` must divide 25 | `K_BLOCK_TILES ∈ {5}` (1 is unit-at-a-time; 25 overflows `cb_x_block`); asserted | The last K block would mix bias and x tiles at the wrong K index: wrong results |
| Resident in1 ring | `cb_weights` capacity must equal `K_TILES * LANE_N_TILES` exactly | Derived from the two knobs. Never a free knob. | Re-pushes of M block m >= 1 land on slots holding other K blocks: wrong results |
| Multicast writes into the same L1 address on every receiver | `cb_x_block` and `cb_weights` must sit at identical addresses on every core | Every CB is declared on the whole grid, in the same order, with the same sizes. That includes `cb_rotary_table` on v cores, where it is unused. | The multicast lands on another CB: silent corruption |
| Per-core CB L1 budget 640 KiB (the task's budget at M = 128) | `block_m_tiles` | `BLOCK_M_TILES_MAX = 3`, so the total is 628 KiB (`l1_ledger.md`). `validate()` refuses a plan whose `circular_buffer_bytes > 640 KiB`. | Allocation failure, or clobbering of the model's L1 tensors |
| x shard geometry (K part width `kp` = 8 or 12 at P = 3 or 2) | in0 read runs per row per K block | A K block's row tiles are read as runs split at `kp` boundaries (at most 2 runs per row at kb = 5) | A run read past a shard's K part reads the neighbouring row's tiles: wrong results |
| Runtime-arg budget (341 words per kernel) | x source table entries per column sender | One entry per (holder, contiguous row run): at most `(core_m_tiles + 1) * P` entries of 5 words (at most 195 words) | Truncated arg table: reads from the wrong holder |
| Packer L1 accumulation needs the partials format to equal DEST's | `cb_partials` page format | Float32 (fp32_dest_acc_en is required) | Accumulation in a 16-bit format: precision loss beyond tolerance |
| Ragged last M block (core_m_tiles not a multiple of block_m_tiles, or the last column short) | rows actually computed | Nominal push/wait counts everywhere. Garbage rows are computed on stale in0 and never written. Their table pages are clamped to page 0. | Narrowing counts in one kernel only: CB hang |

### Regimes

The named memory boundary is DRAM plus cross-core NoC. The **minimum** is: every W byte crosses DRAM once; every x byte leaves its shard holder once; every q/k/v tile is written once; nothing else crosses.

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| `mcast2d_fused_rotary`: 2D split (M over grid columns, N lanes over 9 grid rows). x gathered from the norm's shards by each column's top core and multicast down the column. W (bias as K tile 24) read from DRAM once per row by the row's right core, multicast west, resident after M block 0. K in 5 blocks with packer L1 accumulation. The last K block reloads the partials and adds the bias tile in fp32 DEST. On q/k cores, C and S go into DEST slots 2–3 and the packer thread rotates the head (SFPU) while the math thread runs the next head. Pack once to bf16; the writer scatters each tile to its q/k/v page. | **built** | `32 <= M <= 128` (the entire supported domain; 9 lanes x 8 tiles is the only head-aligned 9-row split, and no other built regime competes there) | `block_m_tiles x LANE_N_TILES x K_BLOCK_TILES` (in0 3x5, in1 5x8, out 3x8 at M >= 48; 2x5 / 3x5 at M <= 39) | DRAM: W once (minimum), x zero (L1), outputs once. NoC: x crosses twice (gather to the column's top core, then one multicast), which is one crossing above the minimum (see `holder_direct_x_mcast`). Tables: each q/k core reads its own rows (see `table_broadcast`). | Per M block: one `matmul_block` init + data-format reconfig, one table handshake, the in0 multicast handshake per K block (amortized over block_m_tiles x 5 tiles), the per-M-block in1 re-push handshakes (5), and the pipeline fill of the first K block. Intended frequency: once per M block (4 at M = 128). The cap is L1, not cost. |
| `holder_direct_x_mcast`: shard holders multicast their own row-runs straight into the column's `cb_x_block` slots (no gather hop) | **deferred**: the gather costs one extra L1→L1 crossing of x (6.3 MB at M = 128, 576 KB per column sender over the op). It is off the critical path: per K block the sender reads `block_m_tiles x 5` tiles (30 KB) while compute spends `block_m_tiles x 8 x 5` tile products on the block. The direct form needs a many-to-many handshake: each slot is filled by up to `(block_m_tiles/r + 1) * P` holders, and each holder must hear "slot free" from 9 receivers. **Reachable:** receivers and slot layout are unchanged; only the sender side of the in0 channel changes. | — | same | −1 crossing of x (−6.3 MB L1→L1 at M = 128) | same |
| `table_broadcast`: append the block's C/S pages to the in0 multicast of the last K block (operand-reuse check: tables do not vary along N) | **deferred**: the 6 q/k cores of a column each read the same 2·block_m_tiles fp32 pages, 24 KB per core per M block (6.3 MB of L1→L1 reads chip-wide at M = 128). Each core's read is on its idle RISCV_0 between in0 blocks, off the critical path. Folding the tables into in0 would make one K block's slot larger than the others, which breaks `cb_x_block`'s uniform ring. **Reachable:** add a separate channel down the column. | — | same | −5.2 MB L1→L1 reads at M = 128 | same |
| `finalize_pass`: K loop with 1x4 subblocks into fp32 partials, then a separate pass (bias add from L1 partials + math-thread rotary + pack) | **rejected**: superseded by `mcast2d_fused_rotary`. It re-reads every q/k tile in fp32 from L1 (2 x 4 KB unpacks per head) and runs the SFPU rotary on the math thread with no FPU work to overlap. Pirotski's bias-only output pass alone measured 5.2 us of its 71.3 us at 8x512 (§7.1). | — | — | +1 fp32 L1 read of every output tile | — |
| `strip_row_10`: the 10th grid row computes a strip of M rows over all 72 N tiles | **rejected**: measured on QKV in PIROTSKI_MATMUL.md §8.1. A strip row needs the whole 3.5 MB weight through one reader (about 115 us against the 71 us call), and several readers into one row take turns. | — | — | +3.5 MB DRAM per strip row | — |
| `nmajor_transposed`: M over 10 rows, N over 11 columns (shares of 7) | **rejected**: measured, PIROTSKI_MATMUL.md §6.4 (QKV at 64 tile rows: 87.7 us streamed / 64.6 us resident, against 56.7 us for minimal_matmul; weight-read bound). §8.1 bounds its gain at 128 tile rows at <= 5% (91 against 96 tiles a core). Heads do not align with shares of 7. | — | — | weight column senders read 28-tile blocks | — |
| `split_k_combine`: split K across cores, combine fp32 partials before bias and rotary | **rejected**: the independent axes already occupy 81–99 cores at every M. The critical path is `ceil(M/11) x 8` tiles, which no column count changes. A split adds a cross-core exchange of the whole fp32 output block (`block_m_tiles x 8 x 4 KB` per extra split per M block) and serializes the rotary behind the combine. Residency needs no split: the weights are already resident whole. | — | — | +fp32 partial exchange of every output tile per extra split | — |
| `weights_streamed_per_m_block` | **rejected**: superseded by the resident ring. Re-streaming costs about 2.2 us per extra M block (PIROTSKI_MATMUL.md §3.3), and the ring fits the budget. | — | — | +(num_m_blocks − 1) x 3.7 MB DRAM | — |
| `single_m_block` (`block_m_tiles = core_m_tiles`) | **rejected**: at core_m_tiles = 12, partials take 12x8x4 KB = 384 KB beside 400 KB of weights (over budget). The whole output then drains at the end: pirotski's M block 12 took 93 us against 79 us for blocks of 5 (§3.3). Smaller blocks are its successor. | — | — | same | — |

Phase 0 has one built regime, so there is no selection function. The regime-pinned tests are the per-M-class acceptance shapes, which pin 1, 2, 3 and 4 M blocks per core and 9, 10 and 11 columns.

### Traffic ranking

Counted per tensor and tier at M = 128 (x 6.3 MB, W+bias 3.7 MB, q/k/v 18.9 MB). No time estimates.

| Rank | Candidate split | DRAM crossings | Cross-core | Verdict |
|------|-----------------|----------------|------------|---------|
| 1 | **2D M x N, x multicast down columns, W multicast along rows, W resident** (chosen) | W x1 (minimum) | x: gather x1 + multicast fan-out 8. W: multicast fan-out 10 (cols − 1). Tables: unicast reads. | Fewest DRAM bytes and the only split that keeps W resident (400 KB per core) |
| 2 | 2D, holder-direct x multicast | W x1 | x: multicast only | −6.3 MB L1→L1. Deferred (regime table): off the critical path, many-to-many handshake |
| 3 | 2D, W streamed per M block | W x num_m_blocks | same | Rejected: up to 4x the DRAM bytes |
| 4 | N-only split (every core takes all M) | W x1 | x: all 6.3 MB to every core, which is impossible in L1. 36 heads leave at most 36 cores busy. | Under-fills the grid and breaks residency |
| 5 | M-only split (every core takes all N) | W x11 or more (3.7 MB does not fit L1, so every column re-reads) | none for W | Dead end: about 40 MB of DRAM |
| 6 | Split K + combine | W x1 | +fp32 partial exchange per extra split | Rejected (regime table) |

"Prefer an independent axis" agrees with rank 1. Occupancy was not used to rank.

### Block schedule

Logical schedule. Reader (RISCV_0), compute and writer (RISCV_1) realize their parts asynchronously.

```cpp
for (uint32_t m_block_idx = 0; m_block_idx < num_m_blocks_this_core; ++m_block_idx) {
    for (uint32_t k_block_idx = 0; k_block_idx < num_k_blocks; ++k_block_idx) {
        load_x_block(m_block_idx, k_block_idx);        // reader: gather (top core) + multicast down the column
        load_weight_block(m_block_idx, k_block_idx);   // writer: DRAM + multicast (m_block 0) / resident re-push
        if (k_block_idx == 0) load_rotary_table_block(m_block_idx);  // reader, q/k cores
    }
    matmul_accumulate_block(m_block_idx);   // compute: K blocks 0..3 → cb_partials (packer L1 acc, fp32)
    matmul_finish_rotate_block(m_block_idx);// compute: last K block → reload + bias tile + rotary → bf16 cb_heads_out
    scatter_heads_block(m_block_idx);       // writer: q/k/v pages, zero padding rows; overlaps block m+1's K loop
}
```

| Operation | Block shape | Resident across it | Intended frequency of fixed costs |
|-----------|-------------|--------------------|-----------------------------------|
| `load_x_block` | block_m_tiles rows x K_BLOCK_TILES tiles (K block 4: 4 x tiles + the ones tile per row) | `cb_x_block` 2 slots | One multicast + one handshake per (M block, K block). On the top core: one read run per (row, holder part) and **one read barrier per K block**. The ones tiles are written by the top core into K block 4's slot once per M block (`fill_l1_range`). |
| `load_weight_block` | K_BLOCK_TILES x LANE_N_TILES (5 lane pages of 16 KB) | `cb_weights` (whole K x lane) | M block 0: one DRAM read batch + one multicast per K block, with one block of read-ahead. M blocks >= 1: one reserve/push per K block, no reads. |
| `load_rotary_table_block` | 2 x block_m_tiles fp32 pages (C, S per row) | `cb_rotary_table` | One read batch + one barrier per M block. Issued after in0 K block 0 of the same M block is pushed. |
| `matmul_accumulate_block` | block_m_tiles x LANE_N_TILES output, K blocks 0..3 | `cb_weights`, `cb_partials` | **One `matmul_block` call per M block covering all 5 K blocks.** The helper owns `matmul_block_init` (once per call plus after each reload), and its first K block packs with L1 acc off. One partials pack per subblock per K block. |
| `matmul_finish_rotate_block` | same block, last K block (5 tiles: 20–23 + bias) | `cb_partials` (reloaded per subblock), `cb_rotary_table` | Inside the same call. Per q/k subblock (one head): one reload (2 tiles), one C/S copy pair into DEST 2–3, the FMAs, one packer-thread rotary, one bf16 pack of 2 tiles. The copy init is once per subblock (it shares the reload's srcA format: fp32 = fp32). The table wait is once per M block, the pop after the call. |
| `scatter_heads_block` | block_m_tiles rows x LANE_N_TILES tiles | `cb_heads_out` (whole block) | One wait/pop and one write flush per row. Padding-row zeroing only on rows with `s_t == S_t − 1` when S % 32 ≠ 0. |

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| L1 q/k subblock 1x2 in **every** K block (the helper has one subblock shape per call) | 1x2 unpacks 1.5 tiles per tile product against 1.25 for 1x4. Catalog `matmul_output_subblock`: the subblock size effect is largest for short inner dims (here 5 a subblock). | 1x4 (or 2x2) for K blocks 0..3 and 1x2 only in the last K block (custom block helper: `MatmulBlockShape` per phase). Partials stay row-major either way because subblock height is 1. **Built (Refinement 2)**: two `matmul_block` calls per M block (Interm 1x4, then the last K block 1x2 via `MatmulBlockShape::reload_on_first_k_block`); knob `WIDE_SUBBLOCK_W`; −0.8 µs at 8x512. |
| L2 rotary SFPU on the packer thread (overlap) | If the SFPU rotate (about 10 SFPU instructions x 32 lanes-iterations per head) exceeds the math thread's per-head last-block work (reload + C/S copy + 10 tile products), the last K block becomes pack-bound | Math-thread SFPU inside the `PostComputeFn` (no helper extension), and/or SFPU instruction scheduling (`SFPLOADMACRO`) **Resolved (Refinement 2)**: the rotate was bound by the packer TRISC building dynamic SFPLOAD/SFPSTORE words (loop-variant `dst_reg[i]`); static offsets + `dst_reg++` hide it (SFPU 9.4 → 0.7 µs at 8x512). The C/S copy now rides the partials reload (`post_reload` hook, knob `CS_ON_RELOAD`). |
| L3 `block_m_tiles` 3 (overlap and tail) | Larger blocks are capped by L1. Smaller blocks shorten the final drain tail but add M-block fixed costs. Pirotski's best first block at share 12 was 5 (§7.1), with a 680 KB footprint. | `BLOCK_M_TILES_MAX = 2`, and non-uniform blocks (e.g. 3,3,3,2+1 garbage vs 4 with a half-block `cb_heads_out`) |
| L4 weight sender read-ahead depth 1 | Small M (2x512: one M block) streams all 3.7 MB of W inside one K loop; per-sender DRAM rate may bound it (SMALL_M_MATMUL.md §4.2–4.3) | read-ahead 2, transaction ids (catalog `trid_double_issue`: depth 3–4 pays), 16 KB lane pages instead of 4 KB head pages. **Measured (Refinement 3)**: 16 KB lane pages kept (−0.5 to −1.1 µs). Read-ahead and transaction ids are built (`WEIGHT_READ_AHEAD_CHUNKS`), but 2 blocks in flight cost +2 to +3 µs: the 9 senders together read ~720 KB per K block at ~240 GB/s, so the stream is DRAM-bound and extra in-flight blocks delay block 0. Chunked multicast (`W_MCAST_CHUNK_K_ROWS`, a new `send_payload` on the mcast sender pipe) makes block 0 land ~0.5 µs earlier and ends compute 0.6–1.1 µs earlier, but the output-write tail absorbs it; parked at one multicast per block. |
| L5 fixed in0/in1 senders | One sender per column/row serializes that line's reads (catalog `mcast_topology`: "each line's sender reads its slice serially") | `McastRotatingSenderConfig` for in1 at small M (K block b sent by column b mod cols). **Measured (Refinement 3)**: built as `W_SENDER` (right / diagonal / rotating with a per-row stagger). Rotating is flat with one block of read-ahead and worse with more (DRAM-bound, as in L4); the diagonal (catalog `noc_placement`) is no faster than the right column. Default stays the right core. |
| L6 q/k/v write order | 99 cores writing 2 KB pages to L1/DRAM banks at once (pirotski §1: about 9–18 GB/s per core when all write together) | Rotate each core's write order by its index (catalog `bank_stagger`, write variant unmeasured) |
| L7 single-RISC drain | The last M block's tiles drain after compute ends (the tail) | Two-RISC drain of the last block (pirotski `split_drain`, §3.3: −9 us for a 12-row block). It needs a semaphore contract so `cb_heads_out` keeps one CB consumer. |
| L8 ragged last block computes garbage rows | core_m_tiles = 5, 8, 11 compute 6, 9, 12 rows (M = 48, 80, 112) | An M-narrowing field in `MatmulBlockShape` (analogue of `last_in1_subblock_w_valid`, `matmul_block_helpers.hpp:158`) with nominal CB counts |
| Grid synchronization | At M = 32 every core runs a single M block: 5 in0 and 5 in1 handshakes, and the first-block DRAM latency is exposed | Fewer columns with larger blocks is impossible (block_m_tiles is at its 3-row cap). Measure the first-block arrival under zones (pirotski §7.2 found no cheaper first block). |

Catalog entries that set knobs: `mcast_topology` (two 1D multicasts for a 2D split → arrangement A), `trid_double_issue` and `double_buffer` (in-flight depth → `X_BLOCK_DEPTH = 2`, read-ahead 1, L4), `matmul_output_subblock` (L1), `compute_block_size` (reconfigs cost 110–150 ns each: tables kept fp32 like partials, so the C/S copy needs no srcA reconfig after the reload), `eltwise_l1_vs_dest_accumulate` (packer L1 accumulation over read-modify-write), `compute_fusion` (no FPU dest-reuse multiply after the matmul; the rotary is SFPU on DEST), `bank_stagger` (L6).

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| x shards → column top core | bf16 tiles, L1 (norm's ND shards) | RISCV_0, NOC0 `async_read` of row runs from holder cores, driven by the **x source table** | Each x tile is read exactly once, by the top core of the column that owns its row |
| top core → column | bf16 tiles in `cb_x_block` slot | `McastArgs` sender `send(slot, slot, bytes)` on NOC0 south to (c, 1..8); receivers `receive()` | The top core pushes its own slot to its compute before sending (both only read the slot) |
| W pages DRAM → row in1 sender (`W_SENDER`, default the right core) | 16 KB lane pages (8 tiles) | RISCV_1, NOC1 `TensorAccessor` reads, one transaction id per chunk, read-ahead 1 block | M block 0 only |
| right core → row | bf16 tiles in `cb_weights` K-block slot | `McastArgs` on NOC1 west to (0..cols−2, y) | M block >= 1: resident re-push, no traffic |
| tables L1 → q/k core | fp32 tiles | RISCV_0 `TensorAccessor` reads into `cb_rotary_table` | — |
| compute | fp32 DEST; fp32 partials in L1 | `matmul_block` (packer L1 acc; Out target with reload), hooks: C/S copy (math), rotary (packer-thread SFPU) | **One rounding to bf16**, at the final pack |
| `cb_heads_out` → q/k/v | bf16 tiles | RISCV_1, NOC1 `TensorAccessor` write per tile (3 accessors), page `((b*12 + head)*S_t + s_t)*2 + half` | Padding rows overwritten with 0 (`fill_l1_range`) in the CB page before the write |

**x source table (host, per column top core).** Read from `x.memory_config().nd_shard_spec`. Shard shape (1, 1, 32·r, 32·kp) gives r and kp in tiles; `P = 24/kp`. Ordered holders: `ttnn.corerange_to_cores(grid, None, row_wise=(orientation == ROW_MAJOR))`. `n_sb = ceil(S_t / r)`. Tile (m, k): `b = m // S_t`, `s_t = m % S_t`, shard index `i = (b*n_sb + s_t // r)*P + k // kp`. The holder is `cores[i mod ncores]`, at `x.buffer_address() + (i // ncores)*r*kp*2048 + ((s_t % r)*kp + k % kp)*2048`. One entry is emitted per (holder, run of consecutive rows in the same shard): (noc_x, noc_y, L1 address of the run's first row at k = part start, row count, row stride = kp·2048). Rows past S_t inside a partial shard are never addressed. Self-check on the host: every (m, k) of the column maps to exactly one entry.

**Tensix-to-Tensix contracts of the deferred regimes.**
- `holder_direct_x_mcast`: each holder multicasts its run into the receivers' slot at byte offset `(row_in_block*K_BLOCK_TILES + k_in_block)*2048`. Receivers wait for a counter equal to the number of contributing holders for that (M block, K block), which the host can precompute.
- `table_broadcast`: a third channel on the column, same sender, fixed 2·block_m_tiles fp32 pages per M block.

**TARGET → scheme**

| TARGET axis value | Realization | Cost class |
|-------------------|-------------|------------|
| placement `ln_layout` | x source table from the ND shard spec (above) | **Phase 0** |
| placement `op_layout` | `input_memory_config()` returns the norm's layout, so op_layout ≡ ln_layout and the golden suite skips its cells. Not listed in SUPPORTED since Refinement 1b: validate() never tags a tensor `op_layout`, so it is served as `ln_layout` | Phase 0 (by construction) |
| placement `model_layout` | Same gather; the x source table is built from BLOCK_SHARDED COL_MAJOR shards (shard (32·bh, 96), bh = ceil(M/11) = core_m_tiles: column c's rows sit in column c's own 8 cores, K part y on grid row y, kp = 3, so up to 3 runs per row per K block) | **knob-turn** refinement (host table builder only; kernels unchanged) |
| output_buffer `l1` / `dram` | q/k/v `TensorAccessor` over the buffer's banks | Phase 0 |
| output_mode `new` / `preallocated` | Host: allocate v, k, q in `memory_config`, or take `output_tensors` | Phase 0 |
| seq_alignment `tile_aligned` / `seq_padded` | Writer zeroes padding rows of tiles at `s_t == S_t − 1` | Phase 0 |

**op_layout decision (what the norm would have to write).** Nothing different: `input_memory_config` returns the norm's own layout. The layout this op would gain most from is the model's split layout (each column's x on that column's 8 cores, K part per grid row), which would turn the gather into short in-column reads or enable holder-direct multicast. The norm itself measured that layout at 34.3 us against 16.9 us for its own layout at 8x512 (`nomic_residual_layer_norm.py`, MODEL/PIPELINE comments: "model 8x512 ... 34.3"). That is +17 us for the norm, against a gather that is off this op's critical path. Not requested. Nothing measured by this op yet: the perf pass re-checks it with the gather's zone time.

**Structural impossibilities.** None to add to `INVALID`.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | An M block: `block_m_tiles` rows x the core's lane (`LANE_N_TILES` = 8 tiles) x all `K_TILES` |
| Grid | `num_columns` x 9: logical cores (c, y) with c in [0, num_columns) and y in [0, 9). Grid row 9 is idle (regime `strip_row_10` rejected). Device grid must be 11 x 10 (otherwise SupportRefusal). |
| Per-core work | Rows `[c*core_m_tiles, min(M, (c+1)*core_m_tiles))`; lane y = N tiles `[8y, 8y+8)`; `num_m_blocks_this_core = ceil(rows_this_core / block_m_tiles)` |
| Remainder | M = B·ceil(S/32), per image (each batch tile-padded independently). The last column holds `M − (num_columns−1)*core_m_tiles` rows. The ragged last block computes garbage rows on stale in0 with nominal CB counts, and the writer skips rows >= the column's end. |
| Roles | Column top core (c, 0): in0 sender. Row right core (num_columns−1, y): in1 sender. Rows 0–5: q/k (rotary) compute kernel. Rows 6–8: v (plain) compute kernel. |

Per-shape plan (Phase 0 rule, every INPUTS M class):

| M | core_m_tiles | num_columns | cores | num_m_blocks | block_m_tiles | rows computed / core |
|---|---|---|---|---|---|---|
| 32, 33 | 3 | 11 | 99 | 1 | 3 | 3 |
| 36 | 4 | 9 | 81 | 2 | 2 | 4 |
| 39 | 4 | 10 | 90 | 2 | 2 | 4 |
| 48 | 5 | 10 | 90 | 2 | 3 | 6 (L8) |
| 64 | 6 | 11 | 99 | 2 | 3 | 6 |
| 80 | 8 | 10 | 90 | 3 | 3 | 9 (L8) |
| 96 | 9 | 11 | 99 | 3 | 3 | 9 |
| 112 | 11 | 11 | 99 | 4 | 3 | 12 (L8) |
| 128 | 12 | 11 | 99 | 4 | 3 | 12 |

One regime, so no selection predicate. Acceptance shapes pin M = 32, 33, 39, 64 and 128 (1, 2 and 4 M blocks; 10 and 11 columns; padded and aligned).

## Circular Buffers

All CBs are declared on every core of the grid, in this order, with these sizes (Mechanism caps: equal addresses for multicast).

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_x_block` | 0 | 2048 | `X_BLOCK_DEPTH * block_m_tiles * K_BLOCK_TILES` (30) | spans M block x K block; streams over K and over M blocks | Float16_b | reader | compute | per K block |
| `cb_weights` | 1 | 2048 | `K_TILES * LANE_N_TILES` (200) | spans K (all) x lane; resident ring | Float16_b | writer | compute | whole op |
| `cb_rotary_table` | 2 | 4096 | `ROTARY_TABLE_DEPTH * 2 * block_m_tiles` (6) | spans M block rows x {C, S}; streams over M blocks | Float32 | reader | compute | per M block (q/k cores; unused on v cores) |
| `cb_partials` | 3 | 4096 | `block_m_tiles * LANE_N_TILES` (24) | spans M block x lane; accumulates over K blocks 0..3 | Float32 | compute | compute | K blocks 0..3 plus the reload in the last K block |
| `cb_heads_out` | 4 | 2048 | `OUT_DEPTH_BLOCKS * block_m_tiles * LANE_N_TILES` (24) | spans M block x lane | Float16_b | compute | writer | from the last K block of block m until drained during block m+1 |

Full ledger, sharing decisions and footprint: `l1_ledger.md` (628 KiB at block_m_tiles = 3).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_x_block` | block_m_tiles x 5 | Mcast helper + raw gather reads + `fill_l1_range` (ones tile) | x shards (L1, not a CB) | `cb_x_block`, push `block_m_tiles*5` | — |
| 2 | `load_weight_block` | 5 x 8 | Mcast helper + `TensorAccessor` reads | W lane pages (DRAM) | `cb_weights`, push 40 | M block >= 1: reserve+push only (resident) |
| 3 | `load_rotary_table_block` | 2·block_m_tiles fp32 pages | `TensorAccessor` reads | tables (L1) | `cb_rotary_table`, push `2*block_m_tiles` | — |
| 4 | `matmul_accumulate_block` + 5 `matmul_finish_rotate_block` | block_m_tiles x 8, K = 5 blocks of 5 | **`matmul_block`, one call per M block**: `packer_l1_acc=true`, `LastBlockTarget::Out`, `SubblockMajor`. q/k cores: `PostComputeFn` = copy C/S of the subblock's row into DEST 2–3; packer-thread rotate functor (helper extension). | `cb_x_block` (wait/pop 15 per K block), `cb_weights` (wait/pop 40 per K block), `cb_partials` (self), `cb_rotary_table` (wait 6 before the last K block, pop 6 after the call) | `cb_heads_out`, push 2 per q/k subblock (4 per v subblock); 24 per M block | `cb_partials` empty after each call; `cb_weights` rd_ptr wrapped to K block 0 |
| 6 | `scatter_heads_block` | block_m_tiles x 8 | `TensorAccessor` writes + `fill_l1_range` (padding rows) | `cb_heads_out`, wait/pop 8 per row | q/k/v pages | — |

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| compute startup | helper | `compute_kernel_hw_startup<SrcOrder::Reverse>(in0, in1, out)` | required by `matmul_block_helpers.hpp:100-106` (mm_block_init removed) | in0 = `cb_x_block`, in1 = `cb_weights`, out = `cb_heads_out` | — | — | — |
| matmul_accumulate / finish | helper | `compute_kernel_lib::matmul_block<false, true, LastBlockTarget::Out, OutputCBLayout::SubblockMajor, InitMode::Short, WaitAndPopPerKBlock, WaitAndPopPerKBlock, PostComputeFn, ...>` | `matmul_block_helpers.hpp:339-370` (decl), `.inl:90` (def); last-block reload `.inl:316-351`, post_compute `.inl:381`, packer-thread slot `.inl:397-405`, Out-target pack with L1 acc off `.inl:418` | `MatmulBlockShape::of(in0_num_subblocks = block_m_tiles, in1_num_subblocks = 8/out_subblock_w, out_subblock_h = 1, out_subblock_w = 2 (q/k) or tuner (v), in0_block_k = K_BLOCK_TILES, num_k_blocks = K_TILES/K_BLOCK_TILES)` (`.hpp:160`); `interm_buf = cb_partials` | `cb_x_block`, `cb_weights`, `cb_partials` | `cb_heads_out` | `block_m_tiles` (in0_num_subblocks), `K_BLOCK_TILES` (in0_block_k), `LANE_N_TILES` (in1_per_core_w). The subblock is pinned on q/k cores by the DEST cap and tuner-chosen on v cores. |
| C/S into DEST (q/k) | raw_api inside the helper's hook | `PostComputeFn`: `reconfig_data_format_srca`, `copy_init(cb_rotary_table)`, `copy_tile(cb_rotary_table, 2*row, 2)`, `copy_tile(..., 2*row+1, 3)` | hook call `matmul_block_helpers.inl:381`; `copy_tile` `tt_metal/hw/inc/api/compute/tile_move_copy.h:126` | row = in0 subblock index within the block | `cb_rotary_table` | DEST 2–3 | — |
| rotary (q/k) | raw_api (custom SFPU) driven on the packer thread | custom `rotate_pair` SFPU: DEST0 = t0·C − t1·S, DEST1 = t1·C + t0·S, fp32, one pass. Packer-thread wrapper modelled on `apply_activation_from_pack` (semwait MATH_PACK, packer dest-offset flip, SFPU, `STALLWAIT(STALL_PACK, WAIT_SFPU)`) | `sfpu_activation_helpers.hpp:159-172`; helper call site `matmul_block_helpers.inl:397-405` | packer init of the SFPU once at kernel start (as `*_tile_init_pack`, `sfpu_activation_helpers.hpp:72-104`) | DEST 0–3 | DEST 0–1 → packed by the helper | — |
| ones / padding fill | helper | `fill_l1_range<4>(addr, n_bytes, 0)`, `fill_l1_range<2>(addr, 2, 0x3F80)` | `ttnn/cpp/ttnn/kernel_lib/l1_helpers.hpp:101` | — | — | — | — |
| in0 / in1 multicast | helper | device `McastArgs<ct, rt>` → `sender(noc).send(src, dst, size)`, `receiver(noc).receive()`; host `ttnn.Mcast(device, McastConfig(noc, handshake=True), receivers, group_size, McastExplicitFixedSenderConfig(senders)).attach(descriptor, prefix, kernels, first_sem)` | `mcast/kernel/mcast_args.hpp:310`; `mcast/kernel/pipes/mcast_pipe.hpp:92`, `:163`; `mcast/host/mcast.hpp:98`, `:124`, `:135`; Python binding `ttnn/cpp/ttnn-nanobind/mcast_host.cpp:55` | in0: NOC0, groups = columns (8 receivers), sender (c, 0). in1: NOC1, groups = rows (cols−1 receivers), sender (cols−1, y). | slot address | same slot address | slot bytes = block tiles x 2048 |
| x gather | raw_api | `Noc::async_read(UnicastEndpoint{}, CoreLocalMem, bytes, {noc_x, noc_y, addr})` + one barrier per K block | pattern: `pirotski_matmul/device/kernels/dataflow/in0_stream.cpp:66-98` | runs from the x source table | x shards | `cb_x_block` slot | runs per row per K block (from `kp`, `K_BLOCK_TILES`) |
| W reads, table reads, q/k/v writes | raw_api | `TensorAccessor` page reads/writes | `tech_reports/tensor_accessor/tensor_accessor.md`; pattern `in1_stream.cpp:109-127` | page ids above | — | — | — |

**Helpers considered and rejected / gaps recorded:**
- Rotary on the packer thread through `matmul_block`'s `Activation` slot. The slot only takes `KernelActivation` enum values (`sfpu_activation_helpers.hpp:111-119` static_assert; dispatched at `matmul_block_helpers.inl:397-405`), so a custom functor cannot be passed. **Gap:** add a generic packer-thread functor template parameter to `matmul_block` (default: none), called where `apply_activation_from_pack` is. The alternative is a custom block helper copying the loop. The math-thread fallback (inside `PostComputeFn`) is lamp L2.
- `eltwise_chain` / `SubBinary`, `MulBinary` (`eltwise/binary/sfpu/basic.hpp:20-30`) for the rotary. A chain owns its own acquire/commit window and cannot run inside the matmul hook (`eltwise/api/chain.hpp:15-16`). The two-operand SFPU binaries need a fifth DEST slot for t0·C − t1·S without clobbering t0 before out1 uses it; half-sync fp32 has 4 (`dest_helpers.hpp:98`). Hence one fused 4-in/2-out SFPU pass.
- `add_bias_bcast_rows` (`bias_add_helpers.hpp:156-168`). Not used: the bias is the K tile 24, accumulated in fp32 by the matmul itself. That removes the bias phase, the bias CB and the reload-then-add round trip.
- FPU rotary (`BinaryFpu` mul with dest accumulation). At HiFi2 srcB keeps 1+6 mantissa bits (`tech_reports/matrix_engine/matrix_engine.md:69`), which truncates either the tf32 data or the table to below bf16 precision. That breaks "no bf16 intermediate before the rotary".

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| bias K tile | matmul product ones x bias | `cb_x_block` K tile 24: column 0 = 1.0, all else 0 (All, by construction) | `cb_weights` K tile 24: row 0 = bias, rows 1–31 = 0 (All) | None (matmul) |
| rotary | SFPU rotate_pair | DEST 0–1 (All) | DEST 2–3 from `cb_rotary_table` (All) | None |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Packer-thread SFPU with a custom functor | The helper accepts only `KernelActivation` on the packer thread | Extend `matmul_block` with a packer functor slot (mirror `apply_activation_from_pack` exactly: `TTI_SEMWAIT` MATH_PACK, `SETC16` dest offset, SFPU, `STALLWAIT` before pack). Fallback: math-thread rotary in `PostComputeFn` (correct; slower). |
| `PostComputeFn` changes unpacker/math state between subblocks | `matmul_block_init` is not re-issued between subblocks except after a reload | On q/k cores every last-block subblock starts with the helper's reload (`.inl:327-351`), which reconfigures srcA and re-inits matmul. The hook must leave srcA in an fp32 format (tables fp32 = partials fp32). Each per-M-block call re-inits. |
| The rotary on a partial instead of the final sum | With `LastBlockTarget::Interm`, DEST at `post_compute` holds only the last K block's partial | Use `LastBlockTarget::Out` with `packer_l1_acc=true`: the partial is reloaded into DEST before the last FMAs (`.inl:327-351`, `:529`), so DEST holds the full fp32 sum + bias |
| Multicast address equality | in0/in1 multicasts write each slot at the sender's address on every receiver | Identical CB declarations on all cores (incl. `cb_rotary_table` on v rows) |
| Resident ring wrap | Re-pushes of M block >= 1 must land on the slots of the same K block | `cb_weights` pages = `K_TILES * LANE_N_TILES` exactly; 5 pushes of 40 close the ring |
| Odd K-block count | 5 K blocks per M block alternate `cb_x_block`'s slot parity each M block | The top core rewrites the ones tiles of K block 4 every M block; they are never pre-placed |
| Table load placement | `cb_rotary_table` is single-slot | Request it after pushing in0 K block 0 of the same M block (Stall-shadow table) |
| Padding rows by multiplication | NaN · 0 = NaN; x padding rows are poisoned (inf, NaN, 3e37) in `test_regression.py` | The writer overwrites rows `[S − 32*(S_t−1), 32)` of every tile at `s_t == S_t−1` (q, k **and** v) with zeros (`fill_l1_range`) in `cb_heads_out` before the write: face rows are 32 B; at most 3 ranges per tile. Valid rows never read padding rows: matmul, bias, rotary and pack are row-local. |
| Garbage rows | Stale or poisoned in0 rows in ragged blocks | Computed, never written. Table page index clamped. No cross-row op exists in the pipeline. |
| x shard plan drift | The norm picks P and r per shape from a cost table that can change | Derived at runtime from x's `NdShardSpec` with a host self-check; never from the numbers in this document |
| RISC/NOC mapping is inverted against the triage default | reader = RISCV_0 (BRISC) + NOC0, writer = RISCV_1 (NCRISC) + NOC1, as in pirotski | State it in the kernel headers. In triage, a hung in0 stream shows on BRISC. |
| Precision headroom | q/k gate RMS 0.0050 against the incumbent's 0.0049 | One bf16 rounding (final pack). Bias exact (bf16 x 1.0 in fp32). q scale exact. Error sources: HiFi2 drops x's last mantissa bit (srcB), as the incumbent does; tf32 truncation of the reloaded partial (rel <= 2^-11) and of C/S (fp32 tables via srcA). |
| Weight/table object identity | `validate()` must recognize prepared objects | `QkvWeights` / `RotaryTables` frozen dataclasses carrying the tensor plus a layout tag; validate checks type, tensor shape/dtype/layout and `tables.seqlen >= S` |
| q/k/v allocation order | The model preallocates v, k, q so that v is highest | With `output_tensors` the op only writes. When allocating it allocates v, k, q (mirrors the model). |
