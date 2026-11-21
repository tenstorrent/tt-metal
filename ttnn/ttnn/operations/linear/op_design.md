# Operation Design: linear

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (tiled matmul + optional row-broadcast bias), one `ttnn.generic_op` dispatch |
| Goal | `out = a @ b (+ bias)` over the last two dims, Phase 0 = the PHASE0 rectangle of `eval/specs/linear/spec.py` (bfloat16, TILE, rank 4, tile-aligned, single activation matrix, activation interleaved DRAM/L1 or width-sharded L1, result in DRAM or width-sharded L1, bias none / row / vector, both DEST widths) |
| Math | `out[..., i, j] = sum_k a[..., i, k] * b[k, j] (+ bias[j])` |
| References | `eval/specs/linear/spec.py` (the requirements: governs over this file); `.claude/references/targets/quasar.md`; `.claude/references/backends/ttsim.md`; `.claude/references/quasar_op_examples/` (README, `toy_row_passes`, `toy_thread_plan`); `tt_metal/impl/dataflow_buffer/DataflowBuffer.md` (Part A, C2); `tt_metal/hw/inc/api/compute/matmul.h`; `tt_metal/hw/inc/api/compute/bcast.h`; `ttnn/ttnn/operations/examples/master.md` (`matmul_output_subblock`, `compute_block_size`, `double_buffer`, `mcast_topology`, `shared_input_reuse`, `split_reader`) |

## Target

| Fact | Value | Source |
|------|-------|--------|
| Target | `quasar` on ttsim simulator | `.claude/references/targets/quasar.md` + `.claude/references/backends/ttsim.md` |
| Arch | quasar | target page |
| Compute grid | 8 × 4 = 32 clusters (fixed on this part); the op queries `device.compute_with_storage_grid_size()` | target page |
| L1 per core | 4,194,304 B (4 MiB), shared by the cluster's 4 NEOs and DM cores | target page |
| Unreserved budget | `ttnn.get_max_worker_l1_unreserved_size() - L1_RESERVE_BYTES` (`L1_RESERVE_BYTES = 65536`); the value is below 4,194,304 B. The target page does not state the reserved part, so the exact number is the query's result on the run | target page (L1); exact value `unknown` until queried |
| Host program | `ttnn.ProgramSpec` (Metal 2.0) through `ttnn.generic_op`: `.claude/references/quasar_op_examples/` (README, then `toy_row_passes/toy_row_passes_program_artifacts.py`) | target page |
| Threads per core | 4 NEOs (C ∈ {1, 2, 4}), 6 user DM harts (R + W ≤ 6); on STRIDED rings R and W divide C; ≤ 24 active DFBs per NEO, 32 per cluster; DFB ring ≤ 1,048,560 B on the compute side | target page, *Cluster model / Thread plan* |
| Unavailable here | **`ttnn.ProgramDescriptor`** and the legacy C++ program factory (refused at kernel creation). **`matmul_block_helpers.hpp`** and **`bias_add_helpers.hpp`**: not used, the matmul and the bias preload are written with the raw compute API. **`matmul_block` spill/reload**: not used, the design never spills. **`matmul_block` with `idst != 0`** (`matmul.h:297`). **`matmul_block_k`'s fused K steps** (Quasar runs them as a loop, `matmul.h:368`). **`llk_unpack_AB_matmul_set_in1_column_stride` ≠ 1** (`llk_unpack_AB_matmul_api.h:16-19`). **Strided indexing in the matmul unpacker**: not used, a matmul operand is never on a STRIDED ring with more than one thread on either side. **`unary_bcast` of a 32-bit tile** (`bcast.h:71,109`, LLK_ASSERT): matters for the float32 refinement, not Phase 0. **`bfloat8_b` / `bfloat4_b`** have no path (family facts). Every compute `KernelSpec` must set `fpu_math_fidelity` explicitly |

Every worked number below is for this target (32 clusters, 4 MiB L1, bfloat16 tiles of 2048 B). Every predicate the op evaluates queries the device.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor_a` | `ttnn.Tensor` | yes | `[..., M, K]`, rank ≥ 2 (X2); Phase 0: rank 4, leading dims 1, bf16, TILE, tile-aligned, interleaved DRAM/L1 or width-sharded L1 | — | tensor |
| `input_tensor_b` | `ttnn.Tensor` | yes | `[K, N]` or `[1, …, 1, K, N]` (X7 refuses a non-1 leading dim; X1 a K mismatch); interleaved DRAM, TILE, dtype of `a` (R6) | — | tensor |
| `bias` | `ttnn.Tensor` / None | no (kw) | `[N]` or `[1, …, 1, N]` (rank 2-4), interleaved DRAM, TILE, dtype of `a`; read with spec `canonical_bias` (X3 when not N wide; `"block"` refused by name) | None | tensor; presence is CT (`has_bias`) |
| `memory_config` | `ttnn.MemoryConfig` / None | no (kw) | read with spec `canonical_memory`: `"DRAM"`, `"L1/width"`, `"L1/width/auto"`; anything else refused by name (R8); a non-config raises `TypeError` (X6) | None → DRAM interleaved | host |
| `dtype` | `ttnn.DataType` / None | no (kw) | read with spec `canonical_dtype`: only `"same"` (R9, X5) | None | host |
| `program_config` | any | no (kw) | accepted and ignored (R10), except `fused_activation` not None → `NotImplementedError` (R11, X4) | None | host |
| `compute_kernel_config` | `ttnn.ComputeConfigDescriptor` / `ttnn.WormholeComputeKernelConfig` / None | no (kw) | read `fp32_dest_acc_en`, `math_fidelity`, `math_approx_mode` as attributes; never refused for its type; extra fields ignored (R4); None → `default_compute_kernel_config()` (R2); float32 operands with `fp32_dest_acc_en=False` refused (R1) | `default_compute_kernel_config()` = HiFi4, `fp32_dest_acc_en=True`, `math_approx_mode=False` | `fp32_dest_acc_en` and fidelity are kernel config (CT) |
| `core_grid` | `ttnn.CoreGrid` / None | no (kw) | accepted and ignored (R10) | None | host |

The op file imports the axis functions: `INPUT_TAGGERS = AXES` (from `eval/specs/linear/spec.py`). `validate()` derives each axis: `alignment`, `rank`, `weight_form`, `batch` from `INPUT_TAGGERS` on `(a.shape, b.shape)`; `memory_config` and `sharding` with `eval.memory.memory_axes(input_tensor_a)` (R5); `output_memory` with `canonical_memory`, `bias` with `canonical_bias`, `dtype_argument` with `canonical_dtype`; `fp32_dest_acc_en` from the resolved compute config; `dtype`, `layout` from `a`. Then it gates every axis on SUPPORTED (= PHASE0) with `NotImplementedError` (R15). It also checks the weight and bias with `memory_axes` (DRAM, interleaved), layout TILE, dtype == `a.dtype`, `NotImplementedError` naming the parameter (R6). The order: the malformed-call checks (X1, X2, X3, X6, X7) first, then R11, then the axis gate.

## Tensors

### Input

| Property | `input_tensor_a` | `input_tensor_b` | `bias` |
|----------|------------------|------------------|--------|
| Shape | `[..., M, K]` | `[K, N]` / `[1,…,1,K,N]` | `[N]` / `[1,…,1,N]` |
| Dtype | bf16 (Phase 0) | = `a` | = `a` |
| Layout | TILE | TILE | TILE (one tile-row: row 0 holds the values) |
| Memory | interleaved DRAM / L1, or width-sharded L1 (any shard grid) | interleaved DRAM | interleaved DRAM |
| Page id of tile (r, c) | `r * tensor_k_tiles + c` (r over all images' tile-rows) | `k * tensor_n_tiles + n` | `n` (both forms are the same tiles) |

### Output

| Property | Value |
|----------|-------|
| Shape | `a.shape[:-1] + [N]` (spec `output_shape`) |
| Dtype | dtype of `a` (R9) |
| Layout | TILE |
| Memory | `"DRAM"`: `ttnn.DRAM_MEMORY_CONFIG`; `"L1/width"`: exactly the config passed; `"L1/width/auto"`: `eval.memory.place(eval.memory.Sharded("L1", "width"), out_shape, ttnn.TILE_LAYOUT, device)` (R7). Allocated with `ttnn.allocate_tensor_on_device(ttnn.TensorSpec(out_shape, dtype, ttnn.TILE_LAYOUT, memory_config), device)` |
| Page id of tile (r, n) | `r * tensor_n_tiles + n` |

## Blocking Model

Semantics: `.claude/references/blocking-model.md`.

**Index space.** Three tile axes: rows (`tensor_m_tiles = batch * ceil(M / 32)`, every image tile-padded on its own; Phase 0: `batch = 1`), columns (`tensor_n_tiles = ceil(N / 32)`), and the contraction (`tensor_k_tiles = ceil(K / 32)`).

**The block.** One block is a `[block_h × block_w]` rectangle of output tiles. One NEO owns it, holds its accumulator in DEST for the whole contraction, and walks `K` through it in chunks of `block_k` tiles: `[block_h × block_k]` activation tiles and `[block_k × block_w]` weight tiles per chunk. One pipeline pass is: preload the bias into DEST, run every K chunk of matmul into DEST, pack once. A **round** is C blocks with the same rows and C consecutive column blocks, one per NEO. Its activation chunks are delivered once to the cluster and seen by all C NEOs.

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| rows (`tensor_m_tiles`) | **independent**: output rows never combine; **the weight is reuse-shared along it**, since every row block needs the same weight columns | `block_h` | from `select_blocking()`: 1 for every Phase 0 shape (M_t ∈ {1, 4}, chosen for occupancy) | `BlockPlan.block_h`, chosen once on the host; passed as a CT arg to all four kernels | row blocks are part of the round index; rounds are dealt to clusters with `split_work_to_cores` | knob-turn (any divisor of `tensor_m_tiles` within the DEST cap) |
| columns (`tensor_n_tiles`) | **independent**: output columns never combine; **the activation is reuse-shared along it**, since every column block needs the same activation rows | `block_w` | from `select_blocking()`: 1 for every Phase 0 shape | `BlockPlan.block_w`, CT arg | C column blocks per round, one per NEO (thread plan); rounds dealt to clusters | knob-turn |
| contraction (`tensor_k_tiles`) | **dependent**: every output sums over all of K | `block_k` | the largest divisor of `tensor_k_tiles` that passes the fit predicate: ResNet 32 (2 chunks), Qwen 8, Llama 2, `64x96` 3, `32x32` 1 | `BlockPlan.block_k`, CT arg; the core's K range is RT (`k_tile_start = 0`, `k_tiles = tensor_k_tiles`) | **not split across cores or NEOs**: each NEO walks the whole K sequentially into DEST (the cheap local accumulate) | scheme-change (split-K + combine: deferred regime rows) |
| batch (leading dims of `a`) | **independent**, and the weight is reuse-shared along it, exactly as for rows | folded into rows: `tensor_m_tiles = batch * ceil(M/32)` | `batch = 1` (PHASE0 `batch: single`) | `tensor_m_tiles` (host, CT arg) | as rows | knob-turn: page ids are already global tile-row indices, so a batch is more rows |
| round (C blocks sharing rows) | derived unit of the thread plan, not a data axis | C (`THREAD_PLAN.compute`) | 4 | `THREAD_PLAN`, CT arg `compute_threads` | whole rounds dealt to clusters (`split_work_to_cores(grid, num_rounds)`) | knob-turn (C ∈ {1, 2, 4}) |

No axis is left undecided. The stages the scheme adds produce data of their own, and each is assigned. The K-chunk partials live only in DEST, so they have no storage and no axis of their own. The bias preload acts on the block's `[block_h × block_w]` tiles inside the same NEO, in DEST. The output block is written by the cluster's writer harts. No core collects or coordinates: every cluster computes and writes its own rounds, and no stage runs on one core while the others wait.

**`select_blocking(tensor_m_tiles, tensor_n_tiles, tensor_k_tiles, fp32_dest_acc_en, C, num_cores, budget)`: the regime-internal extent choice, exact.**

1. `dest_tiles = 4 if fp32_dest_acc_en else 8` (half-sync DEST, `double_buffer_dest=True`).
2. Candidates: every `(block_h, block_w)` with `block_h | tensor_m_tiles`, `block_w | tensor_n_tiles` and `block_h * block_w <= dest_tiles`.
3. For each: `num_row_blocks = tensor_m_tiles / block_h`, `col_blocks = tensor_n_tiles / block_w`, `rounds_per_row_block = ceil(col_blocks / C)`, `num_rounds = num_row_blocks * rounds_per_row_block`.
4. Pick the candidate with the largest key `(min(num_rounds, num_cores) * min(col_blocks, C), block_h * block_w, block_h)`. That is occupancy first (NEOs doing real work at once), then the coarsest block, then the taller one (fewer weight re-reads).
5. `block_k` = the largest divisor `d` of `tensor_k_tiles` with (a) `ACT_DEPTH*block_h*d*tile_bytes <= DFB_RING_CAP_BYTES`, (b) `WEIGHT_DEPTH*C*d*block_w*tile_bytes <= DFB_RING_CAP_BYTES`, (c) `d*block_w <= 1023`, (d) the ledger total `<= budget`. `d = 1` always passes for TARGET extents. If even it fails, `validate()` raises `NotImplementedError` naming the footprint.

Worked Phase 0 values on this target (C = 4, 32 clusters; identical for both DEST widths):

| INPUTS entry | M_t, K_t, N_t | block_h × block_w | block_k | num_rounds | clusters | padded NEO slots |
|---|---|---|---|---|---|---|
| `1x1x32x2048 + 1x1x2048x1024` (ResNet) | 1, 64, 32 | 1 × 1 | 32 (64 fails the ring cap by 16 B) | 8 | 8 | 0 |
| `1x1x32x64 + 1x1x64x128` (Llama) | 1, 2, 4 | 1 × 1 | 2 | 1 | 1 | 0 |
| `1x1x128x256 + (1x1x)256x512` (Qwen) | 4, 8, 16 | 1 × 1 | 8 | 16 | 16 | 0 |
| `1x1x32x32 + 1x1x32x32` | 1, 1, 1 | 1 × 1 | 1 | 1 | 1 | 3 |
| `1x1x64x96 + 96x160` | 2, 3, 5 | 1 × 1 | 3 | 4 | 4 | 6 (3 per row block) |

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_act_tiles` | `ACT_DEPTH` (K chunks in flight) | 2 | reader fills chunk k+1 while the NEOs multiply chunk k |
| `cb_weight_tiles` | `WEIGHT_DEPTH` (K chunks per NEO in flight) | 2 | the same overlap for the weight stream, the dominant byte stream |
| `cb_bias_tiles` | `BIAS_DEPTH` (rounds) | 2 | the next round's bias lands while this round multiplies |
| `cb_output_tiles` | `OUT_DEPTH` (blocks per NEO) | 2 | the writer drains round r while the NEOs pack round r+1 |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| Matmul unpack MOP: `ct_dim * rt_dim <= 8` (16-bit DEST), `<= 4` (32-bit) (`llk_unpack_matmul.h:238`; `matmul.h:214-215`). On Quasar `matmul_block` writes DEST from index 0 only (`matmul.h:297`). The block's accumulator must stay live in DEST across all of K | `block_h * block_w` | `<= dest_tiles` in `select_blocking` step 2 | DEST overflow: wrong tiles, or an LLK assert (`rv32_ecall_ebreak` under `--dev`) |
| `kt_dim`, `ct_dim`, `rt_dim` are baked into the unpack/math MOP at `matmul_block_init` (`llk_unpack_matmul.h:63,80`) | `block_k`, `block_h`, `block_w` must be uniform across the op | divisors only: `block_k \| tensor_k_tiles`, `block_h \| tensor_m_tiles`, `block_w \| tensor_n_tiles` | a ragged last chunk or block would be multiplied with the wrong in0 row stride: wrong results |
| Contiguous indexing of a matmul operand ring (`llk_unpack_AB_matmul_api.h:192,194`) | the access pattern of every matmul operand ring | `cb_act_tiles`: one producer thread, **ALL** consumer (stride 1); `cb_weight_tiles`: one producer thread, **BLOCKED** consumer (each NEO's block contiguous) | — |
| DFB ring ≤ 1,048,560 B on the compute side (target page) | `block_k` (through `cb_weight_tiles`, `cb_act_tiles`) | `select_blocking` step 5 (a), (b). `DFB_RING_CAP_BYTES = 1048560` is a host constant from the target page; there is no run-time query | program build refuses the ring |
| BLOCKED `block_size ≤ 1023` with a Tensix endpoint (`DataflowBuffer.md:311`) | `block_k * block_w` (weight block), `block_w` (bias block) | step 5 (c); `block_w <= 8` already | build refusal |
| STRIDED → BLOCKED: `P \| block_size`, `num_entries % (block_size * C) == 0` (`DataflowBuffer.md:307`) | weight / bias reader thread count, ring capacities | weight reader P = 1; capacities are `depth * C * block_size` | build refusal, or a hang from mis-dealt blocks |
| Implicit-sync threshold on a STRIDED side facing BLOCKED or ALL (`DataflowBuffer.md:431`) | ring `num_entries` | capacities are whole multiples of `C * block_size`; if the host still finds no txn-id count, raise that ring's depth to 3, or use explicit sync on that reader (tagged transfers, ring listed in `disable_dfb_implicit_sync_for`) | build refusal |

### Regimes

| Regime | Status | Predicate | Block | Data movement vs. minimum | What a bigger block buys |
|--------|--------|-----------|-------|---------------------------|--------------------------|
| **R-2D-stream**: 2-D output blocks dealt in rounds of C to clusters; K streamed whole per NEO into DEST; every operand read per cluster through its accessor (any placement); bias folded into DEST; result written through its accessor | **built** | every Phase 0 call (the only built regime) | `[block_h × block_w]` per NEO, K in `block_k` chunks, C blocks per round | Minimum named against DRAM / the activation's home: each input crosses once, the output once. Actual: weight × `num_row_blocks` (+ padded slots), activation × `rounds_per_row_block`, bias × `num_row_blocks`, output once (+ padded duplicates). Phase 0: ResNet weight 1× (4 MiB), activation 8× (1 MiB total); Qwen weight 4× (1 MiB), activation 4× (256 KiB) | `block_h`: fewer weight re-reads, `block_w`: fewer activation re-reads (with C), both: SRC-operand reuse in the matmul MOP. `block_k`: fewer ring handshakes (one `wait_front`/`pop_front` pair per chunk per ring), at a cost in overlap (lamp). Intended frequencies: `compute_kernel_hw_startup` once per kernel; `matmul_block_init` once per kernel without a bias, once per round with one (the bias preload re-inits the unpacker); `unary_bcast_init<ROW>` once per round; DEST acquire/commit once per round; no barriers or semaphores |
| **R-mcast-weight**: one cluster per column round reads the weight chunk and multicasts it to the clusters of the other row blocks (1-D channel along the row-block axis, `mcast_topology`) | **deferred**. It layers onto R-2D-stream without changing the compute kernel: only the weight reader of non-injector clusters becomes an mcast receiver. Phase 0's largest weight (ResNet, 4 MiB) already crosses DRAM once (one row block); multicast between clusters is unexercised on this part at this base (README *Not covered*) | `num_row_blocks > 1` | same | weight 1× instead of × `num_row_blocks` (Qwen: −768 KiB DRAM), adds the same bytes as a NoC multicast | same, plus fewer mcast handshakes per chunk |
| **R-mcast-act**: the clusters of one row block receive its activation chunk by multicast from one reader | **deferred**, same reachability (only the activation reader changes) and the same multicast precondition. Phase 0's activations are ≤ 64 KiB per row block | `rounds_per_row_block > 1` across clusters | same | activation 1× (ResNet: 1 MiB → 128 KiB; from L1 shards in ResNet's call) | same |
| **R-splitK-cores**: K split across clusters, partials combined through the NoC (gather to one cluster, or a reduce-scatter) | **deferred**: a scheme-change. It is the occupancy lever for M_t = 1 shapes (ResNet engages 8 of 32 clusters). Phase 0 keeps it reachable: the K range is already RT (`k_tile_start`, `k_tiles`), so a split-K cluster packs a partial for a sub-range. The partial needs a Float32 page and a combine stage with its own ring | `num_rounds < num_cores` and `tensor_k_tiles >= 2 * splits` | `[block_h × block_w]`, K sub-range | DRAM unchanged (each cluster reads its K slice); adds `(splits−1) × M_t × N_t` fp32 partial tiles over the NoC (ResNet at 4 splits: 32 × 3 × 4 KiB = 384 KiB) | fewer combine rounds |
| **R-splitK-neos**: K split across the NEOs of one cluster, combined in L1 | **deferred**: compute-to-compute rings across NEOs do not exist (intra-tensix DFBs are packer→unpacker of one NEO, `DataflowBuffer.md` C7), so a DM-mediated local combine is needed. Reachable for the same reason as R-splitK-cores | as above | as above | no DRAM change; adds L1-local partial traffic | — |
| **R-rows-to-neos**: NEOs of a round split rows instead of columns (weight ALL, activation BLOCKED) | **deferred**: it serves narrow-N, tall-M shapes outside Phase 0 (TARGET `2048x64 + 64x64`, where N-dealt rounds pad half the NEOs). It is the same kernels with the two rings' access patterns swapped, selected by a host predicate | `col_blocks < C <= num_row_blocks` | same | weight read once per round instead of activation | same |
| **R-out-aligned**: output width-sharded and the assignment aligned to its shard grid; `cb_output_tiles` borrowed from the output tensor (`DataflowBufferSpec.borrowed_from`), so there are no writes | **deferred**: R7/R14 require any pair of grids and the accessor path serves every one. It saves only the output's NoC hop (the smallest tensor), and it pins the split to the caller's grid | `output_memory` in `{"L1/width", "L1/width/auto"}` and output shards are whole multiples of `C * block_w` columns | same | output 0 NoC hops instead of 1 | — |
| **Activation consumed in place from its width shard** | **rejected**: the width shard cuts K (dependent), so in-place consumption *is* R-splitK-cores with the caller's shard grid as the K split. Superseded by that row | — | — | — | — |
| **K-outer with partial spill** (blocks larger than DEST, partials packed to a ring and reloaded per K chunk) | **rejected**: superseded by DEST-resident accumulation, which moves zero partial bytes. A block larger than DEST buys only the SRC reuse that `block_h * block_w = dest_tiles` already captures, at a pack+unpack of every partial per chunk | — | — | — | — |
| **Activation re-streamed per output tile** | **rejected (dead end)**: re-reads `a` `N_t` times; superseded everywhere by the round's shared ALL ring | — | — | — | — |

Regime-pinned tests: R-2D-stream is the only built regime. `select_blocking` is regime-internal, but its outputs vary with the grid and the shape, so a host unit test pins its outputs for every Phase 0 INPUTS entry on a 32-core and a 2-core grid. A device test pins one shape with `block_k < tensor_k_tiles` (ResNet's, 2 chunks) and one with padded rounds (`64x96 + 96x160`).

### Traffic ranking

Counted in tile crossings per tensor (bytes = tiles × 2 KiB), no time estimate.

| Candidate split | Weight crossings (DRAM) | Activation crossings (its home: DRAM or L1 shards) | Cross-core traffic added | Verdict |
|---|---|---|---|---|
| Columns only (`block_h = M_t`) | 1× | × `ceil(N_t / (block_w C))` | none | in the built family; picked by `select_blocking` when it also fills the grid |
| Rows only | × `M_t / block_h` | 1× | none | worse whenever the weight dominates (every Phase 0 shape: weight ≥ activation) |
| **2-D rounds (built, R-2D-stream)** | × `num_row_blocks` | × `rounds_per_row_block` | none | **chosen**: the family containing both 1-D splits; occupancy picks the point in it |
| Split K + combine (R-splitK-cores) | 1× per column set | 1× per row set | `(splits−1) M_t N_t` fp32 partials | more clusters for M_t = 1, more NoC bytes; deferred |
| 2-D + two 1-D multicasts (R-mcast-weight + R-mcast-act) | 1× | 1× | multicast payloads of the same bytes | **cheapest DRAM traffic**; deferred (it layers onto the built structure) |

Within the built split, block extents change bytes: a shared operand is re-fetched per block, which is `blocking-model.md` §4's second case. `select_blocking` puts occupancy first, and the perf lamp names the traffic-minimal alternative.

### Block schedule

Per cluster, logical schedule (reader, compute and writer run asynchronously; adjacent rounds pipeline through the depth-2 rings):

```cpp
for (uint32_t round_idx = 0; round_idx < num_rounds_this_core; ++round_idx) {   // C blocks, one per NEO
    load_round_bias(round_idx);                 // weight reader: C x block_w bias tiles (has_bias only)
    preload_bias_block(round_idx);              // each NEO: DEST[r*block_w + c] = bias row of column c, broadcast
    for (uint32_t k_chunk = 0; k_chunk < num_k_chunks; ++k_chunk) {
        load_act_chunk(round_idx, k_chunk);     // act reader: [block_h x block_k], seen by all C NEOs
        load_weight_chunk(round_idx, k_chunk);  // weight reader: C blocks of [block_k x block_w]
        matmul_chunk(k_chunk);                  // each NEO: DEST += act_chunk x its weight_chunk
    }
    pack_block(round_idx);                      // each NEO: DEST -> cb_output_tiles, block_h*block_w tiles
    store_block(round_idx);                     // writers: C blocks to their output pages
}
```

| Operation | Block shape | Resident across it | Intended fixed-cost frequency |
|---|---|---|---|
| `load_round_bias` | C × `[1 × block_w]` bias tiles | — | one implicit-sync read per tile; no barriers |
| `preload_bias_block` | `[block_h × block_w]` DEST tiles from `block_w` bias tiles | DEST acquired from here to `pack_block` | `unary_bcast_init<ROW>` once per round; one `wait_front(block_w)` / `pop_front(block_w)` per round |
| `load_act_chunk` | `[block_h × block_k]` | — | one tagged read per tile; one ring entry per tile |
| `load_weight_chunk` | C × `[block_k × block_w]` | — | one tagged read per tile |
| `matmul_chunk` | `[block_h × block_w]` accumulator, `block_k` K steps of `matmul_block(ct=block_w, rt=block_h, kt=block_k)` | the block's DEST accumulator (all of K) | `matmul_block_init` once per round (with bias) or once per kernel (without); one `wait_front`/`pop_front` pair per ring per chunk |
| `pack_block` | `block_h * block_w` tiles | — | one `tile_regs_commit/wait/release`, one `reserve_back`/`push_back` per round |
| `store_block` | C × `[block_h × block_w]` | — | one tagged write per tile; the tag handed back once at kernel end |

**Stall-shadow check.** Every NEO stage waits on DRAM/L1 fetches it does not produce. What does not depend on the awaited K chunk is scheduled into that wait. The bias preload depends only on the small bias block, which the weight reader delivers first, so it runs while the first weight chunk is in flight. Packing round r (on the pack thread, out of the other DEST half) overlaps round r+1's matmul. Writes of round r overlap the reads of round r+1. The DEST accumulation over K is not reordered: chunks run in K order, so the floating-point order is fixed. Staggering each cluster's starting K chunk to spread DRAM bank load would change that order, so it is gated on the precision baseline (the `dot` bound of NUMERIC) and not done in Phase 0.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (`block_k` coarsest) | Qwen's whole K (8) is one chunk: each NEO waits for the full chunk before its first matmul, so no DM/compute overlap inside a round | `block_k = tensor_k_tiles / 2` or `/ 4`, or `WEIGHT_DEPTH = 3` |
| Grid synchronization / occupancy-first blocks | `select_blocking` picks 1×1 blocks for Qwen (16 clusters, weight read 4×) over 4×1 (4 clusters, weight read 1×, 4-tile SRC reuse; catalog `matmul_output_subblock`: 1×1 blocks forgo ~1.4× SRC reuse) | key with block area first: `(4, 1)` for Qwen, `(1, 4)` for ResNet |
| Thread plan | M_t = 1 shapes (ResNet, Llama) use each weight tile once | `(R, C, W) = (2, 1, 2)` with 4× the clusters, against `(2, 4, 2)`. Timed backend only (verifier) |
| Weight reader threads | one hart feeds the dominant byte stream (catalog `split_reader`) | 2 weight-reader harts where `2 \| block_k * block_w` |
| DEST sync | half-sync caps a block at 4 (fp32) / 8 tiles | `double_buffer_dest=False` (8 / 16 tiles), giving up pack/math overlap |

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| activation → `cb_act_tiles` | bf16 tiles | `reader_act` (1 hart, NoC0): `noc.async_read<NocOptions::TXN_ID>(acc_a, cb_act, {.page_id = (row_block*block_h + r)*tensor_k_tiles + k_tile_start + k_chunk*block_k + k}, {})`, implicit sync | the same code for interleaved DRAM, interleaved L1 and width-sharded L1 (any shard grid): the accessor resolves the page's home (GUIDANCE, R5, R14) |
| weight → `cb_weight_tiles` | bf16 tiles | `reader_weight` (1 hart, NoC0), implicit sync: per chunk, for NEO t = 0..C−1, for k, for c: page `(k_tile_start + k_chunk*block_k + k) * tensor_n_tiles + colblock(t)*block_w + c` | BLOCKED consumer: NEO t's `[block_k × block_w]` block is contiguous, row-major |
| bias → `cb_bias_tiles` | bf16 tiles | `reader_weight`, before the round's first weight chunk: for t, for c: page `colblock(t)*block_w + c` | only bound when `has_bias` |
| `cb_act_tiles` + `cb_weight_tiles` → DEST | SrcB ← act, SrcA ← weight | `matmul_block` (raw compute API), `compute_kernel_hw_startup<SrcOrder::Reverse>` | accumulator fp32 (`enable_32_bit_dest`) or 16-bit |
| `cb_bias_tiles` → DEST | row 0 broadcast down 32 rows | `unary_bcast<BroadcastType::ROW>` (B2D datacopy) before the matmul | exact copy, no multiply: fidelity-independent |
| DEST → `cb_output_tiles` | bf16 | `pack_tile` into the STRIDED output ring | pack ring stride is handled by the pack LLK (#56194) |
| `cb_output_tiles` → result | bf16 tiles | `writer` (2 harts, NoC1), implicit sync: page `(row_block*block_h + j / block_w) * tensor_n_tiles + colblock(neo)*block_w + j % block_w` | the same code for DRAM interleaved and width-sharded L1 (any grid, R7, R14) |

`colblock(t) = col_round * C + t` if that is `< col_blocks`, else `col_round * C`. A NEO without a real block in the last round of a row block recomputes NEO 0's block, and the writer stores the same bytes to the same pages again. `row_block = round / rounds_per_row_block`, `col_round = round % rounds_per_row_block`.

**TARGET placement values as views of the scheme:**

| TARGET value | Cut axis (character) | Class | Phase 0 realization |
|---|---|---|---|
| `memory_config: DRAM/L1, sharding: interleaved` (activation) | — | — | accessor reads |
| `sharding: width` (activation) | K (dependent) | consuming it in place is a **scheme-change** (= R-splitK-cores) | accessor reads over the NoC: placement only, no algorithm change |
| `output_memory: L1/width`, `L1/width/auto` | N (independent) | aligned zero-copy is a **knob-turn** (R-out-aligned, deferred) | accessor writes over the NoC to whichever cluster holds the shard |

Deferred inter-cluster contracts (for the deferred rows, so the structure stays reachable). R-mcast-*: a host `ttnn.Mcast` channel per operand. The injector reads a chunk into its ring entries and `McastArgs::sender(noc).send()`s it to receivers whose reader becomes `receiver(noc).receive()` into the same ring layout. Compute does not change. R-splitK-cores: each split cluster packs a Float32 partial block into a `cb_partial` ring. A gather cluster receives `splits − 1` partials (semaphore-gated unicast) and adds them into DEST before the bias and the final pack.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a round: C blocks of `[block_h × block_w]` output tiles with the same rows, each with its full K |
| Grid | `device.compute_with_storage_grid_size()` (32 on this target); `split_work_to_cores(grid, num_rounds, row_wise=True)` → `core_set`, two groups with `rounds_per_core_1/2`; `round_start` assigned contiguously. Only the engaged clusters are in the `WorkUnitSpec` |
| Per-core work | `num_rounds_this_core` rounds from `round_start` (RT args to all four kernels); `k_tile_start = 0`, `k_tiles = tensor_k_tiles` (RT, kept for split-K reachability) |
| Remainder | clusters: the second `split_work_to_cores` group takes one round fewer. NEOs: padded slots only in the last round of each row block (≤ C−1 per row block, independent of cluster count). Tiles: every count is `ceil` and per image (`tensor_m_tiles = batch * ceil(M/32)`, `tensor_n_tiles = ceil(N/32)`, `tensor_k_tiles = ceil(K/32)`) |
| Bound | **compute**: each weight tile (one 2 KiB transfer) feeds `block_h` tile-matmuls and each activation tile feeds up to `C * block_w`, every tile-matmul being 4 fidelity passes at the default HiFi4. (M_t = 1 shapes: the thread-plan lamp) |
| Thread plan | `THREAD_PLAN = (act_readers=1, weight_readers=1, compute=4, writers=2)`, i.e. page-form `(R, C, W) = (2, 4, 2)`, one host constant. R is two kernels of one hart each: a matmul operand ring must have one producer thread (ALL capacity is split per producer, `DataflowBuffer.md:274`; contiguous indexing is required, see Mechanism caps). W = 2 divides C; R + W = 4 ≤ 6. Counts reach the kernels as CT args (`compute_threads`, `writer_threads`) |
| Unit dealt to compute threads | a block, larger than a ring entry. Each NEO takes exactly `num_k_chunks` blocks of `cb_weight_tiles`, `num_k_chunks * block_h * block_k` entries of `cb_act_tiles` (ALL), `block_w` of `cb_bias_tiles`, and pushes `block_h * block_w` to `cb_output_tiles`, all per round. The last round of each row block is padded (recompute NEO 0's block); the host deals whole rounds to clusters |
| Tests the plan needs | (1) **every NEO ran**: `1x1x32x64 + 1x1x64x128` (Llama), where the 4 column tiles belong to 4 distinct NEOs of one cluster, checked column by column, plus a `(2, 1, 2)` vs `(2, 4, 2)` run of the same call giving the same result. A missing NEO leaves its column unwritten or hangs the ring. (2) **one device session**: the op after another program (a first `linear` call) and before `ttnn.to_layout(out, ttnn.ROW_MAJOR_LAYOUT)`, in one session, with a bias and with a width-sharded result. (3) The ALL/BLOCKED rings at C = 1 and C = 4 before anything else (the page: matmul has not been run under a plan) |

## Circular Buffers

Rings (`DataflowBufferSpec`), entry = one tile, `tile_bytes = a.buffer_page_size()` (2048 B bf16).

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Sync | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|------|-----------|
| `cb_act_tiles` | 0 | tile_bytes | `ACT_DEPTH * block_h * block_k` | spans rows (`block_h`) and K (`block_k`); streams over K chunks; shared by the round's columns (ALL: not × C) | bf16 (dtype of `a`) | `reader_act` (STRIDED, 1 thread) | `compute` (**ALL**, C threads) | producer: implicit (tagged reads); consumer: compute wait/pop | whole kernel, per chunk |
| `cb_weight_tiles` | 1 | tile_bytes | `WEIGHT_DEPTH * C * block_k * block_w` | spans K (`block_k`) and columns (`block_w`) per NEO, × C; streams over K chunks and rows | bf16 | `reader_weight` (STRIDED, 1 thread) | `compute` (**BLOCKED**, `block_size = block_k * block_w`) | producer: implicit; consumer: compute wait/pop of one block | per chunk |
| `cb_bias_tiles` | 2 | tile_bytes | `BIAS_DEPTH * C * block_w` (bound only when `has_bias`) | spans columns (`block_w`) per NEO, × C; nothing else | bf16 | `reader_weight` | `compute` (**BLOCKED**, `block_size = block_w`) | producer: implicit; consumer: compute wait/pop | per round |
| `cb_output_tiles` | 3 | tile_bytes | `OUT_DEPTH * C * block_h * block_w` | spans rows × columns of one block per NEO, × C; K is folded in DEST | bf16 (output dtype = `a`'s, R9) | `compute` (STRIDED, C threads) | `writer` (STRIDED, 2 threads) | producer: compute reserve/push; consumer: implicit (tagged writes) | per round |

No intermediate ring: the product packs straight into `cb_output_tiles` (L1 discipline Rule 3.1) and the bias is folded into the accumulator (Rule 3.4).

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_round_bias` | C × `block_w` | no (dataflow) | bias tensor | `cb_bias_tiles`, C·`block_w` entries | C blocks posted |
| 2 | `preload_bias_block` | `[block_h × block_w]` DEST | raw | `cb_bias_tiles`, wait `block_w` | DEST 0..`block_h*block_w`−1 | popped `block_w`; DEST held |
| 3 | `load_act_chunk` | `[block_h × block_k]` | no (dataflow) | activation tensor | `cb_act_tiles`, `block_h*block_k` entries | seen by all C NEOs |
| 4 | `load_weight_chunk` | C × `[block_k × block_w]` | no (dataflow) | weight tensor | `cb_weight_tiles`, C blocks | one block per NEO |
| 5 | `matmul_chunk` | DEST `[block_h × block_w]` += `[block_h × block_k] × [block_k × block_w]` | raw | `cb_act_tiles` wait `block_h*block_k`; `cb_weight_tiles` wait `block_k*block_w` | DEST | both popped after the chunk's `block_k` `matmul_block` calls |
| 6 | `pack_block` | `block_h*block_w` tiles | raw | DEST | `cb_output_tiles`, reserve/push `block_h*block_w` | DEST released (Quasar's pack zeroes the half, `llk_pack_common.h:144-151`) |
| 7 | `store_block` | C × `[block_h × block_w]` | no (dataflow) | `cb_output_tiles` | result tensor | drained |

Without a bias, operation 2 is absent and the matmul accumulates onto the zeroed DEST half.

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| (boot) | raw_api | `compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act, cb_weight, cb_out)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:59-60` | matmul source order | — | — | — |
| `matmul_chunk` (init) | raw_api | `matmul_block_init(cb_act, cb_weight, 0, block_w, block_h, block_k)` | `tt_metal/hw/inc/api/compute/matmul.h:220-248` | transpose 0 | — | — | `ct_dim = block_w`, `rt_dim = block_h`, `kt_dim = block_k` |
| `matmul_chunk` | raw_api | `matmul_block(cb_act, cb_weight, k, k*block_w, 0, 0, block_w, block_h, block_k)` for k < `block_k` | `matmul.h:274-302` (Quasar branch 293-301) | `idst = 0` (Quasar) | `cb_act_tiles`, `cb_weight_tiles` | DEST | same three |
| `preload_bias_block` | raw_api | `unary_bcast_init<BroadcastType::ROW>(cb_bias)`; `unary_bcast<BroadcastType::ROW>(cb_bias, c, r*block_w + c)` | `tt_metal/hw/inc/api/compute/bcast.h:37-79, 82-118` (Quasar branches 64-78, 103-117) | bf16 only (32-bit asserts, `bcast.h:71,109`) | `cb_bias_tiles` | DEST | `block_h`, `block_w` |
| `pack_block` | raw_api | `tile_regs_commit/wait`; `pack_tile(i, cb_out, i)`; `tile_regs_release` | `tt_metal/hw/inc/api/compute/pack.h` | `pack_init(cb_out)` once after hw_startup (the output ring never changes) | DEST | `cb_output_tiles` | `block_h*block_w` |
| `load_*`, `store_block` | raw_api (dataflow) | `noc.async_read/async_write<NocOptions::TXN_ID>` + `TensorAccessor`; last statement `noc_async_{read,write}_set_trid(0, noc.get_noc_id())` | pattern: `.claude/references/quasar_op_examples/toy_row_passes/kernels/reader.cpp`, `writer.cpp` | implicit sync | tensors / `cb_output_tiles` | rings / tensor | page-id loops in `block_h`, `block_w`, `block_k`, C |
| host rings | host | `E.all_consumer_of`, `E.blocked_consumer_of(name, acc, block_size)`, `E.producer_of`, `E.consumer_of` | `ttnn/cpp/ttnn-nanobind/program_spec.cpp:798-835` | access patterns above | — | — | `block_size = block_k*block_w` / `block_w` |
| host compute config | host | `E.ComputeHardwareConfig(fpu_math_fidelity=…, enable_32_bit_dest=fp32_dest_acc_en, double_buffer_dest=True, sfpu_precision_mode=Approximate if math_approx_mode else Precise)` | `program_spec.cpp:522-555`; fields `tt_metal/api/tt-metalium/workload/specification/compute_hardware_config.hpp:43-87` | fidelity always explicit | — | — | `enable_32_bit_dest` sets `dest_tiles` |
| host split | host | `ttnn.split_work_to_cores(grid, num_rounds, row_wise=True)`, `ttnn.corerange_to_cores` | as used in `toy_row_passes_program_artifacts.py:split_rows` | — | — | — | `num_rounds` from the block knobs |

**Helpers considered and rejected:**

| Raw entry | Candidate helper | File:line of the mismatch | Reason |
|---|---|---|---|
| `matmul_chunk` | `matmul_block()` (`matmul_block_helpers.hpp`) | — | not used: the matmul is written with the raw compute API |
| `preload_bias_block` | `add_bias_bcast_rows()` (`bias_add_helpers.hpp`) | `bias_add_helpers.hpp:13-14` | includes `matmul_block_helpers.hpp`, which is not used |
| `preload_bias_block` | eltwise `add<…, input(cb_bias, BroadcastDim::Row …)>` / `eltwise_chain` | `ttnn/cpp/ttnn/kernel_lib/eltwise/core/chain.inl:3366,3378,3389` (the chain acquires DEST itself); `eltwise/api/chain.hpp:579-582` (`DestReuseBinary` has no `BroadcastDim`) | a chain cannot add to an accumulator a raw matmul left in DEST, and its DEST-reuse element cannot broadcast a row. Using the chain would need the product packed to an extra ring and re-read: the intermediate Rule 3.4 removes |
| `pack_block` | `PackTile` chain element | `chain.inl:3366-3389` | same DEST ownership: the chain packs only what it computed |
| `load_*` / `store_block` | `mcast_args.hpp`, `dfb_helpers_dataflow.hpp` | `mcast/kernel/mcast_args.hpp:16-37` (a multicast channel: no multicast in the built regime); `dfb_helpers_dataflow.hpp:9` (tile-dimension queries, not transfers) | neither moves a page from an accessor into a ring; the target page mandates implicit-sync tagged transfers for NoC-filled rings |

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| `preload_bias_block` | DEST ← bias (datacopy, then the matmul adds onto it) | DEST (written, not read) | `cb_bias_tiles`: Row0 (`[N]` / `[1,…,1,N]` in TILE layout: row 0 of each tile holds the values) | ROW (row 0 replicated to all 32 rows) |
| `matmul_chunk` | DEST += act × weight | `cb_act_tiles`: All | `cb_weight_tiles`: All | None |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| ALL / BLOCKED rings with a Tensix consumer are not exercised by the pipeline's examples (README *Not covered*) | the whole thread plan rests on them | the first test runs the Llama call at C = 1 and C = 4 (`THREAD_PLAN` turned on the host only); if a pattern fails, fall back to C = 1 with STRIDED rings (correct because P = C = 1 gives stride 1) |
| STRIDED rings with indexed matmul | contiguous indexing is required (`llk_unpack_AB_matmul_api.h:192,194`) | the access patterns in the CB table; never bind a matmul operand STRIDED with more than one thread on either side |
| Host rejects an implicit-sync txn-id count on the STRIDED→BLOCKED/ALL rings | `DataflowBuffer.md:431` | capacities are multiples of `C * block_size`; else depth 3, else explicit sync on that reader with every transfer tagged (`{.trid = 0}`) and the ring in `disable_dfb_implicit_sync_for` |
| Bias preload then matmul in one DEST window | two MOP configurations per round; an unpack-config mismatch would corrupt silently | re-init `matmul_block_init` after the preload every round; test bias cells with `block_k < tensor_k_tiles` (ResNet) and C = 4. Fallback if the fold fails on ttsim: matmul packs to a `cb_product` ring (Float32 page under fp32 DEST) and eltwise `add<…, BroadcastDim::Row>` writes `cb_output_tiles` (call `pack_init` between the two targets); the ledger gains that ring |
| NoC tag after implicit sync | implicit-sync kernels leave a ring's transaction id in the hart's tag register | every DM kernel ends with `noc_async_{read,write}_set_trid(0, noc.get_noc_id())`; the one-session test |
| Width-sharded activation read through its accessor | the README lists sharded inputs as not covered | a probe of a width-sharded-activation cell (ResNet's call, 2-core shard) before the full suite; the accessor gets the tensor's spec through `TensorParameter`, so no per-placement code is written |
| Padded NEOs write the same page twice from two writer harts | duplicate writes of identical bytes | harmless (same bytes); never skip the write (implicit sync needs every entry drained) |
| `L1/width/auto` shard spec | R7 needs all rows and whole-tile column runs | use `eval.memory.place(...)` exactly; the check compares buffer and sharding |
| Default compute config drift | R2 is linted | one `default_compute_kernel_config()`; `None` resolves only through it |
| Later refinements: K not tile-aligned | padding inside the K sum (`alignment: k_non_aligned`) | not in Phase 0; the refinement must zero (mask) the activation's last K-tile columns, or the weight's last K-tile rows, in the reader or compute. Tile counts already use `ceil` |
| Later refinement: float32 | `unary_bcast` asserts on 32-bit (`bcast.h:71,109`); SrcA/B carry 19 bits | the float32 refinement needs another bias path (the raw SFPU `sfpu_add_bcast_row`) and states the matmul precision |

**Structural impossibilities:** none to add to the specification's INVALID.
