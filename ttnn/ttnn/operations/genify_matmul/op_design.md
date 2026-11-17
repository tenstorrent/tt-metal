# Operation Design: genify_matmul

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (tiled matmul + per-element epilogue, one device program) |
| Goal | `out = epilogue(A @ B)` replacing the stock matmuls of Qwen3.8-27B, FLUX.2-dev and Panoptic-DeepLab on Blackhole. There is one public entry point. The op picks its own work split, blocking and data movement from the shapes, dtypes, placement and device grid, and runs one `ttnn.generic_op` dispatch per call. |
| Math | `Y[..., m, n] = Σ_k A[..., m, k] · B[(...,) k, n]`. Then: none → `Y`. bias → `Y + bias[0, n]`. bias_relu → `relu(Y + bias[0, n])`. swiglu → `out[..., m, 32j+c] = silu(Y[..., m, 64j+c]) · Y[..., m, 64j+32+c]`. addcmul → `residual[..., m, n] + gate[0, n] · Y[..., m, n]` |
| References | `.claude/references/blocking-model.md`, `.claude/references/l1-footprint-discipline.md`, `ttnn/cpp/ttnn/kernel_lib/matmul_block_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/bias_add_helpers.hpp`, `ttnn/cpp/ttnn/kernel_lib/eltwise/api/chain.hpp`, `ttnn/cpp/ttnn/kernel_lib/mcast/` (successor regimes), `eval/golden_tests/genify_matmul/{feature_spec,helpers}.py`, catalog `ttnn/ttnn/operations/examples/{matmul_output_subblock,mcast_topology,shared_input_reuse,double_buffer,trid_double_issue,noc_placement,split_reader,read_coalescing,bank_stagger,compute_block_size,compute_fusion,width_split}/README.md` |

### Catalog entries that set knobs

| Entry | Finding used | Knob it sets |
|-------|--------------|--------------|
| `matmul_output_subblock` | 8-tile subblocks run 1.46× faster than 1×1. Under fp32 DEST (4 tiles) a 1×4 or 2×2 subblock gets about 1.40×. Only the area matters, not the shape. | `core_n_tiles` quantum `q_n` = 4, so a 4-tile subblock always exists (see the Work-split rule) |
| `double_buffer` | Throughput is flat from 4 to 64 tiles per batch at buffer depth 2×block. 1-tile blocks are bad. | `depth_act` = `depth_weight` = 2. `block_k_floor` = 8. |
| `trid_double_issue` | When DRAM is the limit, a depth of 3–4 blocks in flight pays; ×2 only ties the baseline. | Perf lamp on `depth_weight` (stream / dram_sharded) |
| `compute_block_size` | Per-pass fixed cost is about 320 ns per phase. Coarser blocks give up to 1.65×. | `block_m_tiles` × `block_n_tiles` = the whole per-core region by default |
| `mcast_topology` | A 2-D split needs A multicast along rows and B down columns: 1.9× on BH. | Deferred successor `interleaved_2d_mcast` |
| `shared_input_reuse` | One injector plus a broadcast: 1.7×. Two-chunk input CB. | Deferred successors `interleaved_stream_mcast_a`, `height_sharded_mcast_b` |
| `noc_placement` | Use row-wise core lines (`row_wise=True`), reads on NoC0 and writes on NoC1. | Core enumeration order; reader/writer NoC |
| `split_reader` | A second reader RISC helps only when one DM RISC is issue-bound. That gives 1.2× at 1–2 KB tiles. | Perf lamp (B reads on the writer RISC) |
| `read_coalescing`, `bank_stagger` | A width-sharded DRAM bank gives contiguous rows, so one call per block saves 3.2×. Staggering the start bank gives 1.0–1.3×. | dram_sharded reader (coalesced bank reads); perf lamp (K-walk stagger) |
| `compute_fusion` | Fuse SFPU epilogues on the matmul output. Do not use DEST-reuse ahead of an FPU op. | swiglu uses SFPU in the chain; addcmul FPU-mul then SFPU-add |
| `width_split` | When Mt is small, split along N. | `grid_m = 1` factorization for M ≤ 32 |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` (A) | ttnn.Tensor | yes | rank 2–4, TILE, (..., M, K) | — | addresses RT |
| `weight` (B) | ttnn.Tensor | yes | (K, N), or (..., K, N) with leading dims == A's | — | address RT |
| `bias` | ttnn.Tensor | no | (1, N) (padded to one tile row) | None | address RT; presence CT |
| `activation` | str | no | None, "relu" (needs bias), "swiglu" (no bias/residual) | None | epilogue CT |
| `residual` | ttnn.Tensor | no | (..., M, N), same leading dims as A. Must be given together with `gate`. | None | address RT |
| `gate` | ttnn.Tensor | no | (1, N). Must be given together with `residual`. | None | address RT |
| `dtype` | ttnn.DataType | no | bf16 / bf8b / fp32 | A's dtype | output CB format CT |
| `memory_config` | ttnn.MemoryConfig | no | see Output | DRAM interleaved | — |
| `compute_kernel_config` | ttnn.ComputeConfigDescriptor | no | any fidelity / approx; fp32_dest_acc_en bool | `default_compute_kernel_config()` = (HiFi4, fp32_dest_acc_en=True, math_approx_mode=False) | compute kernel config |

**Exports from `ttnn.operations.genify_matmul`:** `genify_matmul`, `default_compute_kernel_config`, `INPUT_TAGGERS`, `SUPPORTED`, `EXCLUSIONS`, `validate`. `eval/golden_tests/genify_matmul/axes.py` imports `INPUT_TAGGERS` and `default_compute_kernel_config`. Passing `None` resolves through that one factory, both in the op and in the golden tagger.

### Validation order (entry point)

| Step | Check | Raises |
|------|-------|--------|
| 1 | Contract: A rank < 2; `A[-1] != B[-2]`; B neither 2-D nor (rank == A rank and B[:-2] == A[:-2]); activation ∉ {None, "relu", "swiglu"}; swiglu with `N % 64 != 0`; bias shape ≠ (1, N); gate shape ≠ (1, N); residual shape ≠ A[:-1] + [N]; residual without gate or gate without residual | `ValueError` |
| 2 | `validate()`: build the axes (dtype, weight_dtype, layout, fp32_dest_acc_en, placement, epilogue + `INPUT_TAGGERS`), then the SUPPORTED per-axis check, then EXCLUSIONS | `NotImplementedError` (`UnsupportedAxisValue` / `ExcludedCell`) |

The contract checks run **before** the support gate. A call such as `activation="gelu"` or swiglu with N = 96 must raise `ValueError`, not `NotImplementedError`. Epilogue classification matches `axes.py::classify_epilogue`. Any kwarg combination that `classify_epilogue` maps to `"other"` (bias + swiglu, relu without bias, bias + residual, and so on) is a contract violation → `ValueError`.

### Registry axes (names exactly as in feature_spec TARGET)

| Axis | Derived from | Phase 0 SUPPORTED |
|------|--------------|-------------------|
| `dtype` | `input_tensor.dtype` | `[bfloat16]` |
| `weight_dtype` | `weight.dtype` | `[bfloat16]` |
| `layout` | `input_tensor.layout` | `[TILE_LAYOUT]` |
| `fp32_dest_acc_en` | resolved compute config | `[True]` |
| `placement` | `dram_sharded` if B is DRAM WIDTH_SHARDED, else A's memory layout (`interleaved` / `height_sharded` / `block_sharded`) | `["interleaved"]` |
| `epilogue` | the kwargs (none / bias / bias_relu / swiglu / addcmul) | `["none"]` |
| `alignment` | `tag_alignment((A_shape, B_shape), axes)`: K = A[-1], N = B[-1], M = A[-2]. Returns `"tile_aligned"` if all three are % 32 == 0; otherwise the first non-aligned in order K, N, M → `"k_non_aligned"` / `"n_non_aligned"` / `"m_non_aligned"` | `["tile_aligned"]` |
| `weight_batch` | `tag_weight_batch`: `"batched"` if any of B[:-2] > 1, else `"single"` | `["single"]` |

| EXCLUSIONS (op-side, permanent or for now) | Reason |
|---|---|
| `{"dtype": ttnn.float32, "fp32_dest_acc_en": False}` | Precision convention (task rule). Permanent. |
| `{"placement": "dram_sharded", "weight_batch": "batched"}` | No model uses it (task rule allows it). Permanent. |

`INVALID = []` (feature_spec). **Structural impossibilities:** none to add.

## Tensors

### Input

| Property | A (`input_tensor`) | B (`weight`) | bias / gate | residual |
|----------|----|----|----|----|
| Shape | (..., M, K), rank 2–4 | (K, N) or (..., K, N) | (1, N) | (..., M, N) |
| Dtype | bf16 / bf8b / fp32 (Phase 0: bf16) | bf16 / bf8b / bf4b (Phase 0: bf16) | bias: weight dtype or bf8b; gate: output dtype | output dtype |
| Layout | TILE | TILE | TILE | TILE |
| Memory | interleaved DRAM/L1; L1 height / block / width sharded | interleaved DRAM/L1; DRAM width-sharded (dram_sharded) | interleaved | interleaved |

### Output

| Property | Value |
|----------|-------|
| Shape | A[:-1] + [N_out], with N_out = N/2 for swiglu, otherwise N |
| Dtype | `dtype`, or A's dtype when `dtype` is None |
| Layout | TILE |
| Memory | `memory_config` (default DRAM interleaved). height/block_sharded: exactly the requested shard spec. dram_sharded (`L1_WIDTH_SHARDED_MEMORY_CONFIG` with no spec): the op builds `MemoryConfig(WIDTH_SHARDED, L1, ShardSpec(worker_cores, [M_pad_total, out_shard_w_tiles·32], ROW_MAJOR))` |

Allocation: `ttnn.allocate_tensor_on_device(ttnn.Shape(out_shape), out_dtype, TILE_LAYOUT, device, out_mc)` (positional args). Dispatch: `ttnn.generic_op([A, B, <bias/gate/residual present>, out], program_descriptor)`, with the output last. This is exactly one dispatch.

---

## Blocking Model

### Tile geometry (alignment-aware from Phase 0)

| Symbol | Definition |
|--------|-----------|
| `lead` | `A.shape[:-2]` (empty for rank 2) |
| `img_m_tiles` | `ceil(M / 32)`, per image (each image is tile-padded on its own) |
| `tensor_k_tiles` | `ceil(K / 32)` |
| `tensor_n_tiles` | `ceil(N / 32)` (B's N); `out_n_tiles = tensor_n_tiles / 2` for swiglu, otherwise `tensor_n_tiles` |
| `tensor_batches` | `prod(lead)` if weight_batch == batched, otherwise **1** (the leading dims fold into M) |
| `tensor_m_tiles` | `prod(lead) · img_m_tiles` if single B (folded), otherwise `img_m_tiles` |
| page indices | A: `((b·tensor_m_tiles + r)·tensor_k_tiles + k)`. B: `(b·weight_batch_stride + k·tensor_n_tiles + n)` with `weight_batch_stride = tensor_k_tiles·tensor_n_tiles` (batched) or 0 (single). Out: `((b·tensor_m_tiles + r)·out_n_tiles + n_out)`. Folding is exact because every image is padded separately. |

### Axes

| Axis | Character (+ one-clause reason) | Extent knob | Phase 0 value | Knob source | Core-assignment | Later unlock |
|------|--------------------------------|-------------|---------------|-------------|-----------------|--------------|
| batch (`b`) | **independent**: output batch b depends only on A[b] and B[b]. Under a **single** B it is folded into `m`, because B does not vary along it. | `block_batches` | `core_batches` (the whole per-core batch range). Phase 0 has `tensor_batches = 1`, so the value is 1 because the extent is 1, not because the knob is collapsed. | host `choose_blocking()` → compute CT arg `block_batches` (= `MatmulBlockShape.batch`) | logical grid dim `grid_b`, from the factorization below | knob-turn (batched B: `grid_b > 1`, `weight_batch_stride` RT arg) |
| m (rows of A / out) | **independent**: every output row uses only its own A row. **Reuse-shared w.r.t. an N-split**: B does not vary along m, so every core row re-reads the same B columns. | `block_m_tiles` | `core_m_tiles` (whole per-core range), unless the L1 solve shrinks it | host `choose_blocking()` → CT `block_m_tiles` | logical grid dim `grid_m` | knob-turn (extent). The B column-mcast is a scheme-change (`interleaved_2d_mcast`). |
| n (cols of B / out) | **independent**: every output column uses only its own B column. **Reuse-shared w.r.t. an M-split**: A does not vary along n. | `block_n_tiles` | `core_n_tiles` (whole per-core range, quantized to `q_n`), unless the L1 solve shrinks it | host `choose_blocking()` → CT `block_n_tiles` | logical grid dim `grid_n` | knob-turn (extent). The A row-mcast is a scheme-change. |
| k (reduction) | **dependent**: every output element sums over all of K. | `block_k_tiles` (+ derived `num_k_blocks = ceil(tensor_k_tiles / block_k_tiles_max)`, `block_k_tiles = ceil(tensor_k_tiles / num_k_blocks)`, `last_block_k_tiles = tensor_k_tiles − (num_k_blocks−1)·block_k_tiles`) | `tensor_k_tiles` (one K-block, resident) if it fits. Otherwise the largest that fits with `block_k_tiles ≥ block_k_floor = min(tensor_k_tiles, 8)`. | host `choose_blocking()` → CT `block_k_tiles`, `num_k_blocks`, `last_block_k_tiles` | **not split across cores**: one core owns all of K for each of its output tiles. This is a decision: Phase 0 keeps the result deterministic with no cross-core combine. | scheme-change (`interleaved_split_k`, deferred) |
| out-subblock (inside a block, owned by the helper) | not a blocking axis. This is the helper's DEST decomposition. | derived `sb_h`, `sb_w` | the largest `sb_h·sb_w ≤ dest_limit` with `sb_h ∣ block_m_tiles`, `sb_w ∣ block_n_tiles`; ties go to larger `sb_w` | host derivation from `block_*` + `fp32_dest_acc_en` (`dest_limit = 4` if fp32 DEST else `8`, matching `DEST_AUTO_LIMIT`) | — | none (tuning) |

**Intermediate data re-tabled.** The partial-sum block and the epilogue output have the same (m, n) axes as the output. Both are produced and consumed **on the core that owns the block**. No stage is centralised on one core, and no core acts only as a coordinator. The swiglu epilogue halves n (`block_n_tiles/2` output columns). That axis is still independent and is owned by the same core.

### Work-split rule (core-assignment knob, host `choose_split()`)

`C = grid.x · grid.y`, with `grid = device.compute_with_storage_grid_size()`. The grid is never hardcoded.

Candidates `(grid_b, grid_m, grid_n)` must satisfy `grid_b·grid_m·grid_n ≤ C`, `grid_b ≤ tensor_batches`, `grid_m ≤ tensor_m_tiles` and `grid_n ≤ tensor_n_tiles`. When `grid_b == 1`, either `(grid_m ≤ grid.y and grid_n ≤ grid.x)` or `(grid_m ≤ grid.x and grid_n ≤ grid.y)` must hold, so the logical rectangle maps onto a physical rectangle; this keeps the mcast successors reachable. As an exception, `grid_m == 1` or `grid_n == 1` may use up to `C` cores as a flat list.

For each candidate:
- `core_batches = ceil(tensor_batches / grid_b)`
- `core_m_tiles = ceil(tensor_m_tiles / grid_m)`
- `core_n_tiles = round_up(ceil(tensor_n_tiles / grid_n), q_n)`, with `q_n = min(4, pow2_floor(ceil(tensor_n_tiles/grid_n)))`, forced to `≥ 2` for swiglu (whole gate/up pairs per core; N % 64 == 0 makes `tensor_n_tiles` even)
- the used grid is then `grid_x' = ceil(extent / core_extent)` for each axis

Selection is lexicographic min of:
1. `W = core_batches·core_m_tiles·core_n_tiles`: per-core output tiles, i.e. the critical path, since each tile costs `tensor_k_tiles` matmul steps.
2. `R = core_batches·tensor_k_tiles·(core_m_tiles·a_tile_bytes + core_n_tiles·b_tile_bytes)`: per-core DRAM read bytes. Summed over cores this is the aggregate DRAM traffic.
3. Used core count.

> **Verifier amendment (Phase 0 verification).** The strict `(W, R, cores)` order is replaced, because R1 has no cross-core operand reuse and is DRAM-bound on every measured shape (HiFi2 ≈ HiFi4 device time). Ranking W strictly first picked a 1×108 N-only split for M=9472, K=5120, N=3456 (W 296 vs 324 for the 10×9 rectangle): 1×1 subblocks and every core reading all of A, 10.5 GB of DRAM reads instead of 1.3 GB, 4.6× slower. Implemented rule (`choose_split()`): keep the candidates with `W ≤ (1 + GRID_W_SLACK)·W_min` (`GRID_W_SLACK = 0.25`, a named knob), run `choose_blocking()` on each, and take the lexicographic min of **(aggregate A+B read bytes over the used cores *including* the block re-reads `nb_n`/`nb_m` the L1 solve adds, W, used cores)**. Blocking-unaware traffic was tried first and regressed 2048×9216×6144 by 1.2× (the S3 shrink doubled A re-reads), so the blocking is part of the score. Measured on the 14-shape guard set (`tests/.../genify_matmul/perf/`): no regression beyond ±1% noise; 0.21–0.46× device time on the shapes the old rule mis-split. Across the golden suite the geomean is 0.95. The worst regression is +7% (`test_matmul_with_core_grid[7]`, a small-K shape whose W sits at the cap); it is a `GRID_W_SLACK` tuning point for the perf refinements.

> **Refinement 11 amendment (split cost model).** The W-slack filter above never let a split trade per-core work for traffic or injector load: FLUX proj_out 2048×3072×128 stayed on a 64×1 R1 split that re-read all of B on every core (161 µs), text-encoder qkv on an R1b split capped by its one A injector (233 µs), the FLUX single blocks on 9×9 / 9×10 splits doing 12–18% more work per core than 11×10 / 11×9. `choose_split()` now (`SPLIT_COST_MODEL = True`) blocks **every** candidate and ranks by `split_cost` = max(padded K-walk, (operand reads + output writes) / DRAM rate, busiest core's reads / one core's read rate, busiest core's writes / one core's write rate), then traffic, W, cores. The K-walk counts the padded nominal blocks every core walks × K × the tile-K cost (`FPU_CYCLES_PER_TILE_K` × fidelity passes, or the subblock's operand unpacks when they bind); the math fidelity is now part of `Geometry`. Candidates are visited in W order and the walk stops once W alone exceeds the best cost. Candidates also use the epilogue's own N quantum `q_n_min` beside the 4-tile subblock quantum, and the S3 block search ranks by the same cost (padding vs re-reads). The legacy ranking stays reachable (`SPLIT_COST_MODEL = False`). Guard set (140 cells): no cell slower beyond noise; e.g. 4096³ −16% / −11%, 9472×5120×3456 −8%, FLUX addcmul 2048×6144×3072 −8% / −15%.

Cores are enumerated row-wise (`ttnn.num_cores_to_corerangeset(n, grid, row_wise=True)` / `ttnn.grid_to_cores(..., row_wise=True)`, per `noc_placement`). The logical index is `(b, i, j)`. For a rectangle placement core `(x=j, y=i)` owns region `(i, j)`, transposed if the transposed orientation was chosen. Otherwise core `p = (b·grid_m' + i)·grid_n' + j` in row-wise order. Ragged edge regions are narrower in real tiles; their compute still runs the **nominal** block shape (see Key Risks).

### L1 solve (extent knob values, host `choose_blocking()`)

**The inventory comes first** (see the Circular Buffers section and `l1_ledger.md`). Phase 0 has four CBs: two streams, the K-accumulator (only when `num_k_blocks > 1`), and the output row stream. Each is irreducible:
- The two input streams are separate operands.
- The accumulator cannot share with the output: Float32 vs the output format, and both are live together on the last K-block.
- The output already packs into the destination stream.

A solve is still unavoidable. The op's extents span `tensor_k_tiles` = 4..512 and per-core regions from 1 to ~200 tiles, so no fixed extent is both legal and coarse. The solve picks extents **coarsest-first** against the footprint `F` (closed form in `l1_ledger.md`) and the budget

`l1_cb_budget = ttnn.get_max_worker_l1_unreserved_size() − l1_tensor_bytes_per_core − l1_reserve_bytes`

> **Implemented (implementer + verifier).** `get_max_worker_l1_unreserved_size()` counts from below the CB base and over-states the room. The code takes the L1 memory view instead: `l1_cb_budget = min(lowest allocated L1 block offset, bank − l1_tensor_bytes_per_core) − l1_reserve_bytes`. The **lowest allocated block** matters (verifier fix). L1 tensors are placed top-down, and a freed tensor above a live one leaves a hole. `bank − allocated bytes` then over-states the room by the hole, and program.cpp throws "circular buffers … clash with L1 buffers". This is reproduced in `test_genify_matmul_extended.py::test_fragmented_l1`.

Here `l1_tensor_bytes_per_core` = Σ over this op's L1-resident tensors: an interleaved-L1 tensor costs `ceil(pages / C)·aligned_page_size`, a sharded tensor costs its shard bytes. `l1_reserve_bytes` is a named knob with Phase 0 value 0.

| Step | Rule |
|------|------|
| S1 | `block_batches = core_batches` (costs no L1: the helper's batch loop reuses the per-batch CBs). `block_m_tiles = core_m_tiles`, `block_n_tiles = core_n_tiles`, `block_k_tiles = tensor_k_tiles` (`num_k_blocks = 1`, no `cb_partials` for epilogue none). Accept if `F ≤ budget`. |
| S2 | Keep the whole per-core region. Enable `cb_partials`. `block_k_tiles_max` = the largest `bk` with `F(core_m, core_n, bk) ≤ budget`. Accept if `block_k_tiles_max ≥ block_k_floor`. Then `num_k_blocks = ceil(tensor_k_tiles / block_k_tiles_max)` and `block_k_tiles = ceil(tensor_k_tiles / num_k_blocks)` (balanced; the ragged last block runs through `KBlockInnerDimFn`). |
| S3 | Shrink the output block. Search `(nb_m, nb_n)` (blocks per core along m, n) over `block_m_tiles = ceil(core_m_tiles/nb_m)` and `block_n_tiles = round_up(ceil(core_n_tiles/nb_n), q_n)`. Keep the candidates where `F(block_m, block_n, block_k_floor) ≤ budget`. Minimize the re-read bytes `tensor_k_tiles·(core_m_tiles·a_tile_bytes·(nb_n−1) + core_n_tiles·b_tile_bytes·(nb_m−1))`; ties go to the larger `block_m·block_n`. Then grow `block_k_tiles` into the remaining budget as in S2. |
| S4 | If even `(block_m, block_n, block_k) = (sb-legal minimum, q_n, 1)` does not fit, raise `RuntimeError` naming the budget. This cannot happen for any INPUTS / LOOSE_CASES interleaved shape: the minimum footprint is < 100 KB. |

**Every knob is a parameter.** Kernel loops are `for (block_idx = 0; block_idx < num_blocks_this_core; ++block_idx)` and `for (kb = 0; kb < num_k_blocks; ++kb)` over CT/RT values. No tensor dimension is baked into a kernel. Every CB page count is computed on the host from the same `block_*` / `depth_*` variables that are passed as CT args. Each knob is defined once.

### Buffer-depth knobs

| CB | Depth knob | Phase 0 value | What the depth buys |
|----|------------|---------------|---------------------|
| `cb_act_tiles` | `depth_act` (K-blocks in flight) | 2 | The reader fetches A K-block kb+1 while compute multiplies kb. Across output blocks, the next block's first K-block also overlaps the current block's tail. |
| `cb_weight_tiles` | `depth_weight` | 2 | Same, for B. This is the operand the DRAM-bound decode/FF1 shapes stream. |
| `cb_output_tiles` | `depth_out` (row groups of `sb_h × out_block_n_tiles`) | 2 | Compute packs row group r+1 while the writer drains r. |
| `cb_partials` | none: capacity is exactly one block (mechanism cap) | 1 block | — |
| `cb_bias_row`, `cb_gate_row` (later) | none: one N-block row, refreshed per block | 1 row | — |
| `cb_residual_tiles` (later) | `depth_residual` | 2 row groups | The residual read overlaps the epilogue. |

### Mechanism caps

| Mechanism | Cap on which extent | Clamp | What happens unclamped |
|-----------|--------------------|-------|------------------------|
| DEST capacity (half-sync; fp32 DEST halves it) | `sb_h·sb_w ≤ DEST_AUTO_LIMIT` (`dest_helpers.hpp:103`, asserted at `matmul_block_helpers.inl:158`) | host subblock derivation with `dest_limit = 4` (fp32) / `8` | ASSERT under `--dev`; otherwise DEST overflow and corrupt tiles |
| Subblock grid tiles the block | `sb_h ∣ block_m_tiles`, `sb_w ∣ block_n_tiles` (`in0_num_subblocks = block_m/sb_h`, `in1_num_subblocks = block_n/sb_w`) | `block_n_tiles` quantized to `q_n`; the subblock is chosen among divisors | Missing or duplicate output tiles |
| In-place packer L1 accumulation (TileRowMajor + `packer_l1_acc`) | `cb_partials` capacity == `block_m_tiles·block_n_tiles` **exactly** | allocate exactly one block | The ring does not return to the block base each K-block, so accumulation lands at shifting addresses and produces wrong sums |
| Ragged K (balanced split) | last K-block valid width `last_block_k_tiles ≤ block_k_tiles` | `KBlockInnerDimFn` returns `last_block_k_tiles` on the last block. The reader pushes the **nominal** `block_k_tiles` pages (it reads only valid ones). | Garbage K tiles get summed into every output, giving silently wrong results |
| swiglu pairing | `block_n_tiles` even, N-block start even | `q_n ≥ 2`. Every N-region and block boundary falls on an even tile. | A gate/up pair straddles two blocks or cores, giving wrong output |
| Sharded A in place (later regimes) | HS: `block_m_tiles = shard_m_tiles`, `block_k_tiles = tensor_k_tiles` (a K-sub-block of a multi-row shard is not contiguous). BS/DS: `block_k_tiles` divides the sender shard's K width. | Taken from the shard spec | The K-block reads non-contiguous tiles → wrong operand |
| Sharded output in place (later) | zero-copy `cb_output_shard` only if `block_n_tiles == core_n_tiles` (whole shard width) | otherwise a streaming `cb_output_tiles` plus a local NoC write into the shard | Row-group reserves walk past this N-block's columns and corrupt the shard |
| `matmul_block` interm placeholder | when `num_k_blocks == 1` and epilogue none, pass `cb_output_tiles` as `interm_buf` (`matmul_block_helpers.hpp:307-314`) | host omits `cb_partials`; the kernel binds the placeholder | The packer is configured for the wrong format → corrupt output |

### Regimes

Selection key: placement (from the tensors), then the factorization outcome. Phase 0 builds **R1** only. Every other row is designed here with its topology.

| Regime | Status | Predicate | Block | Data movement vs. minimum (DRAM boundary: A, B once; out once) | What a bigger block buys |
|--------|--------|-----------|-------|----------------------------------------------------------------|--------------------------|
| **R1 `interleaved_out_split`** (one kernel family covers 2-D and 1-D grids) | **built** (Phase 0) | `placement == "interleaved"` | `block_batches × block_m_tiles × block_n_tiles`, K-walk `num_k_blocks × block_k_tiles` | A crosses DRAM `grid_n'·nb_n` times and B crosses `grid_m'·nb_m` times (each core reads its own rows/cols; no reuse across cores). Above minimum by the factor that the mcast successors remove. Out once. Example, prefill 2048×5120×8704: ≈10× A + 10× B. | Fewer A/B re-reads (`nb_n`, `nb_m` → 1). Per K-block: one CB handshake per operand, one barrier per operand, one partial pack of the whole block. Per block: one `matmul_block_init` (intended once per block; once per kernel for epilogue none, see lamp). |
| R1a `interleaved_2d_mcast` | **built** (Refinement 3). It keeps R1's per-core region, CBs, compute kernel and block walk; only the reader's operand **source** changes from DRAM to a line injector per grid row-line (A) and column-line (B), `ttnn.Mcast` + `McastArgs`, injectors staggered (`STAGGER_INJECTORS`) so no core injects both. The block walk is grid-uniform (nominal `nb_m`, `nb_n` on every core; padded-tail blocks carry 0 valid tiles). PERF FOCUS 1: 1673 → 463 µs. | `R1 and grid_b'==1 and grid_m'>1 and grid_n'>1` (rectangle placement) `and block_k·max(block_m·a, block_n·b) ≥ MCAST_MIN_KBLOCK_BYTES` (32 KB: tiny K-blocks are handshake-latency-bound, R1 wins) | same as R1 | A once and B once from DRAM. NoC: A multicast to `grid_n'−1` receivers per row, B to `grid_m'−1` per column (`mcast_topology`). | Per K-block adds one mcast and two semaphore round-trips per operand; a larger `block_k_tiles` amortizes them. Grid-sync lamp applies. |
| R1b `interleaved_stream_mcast_a` | **built** (Refinement 9). A is reuse-shared by all `grid_n'` cores of a 1-D N split; ONE injector reads each A K-block and broadcasts it to the whole split (`shared_input_reuse`), from the **writer RISC** (BRISC / NoC1, which produces `cb_act_tiles`) so the A handshake never stalls the reader's B stream; the injector prefetches K-block kb+1 (trid) before multicasting kb (`ACT_STREAM_AHEAD`). Operand rings `ACT_STREAM_DEPTH = 4` K-blocks; N quantum row-aware on 1-D splits (`Q_N_ROW_AWARE`: a `core_m`-tall subblock needs only `4/h` columns). Ranked at `max(A + B, ACT_STREAM_INJECTOR_COST · A)` (single-injector throughput). PERF FOCUS 4: 1125 → ~640 µs. | `R1 and grid_b'==1 and tensor_batches==1 and grid_m'==1 and grid_n' > 1 and block_k·block_m·a ≥ ACT_STREAM_MIN_KBLOCK_BYTES (32 KB) and B bytes ≥ output bytes` (weight-streaming: the A rounds share the writer RISC with the output) | `block_m_tiles = tensor_m_tiles` (≤ 1 for M ≤ 32), `block_n_tiles = core_n_tiles` | R1: A crosses DRAM `grid_n'` times (e.g. FLUX timestep linear_2: 96 × 384 KB = 37 MB against 75 MB of B, +49%). R1b: A once, plus a NoC broadcast of A to `grid_n'` cores. | Per K-block: one broadcast, one handshake |
| R1c `interleaved_split_k` (combine) | **deferred**: R1 already gives every shape a correct, deterministic result. The only shapes it would serve under-fill the grid (`grid_b'·grid_m'·grid_n' < C/2`): the regression deep-K probe (32 output tiles, K = 16384) and the PDL ASPP-pool cell (block-sharded, R3). Neither carries a PERF FOCUS note. Reachable: a K-slice index is one more outer axis on R1's K-walk (`kb` range per core). The combine is a fixed-order gather of `block_m·block_n` fp32 partials to the slice-0 core, so results stay bit-identical (`test_regression.py`). | `R1 and grid_b'·grid_m'·grid_n' < C/2 and tensor_k_tiles ≥ 2·block_k_floor` | R1 block × K-slice `core_k_tiles = ceil(tensor_k_tiles / grid_k)` | A, B once each. NoC adds `(grid_k−1)·block_m·block_n·4 KB` per output block. | Fewer slices mean fewer partial transfers |
| R1d `interleaved_batched` | **built** (Refinement 7; PERF FOCUS 3: 70.8 µs vs stock 123.0 µs). It is a **knob-turn** of R1: `grid_b > 1`, the `weight_batch_stride` RT arg, `MatmulBlockShape.batch = block_batches`. The block walk already carries the batch axis. | `R1 and weight_batch == "batched"` | `block_batches × block_m_tiles × block_n_tiles` | Same as R1 per batch. A and B per batch never cross batches. | Fewer inits (one `matmul_block` call spans `block_batches`) |
| R2 `height_sharded` | **built** (Refinement 7) as R3's single-K-shard case on A's shard cores (`height_sharded_topology`; `cb_act_shard` is compute's A operand, no `cb_act_tiles`; batched B mapped per shard by `ShardRows`). Each core's work is self-contained: the shard holds all of K for its rows, so this is a **knob-turn** of R1's compute with a different A source. | `placement == "height_sharded"` | `block_m_tiles = shard_m_tiles`, `block_k_tiles = tensor_k_tiles`, `block_n_tiles = core_n_tiles = out_n_tiles` (coarsest that fits, otherwise S3 on n) | A: 0 DRAM (resident; `cb_act_shard` backed on the shard, zero-copy). Out: 0 DRAM (`cb_output_shard` backed on the output shard when `block_n == out shard width`). B: crosses DRAM once **per core** (`C_A` times), the stepping stone. | Whole-shard block: one pass, one init |
| R2a `height_sharded_mcast_b` | **built (Refinement 10)**, the default R2 path (`SHARDED_B_MCAST`). B is reuse-shared across all of A's cores, so one fixed injector (shard core `SHARDED_B_INJECTOR`) reads each B round once and multicasts it to the shard set (an irregular receiver set of ≤ 3 rectangles; more falls back to R2's per-core reads), and the walk's first bias / gate row rides one row round on the same channel. Single B only (batched B varies along the rows). PDL HS: 0.38–0.80× of R2's time, 1.02–1.26× stock. | `R2` | as R2 | B once from DRAM plus a NoC broadcast to `C_A` cores | Per B chunk: one mcast |
| R3 `block_sharded` | **built (Refinement 5)**, with R3a's rotating A row multicast as the default exchange (`BS_ROW_MCAST`; unicast peer reads remain as the knob's other setting); B stays per-core DRAM reads. Originally deferred: outside Phase 0 SUPPORTED. A's block shard splits **K** across grid columns, so each core must gather its row peers' A K-slices: a **scheme-change** (cross-core movement of A). Stepping stone: the own K-slice comes zero-copy from `cb_act_shard` via `In0SourceFn`; peer K-slices are read by NoC unicast into `cb_act_tiles`. | `placement == "block_sharded"` | `block_m_tiles = shard_m_tiles`, `block_k_tiles = shard_k_tiles` (one peer shard per K-block, `num_k_blocks = grid cols`), `block_n_tiles = out shard width` | A: 0 DRAM, each shard sent over NoC to `cols−1` row peers. B: column slice per core, crossing DRAM `rows` times. Out: in place. | Per K-block: one peer read / mcast |
| R3a `block_sharded_mcast` | **built**: the A row-mcast (Refinement 5) and the B column-mcast (Refinement 10: one fixed injector per grid column, staggered over the rows; B rounds of `g` K-blocks (`SHARDED_B_ROUND_BYTES`); the first row on a row round). Every core runs A round → B round per K-block (no channel deadlock). The B column-mcast is gated on the B re-reads it saves (`BS_B_MCAST_MIN_SAVED_BYTES` = 3 MiB: below it the extra round on the reader's chain measured 0–9% slower). PDL res5: 0.39–0.74× of R3. | `R3` | as R3 | B once from DRAM. A row mcast. | as R1a |
| R4 `dram_sharded` | **built** (Refinement 4; see the R4 note under Inter-Tensix contracts). A is width-sharded over **K**, so the bank workers need all of A: a **scheme-change** (A broadcast from rotating senders). Successor for PERF FOCUS 2. | `placement == "dram_sharded"` | per worker: `block_m_tiles = tensor_m_tiles` (≤ 1 for decode), `block_n_tiles = bank shard width (out cols)`, `block_k_tiles = a_shard_k_tiles / s` (s = smallest divisor making the B double buffer fit) | B: once, bank-local (worker w = `get_optimal_dram_bank_to_logical_worker_assignment(device, noc)[w]` reads bank w's shard with coalesced reads, `read_coalescing`). A: each A shard crosses NoC once, broadcast to the bank workers (`Mcast`, rotating/explicit senders = A's cores, irregular receiver set = workers). Out: written once to its L1 shard (local if the worker owns it, otherwise a NoC write through the output accessor). | Per K-block: one A round (mcast + handshake). B stream depth 3 (lamp). |
| R5 `k_split_full` | **rejected**: superseded by R1 / R1c. It splits only K across all C cores for the full M×N. | — | — | Combine traffic `C·M·N·4 B` (prefill: 110 × 71 MB) dwarfs the A/B saving | — |
| R6 `per_tile_dot` (1×1 output blocks, A row and B col re-fetched per tile) | **rejected** dead end: superseded everywhere by R1's coarse region. | — | 1×1 | A re-read `tensor_n_tiles` times, B `tensor_m_tiles` times | — |
| R7 `reshard_then_R1` | **rejected**: violates "read A where it lives" and one-dispatch. Superseded by R2/R3/R4. | — | — | An extra full copy of A | — |

### Traffic ranking (interleaved, all candidate splits)

| Rank | Split | DRAM crossings (A, B, out) | Cross-core traffic | Verdict |
|------|-------|---------------------------|--------------------|---------|
| 1 | 2-D (M×N) + row/col mcast (R1a) | 1, 1, 1 | A to `grid_n'−1` per row, B to `grid_m'−1` per column | cheapest. Deferred successor of R1. |
| 2 | 1-D N split + A broadcast (R1b), when `tensor_m_tiles` small | 1, 1, 1 | A (≤ 0.4 MB) to all cores | cheapest for decode-shaped interleaved. **Built** (Refinement 9). |
| 3 | 2-D (M×N), per-core reads (**R1, built**) | `grid_n'`, `grid_m'`, 1 | none | **implemented** (stepping stone of 1 and 2) |
| 4 | split-K + combine on top of R1 (R1c) | 1, 1, 1 | `(grid_k−1)·out_block·4 B` per block | only pays when `grid_m'·grid_n' < C/2`. Deferred. |
| 5 | 1-D M split (`grid_n=1`) for large N | 1, C, 1 | none | dominated by 3 (it is one of R1's factorization candidates, so it is picked only when it wins on W, then R) |
| 6 | K-only split (R5) | 1, 1, 1 | `C·M·N·4 B` | rejected |

The primary split is the **independent M×N output split**. It needs no combine, it is the base the cheapest (mcast) successors build on, and it fills the grid on every INPUTS / LOOSE_CASES interleaved shape except the deep-K regression probe.

### Block schedule (per core; logical schedule across reader / compute / writer)

```cpp
for (uint32_t block_idx = 0; block_idx < num_blocks_this_core; ++block_idx) {   // (b, m-block, n-block), n fastest
    load_kblocks(block_idx);          // reader: for kb in num_k_blocks: A block_m×block_k, B block_k×block_n
    matmul_accumulate_block(block_idx);  // compute: one matmul_block call over all K-blocks
    epilogue_block(block_idx);        // compute: only when epilogue != none (later phases)
    store_block(block_idx);           // writer: row groups → output
}
```

| Operation | Block shape | Resident across it | Intended fixed-cost frequency |
|-----------|-------------|--------------------|-------------------------------|
| `load_kblocks` | per K-block: A `block_m_tiles × block_k_tiles`, B `block_k_tiles × block_n_tiles` (swiglu: B columns permuted so all gate tiles of the N-block precede all up tiles) | nothing (streams) | One `noc_async_read_barrier` per K-block (covering both operands); one reserve/push per operand per K-block. Bias/gate rows (later): one read per block. |
| `matmul_accumulate_block` | `block_batches × block_m_tiles × block_n_tiles × (num_k_blocks × block_k_tiles)` | partial sums of the whole block in `cb_partials` (Float32 under fp32 DEST) across all K-blocks | `compute_kernel_hw_startup<SrcOrder::Reverse>` once per kernel; `matmul_block_init` + data-format reconfig once per block; one partial pack of `block_m·block_n` tiles per K-block (L1 accumulation); output packed once per block |
| `epilogue_block` (later) | `block_m_tiles × block_n_tiles` in, `block_m_tiles × out_block_n_tiles` out | the bias / gate row for this N-block | Chain init once per block; PACK_RELU configured once per block (bias_relu) |
| `store_block` | row groups of `sb_h × out_block_n_tiles` | — | One `noc_async_write_barrier` per row group (Phase 0); `depth_out = 2` row groups in flight |

**Stall-shadow.**
- **Phase 0.** Compute waits on the reader's K-blocks. The only work independent of that wait is the next block's prefetch and the previous block's store. Both already overlap through `depth_act`, `depth_weight` (2) and the separate writer kernel. There is no other independent work.
- **Successors.** The injector in R1a/R1b issues its DRAM reads for K-block kb+1 **before** waiting on the receivers' ready-semaphore for kb. In R4 each worker issues its bank reads for B K-block kb before waiting for A round kb, because B does not depend on A's arrival.
- **R1c.** The K-sum commutes, so the combine order is free algebraically. It is still fixed (slice 0 … grid_k−1) for bit-identity. In fp32 the reordering against the single-core sum is within the precision baseline.

### Perf lamps

| Lamp | Why the default may be wrong here | Nearby alternative to measure |
|------|-----------------------------------|-------------------------------|
| Overlap (K-block size) | S1/S2 take the coarsest `block_k_tiles`. For DRAM-bound decode-shaped R1 (e.g. N=6144, K=6144: 2 K-blocks of 288 tiles) compute cannot start until a whole K-block lands. | `block_k_tiles` capped at 32 with `depth_weight` 3 (`trid_double_issue`) |
| Grid factorization priority | `W` (critical path) is ranked before `R` (traffic). Without mcast, Phase 0 prefill is DRAM-heavy (A, B ×10), and a factorization with fewer cores and fewer re-reads may win. | Swap the priority to `(R, W)`; or cap `grid_m'·grid_n'` at C/2 for the prefill shapes |
| Subblock quantum `q_n = 4` | Pads `core_n_tiles` up to a multiple of 4 (computed-and-dropped columns on edge cores) in exchange for a 1×4 subblock (≈1.4×) | `q_n = 1` (exact split, possibly 1×1 subblocks) on the prefill and FLUX shapes |
| Split reader | One NCRISC reader issues both A and B. For R1b/R4 the B stream may make one DM RISC issue-bound. | B reads on the writer RISC (BRISC) beside the output writes (stock arrangement), `split_reader` |
| `block_k_floor = 8` | The partial pack per K-block costs `block_m·block_n` fp32 tile packs per `block_k` multiplies. For HiFi4 a smaller floor may be free; for LoFi a larger one may matter. | `block_k_floor` ∈ {4, 16} on prefill gate_up |
| Matmul re-init per block | Epilogue none needs no re-init between blocks (nothing dirties the matmul state) | `InitMode::None` after the first block (`matmul_block_helpers.hpp:108`) |
| K-walk stagger | All cores start at kb = 0 and hit the same DRAM banks together | Start each core's K walk at `core_idx % num_k_blocks` (`bank_stagger`). This changes the fp32 accumulation order per core, is still deterministic per call, and is gated on the precision baseline. |

---

## Dataflow Strategy

| Stage | Format | Mechanism | Notes |
|-------|--------|-----------|-------|
| A: DRAM/L1 interleaved → `cb_act_tiles` | A dtype tiles | reader (NCRISC, NoC0), `TensorAccessor(A)`, `noc_async_read_tile` per page, one barrier per K-block | Rows `r ∈ [r0, r0+block_m)`, tiles `k ∈ [kb·block_k, +block_k_valid)`. Rows ≥ `tensor_m_tiles` and K tiles past `last_block_k_tiles` are **not read**, but the nominal page count is pushed. |
| B: DRAM/L1 interleaved → `cb_weight_tiles` | B dtype tiles | same reader, `TensorAccessor(B)` | Row-major within the K-block (`k`-major, `n` contiguous), matching `matmul_block`'s in1 layout (`in1_per_core_w = block_n_tiles`). Columns ≥ `tensor_n_tiles` are not read. swiglu (later): CB column `c` ← B column `n0 + (c < h ? 2c : 2(c−h)+1)`, `h = block_n/2`. |
| compute: `cb_act_tiles` × `cb_weight_tiles` → `cb_partials` (K-blocks 0..n−2, L1 acc) → `cb_output_tiles` (last K-block, reloads the partials) | partials: Float32 (fp32 DEST) / Float16_b; out: output dtype | `matmul_block<false, packer_l1_acc=true, LastBlockTarget::Out, OutputCBLayout::TileRowMajor, ...>` | When `num_k_blocks == 1`: no partials, direct pack |
| compute epilogue (later): `cb_partials` (`LastBlockTarget::Interm`) → `cb_output_tiles` | output dtype | `add_bias_bcast_rows` / `eltwise_chain` | see Block Operation Realization |
| `cb_output_tiles` → DRAM/L1 interleaved output | output dtype | writer (BRISC, NoC1), `TensorAccessor(out)`, one barrier per row group | Skips tiles with row ≥ `tensor_m_tiles` or col ≥ `out_n_tiles` |
| bias / gate rows (later) → `cb_bias_row` / `cb_gate_row` | their dtypes | reader, once per block (`block_n_tiles` tiles of the single tile row) | |
| residual (later) → `cb_residual_tiles` | output dtype | reader, row groups in lockstep with the epilogue | |

### Inter-Tensix contracts of the deferred regimes (to keep them reachable)

| Regime | Contract |
|--------|----------|
| R1a | Row injector = core `(x=0, y=i)` reads A K-block kb and multicasts it into `cb_act_tiles` of row i's cores (`ttnn.Mcast`, RowMajor groups = grid rows, `McastFixedSenderConfig(sender_index=0)`). Column injector = core `(x=j, y=0)` does the same for B, ColumnMajor. Separate semaphore ids. Receivers `cb_reserve_back` → signal ready → wait data → `cb_push_back`. The compute kernel is unchanged. |
| R1b | **Built (Refinement 9)**: one fixed injector (`ACT_STREAM_INJECTOR`, row-wise index 0) on the writer RISC (BRISC / NoC1) reads each A K-block and multicasts it into every core's `cb_act_tiles` slot (one `ttnn.Mcast` group = the whole split; a flat split's partial row is a second rectangle of the irregular receiver set). The writer is `cb_act_tiles`' producer on every core (reserve → ack / send → push); the reader streams B only (reader CT `operand_mode = 2`, writer CT `act_mode = 2`). The compute kernel is unchanged. B stays per-core. |
| R1c | Slice s ≠ 0 packs its fp32 block into a `cb_partials_out` and unicasts it to slice 0's `cb_partials_in` (semaphore-gated). Slice 0 adds the partials **in slice order** (eltwise_chain Add) before the epilogue. |
| R2/R2a | `cb_act_shard = ttnn.cb_descriptor_from_sharded_tensor(A)`; the reader only `cb_push_back(shard tiles)` once. No NoC read of the own shard. R2a (built, Refinement 10): `weight` channel = one group of all shard cores, `McastFixedSenderConfig(SHARDED_B_INJECTOR, Staggered)`; the B round lands in every core's reserved `cb_weight_tiles` slot (lockstep pointers), the first row round in `cb_row`. |
| R3/R3a | Own K-slice: `In0SourceFn` selects `cb_act_shard` at `kb == own column` (same dataformat as `cb_act_tiles`). Peer K-slices: unicast reads of the peer shard (R3) → rotating row mcast (R3a). B (built, Refinement 10): `weight` channel with ColumnMajor groups of `rows` (= grid columns), staggered fixed injector, semaphores after the `act` channel's. |
| R4 | Senders = A's shard cores (shard s holds K-slice s), receivers = bank workers. `McastRotatingSenderConfig` (or one `McastExplicitFixedSenderConfig` channel per A core if rotating senders must lie in the receiver set). Irregular receiver set via `McastConfig(irregular_receiver_set_mode=…)`. Round s ↔ K-blocks of slice s. |

> **Implemented (Refinement 4, R4).** Topology as specified, with one helper-driven deviation in the receiver set:
> - **Workers**: bank w's optimal worker (`get_optimal_dram_bank_to_logical_worker_assignment(device, NOC_0)`, the reader's NoC) for every bank holding a valid B column. Each streams its bank shard with one coalesced read per K-block through the sharded `TensorAccessor` (the K-block's rows of a bank shard are contiguous), issued before the K-block's A round (stall-shadow).
> - **A broadcast**: one `act` channel, `McastRotatingSenderConfig(sender_cores = A's shard cores, sender_order = A's orientation)` (a single shard: `McastExplicitFixedSenderConfig`). Round r = A shard r. The helper's rotating mode requires a rectangular receiver group and its irregular multicast is capped at three rectangles, while the Blackhole bank workers are eight scattered single cores (columns 0 and 6). So the receiver group is the workers' **bounding rectangle**, with `McastConfig(handshake_cores = workers)`: only workers ack, and the other rectangle cores just land the payload in a reserved, unused `cb_act_tiles` slot. A worker that owns an A shard sends its round instead of receiving. A shard core outside the workers runs the reader with `ds_worker = 0` and only sends.
> - **K-blocks**: `block_k = kw / s`, where `s` divides the shard width. A shard core's K-block must be one contiguous run of its row-major shard, so `s > 1` requires `block_m == 1`.
> - **Output**: the op builds `ShardSpec(workers in bank order, [padded rows, out_sw·32])`; `CoreRangeSet` keeps insertion order, so every worker owns its shard and the compute packs in place (`cb_output_shard`). The remote-write branch is never needed.
> - **swiglu**: coalesced bank reads keep B's natural (gate, up) tile pairs, so the epilogue walks the block flat with Strided row stride 2 (`swiglu_natural_pairs`) instead of the reader's per-tile permutation.
>
> **Refinement 6 (PERF FOCUS 2).** Three R4 knobs, measured by ablation (one-core-per-bank decode was compute-bound: 38.7 µs of matmul vs a 28-29 µs B stream):
> - **`cores_per_bank` split** (`DS_CORES_PER_BANK`, auto via `DS_CORE_N_TILES_MIN`): each bank's columns are split over C workers (the bank's optimal worker + its C − 1 nearest free cores), each streaming its column slice of every K row (`ds_row_reads`). C = 1 is the Refinement 4 topology (in place); C > 1 streams the output through `cb_output_tiles` and the output's sharded accessor into the bank's shard on its primary worker. The output placement the op builds is unchanged.
> - **A rounds on the writer** (`DS_ACT_ON_WRITER`): the shared round implementation (`kernels/genify_matmul_ds_act.hpp`) runs on BRISC / NoC1, so the reader's B issues and barriers on NCRISC / NoC0 no longer wait out each round (`split_reader`). The Mcast `act` channel attaches to the writers; non-worker broadcast cores run only the writer.
> - **trid B ring** (`DS_B_AHEAD`, `DS_DEPTH_WEIGHT`): K-blocks in flight on per-K-block transaction ids (`trid_double_issue`). Parked at 1 / 2: once C slice readers share a bank it is saturated, and depth 2–3 measured flat to +3%.

### TARGET values as regimes / unlock class

| TARGET axis value | Unlock class | Why (axis character) |
|---|---|---|
| dtype `bfloat8_b`, `float32` | knob-turn | CB page format = A dtype; tile bytes from the tensor. fp32 + acc False is an EXCLUSION. |
| weight_dtype `bfloat8_b`, `bfloat4_b` | knob-turn | CB page format = B dtype; the unpacker reconfig is done by the helper (`DataFormatReconfig::INPUT_AND_OUTPUT`) |
| fp32_dest_acc_en `False` | knob-turn | `cb_partials` page format becomes Float16_b; `dest_limit` = 8 |
| placement `height_sharded` | knob-turn (R2) → R2a scheme-change for B reuse | The shard cuts M (independent); each core's work is self-contained |
| placement `block_sharded` | scheme-change (R3) | The shard cuts A's **K** (dependent for A): row peers must exchange A |
| placement `dram_sharded` | scheme-change (R4) | A is cut along K; B is cut along N per bank (independent). Needs an A broadcast to the bank workers. |
| epilogue `bias`, `bias_relu`, `swiglu`, `addcmul` | phase addition, no blocking change | `LastBlockTarget::Interm` + one epilogue block op. swiglu also adds the reader's column permutation and halves the output width. |
| alignment `m_non_aligned` | knob-turn | `ceil` per image is already in Phase 0's geometry; padded A rows are zero, so padded output rows are discarded by the tensor's padding |
| weight_batch `batched` | knob-turn (R1d) | Batch is independent; `weight_batch_stride` and `grid_b` |

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | a block: `block_batches × block_m_tiles × block_n_tiles` output tiles with the full K-walk |
| Grid | `device.compute_with_storage_grid_size()`; logical `(grid_b', grid_m', grid_n')` from the Work-split rule, enumerated row-wise |
| Per-core work | region `core_batches × core_m_tiles × core_n_tiles`; `num_blocks_this_core = nb_b·nb_m·nb_n` (Phase 0 usually 1). Walk order: b, then m-block, then n-block (n fastest). |
| Remainder | Edge regions: `rows_valid = min(core_m_tiles, tensor_m_tiles − i·core_m_tiles)`, `cols_valid = min(core_n_tiles, tensor_n_tiles − j·core_n_tiles)` (RT args). Compute always runs the nominal block. The reader skips out-of-range pages; the writer skips out-of-range tiles. Tile counts use `ceil` per image. |
| Unused cores | Cores beyond `grid_b'·grid_m'·grid_n'` get no kernels (the core range set covers only the used cores) |
| Regime selection | `select_regime()`: R1 iff placement == interleaved (Phase 0). R2/R3/R4 by placement when built. R1a/R1b/R1c sub-select by the factorization predicates in the regime table. |
| Regime-pinned tests | The acceptance test pins R1 with both a 2-D grid (`grid_m'>1`, `grid_n'>1`) and a 1-D grid (`grid_m'=1`). It also covers a multi-block K-walk (`num_k_blocks>1`, ragged last K-block) and a single K-block (`num_k_blocks==1`, no `cb_partials`). Shapes are chosen so these hold on any grid ≥ 4×4. |

## Circular Buffers

Phase 0 = the first four rows. The remaining rows are reserved for later refinements so their indices do not churn.

| Semantic Name | Index | Page Size | Num Pages | Sizing rationale | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|------------------|--------|----------|----------|-----------|
| `cb_act_tiles` | 0 | A tile bytes | `depth_act · block_m_tiles · block_k_tiles` | spans m, k; streams b, n | A dtype | reader | compute | whole kernel (streams) |
| `cb_weight_tiles` | 1 | B tile bytes | `depth_weight · block_k_tiles · block_n_tiles` | spans k, n; streams b, m | B dtype | reader | compute | whole kernel (streams) |
| `cb_output_tiles` | 16 | out tile bytes | `depth_out · sb_h · out_block_n_tiles` | spans n (out width); streams m in `sb_h` row groups; streams b, k | output dtype | compute | writer | whole kernel (streams) |
| `cb_partials` | 24 | partial tile bytes (4096 fp32 DEST / 2048) | `block_m_tiles · block_n_tiles` (exact, mechanism cap); **allocated only if** `num_k_blocks > 1 or epilogue != none` | spans m, n; streams k (accumulates in place), b | Float32 if fp32_dest_acc_en else Float16_b | compute | compute | one block (all K-blocks) |
| `cb_bias_row` (later) | 2 | bias tile bytes | `block_n_tiles` | spans n | bias dtype | reader | compute | one block |
| `cb_gate_row` (later) | 3 | gate tile bytes | `block_n_tiles` | spans n | gate dtype | reader | compute | one block |
| `cb_residual_tiles` (later) | 4 | residual tile bytes | `depth_residual · sb_h · block_n_tiles` | spans n; streams m | output dtype | reader | compute | streams |
| `cb_act_shard` (later, R2/R3) | 5 | A tile bytes | the shard (`cb_descriptor_from_sharded_tensor`) | the tensor's own L1 | A dtype | reader (one push) | compute | kernel |
| `cb_output_shard` (later, R2/R3/R4) | 17 | out tile bytes | the output shard | the tensor's own L1 | output dtype | compute | writer (a wait/pop only, to retire) | kernel |

`cb_partials` has the same kernel (compute) as producer and consumer; this is the helper's own spill/reload loop (`matmul_block_helpers.inl:316-339`, `:501-535`). No dataflow kernel touches it.

## Block Operation Realization

| # | Block operation | Block shape | Helper? | Input CB (semantic name, pages, state) | Output CB (semantic name, pages) | CB state after |
|---|-----------------|-------------|---------|----------------------------------------|----------------------------------|----------------|
| 1 | `load_kblocks` | per K-block: A `block_m × block_k`, B `block_k × block_n` | dataflow (`TensorAccessor`) | DRAM / L1 interleaved A, B | `cb_act_tiles` (+`block_m·block_k` per push), `cb_weight_tiles` (+`block_k·block_n` per push) | pushed `num_k_blocks` times per block |
| 2 | `matmul_accumulate_block` (epilogue none) | `MatmulBlockShape::of(block_m/sb_h, block_n/sb_w, sb_h, sb_w, block_k, num_k_blocks, block_batches)` | `matmul_block<false, true, LastBlockTarget::Out, OutputCBLayout::TileRowMajor, InitMode::Short, WaitAndPopPerKBlock, WaitAndPopPerKBlock, NoPostCompute, NoPreKBlock, NoPostKBlock, 0, LastKBlockInnerDim>` | `cb_act_tiles`, `cb_weight_tiles` (waited and popped per K-block); `cb_partials` (helper-internal, in-place) | `cb_output_tiles`: `block_m/sb_h` pushes of `sb_h·block_n` per batch | `cb_partials` drained (empty) |
| 2' | `matmul_accumulate_block` (epilogue ≠ none, later) | same | same, with `LastBlockTarget::Interm` | same | `cb_partials`: `block_m·block_n` pushed (row groups), retained for #3 | full block fronted in `cb_partials` |
| 3a | `epilogue_bias_block` (bias / bias_relu) | `BiasAddShape::of(block_m/sb_h, block_n/sb_w, sb_h, sb_w, block_n)` | `add_bias_bcast_rows<RowBroadcast, TileRowMajor>`; bias_relu: the caller sets `llk_pack_relu_config(ReluConfig::zero())` before and resets it after | `cb_partials` (block, popped by the helper per row group), `cb_bias_row` (`block_n`, caller wait before / pop after) | `cb_output_tiles` (per row group) | `cb_partials` empty |
| 3b | `epilogue_swiglu_block` | grid `(block_m, block_n/2)` | `eltwise_chain`: `CopyTile<Strided, D0>{gate base 0, stride block_n}`, `CopyTile<Strided, D1>{gate}`, `Sigmoid<D1>`, `MulBinary<D0,D1,D0>`, `CopyTile<Strided, D1>{up base block_n/2}`, `MulBinary<D0,D1,D0>`, `PackTile<output(cb_output_tiles, PerOuter, PerOuter)>` | `cb_partials` (caller `wait_front(block_m·block_n)` before, `pop_front` after; the Strided addressing requires `(None, None)`) | `cb_output_tiles` (`block_n/2` per row) | `cb_partials` empty |
| 3c | `epilogue_addcmul_block` | grid `(block_m, block_n)` | `eltwise_chain`: `BinaryFpu<Mul, input(cb_partials, None, None, Block), input(cb_gate_row, BroadcastDim::Row, None, None, Row)>{D0}`, `CopyTile<input(cb_residual_tiles), D1>`, `AddBinary<D0,D1,D0>`, `PackTile<output(cb_output_tiles, PerOuter, PerOuter)>` | `cb_partials` (caller-managed), `cb_gate_row` (caller-managed), `cb_residual_tiles` (per tile) | `cb_output_tiles` | `cb_partials` empty |
| 4 | `store_block` | row groups `sb_h × out_block_n` | dataflow | `cb_output_tiles` (wait/pop `sb_h·out_block_n`) | DRAM / L1 interleaved out | — |

> **Implemented (Refinement 2).** Row 3b is built as `CopyTile<Strided, D0>{gate base 0, stride block_n}`, `Silu<D0>`, `CopyTile<Strided, D1>{up base block_n/2}`, `MulBinary<D0,D1,D0>`, `PackTile<output(cb_output_tiles, PerOuter, PerOuter)>`.
> - `Silu` is a kernel-local chain element on the chain's documented `UnaryOp` CRTP extension point (`silu_tile` / `silu_tile_init`). It replaces Sigmoid + MulBinary: one SFPU pass and one DEST copy fewer per output tile.
> - Its sigmoid precision is a host knob, `SWIGLU_FAST_SILU`. A non-fp32 output uses the bf16-precision sigmoid; a float32 output keeps the fp32 one. The epilogue is SFPU-bound, so this cut its cost from 124 to 83 µs on PERF FOCUS 1.
> - The chain's DEST lanes per acquire are the knob `EPILOGUE_BLOCK_TILES` = 4 (clamped by the chain to DEST). It measured flat.
> - Rows 3a and 3c are as tabled.
> - The bias / gate row is waited before the first batch's epilogue of a block and popped after the last. The compute calls `matmul_block` once per batch slice (batch = 1), because `cb_partials` holds exactly one block, so the lifecycle stays per block under `block_batches > 1`.
> - After each Interm matmul the caller turns packer L1 accumulation off (`pack_reconfig_l1_acc(0)`, the bias-helper prerequisite).

`LastKBlockInnerDim` is a functor returning `last_block_k_tiles` when `block == num_k_blocks−1`, else `block_k` (the hook described at `matmul_block_helpers.hpp:206-212`).

## API Mapping

| Block operation | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Which params are block knobs |
|-----------------|------|----------|-----------|------------------------|----------|-----------|------------------------------|
| boot | helper prerequisite | `compute_kernel_hw_startup<SrcOrder::Reverse>(cb_act_tiles, cb_weight_tiles, cb_output_tiles)` | contract at `matmul_block_helpers.hpp:100-106` | Reverse is mandatory (matmul maps in0 → SrcB) | — | — | — |
| `matmul_accumulate_block` | helper | `compute_kernel_lib::matmul_block` | `matmul_block_helpers.hpp:339-371` (shape `:139-170`; `LastBlockTarget` `:62`; `OutputCBLayout` `:47`; `InputPolicy` `:77`; `InitMode` `:108`; interm placeholder rule `:307-314`; K-inner-dim hook `:206-212`) | `packer_l1_acc=true`, `Out`/`Interm`, `TileRowMajor`; `in1_per_core_w = block_n_tiles`, `out_row_width = block_n_tiles` | `cb_act_tiles`, `cb_weight_tiles`, `cb_partials` | `cb_output_tiles` / `cb_partials` | `MatmulBlockShape` fields `in0_block_k`, `num_k_blocks`, `batch`, and through `sb_*` the counts `in0/in1_num_subblocks` (block size; the subblock factors are derived) |
| `epilogue_bias_block` | helper | `compute_kernel_lib::add_bias_bcast_rows` | `bias_add_helpers.hpp:156-168` (shape `:38-53`; PACK_RELU / bias-CB lifecycle are the caller's, `:101-105`) | `RowBroadcast`, `TileRowMajor` (must match matmul, `:111-112`) | `cb_partials`, `cb_bias_row` | `cb_output_tiles` | `BiasAddShape` (same block) |
| `epilogue_swiglu_block` | helper | `compute_kernel_lib::eltwise_chain` | `eltwise/api/chain.hpp:619-620`; `CopyTile` `:553-557`; `PackTile` `:588-592`; `TileAddressing::Strided` / `StridedTileRange` `:286-291`; `IterationShape` `:146`; `Sigmoid` `eltwise/unary/activations.hpp:24`; `MulBinary` `eltwise/binary/sfpu/basic.hpp:28` | silu(g)·u = g·sigmoid(g)·u (the chain library has no Silu element; `sfpu_activation_helpers.hpp:44` `SiluActivation` applies to **every** packed tile, so it cannot select gate tiles only) | `cb_partials` | `cb_output_tiles` | `IterationShape::grid(block_m, block_n/2)` |
| `epilogue_addcmul_block` | helper | `eltwise_chain` | `chain.hpp:563-573` (`BinaryFpu`), `:330` (`BroadcastDim`), `AddBinary` `basic.hpp:20` | FPU mul with gate row-broadcast, then SFPU add of the residual (`compute_fusion`: no DEST-reuse ahead of an FPU op) | `cb_partials`, `cb_gate_row`, `cb_residual_tiles` | `cb_output_tiles` | `IterationShape::grid(block_m, block_n)` |
| R1a/R1b/R2a/R3a/R4 mcast (later) | helper | `McastArgs<>` / `mc.sender(noc).send()` / `mc.receiver(noc).receive()`; host `ttnn.Mcast` | `mcast/kernel/mcast_args.hpp:18-52`; `mcast/host/mcast.hpp:98` (`McastConfig`), `:110` (fixed), `:117` (rotating), `:133` (`Mcast`) | — | — | `cb_act_tiles` / `cb_weight_tiles` on the receivers | the payload = one K-block |
| `load_kblocks`, `store_block` | raw dataflow API (no compute helper applies) | `TensorAccessor` + `noc_async_read_tile` / `noc_async_write_tile` + barriers | `tech_reports/tensor_accessor/tensor_accessor.md` | — | — | — | pages per K-block / row group |

No compute phase uses a raw API. The swiglu epilogue builds silu from `Sigmoid` + `MulBinary` inside a helper chain. That is a composition, not a bypass.

## Broadcast Verification

| Phase | Op | CB_A (semantic name) Valid Region | CB_B (semantic name) Valid Region | Broadcast Dim |
|-------|-----|-----------------------------------|-----------------------------------|---------------|
| bias | add | `cb_partials`: All | `cb_bias_row`: Row0 (bias (1, N) → row 0 of each tile) | Row (`add_tiles_bcast_rows`) |
| addcmul mul | mul | `cb_partials`: All | `cb_gate_row`: Row0 | Row |
| addcmul add | add (SFPU, DEST) | product: All | `cb_residual_tiles`: All | None |
| swiglu | mul (SFPU, DEST) | gate·sigmoid(gate): All | up: All | None |

## Key Risks and Gotchas

| Risk | Why it bites here | Mitigation in this design |
|------|-------------------|---------------------------|
| Edge cores compute garbage rows/cols | The nominal block runs on a ragged region (`rows_valid < block_m`). Unread pages hold stale L1, possibly NaN/Inf. | Matmul rows and columns are independent, so garbage A rows touch only garbage output rows, and the writer skips them. K garbage is excluded by `KBlockInnerDimFn`. Never let a ragged **K** tile be summed. |
| Ragged K-block | A balanced `block_k_tiles` leaves the last K-block narrower | `LastKBlockInnerDim` + the reader pushes nominal counts (CB ring stays closed) |
| `cb_partials` sizing | In-place L1 accumulation + TileRowMajor requires the ring to return to the block base | capacity == exactly `block_m·block_n`; never round it up "for safety" |
| fp32 partial reload via SrcA | The Out-target reload (`copy_subblock_row_strided`, `.inl:316-339`) unpacks Float32 partials through SrcA (tf32 mantissa) | Fine for bf16/bf8b outputs. For `dtype=float32` outputs, check against `(float32, True)` = (0.99999, 0.005). If it misses, enable unpack-to-dest fp32 for `cb_partials`. |
| HiFi4 + fp32 DEST with bf16 inputs | Helper note `matmul_block_helpers.hpp:263-264` (Wormhole B0 accumulator corruption, #38306) | Target is Blackhole; the default config is task-mandated. If the acceptance test fails only at HiFi4, this note is the first suspect. |
| `matmul_block` Reverse boot | Without `SrcOrder::Reverse` the operands land in the wrong sources | Boot exactly as in the API Mapping |
| Placeholder interm format | `num_k_blocks == 1` with no `cb_partials`: the helper still pack-reconfigs to the interm format | Pass `cb_output_tiles` as `interm_buf` |
| Factorization differs per device | Grid 11×10 on p300c but harvested elsewhere: the R1 grid shape (2-D vs 1-D) depends on C | Regime-pinned acceptance shapes hold for any grid ≥ 4×4; the factorization never assumes a fixed size |
| L1 budget vs other live L1 tensors | Model cases keep A or the output in L1; CBs grow from the bottom of L1 | Subtract this op's L1 tensors per core (`l1_tensor_bytes_per_core`); `l1_reserve_bytes` knob for foreign tensors |
| swiglu column permutation | The gate/up order inside the CB is a reader decision the epilogue strides depend on | The reader (permute) and the compute (strided bases 0 and `block_n/2`) both derive from `block_n_tiles` |
| Determinism | `test_regression.py` requires bit-identical repeats | Phase 0 has no cross-core K combine; the per-core K order is fixed |
| Contract vs support errors | swiglu N%64, bad activation, etc. must be `ValueError` even though the cell is outside SUPPORTED | Contract checks run before `validate()` |
