# Operation Design: glu_fused

## Overview

| Field | Value |
|-------|-------|
| Classification | compute (fused elementwise — split + activation + multiply) |
| Goal | Compute `torch.nn.functional.glu(x, dim=-1)` as a **single fused TTNN kernel**, by folding the reference TTNN composite (`slice` × 2 + `sigmoid` + `multiply`) into one `ttnn.generic_op` dispatch. No `ttnn::*` op chaining in the Python entry point; no `ttnn::slice` anywhere. |
| Math | For input `x` of shape `(N, C, H, W)` with `W` divisible by 64: `output[n, c, h, j] = x[n, c, h, j] * sigmoid(x[n, c, h, j + W/2])` for `j ∈ [0, W/2)`. |
| Mode | Derivative — kernel implements the math directly, no autograd. |
| References | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp` (chain combinator, Sigmoid, SfpuMul, Load, sfpu_pipeline), `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/mish_kernel.cpp` (FP32 SFPU chain pattern with two Loads + SfpuMul into DST), `ttnn/ttnn/operations/multigammaln_lanczos/` (Python `ProgramDescriptor` shape, multi-core `split_work_to_cores` over a float32 + HiFi4 + fp32-dest-acc kernel), `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp` (DEST_AUTO_LIMIT = 4 in fp32 half-sync). |

### Translation of the reference composite

The reference TTNN composite (reproduced in the requirements) performs:

```
t_a = slice(x, last_dim=[0, W/2])           # first half along the last dim
t_b = slice(x, last_dim=[W/2, W])           # second half along the last dim
s_b = sigmoid(t_b, mode=ACCURATE)           # accurate (not fast-approx)
out = multiply(t_a, s_b)
```

The fused kernel collapses these four `ttnn::*` calls into one program:

```
For each output tile out_idx ∈ [0, total_output_tiles):
    a_tile  = read input[a_tile_idx(out_idx)]      # reader, expressed in tile IDs
    b_tile  = read input[b_tile_idx(out_idx)]      # reader, same tensor different tile
    DST[D0] = a_tile                                # CompactLoad
    DST[D1] = b_tile                                # CompactLoad
    DST[D1] = sigmoid(DST[D1])                      # Sigmoid<Approx::Exact, D1>
    DST[D0] = DST[D0] * DST[D1]                     # SfpuMul<D0, D1, D0>
    write output[out_idx] = pack(DST[D0])           # sfpu_pipeline pack
```

The "slice" is expressed at the **tile-id level inside the reader** — each output tile pulls one A tile and one B tile at the correct offset within the input. No `ttnn::slice` call, no intermediate tensor allocation, no second program dispatch for sigmoid or multiply.

Because the spec guarantees `W % 64 == 0`, each half is `W/2` elements wide, which is divisible by 32 — i.e. each half is tile-aligned. The split is *always* at a tile boundary, so the reader can address A and B halves by tile index alone, without any sub-tile masking or padding logic.

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | float32, TILE_LAYOUT, rank == 4, `shape[-1] % 64 == 0`, `shape[-2] % 32 == 0`, on-device | — | runtime (buffer addr); `Wt_half` baked at compile-time per program |

The op exposes **no other parameters**. In particular:
- No `dim` argument (the split is fixed to the last dim, matching the narrowest variant of the TTNN composite).
- No `memory_config` argument (output inherits the input's memory config).
- No `compute_kernel_config` argument (HiFi4 + fp32 dest acc is hard-coded internally — see below).

### Compute Config (hard-coded internally — NOT a caller parameter)

| Field | Value |
|-------|-------|
| `math_fidelity` | `ttnn.MathFidelity.HiFi4` |
| `fp32_dest_acc_en` | `True` |
| `dst_full_sync_en` | `False` (default — half-sync) |
| Effective DEST capacity | **4 tiles** (half-sync + fp32 dest acc), confirmed at `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:88-99`. `DEST_AUTO_LIMIT == 4`. |

Phase 0 is the maximum-precision configuration. Lower-precision variants (bfloat16, bfloat8_b) and a caller-supplied compute config are explicit precision *downgrades* deferred to later refinements.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | rank == 4 `(N, C, H, W)` |
| Dtype | `float32` (Phase 0). Any other dtype → `ValueError`. |
| Layout | `TILE_LAYOUT`. Any other layout → `ValueError`. |
| Memory | DRAM or L1 interleaved (`input_tensor.memory_config()` is preserved on the output). |
| Tile-alignment, last dim | `W % 64 == 0` (each half = `W/2` must be tile-aligned). Failure → `ValueError`. |
| Tile-alignment, second-to-last dim | `H % 32 == 0`. Failure → `ValueError`. |
| Storage | must be `ttnn.StorageType.DEVICE`. Host tensor → `ValueError`. |

### Output

| Property | Value |
|----------|-------|
| Shape | `(N, C, H, W/2)` — last dim halved, all others identical |
| Dtype | `float32` |
| Layout | `TILE_LAYOUT` |
| Memory | inherits `input_tensor.memory_config()` |

### Validation (Python side, before launch)

| Check | Failure |
|-------|---------|
| `input_tensor.storage_type() == ttnn.StorageType.DEVICE` | `ValueError` |
| `input_tensor.dtype == ttnn.float32` | `ValueError` |
| `input_tensor.layout == ttnn.TILE_LAYOUT` | `ValueError` |
| `len(input_tensor.shape) == 4` | `ValueError` (rank < 2 ⇒ ValueError; rank ≠ 4 in Phase 0 ⇒ ValueError) |
| `input_tensor.shape[-1] % 64 == 0` | `ValueError` |
| `input_tensor.shape[-2] % 32 == 0` | `ValueError` |

The validator runs **before** any `generic_op` dispatch — no half-launched program if validation fails.

## Dataflow Strategy

The operation is **elementwise** in the surviving axes `(N, C, H)` with a fixed offset in the last axis: each output element `out[n,c,h,j]` reads exactly two input elements (`x[n,c,h,j]` and `x[n,c,h,j+W/2]`). There is no reduction, no row/col broadcast, and no inter-Tensix communication.

The split happens **at the tile-id level inside the reader kernel**. The input tensor in TILE_LAYOUT has `Wt = W/32` tile-columns per tile-row. The first half (`x[..., :W/2]`) occupies the first `Wt_half = W/64` tile-columns of each tile-row; the second half occupies the last `Wt_half` tile-columns. For each output tile (the output also has `Wt_half` tile-columns per row), the reader fetches one A tile from the first half and one B tile from the second half of the *same* input tile-row.

| Stage | Role | Data path |
|-------|------|-----------|
| **DRAM → reader** | Reader (NCRISC, NoC0). For each output tile index `out_idx ∈ [start, start+num_per_core)`: computes `a_tile_idx`, `b_tile_idx` (formulas below), reads the A tile into `cb_input_a` and the B tile into `cb_input_b`. Both CBs are double-buffered so the reader can stage tile `i+1` while compute consumes tile `i`. | `input_tensor` DRAM → `cb_input_a` & `cb_input_b` (two pushes per output tile) |
| **compute (single Tensix per core, all 3 TRISCs)** | Auto-batched `sfpu_pipeline` over the chain `Load A → Load B → Sigmoid(B) → SfpuMul(A, B)`. Each chain iteration consumes 1 A tile and 1 B tile, applies accurate sigmoid to B in DST, multiplies A×sigmoid(B) in DST, packs to `cb_output_tiles`. Auto-batching fills DEST (4 slots ÷ chain stride 2 = 2 iterations per acquire). | `cb_input_a` + `cb_input_b` → DST → `cb_output_tiles` |
| **compute → writer** | Writer (BRISC, NoC1) drains `cb_output_tiles` one tile at a time and writes to DRAM at the matching output tile-id. Output tile-ids run from `start_tile_id` to `start_tile_id + num_per_core` in row-major tile order. | `cb_output_tiles` → output DRAM |

Tensor format does not change. Input arrives tiled; every CB holds tiles; output is written tiled. No tilize/untilize step.

### Tile-id arithmetic (reader)

Let `Wt = input_tensor.shape[-1] / 32` (tile-cols per row of input) and `Wt_half = Wt / 2` (tile-cols per row of output). Total output tiles = `output_tensor.buffer_num_pages() = N * C * Ht * Wt_half` where `Ht = H / 32`.

For each output tile-id `out_idx`:
```
row_idx     = out_idx / Wt_half                          # which tile-row (0 .. N*C*Ht - 1)
col_in_half = out_idx % Wt_half                          # which tile-col within the half (0 .. Wt_half - 1)
a_tile_idx  = row_idx * Wt + col_in_half                 # first half: same col offset
b_tile_idx  = row_idx * Wt + Wt_half + col_in_half       # second half: + Wt_half
```

`Wt_half` is the same for every output tile in a given program, so it is passed as a **compile-time** arg to the reader. Output tiles are processed strictly in increasing `out_idx` order on each core; tile-id ↔ `(n, c, h, w)` mapping is the canonical row-major tile-layout indexing implemented by `TensorAccessor`.

## Work Distribution

The work unit is **one output tile**. Output tiles are independent (each depends on exactly two input tiles, distinct tile-ids per output), so distribution is purely a parallelization decision.

| Field | Value |
|-------|-------|
| Work unit | One float32 output tile = 32×32 elements of `output_tensor`. |
| Grid | `device.compute_with_storage_grid_size()` — full compute grid (e.g. 8×8 on Wormhole). |
| Total work | `total_output_tiles = output_tensor.buffer_num_pages() = N * C * Ht * Wt_half`. |
| Per-core work | `ttnn.split_work_to_cores(grid_size, total_output_tiles)` returns `(num_cores, all_cores, core_group_1, core_group_2, tiles_per_core_g1, tiles_per_core_g2)`. Two groups differ by at most one tile per core. |
| Remainder | Handled by the two-group split: `group_1` cores get `tiles_per_core_g1` tiles each, `group_2` cores get `tiles_per_core_g2` tiles each (one fewer). When `total_output_tiles ≤ num_grid_cores`, group_2 is empty and unused cores are simply not in `all_cores`. |
| Per-core RT args (reader) | `(input_buffer_address, num_output_tiles_per_core, start_out_tile_id)` |
| Per-core RT args (writer) | `(output_buffer_address, num_output_tiles_per_core, start_out_tile_id)` |
| Per-core RT args (compute) | `(num_output_tiles_per_core,)` |
| Inter-core comms | none |

`all_cores` (the union returned by `split_work_to_cores`) is used as the `core_ranges` for every CB descriptor and every kernel descriptor. Per-core RT args walk `core_group_1` first, then `core_group_2`, in the same order — see `ttnn/ttnn/operations/multigammaln_lanczos/multigammaln_lanczos_program_descriptor.py:100-121` for the canonical walking pattern.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_a` | 0 | `ttnn.tile_size(ttnn.float32)` = 4096 B | 2 (double-buffer for reader↔compute streaming) | float32 | reader | compute (1 wait + 1 pop per output tile, via `CompactLoad<cb_input_a, D0>` in the chain) | one tile alive at a time; standard streaming |
| `cb_input_b` | 1 | 4096 B | 2 (double-buffer) | float32 | reader | compute (1 wait + 1 pop per output tile, via `CompactLoad<cb_input_b, D1>`) | one tile alive at a time; standard streaming |
| `cb_output_tiles` | 16 | 4096 B | 2 (double-buffer for compute↔writer streaming) | float32 | compute (1 push per output tile via `sfpu_pipeline` `SfpuOutputPolicy::PerTile`) | writer | one tile alive at a time; standard streaming |

CB descriptor `core_ranges` for **every** CB is `all_cores` (the union returned by `split_work_to_cores`). Indices follow convention: 0–7 input, 16–23 output. No intermediate CB is needed: the sigmoid result lives in DST (D1) for the entire chain, then is consumed by `SfpuMul<D0, D1, D0>` in the same `tile_regs_acquire` cycle.

### CB sync — push count = wait count

| CB | Pushes per output tile | Waits per output tile | Pops per output tile | Match |
|----|------------------------|------------------------|----------------------|-------|
| `cb_input_a` | reader: 1 | compute (CompactLoad): 1 | compute (CompactLoad): 1 | ✓ |
| `cb_input_b` | reader: 1 | compute (CompactLoad): 1 | compute (CompactLoad): 1 | ✓ |
| `cb_output_tiles` | compute (sfpu_pipeline PerTile): 1 | writer: 1 | writer: 1 | ✓ |

### CB sizing rationale

| CB | Reason for sized pages |
|----|------------------------|
| `cb_input_a` (2) | Standard double-buffer for reader↔compute streaming. While compute consumes A-tile `i`, the reader fetches A-tile `i+1`. Single-page would serialize reader and compute on the same NoC transaction; 2 pages restores pipeline parallelism. |
| `cb_input_b` (2) | Same reasoning as `cb_input_a`. |
| `cb_output_tiles` (2) | Standard double-buffer for compute↔writer streaming. While the writer drains output tile `i`, compute packs output tile `i+1`. |

No CB needs to hold more than 2 pages: there are no sequential helper-to-helper hand-offs (the chain is a single `sfpu_pipeline` call), no intra-tile accumulator ping-pong, and no in-place CB modifications.

## API Mapping

Every mechanism — helper or raw — has an exact file:line reference.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB | Output CB | Requirements |
|-------|------|----------|-----------|------------------------|----------|-----------|--------------|
| **Reader, per output tile** | raw_api | `cb_reserve_back` / `noc_async_read_tile` (with `TensorAccessor::get_noc_addr(tile_id)`) / `noc_async_read_barrier` / `cb_push_back`, called twice per loop (once for A, once for B) | dataflow primitives in `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | — | input DRAM via `TensorAccessor` | `cb_input_a` (push 1) and `cb_input_b` (push 1) per output tile | **Helpers considered and rejected:** `tilize_helpers_dataflow.hpp` covers RM→tile stick-batching, not relevant (input is TILE_LAYOUT). `cb_helpers_dataflow.hpp` (file `ttnn/cpp/ttnn/kernel_lib/cb_helpers_dataflow.hpp:1-end`) provides only compute-thread CB coordination primitives, not DRAM reads. There is no helper that reads two different tile-ids of the *same* input tensor into *two different* CBs per iteration — the standard tiled streaming reader pattern (as in `ttnn/ttnn/operations/multigammaln_lanczos/kernels/multigammaln_lanczos_reader.cpp:30-37`) is the direct primitive and the only thing that expresses the per-iter "two tile-ids → two CBs" semantics. |
| **Compute, init** | raw_api | `init_sfpu(cb_input_a, cb_output_tiles)` | `tt_metal/hw/inc/api/compute/eltwise_unary/eltwise_unary.h:46` | input CB anchor = `cb_input_a` (its float32 data format also matches `cb_input_b`, so the unpacker config covers both); output CB anchor = `cb_output_tiles` | — | — | Call **once** at kernel start. `init_sfpu` internally invokes `unary_op_init_common` which does the full hw setup; it replaces `compute_kernel_hw_startup` for SFPU-only kernels. `sfpu_helpers.hpp:72` documents this as the prerequisite for the chain/pipeline functions. |
| **Compute, full pipeline (Load A + Load B + Sigmoid(B) + Mul(A,B) + pack)** | helper | `sfpu_pipeline<SfpuBatching::Auto, SfpuInputPolicy::WaitAndPopPerTile, SfpuOutputPolicy::PerTile, SfpuDataFormatReconfig::INPUT_AND_OUTPUT>(chain, cb_output_tiles, num_output_tiles)` where `chain = sfpu_chain(Load<cb_input_a, Dst::D0>{}, Load<cb_input_b, Dst::D1>{}, Sigmoid<Approx::Exact, Dst::D1>{}, SfpuMul<Dst::D0, Dst::D1, Dst::D0>{})` | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:1406-1412` (`sfpu_pipeline` declaration), `:1363-1371` (`sfpu_chain` factory), `:370-376` (`Load`), `:527-531` (`Sigmoid<Approx, Slot>`), `:1047-1051` (`SfpuMul<In0, In1, Out>`). Implementations: `sfpu_helpers.inl:559-638` (`sfpu_pipeline`), `:174-177` (`Sigmoid::init`/`call`), `:496-515` (`CompactLoad::exec`). | `Approx::Exact` selects the accurate sigmoid (`sigmoid_tile<RC, false>` per `sfpu_helpers.inl:177` — `Approx::Exact` = `false` per `sfpu_helpers.hpp:201`), matching the spec's mandatory ACCURATE sigmoid mode. Chain stride = 2 (D0 + D1); auto-batch size = `DEST_AUTO_LIMIT / 2 = 4 / 2 = 2` tiles per acquire (fp32 half-sync). Both Loads use distinct CBs → `sfpu_chain` produces two separate `CompactLoad` elements (no compaction); the multi-group-CB static_assert at `sfpu_helpers.hpp:1366-1369` does **not** fire because each CB appears in exactly one CompactLoad. `WaitAndPopPerTile` means each per-tile `CompactLoad::exec` calls `cb_wait_front(cb, 1)` + `copy_tile(cb, 0, slot+offset)` + `cb_pop_front(cb, 1)` (`sfpu_helpers.inl:507-514`). `PerTile` output policy reserves and pushes 1 output tile per chain iteration (`sfpu_helpers.inl:622-628`). `INPUT_AND_OUTPUT` reconfig calls `reconfig_data_format_srca(FirstLoadCB)` + `pack_reconfig_data_format(cb_output_tiles)` once before the tile loop (`sfpu_helpers.inl:580-585`); since both inputs are float32 and the output is float32, these are effectively no-ops at runtime, but the call is harmless and matches the helper's documented invariants. | `cb_input_a`, `cb_input_b` | `cb_output_tiles` | This is **the** mechanism; the entire compute kernel body after `init_sfpu` is a single call. No raw `tile_regs_acquire` / `pack_tile` / `cb_*` calls in the compute kernel — the helper manages all of that. |
| **Writer, per output tile** | raw_api | `cb_wait_front` / `noc_async_write_tile` (with `TensorAccessor::get_noc_addr(tile_id)`) / `noc_async_write_barrier` / `cb_pop_front` | dataflow primitives in `tt_metal/hw/inc/api/dataflow/dataflow_api.h` | — | `cb_output_tiles` (1 wait + 1 pop per iter) | output DRAM via `TensorAccessor` | **Helpers considered and rejected:** no helper for "drain a tiled CB to DRAM in tile-id order". `untilize_helpers_dataflow.hpp` is RM-output-only (our output is TILE_LAYOUT). The standard tiled writer pattern matches `ttnn/ttnn/operations/multigammaln_lanczos/kernels/multigammaln_lanczos_writer.cpp` (canonical idiom for this exact path). |

### Helper coverage summary

| Compute phase in the algorithm | Helper used? | Reason |
|--------------------------------|--------------|--------|
| Load A tile, Load B tile, accurate sigmoid on B, multiply A × sigmoid(B), pack | ✅ `sfpu_pipeline` + `sfpu_chain` | One helper call covers the entire compute body. The chain abstraction is *exactly* designed for "load N tiles into DST, transform, pack" patterns. The `mish_kernel.cpp:35-47` reference is the closest analogue in the codebase (different ops, same shape: two Loads + activation + SfpuMul). |
| Tile-id math in the reader (split offset) | ❌ no helper applies | No helper expresses "for each output tile, read two distinct tile-ids of the same input tensor into two distinct CBs". The compute-side `binary_op_helpers::mul` operates on two whole CBs but assumes a producer already populated those CBs — which is what our reader is doing. The split logic is a property of the reader's tile-id arithmetic, not of any compute step, so no compute helper applies; no dataflow helper exists for this case. |

## Compute Phases

The compute kernel is a **single helper call** after init. Sequential phases shown for the work each core does:

| # | Operation | Helper? | Input CB (state) | Output CB (state after) | CB State After |
|---|-----------|---------|------------------|--------------------------|----------------|
| 0 | `init_sfpu(cb_input_a, cb_output_tiles)` — runs **once** at kernel start | raw_api | — | — | hw configured for float32 unpack + float32 pack |
| 1 | `sfpu_pipeline(chain, cb_output_tiles, num_output_tiles)` where `chain = sfpu_chain(Load<cb_input_a, D0>{}, Load<cb_input_b, D1>{}, Sigmoid<Exact, D1>{}, SfpuMul<D0, D1, D0>{})` — runs **once** per kernel, internally loops over all `num_output_tiles` | helper | `cb_input_a`, `cb_input_b`: each waits for 1 tile per iter, pops it after use | `cb_output_tiles`: 1 tile pushed per iter | both input CBs fully drained at kernel exit; `cb_output_tiles` has been pushed `num_output_tiles` times (all consumed by writer in parallel) |

### Inside the pipeline (informative — implementer does not write this loop)

Per auto-batched iteration over `actual = min(batch_size, remaining)` tiles (`batch_size = 2` for fp32 half-sync):

| Step | Action |
|------|--------|
| 1 | `tile_regs_acquire()` |
| 2 | For `k ∈ [0, actual)`: `CompactLoad<cb_input_a, ..., D0>::exec(k*2)` → `wait_front(cb_input_a, 1)`, `copy_tile(cb_input_a, 0, k*2 + 0)`, `pop_front(cb_input_a, 1)` |
| 3 | For `k ∈ [0, actual)`: `CompactLoad<cb_input_b, ..., D1>::exec(k*2)` → `wait_front(cb_input_b, 1)`, `copy_tile(cb_input_b, 0, k*2 + 1)`, `pop_front(cb_input_b, 1)` |
| 4 | For `k ∈ [0, actual)`: `Sigmoid<Exact, D1>::call(k*2 + 1)` → accurate `sigmoid_tile` on DST slot `k*2 + 1` |
| 5 | For `k ∈ [0, actual)`: `SfpuMul<D0, D1, D0>::call(k*2 + 0, k*2 + 1, k*2 + 0)` → in-DST multiply |
| 6 | `tile_regs_commit()`; `tile_regs_wait()` |
| 7 | For `k ∈ [0, actual)`: `cb_reserve_back(cb_output_tiles, 1)`, `pack_tile(k*2 + 0, cb_output_tiles)`, `cb_push_back(cb_output_tiles, 1)` |
| 8 | `tile_regs_release()` |

DEST occupancy: `actual * 2 ≤ DEST_AUTO_LIMIT (= 4)`, i.e. `actual ∈ {1, 2}`. The `static_assert(batch_size >= 1, "chain stride exceeds DEST capacity")` in `sfpu_helpers.inl:575` guarantees this fits.

## Broadcast Verification

Not applicable — the operation has no broadcast. The internal multiply uses `SfpuMul<D0, D1, D0>` (in-DST, slot-on-slot), which is a per-tile pairwise op with no broadcast dimension.

## Build Order

Suggested incremental bring-up sequence for the implementer's mental model. Each step has a clear failure mode and a deterministic-input check to verify before moving on.

| Step | What to bring up | Deterministic input verification |
|------|-------------------|-----------------------------------|
| 1. Skeleton + passthrough | Stub reader: read A tile only (ignore B), push to `cb_input_a`. Compute: copy A tile from `cb_input_a` to `cb_output_tiles` via `sfpu_pipeline(sfpu_chain(Load<cb_input_a, D0>{}, Identity<D0>{}), cb_output, num_tiles)` (or any pass-through chain). Writer: stream `cb_output_tiles` to DRAM. **Goal:** verify the program descriptor, work distribution, and CB layout work end-to-end before adding the split. | Feed `x = torch.ones(N,C,H,W)` (or `torch.arange(...)` reshaped). Expected output (this stub): `x[..., :W/2]`. DPRINT the first 4 elements of the first output tile to confirm they match the first 4 elements of input. |
| 2. Add B-half tile reads | Extend reader to compute `b_tile_idx = a_tile_idx + Wt_half` and push to `cb_input_b`. Compute: pass through B instead of A (`Load<cb_input_b, D0>` + identity). **Goal:** verify the split offset is correct — output should match `x[..., W/2:]`, not `x[..., :W/2]`. | Same `torch.arange`-style input. Output's first 4 elements should equal `x[..., W/2:][..., :4]`, NOT `x[..., :4]`. If they match the first half, the offset arithmetic is wrong. |
| 3. Wire the full chain | Replace the pass-through chain with the real one: `sfpu_chain(Load<cb_input_a, D0>{}, Load<cb_input_b, D1>{}, Sigmoid<Approx::Exact, D1>{}, SfpuMul<D0, D1, D0>{})`. **Goal:** end-to-end glu. | Random `torch.randn(N,C,H,W, generator=g)` with `manual_seed(42)`. Compare against `torch.nn.functional.glu(x, dim=-1)` — PCC ≥ 0.999, max abs err ≤ 0.05. |
| 4. Multi-core scale-out | Confirm work distribution across the full grid (e.g. 8×8 on Wormhole). **Goal:** correctness on shapes that span multiple cores. | The acceptance test shapes already exercise multi-core. Watch for off-by-one in tile-id arithmetic when `total_output_tiles / num_cores` has a remainder — the two-group `split_work_to_cores` handles this if the per-core RT args walk groups in the same order. |

DPRINT hints: use `torch.arange(N*C*H*W).reshape(N,C,H,W).float()` as a *deterministic* input — every input element has a unique value, so off-by-one errors in tile-id arithmetic immediately produce a wrong output value at a recognizable position. Random input drowns these errors in noise.

## Key Risks and Gotchas

| Risk | Mitigation |
|------|------------|
| **Tile-id arithmetic off-by-one in the reader.** Easy to get `a_tile_idx` and `b_tile_idx` swapped, or to use `Wt` where `Wt_half` was intended. | Pass `Wt_half` as a *named* compile-time arg. Use the `arange`-input bring-up in Build Order step 2 — wrong offset shows up as the first 32 output elements containing input values from columns `[0, 32)` instead of `[W/2, W/2+32)`. |
| **Sigmoid precision mode.** The chain element `Sigmoid<Approx::Exact, D1>` MUST use `Approx::Exact` (= the accurate `sigmoid_tile<RC, false>`), NOT `Approx::Fast`. Spec mandates ACCURATE. `Approx::Exact` is the *default* template arg (`sfpu_helpers.hpp:527`), but the implementer should write it explicitly to make the intent obvious. | Code review checkpoint: grep the compute kernel for `Sigmoid<` and verify `Approx::Exact` is the first template arg or that the default is being relied on. Confirm at `sfpu_helpers.hpp:201` that `Approx::Exact == false`. |
| **DEST budget with `fp32_dest_acc_en=True`.** DEST has only **4 tiles**, not 8. The chain stride is 2 (D0 + D1), so batch_size = 2 is the maximum. Any further DST consumer (e.g. an extra scratch slot) would overflow. | The chosen chain uses exactly D0 and D1, no more. The implementer must not add intermediate DST slots. The `static_assert` at `sfpu_helpers.inl:575` enforces this at compile time — if it fires, redesign the chain (not the DEST budget). |
| **Same-CB-in-two-CompactLoads static_assert.** `sfpu_chain` forbids the same CB index appearing in two non-adjacent Load groups (`sfpu_helpers.hpp:1366-1369`). If we ever try to also load A again later in the chain, the chain fails to compile. | Not a concern for the current design (one Load per CB), but the implementer should not "optimize" by adding a re-Load of A. |
| **Compute config NOT exposed.** Hard-coded inside `create_program_descriptor`. Tests must not pass a `compute_kernel_config` kwarg — `glu_fused` has none. | Validation step in the entry point rejects unexpected kwargs by virtue of the function signature (`def glu_fused(input_tensor: ttnn.Tensor) -> ttnn.Tensor`); extra kwargs raise `TypeError` from Python. |
| **Output `memory_config` inherits input.** No `memory_config` parameter at this phase. If the input is in L1, the output will be in L1 — which may not fit for large shapes. | Tests use DRAM-only inputs (matches the `ttnn.DRAM_MEMORY_CONFIG` standard in unit tests). Phase 0 spec doesn't require a knob for this. |
| **Reader interleaves two CBs.** The reader pushes A then B for *each* output tile, alternating between `cb_input_a` and `cb_input_b`. Both CBs must have matching producer/consumer cadence (1 push per iter, 1 wait+pop per iter). The compute chain alternates Load<cb_input_a> and Load<cb_input_b>, so this is symmetric — but the reader must not, e.g., push N A-tiles then N B-tiles in a burst (that risks `cb_reserve_back` deadlock if N > num_pages). | The standard per-iter push-A-then-push-B order matches the compute's per-iter wait-A-then-wait-B order. Both CBs have 2 pages, so they can buffer up to 1 iter of leading by the reader before backpressure stops it — exactly the intended pipeline depth. |

## References to existing patterns

| Pattern | File |
|---------|------|
| Multi-core `split_work_to_cores` driving a float32 + HiFi4 + fp32-dest-acc op via `ttnn.generic_op` | `ttnn/ttnn/operations/multigammaln_lanczos/multigammaln_lanczos_program_descriptor.py` |
| Standard tiled DRAM streaming reader / writer | `ttnn/ttnn/operations/multigammaln_lanczos/kernels/multigammaln_lanczos_reader.cpp`, `..._writer.cpp` |
| FP32 SFPU chain with two Loads + SfpuMul (mish: `x * tanh(log1p(exp(x)))`) | `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/mish_kernel.cpp:35-47` |
| Python entry point + validation pattern | `ttnn/ttnn/operations/multigammaln_lanczos/multigammaln_lanczos.py` |
| Chain helper API documentation (with `Sigmoid<Approx, Slot>` and `SfpuMul<In0, In1, Out>` examples) | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:97-169` |
