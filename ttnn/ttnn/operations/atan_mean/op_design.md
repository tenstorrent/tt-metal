# Operation Design: atan_mean

## Overview

| Field | Value |
|-------|-------|
| Classification | fused (SFPU elementwise + REDUCE_ROW AVG) |
| Goal | Collapse the two-step ttnn chain `ttnn.atan(x)` → `ttnn.mean(y, dim=-1)` into a single fused kernel that streams input tiles, applies `atan` on the SFPU, and reduces along the row, with the intermediate `atan(x)` never materialised to DRAM. |
| Math | `output[n, c, h] = (1 / W) · Σ_{w=0..W-1} atan(input[n, c, h, w])` — equivalently `torch.atan(input).mean(dim=-1)`. |
| Mode | Derivative (fusion of two existing ttnn ops; no novel algorithm). |
| References | `ttnn/ttnn/operations/toy_variance/` (REDUCE_ROW + scaler-driven AVG pattern), `ttnn/ttnn/operations/toy_reduce_partial/` (standalone REDUCE_ROW pattern), `ttnn/ttnn/operations/multigammaln_lanczos/` (SFPU-heavy fused op with `split_work_to_cores`), `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp` (`sfpu_atan`, `Atan`), `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` (unified `reduce<>` helper), `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp` (pool-type-aware scaler prep). |

## Parameters

| Name | Type | Required | Valid Range | Default | CT/RT |
|------|------|----------|-------------|---------|-------|
| `input_tensor` | `ttnn.Tensor` | yes | float32, TILE_LAYOUT, rank == 4, H % 32 == 0, W % 32 == 0, on-device | — | runtime (buffer address) |

The reduction dimension is **fixed at `-1`** by spec — it is not exposed as a parameter. No `dim`, `keepdim`, `p`, or `memory_config` are accepted.

### Compute config (hard-coded internally — NOT a caller parameter)

| Field | Value |
|-------|-------|
| `math_fidelity` | `ttnn.MathFidelity.HiFi4` |
| `fp32_dest_acc_en` | `True` |
| Effective DEST capacity | 4 tiles (half-sync + fp32 acc), surfaced through `DEST_AUTO_LIMIT` (`ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp:102`). All helpers auto-honor this. |

The compute kernel never reads a caller-supplied compute config.

## Tensors

### Input

| Property | Requirement |
|----------|-------------|
| Shape | `(N, C, H, W)` — rank == 4 (validated). |
| Dtype | `float32`. Mismatched dtype → `ValueError`. |
| Layout | `TILE_LAYOUT`. `ROW_MAJOR_LAYOUT` → `ValueError`. |
| Memory | DRAM or L1 interleaved. |
| Tile-alignment | `H % 32 == 0` and `W % 32 == 0`. Non-aligned → `ValueError`. |
| Value domain | Any finite float (atan is well-conditioned over the real line; image `[-π/2, π/2]`). |

### Output

| Property | Value |
|----------|-------|
| Logical shape (post-squeeze, returned to caller) | `(N, C, H)` — rank 3, last dim dropped (keepdim=False). |
| Allocation shape (rank-4 intermediate, pre-squeeze) | `(N, C, H, 1)` TILE_LAYOUT. Trailing dim of 1 pads to a full tile of width 32; only column 0 of each output tile holds the row-mean value (this is the standard REDUCE_ROW packer-mask output region). |
| Dtype | `float32`. |
| Layout | `TILE_LAYOUT` (allocation), preserved through `ttnn.squeeze(out, dim=-1)` which is a metadata-only view (no copy). |
| Memory | DRAM interleaved (matches input default; the API does not expose `memory_config`). |

## Validation (Python side, before launch)

| Check | Failure |
|-------|---------|
| `input_tensor.storage_type() == ttnn.StorageType.DEVICE` | `ValueError` |
| `input_tensor.dtype == ttnn.float32` | `ValueError` |
| `input_tensor.layout == ttnn.TILE_LAYOUT` | `ValueError` |
| `len(input_tensor.shape) == 4` | `ValueError` (rank must be exactly 4 in Phase 0) |
| `input_tensor.shape[-2] % 32 == 0` | `ValueError` |
| `input_tensor.shape[-1] % 32 == 0` | `ValueError` |

The entry point dispatches a single `ttnn.generic_op(...)` call. It does NOT call any other `ttnn.*` op (no `ttnn.atan`, `ttnn.mean`, `ttnn.sum`, `ttnn.reduce`, `ttnn.atan2`). The host-side `ttnn.squeeze(out, dim=-1)` is a tensor-metadata operation — it does not dispatch a program.

## Dataflow Strategy

The fusion eliminates a full-tensor DRAM round-trip between `atan` and `mean`. The intermediate `atan(x)` tiles live in a per-core L1 CB sized to **exactly one row of tiles (Wt tiles)** — never the full tensor — so the prohibition against materialising the intermediate to DRAM, L1, or any cross-core buffer is honored by construction.

| Stage | Role | Data path |
|-------|------|-----------|
| **Program startup, per core** | Reader writes a single scaler tile holding `1/W` (in matmul col-0 layout — see "API Mapping" below) into `cb_scaler` exactly once. It is never popped; every subsequent `reduce<>` call waits for it. | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_scaler, AVG, REDUCE_ROW, W>()` → `cb_scaler` |
| **DRAM → reader** | NCRISC reader streams the input's `Wt` tiles for the current row-tile from DRAM into `cb_input_tiles`, one tile at a time. | `input_tensor` DRAM → `cb_input_tiles` |
| **Compute, atan stage** | TRISCs run `sfpu_atan<cb_input_tiles>(cb_atan_tiles, Wt)`. The helper loads each input tile into DEST, applies the SFPU `atan` instruction, packs to `cb_atan_tiles`, and pops `cb_input_tiles`. After the call, `cb_atan_tiles` holds the row's `Wt` post-atan tiles. | `cb_input_tiles` → `cb_atan_tiles` |
| **Compute, reduce stage** | TRISCs run `reduce<AVG, REDUCE_ROW>(cb_atan_tiles, cb_scaler, cb_output_tiles, row(Wt))`. The helper dispatches to the matmul-based REDUCE_ROW path (since AVG/REDUCE_ROW uses matmul internally — see `reduce_helpers_common.hpp:16-19`), consuming `Wt` tiles from `cb_atan_tiles`, the persistent scaler tile from `cb_scaler`, and emitting 1 output tile with the row-mean at column 0. | `cb_atan_tiles`, `cb_scaler` → `cb_output_tiles` |
| **Compute → writer** | BRISC writer drains 1 tile from `cb_output_tiles` and writes it to DRAM at the row-tile's output tile id. | `cb_output_tiles` → `output_tensor` DRAM |
| **(repeat per row-tile assigned to this core)** | The reader/compute/writer triple iterates over the core's assigned row-tiles. All three RISCs stay busy across iterations via the standard double-buffered streaming pattern. | — |

**No inter-Tensix communication.** Each output row-tile is computed end-to-end on a single core. No multicast, semaphores, or ring topology.

**No intermediate full-tensor materialisation.** `cb_atan_tiles` is sized to **`Wt` pages** — the smallest size that satisfies the sequential-helper sizing rule (helpers `sfpu_atan` and `reduce` both own all 3 TRISCs, so they cannot pipeline within the compute kernel; `cb_atan_tiles` must hold every tile `sfpu_atan` produces before `reduce` starts consuming). `cb_output_tiles` is double-buffered (2 pages) to overlap compute with the writer drain. No CB sized to NC × Ht × Wt × tile_size exists anywhere.

## Work Distribution

| Field | Value |
|-------|-------|
| Work unit | **One row-tile** — the slice of `Wt` input tiles whose values collapse to a single output tile. Each row-tile = one `(n, c, ht)` triple. |
| Total work units | `total_row_tiles = N * C * Ht` (where `Ht = H / 32`). |
| Grid | `device.compute_with_storage_grid_size()` — the full Tensix grid returned by the device. The kernel does not hardcode a subset. |
| Per-core split | `ttnn.split_work_to_cores(grid_size, total_row_tiles)` returns `(num_cores, all_cores, core_group_1, core_group_2, row_tiles_per_core_g1, row_tiles_per_core_g2)`. Group 1 cores each get `row_tiles_per_core_g1` row-tiles; group 2 (if non-empty) each get `row_tiles_per_core_g2` (= `row_tiles_per_core_g1 - 1` or `0`). Per-core counts differ by at most 1, so load is balanced regardless of `total_row_tiles`. |
| Remainder | Absorbed by the two-group split — `split_work_to_cores`'s standard contract. Remainder is never dropped onto one core. |
| Per-core RT args (all kernels) | `start_row_tile`, `num_row_tiles_this_core` (computed by the program factory by walking the two core groups in their split order, accumulating row-tile offsets). |

### Tile-id formula (reader and writer)

For input tensor `(N, C, H, W)` in tile layout with `Wt = W/32`, `Ht = H/32`:
- Input tile id for row-tile `r` (where `r ∈ [0, N*C*Ht)`) and intra-row index `wt ∈ [0, Wt)`:
  `input_tile_id = r * Wt + wt`
  (Tiles in the input are stored row-major over `(N, C, Ht, Wt)`.)
- Output tile id for row-tile `r` (output is shape `(N, C, H, 1)` padded to `(N, C, H, 32)` → 1 W-tile per row-tile):
  `output_tile_id = r`

### Why this strategy works for both shape regimes

| Regime | Example shape | `total_row_tiles = N·C·Ht` | Per-core load (8×8 grid, 64 cores) |
|--------|---------------|---------------------------|------------------------------------|
| "Tall" | `(1, 1, 2048, 64)` | 64 | 1 row-tile per core (perfect balance) |
| "Tall" | `(1, 1, 2048, 32)` | 64 | 1 row-tile per core |
| "Tall" | `(1, 1, 1024, 32)` | 32 | 1 row-tile on 32 cores, 0 on the rest |
| "High-channel" | `(1, 256, 64, 64)` | 512 | 8 row-tiles per core |
| "High-channel" | `(256, 1, 64, 64)` | 512 | 8 row-tiles per core |
| "High-channel" | `(1, 128, 128, 128)` | 512 | 8 row-tiles per core |
| "High-channel" | `(128, 1, 128, 128)` | 512 | 8 row-tiles per core |

No code-path branches on the regime. The same compile-time kernel + the same RT-arg shape work in every case; only `Wt`, `W`, `start_row_tile`, `num_row_tiles_this_core` differ.

## Circular Buffers

| Semantic Name | Index | Page Size | Num Pages | Format | Producer | Consumer | Lifetime |
|---------------|-------|-----------|-----------|--------|----------|----------|----------|
| `cb_input_tiles` | 0 | `tile_size(float32)` = 4096 B | 2 (double-buffer for streaming reader → compute) | float32 | reader (`noc_async_read_tile` per input tile) | compute (`sfpu_atan` via internal `Load`) | per-row-tile streaming; reader pushes Wt per row-tile, compute pops Wt per row-tile |
| `cb_scaler` | 8 | `tile_size(bfloat16)` = 2048 B | 1 (one shared scaler tile, lifetime = entire program) | bfloat16 | reader (writes 1/W once at startup via `calculate_and_prepare_reduce_scaler<cb_scaler, AVG, REDUCE_ROW, W>()`) | compute (`reduce<>` calls `cb_wait_front(cb_scaler, 1)` on every row-tile; never pops) | persistent: 1 push at startup, never popped, every row-tile's `reduce<>` re-waits for it |
| `cb_output_tiles` | 16 | `tile_size(float32)` = 4096 B | 2 (double-buffer compute → writer) | float32 | compute (`reduce<>` packs 1 tile per row-tile) | writer (drains 1 tile per row-tile) | streamed; compute pushes 1 per row-tile, writer pops 1 per row-tile |
| `cb_atan_tiles` | 24 | `tile_size(float32)` = 4096 B | `Wt` (minimum size that lets `sfpu_atan` push all Wt tiles before `reduce` starts consuming — sequential helpers own all 3 TRISCs, no pipelining possible) | float32 | compute (`sfpu_atan` packs Wt tiles per row-tile) | compute (`reduce<>` consumes Wt tiles per row-tile) | one row-tile of life: filled by `sfpu_atan`, drained by `reduce<>`, then reused for next row-tile |

CB ranges: `core_ranges = all_cores` (the union returned by `split_work_to_cores`) for every CB.

### CB sync (push count = wait count, per row-tile)

| CB | Producer pushes per row-tile | Consumer waits per row-tile | Match |
|----|------------------------------|------------------------------|-------|
| `cb_input_tiles` | reader: `Wt` (1 push per tile in the inner loop) | compute (`sfpu_atan` with default `WaitAndPopPerTile` input policy): `Wt` (1 wait/pop per tile) | ✓ |
| `cb_atan_tiles` | compute (`sfpu_atan` with default `PerTile` output policy): `Wt` (1 push per tile) | compute (`reduce<>` with default `WaitAndPopPerTile` input policy): `Wt` (1 wait/pop per tile) | ✓ |
| `cb_output_tiles` | compute (`reduce<>` with default `WaitAndPopPerTile` policy emits 1 tile per `REDUCE_ROW` row): 1 | writer: 1 (1 wait/pop per row-tile) | ✓ |
| `cb_scaler` | reader: 1 push at program startup (NOT per row-tile) | compute: `cb_wait_front(cb_scaler, 1)` per `reduce<>` call (never pops) | ✓ — push count 1, pop count 0; `wait_front` does not consume |

### Scaler CB format note

`cb_scaler` is **bfloat16**, not float32. The `calculate_and_prepare_reduce_scaler<cb_scaler, AVG, REDUCE_ROW, W>` helper deduces the data format from the CB's configuration and packs the `1/W` constant in the LLK-required tile layout (col-0 fill for AVG/REDUCE_ROW, since `reduce_uses_matmul<AVG, REDUCE_ROW>() == true` per `reduce_helpers_common.hpp:16-19`). The reduce LLK accepts a bf16 scaler against an fp32 input — the unpacker reconfig (driven by the helper's default `ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT`) handles the format mismatch. The bf16 scaler's quantisation of `1/W` is the only error source acknowledged in the acceptance tolerance.

## API Mapping

Every mechanism — helper or raw — has an exact file:line reference. **All compute phases use a kernel-lib helper.** No raw-API compute fallbacks exist.

| Phase | Type | Function | File:Line | Template Params / Args | Input CB (semantic name) | Output CB (semantic name) | Requirements |
|-------|------|----------|-----------|------------------------|--------------------------|---------------------------|--------------|
| **Reader, program startup (once per core)** | helper | `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<cb_id, PoolType, ReduceDim, reduce_factor>()` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp:94-101` | `<cb_scaler, PoolType::AVG, ReduceDim::REDUCE_ROW, W>` where `W` is a compile-time arg = `input_tensor.shape[-1]`. The helper deduces data format and tile shape from `cb_scaler` (bfloat16, full tile), computes the value `1/W`, and writes it in matmul col-0 layout (because `AVG + REDUCE_ROW` uses the matmul reduce path, see `reduce_helpers_common.hpp:16-19`). `valid_reduce_dim_elements_in_tile` defaults to 32 (tile-aligned). | — | `cb_scaler` (push 1 tile) | Pool-type-aware overload (mandatory — different `PoolType`/`ReduceDim` combos need different fill patterns; the legacy `prepare_reduce_scaler<cb>` overload is forbidden). Must run before any input tiles are streamed (compute `reduce<>` waits on `cb_scaler` immediately). |
| **Reader, per row-tile** | raw_api | `cb_reserve_back` / `noc_async_read_tile` / `noc_async_read_barrier` / `cb_push_back` (paired against `cb_input_tiles`, looped `Wt` times) | `tt_metal/hw/inc/dataflow_api.h` (standard NoC primitives); reader-side `TensorAccessor` per `tech_reports/tensor_accessor/tensor_accessor.md` | — | DRAM (`input_tensor` via `TensorAccessor`) | `cb_input_tiles` (push 1 per inner-loop tile) | **Helpers considered and rejected:** No reader helper exists for "stream a tiled tensor's row-tiles into a CB". `tilize_helpers_dataflow.hpp` covers RM → tile conversion (not relevant — input is already TILE_LAYOUT). `reduce_helpers_dataflow.hpp` only emits scaler tiles. The standard `TensorAccessor` + `noc_async_read_tile` loop is the canonical idiom (verified against `ttnn/ttnn/operations/multigammaln_lanczos/kernels/multigammaln_lanczos_reader.cpp:32-58` and `ttnn/ttnn/operations/toy_variance/kernels/reader.cpp:57-70`). |
| **Compute, init (once per kernel)** | raw_api | `compute_kernel_hw_startup(icb0, icb1, ocb)` | `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h:41` | `compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output_tiles)`. `icb0` is the primary input format anchor (fp32), `icb1` is the scaler format anchor (bfloat16), `ocb` is the pack target (fp32). Subsequent helpers reconfig as needed. | passes the three CBs whose data formats anchor unpacker/packer config | — | — | Documented as the required prologue for all kernel_lib helpers (`reduce_helpers_compute.hpp:29-31` and `sfpu_helpers.hpp:72-73`). Must be called exactly once before any helper. The SFPU helpers internally call `copy_tile_to_dst_init_short` and per-op `*_tile_init()`; the reduce helper internally calls `reduce_init`. No `init_sfpu` separate call required — `compute_kernel_hw_startup` covers the HW configure and the SFPU helpers handle their own datacopy-init. |
| **Compute, per row-tile — atan** | helper | `compute_kernel_lib::sfpu_atan<ICB>(ocb, num_tiles)` | `ttnn/cpp/ttnn/kernel_lib/sfpu_helpers.hpp:1587-1593` (declaration); inline body wraps `sfpu_op<ICB, ...>(ocb, num_tiles, Atan<>{})` per `sfpu_helpers.hpp:1435-1442` and `sfpu_helpers.inl:651-654`. | `<ICB = cb_input_tiles>`. Other template params take defaults: `SfpuBatching::Auto` (the pipeline fills DEST with `DEST_AUTO_LIMIT / 1 = 4` tiles per acquire/release in fp32 half-sync mode), `SfpuInputPolicy::WaitAndPopPerTile` (consumes one input tile per atan op), `SfpuOutputPolicy::PerTile` (push one output tile per atan op), `SfpuDataFormatReconfig::INPUT_AND_OUTPUT` (reconfig unpacker srcA to `cb_input_tiles` and packer to `cb_atan_tiles`). Args: `(cb_atan_tiles, Wt)`. The underlying op struct `Atan<Slot=Dst::D0>` (`sfpu_helpers.hpp:632-636`) calls `atan_tile_init()` + `atan_tile(d0)` from the SFPU LLK. | `cb_input_tiles` (pops `Wt` tiles) | `cb_atan_tiles` (pushes `Wt` tiles) | Helper owns DEST acquire/commit/wait/release, CB wait/pop/reserve/push, and data format reconfig. No external CB ops needed. Reading the .inl confirms: each iteration does `cb_wait_front(icb,1)/copy_tile/cb_pop_front(icb,1)` then `cb_reserve_back(ocb,1)/pack_tile/cb_push_back(ocb,1)`. With Wt ≤ 4 and `DEST_AUTO_LIMIT = 4`, one batch covers the full row-tile. |
| **Compute, per row-tile — reduce** | helper | `compute_kernel_lib::reduce<PoolType, ReduceDim, InputPolicy, ReconfigMode, AccumulateT, PostReduceOp>(...)` | `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp:402-410` (declaration); inline body in `reduce_helpers_compute.inl:166-435` with the REDUCE_ROW path at lines 337-434. | `<PoolType::AVG, ReduceDim::REDUCE_ROW>`. Other template params take defaults: `ReduceInputPolicy::WaitAndPopPerTile` (per-tile streaming wait/pop on `cb_atan_tiles`), `ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT` (reconfig srcA to scaler then to `cb_atan_tiles` for the matmul path; reconfig packer to `cb_output_tiles`), `AccumulateT = NoAccumulation` (single block per row-tile, no Accumulate needed), `PostReduceOp = NoOp` (no post-op required — the scaler already encodes `1/W` so the matmul output is the mean directly). Args: `(cb_atan_tiles, cb_scaler, cb_output_tiles, ReduceInputBlockShape::row(Wt))`. Because `reduce_uses_matmul<AVG, REDUCE_ROW>() == true`, the helper dispatches to `reduce_with_matmul_init` + `reduce_matmul_tiles` (`reduce_helpers_compute.inl:25-41`, REDUCE_ROW loop at lines 357-428), which gives the matmul-precision col-0 reduction that AVG/REDUCE_ROW expects. | `cb_atan_tiles` (pops `Wt` tiles), `cb_scaler` (`cb_wait_front(1)`, never pops) | `cb_output_tiles` (pushes 1 tile per row-tile) | Helper handles DEST acquire/commit/wait/release, the matmul-mode reduce init, post-tile pack to `cb_output_tiles`, and CB reserve/push of output. Critical: must pass `partial_scaler = ReducePartialScaler::none()` (the default) because the W dimension is tile-aligned in Phase 0 — no partial scaler tile is emitted by the reader. |
| **Writer, per row-tile** | raw_api | `cb_wait_front` / `get_read_ptr` / `noc_async_write_tile` / `noc_async_write_barrier` / `cb_pop_front` | `tt_metal/hw/inc/dataflow_api.h` | — | `cb_output_tiles` (waits/pops 1 per row-tile) | DRAM (`output_tensor` via `TensorAccessor`) | **Helpers considered and rejected:** No helper for "drain a tiled CB to DRAM in tile_id order". `untilize_helpers_dataflow.hpp` is RM-output-only; output is TILE_LAYOUT. The standard tiled writer loop matches `ttnn/ttnn/operations/multigammaln_lanczos/kernels/multigammaln_lanczos_writer.cpp:30-54` and `ttnn/ttnn/operations/toy_variance/kernels/writer.cpp:21-29`. |

### Why the SFPU + reduce pair cannot be replaced with one helper

There is no kernel-lib helper that fuses "SFPU per-tile transform" and "REDUCE_ROW AVG" into a single primitive. The `reduce<>` helper offers a `post_reduce_op` hook (`reduce_helpers_compute.hpp:304-308`) but it runs *after* reduction — too late to insert an `atan` on the per-element input. There is no `pre_reduce_op` hook. Hence the two-helper composition is the smallest expression of the operation within the kernel library.

The intermediate `cb_atan_tiles` is the canonical pattern for this composition (mirrors `toy_variance`'s `cb_centered` / `cb_centered_sq` intermediate between `sub`+`square_in_place` and `accumulate_reduce_block`). Its size is bounded by `Wt ≤ 4` in Phase 0 — far below any "full atan(x)" memory footprint.

## Compute Phases

Per row-tile on each core, the compute kernel executes the following sequence. After the final row-tile in a core's range, the kernel exits — `cb_scaler` is not popped (it's a persistent program-lifetime asset; no cleanup required because the program ends).

| # | Operation | Helper? | Input CB (semantic name, tiles, state) | Output CB (semantic name, tiles) | CB State After |
|---|-----------|---------|----------------------------------------|----------------------------------|----------------|
| 0 | `compute_kernel_hw_startup(cb_input_tiles, cb_scaler, cb_output_tiles)` (once, before the row-tile loop) | raw_api | — | — | (init only; no CB state changes) |
| 1 | `sfpu_atan<cb_input_tiles>(cb_atan_tiles, Wt)` | helper (`compute_kernel_lib::sfpu_atan`) | `cb_input_tiles`: pops Wt tiles | `cb_atan_tiles`: pushes Wt tiles | `cb_input_tiles` empty, `cb_atan_tiles` holds Wt atan tiles ready for reduce |
| 2 | `reduce<AVG, REDUCE_ROW>(cb_atan_tiles, cb_scaler, cb_output_tiles, ReduceInputBlockShape::row(Wt))` | helper (`compute_kernel_lib::reduce`) | `cb_atan_tiles`: pops Wt tiles; `cb_scaler`: waits 1 tile, no pop | `cb_output_tiles`: pushes 1 tile (row mean at col 0 of the tile) | `cb_atan_tiles` empty (ready for next row-tile); `cb_scaler` unchanged (still holds the persistent 1/W tile) |
| — | (loop 1 → 2 for `r in [0, num_row_tiles_this_core)`) | | | | |

The sequence (steps 1 → 2) repeats once per row-tile assigned to this core. `cb_scaler` survives across all iterations because the reduce helper only `cb_wait_front`s the scaler (never pops it). The intermediate `cb_atan_tiles` is fully consumed by the reduce in step 2, so it is empty again at the start of the next iteration.

## Build Order

The implementer should bring the op up incrementally rather than implementing the full kernel and debugging end-to-end. Order:

| Stage | Goal | What to verify | DPRINT / probe hints |
|-------|------|----------------|----------------------|
| 1. **Data pipeline only — single core** | Reader streams Wt input tiles per row-tile through `cb_input_tiles`; writer drains 1 dummy output tile per row-tile from `cb_output_tiles`. Compute is a passthrough: takes the first input tile of each row-tile and copies it to `cb_output_tiles` (using `compute_kernel_lib::copy_tiles` from `copy_tile_helpers.hpp`). Hard-code 1 core; run shape `(1, 1, 32, 32)` (Wt=1, total_row_tiles=1). | Output tile equals input tile 0. Confirms tile-id arithmetic, CB sync, and the reader/writer accessor wiring. Output buffer is shape `(1,1,32,1)` allocated in TILE_LAYOUT — verify squeeze on host gives `(1,1,32)`. | `DPRINT << "r=" << r << " base=" << base_tile << ENDL();` in reader. Run via `scripts/tt-probe.sh atan_mean` with `torch.arange(...)` input to make tile contents predictable. |
| 2. **Atan stage — single core, Wt=1** | Replace the passthrough with `sfpu_atan<cb_input_tiles>(cb_atan_tiles, 1)`, then `copy_tiles(cb_atan_tiles, cb_output_tiles, 1)` to drain. Still shape `(1, 1, 32, 32)`. | Output equals `torch.atan(input)`. Confirms SFPU atan wiring, `cb_atan_tiles` CB sync, and the data-format reconfig between `sfpu_atan` and `copy_tiles`. | Use input `torch.full((1,1,32,32), 1.0)` → expected output ≈ π/4 ≈ 0.7854 in every element. DPRINT a few output elements. |
| 3. **Reduce stage — single core, scaler-only sanity** | Drop the atan; reader writes the scaler via `calculate_and_prepare_reduce_scaler<cb_scaler, AVG, REDUCE_ROW, 32>()` then streams Wt input tiles. Compute does `reduce<AVG, REDUCE_ROW>(cb_input_tiles, cb_scaler, cb_output_tiles, row(1))`. Shape `(1, 1, 32, 32)`. | Output equals `input.mean(dim=-1)`. Confirms scaler format/layout, matmul-mode REDUCE_ROW dispatch, and per-row scaling. | Use input `torch.full((1,1,32,32), 4.0)` → expected = 4.0 at every col-0 position of the output tile. DPRINT output tile[0,0]. |
| 4. **Full compute — single core, Wt=1** | Combine: `sfpu_atan` → `cb_atan_tiles` → `reduce<AVG, REDUCE_ROW>` → `cb_output_tiles`. Verify against `torch.atan(x).mean(dim=-1)` for shape `(1, 1, 32, 32)`. | Output matches reference within tolerance. End-to-end fusion correctness on a single tile. | Compare against torch on a small sample. |
| 5. **Multi-tile W — Wt > 1** | Bump shape to `(1, 1, 32, 64)` (Wt=2) then `(1, 1, 32, 128)` (Wt=4). Verify per-row reduce over multiple W-tiles works. Confirm `cb_atan_tiles` is sized to Wt pages (not 1) — sizing it to 1 would deadlock on the second tile push from `sfpu_atan` because reduce hasn't started. | Tolerance match across Wt ∈ {1, 2, 4}. | Inject `cb_atan_tiles = 1` deliberately on one debug build to observe the hang — confirms the sizing rule. |
| 6. **Multi-tile H — multi-row-tile per core** | Single core processing multiple row-tiles back-to-back. Shape `(1, 1, 256, 64)` (Ht=8, Wt=2, total_row_tiles=8). | Each output tile's col-0 elements match the corresponding row's mean. Confirms the `for r in [0, num_row_tiles)` loop works and CBs reset cleanly between row-tiles. | DPRINT `r` in each kernel before the inner work. |
| 7. **Multi-batch — non-trivial NC** | Single core still, shape `(2, 4, 32, 64)` (NC=8, Ht=1, Wt=2, total_row_tiles=8). Verifies the tile-id formula `input_tile_id = r * Wt + wt` and `output_tile_id = r` for non-trivial NC packing. | Tolerance match. | — |
| 8. **Multi-core via `split_work_to_cores`** | Wire `core_group_1` / `core_group_2` and per-core RT args. Use the full Tensix grid. Run a high-channel shape, e.g. `(1, 256, 64, 64)` (total_row_tiles = 512 → ~8 per core on an 8×8 grid). | Same correctness with multiple cores; observably faster. | DPRINT `start_row_tile` and `num_row_tiles` per core in the reader (gated by core x/y) — verify the split sums to `total_row_tiles` and that no overlap exists. |
| 9. **Both shape regimes — acceptance run** | Run the full acceptance test (`tests/ttnn/unit_tests/operations/atan_mean/test_atan_mean.py`). Verify "tall" (e.g. `(1,1,2048,64)`) and "high-channel" (e.g. `(1,256,64,64)`) shapes both pass. | All test cases pass with PCC ≥ 0.9995 and max-abs ≤ 1e-2. | — |
| 10. **Negative-case validation** | Run negative parametrize cases (bf16 input, ROW_MAJOR layout, rank != 4, H not tile-aligned, W not tile-aligned). Verify each raises `ValueError`/`RuntimeError` from the Python validator before launch. | All raise — no kernel ever dispatched. | — |

## Hardware Constraints

- [x] CB sync: push count = wait count for every CB (per row-tile for streaming CBs; lifetime for the scaler) — see CB sync table above.
- [x] Reduce scaler CB is **bfloat16** (matches helper expectation; `calculate_and_prepare_reduce_scaler` deduces from CB format).
- [x] Reduce scaler uses the **pool-type-aware** API (`calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::AVG, ReduceDim::REDUCE_ROW, W>`) — NOT the legacy `prepare_reduce_scaler<cb>` overload. AVG+REDUCE_ROW uses matmul col-0 fill (`reduce_helpers_common.hpp:16-19`); the pool-type-aware overload picks the correct fill pattern automatically.
- [x] DEST: `fp32_dest_acc_en = True` + half-sync ⇒ **4 tiles** capacity, surfaced through `DEST_AUTO_LIMIT` (`dest_helpers.hpp:88-102`). Both `sfpu_atan` (batch_size = `DEST_AUTO_LIMIT / 1 = 4`) and `reduce<>` (internal DST management) honor this automatically — no raw `tile_regs_acquire` loops.
- [x] Sequential helper intermediates sized to full block: `cb_atan_tiles` is sized to `Wt` pages (the full row of tiles `sfpu_atan` produces) before `reduce<>` starts consuming. Smaller would deadlock; larger is unnecessary and wastes L1.
- [x] Page sizes aligned to tile size: all tile CBs use `tile_size(dtype)` (4096 B for float32, 2048 B for bfloat16). No row-major CBs in this op.
- [x] All `cb_wait_front` calls on the same CB use the same page count: `cb_input_tiles` waits 1 per inner iteration; `cb_atan_tiles` waits 1 per inner iteration of `reduce<>`; `cb_scaler` waits 1 per `reduce<>` call; `cb_output_tiles` waits 1 per writer iteration. All consistent.
- [x] Helpers are not wrapped with extra CB operations: every helper in this design manages its own `cb_reserve_back`/`cb_push_back`/`cb_wait_front`/`cb_pop_front`. The compute kernel body is essentially the two helper calls inside the row-tile loop.
- [x] Every compute phase uses a helper (`sfpu_atan` for atan, `reduce<>` for the mean-via-AVG-REDUCE_ROW). The only raw-API entries are reader/writer dataflow (no helper exists for tile-streaming NoC I/O), the program-startup scaler-prep (which itself is a helper), and `compute_kernel_hw_startup` (the required prologue per helper docs). Each raw-API entry has a "helpers considered and rejected" justification above.
- [x] `compute_kernel_hw_startup` called once before any helper usage (top of compute kernel).

## Key Risks and Gotchas

| # | Risk | Mitigation |
|---|------|------------|
| 1 | **Undersized `cb_atan_tiles` deadlocks.** `sfpu_atan` and `reduce<>` are both synchronous helpers owning all 3 TRISCs — they cannot overlap. If `cb_atan_tiles` holds fewer than `Wt` pages, `sfpu_atan` blocks on `cb_reserve_back` waiting for space that `reduce<>` will not free until it starts (and it cannot start until `sfpu_atan` finishes). | Size `cb_atan_tiles = Wt` pages. The factory computes `Wt = W // 32` and uses it for the CB's `num_pages`. Asserts on `num_pages` are enforced inside the reduce helper (`reduce_helpers_compute.inl:344` checks `get_cb_num_pages(input_cb_id) >= Wt`). |
| 2 | **Wrong scaler fill pattern.** AVG+REDUCE_ROW uses the matmul reduce path with col-0 scaler fill; AVG+REDUCE_COL uses the standard reduce path with row-0 fill. Using a row-0 scaler for AVG+REDUCE_ROW silently produces an output 32× too small (only one input element contributes per output position). | Use `calculate_and_prepare_reduce_scaler<cb_scaler, PoolType::AVG, ReduceDim::REDUCE_ROW, W>()` — the pool-type-aware overload selects the correct fill pattern. The legacy single-template overload is explicitly forbidden by the planner's hardware checklist. |
| 3 | **Scaler `cb_scaler` push/pop mismatch.** The scaler is pushed once at startup; `reduce<>` calls `cb_wait_front(cb_scaler, 1)` every row-tile but never `cb_pop_front`. Calling `cb_pop_front(cb_scaler, 1)` after the first row-tile would empty it and the next `reduce<>` would hang on `cb_wait_front`. | Do NOT pop `cb_scaler` between row-tiles or at the end of the kernel. The helper's `cb_wait_front` is idempotent (returns immediately once 1 page is available), and the scaler tile persists for the program's lifetime. |
| 4 | **`fp32_dest_acc_en=True` halves DEST.** Hand-rolled DST loops sized for 8 (the bf16 half-sync limit) would overrun the 4-tile fp32 limit. | Use kernel-lib helpers exclusively — both `sfpu_atan` and `reduce<>` reference `DEST_AUTO_LIMIT` (`dest_helpers.hpp:102`) which evaluates to 4 under this config. No raw `tile_regs_acquire` calls in the compute kernel. |
| 5 | **Output tensor metadata vs physical layout.** The output is allocated as rank-4 `(N, C, H, 1)` TILE_LAYOUT (padded to `(N, C, H, 32)` internally — 1 W-tile per row-tile). Only column 0 of each output tile holds valid data. Squeezing to rank-3 on the host is a metadata view. | The entry point allocates rank-4 `(N, C, H, 1)`, dispatches `ttnn.generic_op`, then returns `ttnn.squeeze(out, dim=-1)`. Verified valid for TILE_LAYOUT via `tests/ttnn/unit_tests/base_functionality/test_squeeze.py`. |
| 6 | **Reader's per-core start offset.** With `split_work_to_cores`'s two-group split, group 1 cores get `row_tiles_per_core_g1` row-tiles and group 2 gets `row_tiles_per_core_g2 = row_tiles_per_core_g1 - 1` (or 0). The program factory must walk both groups in the SAME iteration order used to assign offsets — group 1 first (larger work), then group 2 — and feed each core its correct `start_row_tile`. Misaligning the two walks produces overlapping or missing work. | Follow the multigammaln_lanczos pattern (`multigammaln_lanczos_program_descriptor.py:100-121`): single `current_tile = 0` accumulator; iterate `(group_1, tiles_per_core_g1)` then `(group_2, tiles_per_core_g2)`; for each core in each group, assign `[buffer_addr, tiles_per_core, current_tile]` and increment `current_tile += tiles_per_core`. |
| 7 | **Tile-id formula correctness.** Reader needs `input_tile_id = r * Wt + wt` where `r` is the global row-tile index across `(N, C, Ht)`. Writer needs `output_tile_id = r`. Tiles in TILE_LAYOUT interleaved tensors are stored row-major over `(N, C, Ht, Wt)` — the reader's flat tile index walks them in that order. | The same `r` (range = `[start_row_tile, start_row_tile + num_row_tiles_this_core)`) feeds both reader and writer kernels via the same RT args. Reader multiplies by `Wt`; writer doesn't. Verified against the multigammaln_lanczos pattern (where the work unit is a single tile, so its formula simplifies to identity). |
| 8 | **W must be a compile-time arg.** `calculate_and_prepare_reduce_scaler` takes `reduce_factor` as a **template parameter** (`reduce_helpers_dataflow.hpp:94-101`). Passing W as a runtime arg won't compile. | Pass `W` as a compile-time arg to the reader kernel (`reader_ct_args`). The program factory reads it from `input_tensor.shape[-1]` and bakes it into the kernel descriptor. Same for `Wt` to the compute kernel. The implementer can re-instantiate the kernel for each call (the JIT cache keys on CT args, so distinct W values compile separately — same pattern as toy_variance and multigammaln_lanczos). |
| 9 | **Scaler bf16 quantisation of `1/W`.** For W=32 → 1/32 = 0.03125 (exact in bf16). W=64 → 0.015625 (exact). W=128 → 0.0078125 (exact). All powers-of-two W are bit-exact in bf16. Non-power-of-2 W (e.g. W=96) would round, contributing a small error, but Phase 0 shapes all have W ∈ {32, 64, 128}, so this is not an active concern. | Acceptance tolerance (max-abs ≤ 1e-2) accommodates the worst-case bf16 quantisation across the Phase-0 shape set; the policy already accounts for it. |

## Reduce Direction Verification

The op is restricted to `dim=-1` only (REDUCE_ROW). No alternate path is defined — this op never reduces over H/C/N. The single-direction restriction is part of the spec.

| Logical Dim | Tile ReduceDim | Output Valid Region | BroadcastDim | ReduceInputBlockShape | Scaler Fill Pattern (matmul col-0 because AVG+REDUCE_ROW uses matmul) |
|-------------|----------------|---------------------|--------------|-----------------------|----------------------------------------------------------------------|
| `-1` (W) | `ReduceDim::REDUCE_ROW` | Column 0 of each output tile (matmul-mode REDUCE_ROW packs all 32 row results into column 0) | — (no broadcast; reduce only) | `ReduceInputBlockShape::row(Wt)` = `{rows=1, cols=Wt, batches=1}` | Col-0 fill, value = `1/W` in bfloat16 — emitted by `calculate_and_prepare_reduce_scaler<cb_scaler, AVG, REDUCE_ROW, W>` |
