# Reduce W (Multi-Core) Implementation Analysis

## Overview

The reduce_w multi-core program factory performs reduction along the W (width) dimension of a tiled tensor. Each tile-row of width Wt is reduced to a single output tile, producing an output tensor with shape `[N, C, H, 1]` (in tiles: `NC * Ht` output tiles total). The operation supports SUM, AVG, and MAX reduction types, with an optional negation variant (`reduce_w_neg.cpp`) used for implementing reduce-min as `-reduce_max(-x)`.

**Program factory path**: `ttnn/cpp/ttnn/operations/reduction/generic/device/reduce_op_multi_core_w_program_factory.cpp`

**Focus**: Compute kernel structure, CB layout for intermediates, multi-pass data reuse patterns, scalar/constant CB setup, reduce helper parameters, and binary op broadcast patterns.

---

## Work Unit Definition

| Attribute | Value |
|-----------|-------|
| **Granularity** | tile-row |
| **Unit size** | Wt input tiles -> 1 output tile |
| **Total units** | NC * Ht tile-rows (distributed across cores) |
| **Loop structure** | For each assigned tile-row: reduce Wt tiles along W to 1 output tile |

A single "work unit" is one tile-row: the compute kernel reads Wt consecutive input tiles (one row of the W dimension) and reduces them to a single output tile. Each core is assigned `num_rows_per_core` such tile-rows.

---

## Tensor Format and Layout

### Input Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N, C, H, W] |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT (tilized before reduce if originally row-major) |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | Configurable (bfloat16, float32, etc.) |

### Output Tensor

| Property | Value |
|----------|-------|
| **Logical shape** | [N, C, H, 1] (padded to tile: [N, C, Ht*32, 32]) |
| **Dimension convention** | NCHW |
| **Tensor layout** | TILE_LAYOUT |
| **Memory layout** | INTERLEAVED |
| **Buffer type** | DRAM (or L1) |
| **Data type** | Configurable (matches or overrides input dtype) |

### Layout Transformations

The higher-level `reduce()` function in `reduce_op.cpp` (line 104-105) tilizes the input tensor before calling the program factory:
```cpp
auto tilized_input = ttnn::tilize_with_val_padding(input_tensor, padded_shape, pad_value, ...);
```
The program factory itself operates exclusively on TILE_LAYOUT data. No untilize is performed within the factory.

---

## Compute Kernel Variants and Dispatch

The program factory selects between two compute kernels based on the `negate` parameter:

1. **Standard path** (`reduce_w.cpp`): Used for SUM, AVG, MAX reductions.
2. **Negation path** (`reduce_w_neg.cpp`): Used for reduce-min, which is implemented as `-max(-x)`.

Within the standard path, there is a further compile-time dispatch based on the `REDUCE_ROW_SUM_VIA_MM` define:

### Define-based Dispatch (`get_defines()` in `reduce_op.cpp`)

| Reduce Math | Reduce Dim | Defines Set |
|-------------|------------|-------------|
| SUM | W | `REDUCE_OP=PoolType::SUM`, `REDUCE_DIM=ReduceDim::REDUCE_ROW`, **`REDUCE_ROW_SUM_VIA_MM=1`** |
| AVG | W | `REDUCE_OP=PoolType::AVG`, `REDUCE_DIM=ReduceDim::REDUCE_ROW`, **`REDUCE_ROW_SUM_VIA_MM=1`** |
| MAX | W | `REDUCE_OP=PoolType::MAX`, `REDUCE_DIM=ReduceDim::REDUCE_ROW` (no MM flag) |

When `REDUCE_ROW_SUM_VIA_MM` is defined, the compute kernel uses **matmul_tiles** instead of **reduce_tile** for the reduction. This is a precision workaround: matmul accumulation in DST has better numerical behavior than the reduce LLK for SUM/AVG operations.

---

## Circular Buffer Configuration

### Standard Path (no negate)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_src0 | Input tiles from DRAM | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| c_2 | cb_scaler | Scaler/constant tile for reduce | 2 tiles | 1 tile | Double | Reader | Compute | **Program** (persists entire kernel) |
| c_3 | cb_output | Output tiles to DRAM | 2 tiles | 1 tile | Double | Compute | Writer | Block (per row output) |

### Negation Path (negate=true)

| CB ID | Name | Purpose | Capacity | Block Size | Buffering | Producer | Consumer | Lifetime |
|-------|------|---------|----------|------------|-----------|----------|----------|----------|
| c_0 | cb_input | Input tiles from DRAM | 2 tiles | 1 tile | Double | Reader | Compute | Block (per tile) |
| c_2 | cb_scaler | Scaler/constant tile for reduce | 2 tiles | 1 tile | Double | Reader | Compute | **Program** (persists entire kernel) |
| c_3 | cb_output | Output tiles to DRAM | 2 tiles | 1 tile | Double | Compute | Writer | Block (per row output) |
| c_4 | cb_acc | Accumulator for partial row sums | 1 tile | 1 tile | Single | Compute | Compute | **Row** (built up across Wt iterations, consumed at row end) |
| c_5 | cb_ineg | Negated input tile (intermediate) | 1 tile | 1 tile | Single | Compute | Compute | Block (per tile, produced and consumed within same iteration) |

### CB Persistence and Multi-Pass Reuse

**CB c_2 (scaler)**: This is the most critical persistent CB. The scaler tile is written once by the reader kernel at startup and remains in the CB for the entire program duration. The compute kernel issues a single `cb_wait_front(cb_scaler, 1)` at the top (before the main loop) and never pops it. This means:
- The scaler tile is **read-only** from compute's perspective.
- It persists across all NC batches and all Ht rows.
- Every `reduce_tile` or `matmul_tiles` call references it at index 0.

**CB c_4 (accumulator, negate path only)**: This CB acts as a **row-scoped** accumulator. Within a single tile-row reduction (Wt iterations):
- Iteration 0 (wt==0): The accumulator is not read (no prior partial sum). The reduce result is packed to c_4.
- Iterations 1..Wt-1: The previous partial sum is loaded from c_4, the new negated tile is reduced into it, and the updated sum is packed back to c_4.
- After all Wt iterations: The final accumulated tile is read from c_4, negated, and written to c_3 (output).
- c_4 is then empty, ready for the next row.

**CB c_5 (negated intermediate, negate path only)**: This is a **tile-scoped** intermediate. Each iteration produces one negated tile into c_5, which is immediately consumed by the reduce operation in the same iteration.

---

## Scaler/Constant CB Setup

### Standard Path with `REDUCE_ROW_SUM_VIA_MM`

When `REDUCE_ROW_SUM_VIA_MM` is defined, the reader uses `generate_mm_scaler()` from `ttnn/cpp/ttnn/kernel/dataflow/generate_mm_scaler.hpp`:

```cpp
// Reader kernel (line 28-30):
constexpr uint16_t bf16_val = static_cast<uint16_t>(scaler_bits >> 16);
constexpr uint32_t packed_bf16 = static_cast<uint32_t>(bf16_val) | (static_cast<uint32_t>(bf16_val) << 16);
generate_mm_scaler(cb_id_in2, packed_bf16);
```

The `generate_mm_scaler` function:
1. Zeroes the entire tile (2048 bytes for bf16).
2. Fills specific positions with the packed bf16 scaler value. The pattern fills `ptr[i]` for `i in 0..127 step 8` (face 0 first column) and `i in 256..383 step 8` (face 2 first column).
3. This creates a scaler tile where **only the first column of each left face** contains the scaler value, which is the correct format for matmul-based row reduction (the matmul accumulates across the column dimension).

### Standard Path without `REDUCE_ROW_SUM_VIA_MM` (MAX reduction)

When the flag is not set, the reader uses `prepare_reduce_scaler()` from `reduce_helpers_dataflow.hpp`:

```cpp
// Reader kernel (line 24-25):
float scaler_f = __builtin_bit_cast(float, scaler_bits);
dataflow_kernel_lib::prepare_reduce_scaler<cb_id_in2>(scaler_f);
```

The `prepare_reduce_scaler` function:
1. Converts float to bf16 (or f32 depending on CB data format).
2. Zeroes all faces of the tile.
3. Fills **row 0 of every face** with the scaler value (all 16 columns per face, all 4 faces).
4. This format is correct for the `reduce_tile` LLK, which reads scaler values from row 0 of each face during the SRCB broadcast.

### Scaler Value Semantics

The scaler value is passed from host as `operation_attributes.scaler` (a float). Its meaning depends on the operation:
- **SUM**: scaler = 1.0
- **AVG**: scaler = 1/N (where N is the number of elements being averaged, typically W for W reduction)
- **MAX**: scaler = 1.0

The scaler is embedded as a compile-time argument to the reader kernel (index 0): `std::bit_cast<uint32_t>(operation_attributes.scaler)`.

---

## Compute Kernel Structure: Standard Path (`reduce_w.cpp`)

### Initialization

```cpp
compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);
```

**Signature**: `compute_kernel_hw_startup(uint32_t icb0, uint32_t icb1, uint32_t ocb)`

This is called exactly once at kernel start. It configures:
- **Unpacker** hardware for data formats of c_0 (SRCA) and c_2 (SRCB)
- **Math** engine sync and data format registers
- **Packer** hardware for output format of c_3

**Critical rule**: Must be called before any other compute API. It performs MMIO writes requiring idle execution units.

### Path A: `REDUCE_ROW_SUM_VIA_MM` (SUM/AVG via matmul)

When `REDUCE_ROW_SUM_VIA_MM` is defined, the kernel uses matmul instead of reduce:

```cpp
mm_init(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);

cb2.wait_front(1);  // scaler tile persists
for (uint32_t nc = 0; nc < NC; nc++) {
    for (uint32_t ht = 0; ht < Ht; ++ht) {
        acquire_dst();
        for (uint32_t wt = 0; wt < Wt; ++wt) {
            cb0.wait_front(onetile);
            matmul_tiles(tt::CBIndex::c_0, tt::CBIndex::c_2, 0, 0, 0);
            cb0.pop_front(onetile);
        }
        cb3.reserve_back(onetile);
        pack_tile(reduce_dst_idx, tt::CBIndex::c_3);
        cb3.push_back(onetile);
        release_dst();
    }
}
```

**Key API calls**:

| Call | Signature | Purpose |
|------|-----------|---------|
| `mm_init(in0, in1, out)` | `void mm_init(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t out_cb_id, uint32_t transpose=0)` | Initialize matmul hardware (unpacker, math, packer) for tile multiplication |
| `acquire_dst()` | `void acquire_dst()` | Acquire DST registers, zeroing them (deprecated but still used here) |
| `matmul_tiles(in0, in1, i0, i1, idst)` | `void matmul_tiles(uint32_t in0_cb_id, uint32_t in1_cb_id, uint32_t in0_tile_index, uint32_t in1_tile_index, uint32_t idst)` | Tile matmul C=A*B with accumulation: DST[idst] += A[i0] * B[i1]. Each call accumulates into the same DST[0] |
| `pack_tile(dst_idx, ocb)` | `void pack_tile(uint32_t dst_idx, uint32_t ocb)` | Pack DST[dst_idx] to output CB |
| `release_dst()` | `void release_dst()` | Release DST registers for next iteration |

**How matmul achieves row reduction**: The input tile (32x32) is multiplied by the scaler tile which has values only in the first column. The matmul operation `C = A * B` with B having non-zero values only in column 0 effectively sums all columns of A into column 0 of C, weighted by the scaler. Multiple matmul calls accumulate across Wt tiles in DST[0].

**DST register lifecycle per row**: `acquire_dst()` zeros DST, then Wt matmul calls accumulate, then `pack_tile` extracts the result.

### Path B: Standard `reduce_tile` (MAX via reduce LLK)

When `REDUCE_ROW_SUM_VIA_MM` is NOT defined, the kernel delegates to the reduce helper library:

```cpp
compute_kernel_hw_startup(tt::CBIndex::c_0, tt::CBIndex::c_2, tt::CBIndex::c_3);

compute_kernel_lib::reduce<
    REDUCE_OP,                                          // PoolType (e.g., PoolType::MAX)
    REDUCE_DIM,                                         // ReduceDim::REDUCE_ROW
    compute_kernel_lib::ReduceInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::ReduceDataFormatReconfigMode::NONE>(
    tt::CBIndex::c_0,                                   // input CB
    tt::CBIndex::c_2,                                   // scaler CB
    tt::CBIndex::c_3,                                   // output CB
    compute_kernel_lib::ReduceInputBlockShape::of(Ht, Wt, NC));  // block dimensions
```

**Full template signature of `reduce<>()`**:

```cpp
template <
    PoolType reduce_type,
    ReduceDim reduce_dim,
    ReduceInputPolicy input_policy = ReduceInputPolicy::WaitAndPopPerTile,
    ReduceDataFormatReconfigMode reconfig_mode = ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT,
    typename AccumulateT = NoAccumulation,
    typename PostReduceOp = NoOp>
void reduce(
    uint32_t input_cb,
    uint32_t scaler_cb,
    uint32_t output_cb,
    ReduceInputBlockShape input_block_shape,
    ReduceInputMemoryLayout input_memory_layout = ReduceInputMemoryLayout::contiguous(),
    AccumulateT accumulate = AccumulateT{},
    PostReduceOp post_reduce_op = PostReduceOp{});
```

**Template parameters used here**:

| Parameter | Value | Meaning |
|-----------|-------|---------|
| `reduce_type` | `REDUCE_OP` (from define) | PoolType::MAX for MAX, PoolType::SUM for SUM, etc. |
| `reduce_dim` | `REDUCE_DIM` (from define) | `ReduceDim::REDUCE_ROW` -- reduces along W dimension |
| `input_policy` | `WaitAndPopPerTile` | Stream tiles one at a time: wait 1, process 1, pop 1 |
| `reconfig_mode` | `NONE` | Skip data format reconfiguration (first op after hw_startup, formats already correct) |
| `AccumulateT` | `NoAccumulation` (default) | No cross-call accumulation needed |
| `PostReduceOp` | `NoOp` (default) | No post-reduce transformation |

**`ReduceInputBlockShape::of(Ht, Wt, NC)`**: Describes the input as NC batches of Ht rows and Wt columns. For REDUCE_ROW, this means:
- Outer loop: NC batches
- Middle loop: Ht rows per batch
- Inner loop: Wt tiles per row (reduced to 1 output tile)
- Total output: Ht * NC tiles

**Inside the helper (REDUCE_ROW path in `reduce_helpers_compute.inl`, lines 266-348)**:

The helper internally calls these APIs per row:

```
reduce_init<reduce_type, reduce_dim>(input_cb, scaler_cb, output_cb)   // once
cb_wait_front(scaler_cb, 1)                                            // once

for each batch (NC):
  for each row (Ht):
    tile_regs_acquire()
    for each col (Wt):
      cb_wait_front(input_cb, 1)
      reduce_tile<reduce_type, reduce_dim>(input_cb, scaler_cb, 0, 0, dst_idx=0)
      cb_pop_front(input_cb, 1)
    post_reduce_op(dst_idx)          // NoOp in this case
    cb_reserve_back(output_cb, 1)
    tile_regs_commit()
    tile_regs_wait()
    pack_tile(dst_idx, output_cb)
    tile_regs_release()
    cb_push_back(output_cb, 1)

reduce_uninit<fp32_acc>()
```

**`reduce_tile` signature**:
```cpp
template <PoolType reduce_type, ReduceDim reduce_dim, bool enforce_fp32_accumulation = false>
void reduce_tile(uint32_t icb, uint32_t icb_scaler, uint32_t itile, uint32_t itile_scaler, uint32_t idst);
```

Parameters: `icb` = input CB, `icb_scaler` = scaler CB, `itile` = tile index in input CB (0 for WaitAndPopPerTile), `itile_scaler` = scaler tile index (always 0), `idst` = DST register index for accumulation.

The hardware performs: `DST[idst] += reduce_func(unpack(icb[itile]) * unpack(icb_scaler[itile_scaler]))` where the reduce function collapses the row dimension based on `reduce_dim`.

**`reduce_init` / `reduce_uninit` pair**:
```cpp
template <PoolType reduce_type, ReduceDim reduce_dim, bool enforce_fp32_accumulation = false>
void reduce_init(uint32_t icb, uint32_t icb_scaler, uint32_t ocb);

template <bool enforce_fp32_accumulation = false>
void reduce_uninit(uint32_t icb = 0);
```

`reduce_init` configures unpacker (SRCA/SRCB for reduce), math (reduce mode), and **packer edge mask** (critical: `llk_pack_reduce_mask_config` sets which columns/rows of the output tile are valid based on reduce dimension). `reduce_uninit` clears the packer mask so subsequent operations are not affected.

### DST Register Synchronization Protocol

The tile_regs_* functions implement a half-sync double-buffered protocol for DST registers:

| Function | Thread | Purpose |
|----------|--------|---------|
| `tile_regs_acquire()` | MATH/UNPACK | Acquire DST half for compute. Zeros DST registers. |
| `tile_regs_commit()` | MATH | Signal that compute results are ready, transfer ownership to packer |
| `tile_regs_wait()` | PACK | Wait until DST registers are available for packing |
| `tile_regs_release()` | PACK | Release DST registers, making them available for next acquire |

The older `acquire_dst()`/`release_dst()` in the MM path are deprecated equivalents.

---

## Compute Kernel Structure: Negation Path (`reduce_w_neg.cpp`)

This kernel is significantly more complex because it manually implements a negate-reduce-negate pipeline using explicit DST management and two additional intermediate CBs.

### Initialization

```cpp
compute_kernel_hw_startup(cb_input, cb_scaler, cb_output);
```

Same as standard path: configures hardware for c_0 (input), c_2 (scaler), c_3 (output).

### Main Loop Structure

```
cb_scaler_obj.wait_front(1);          // scaler persists
for each batch (NC):
  for each row (Ht):
    for each tile (Wt):
      // Phase 1: Negate input tile -> cb_ineg
      tile_regs_acquire();
      copy_tile_init(cb_input);
      copy_tile(cb_input, 0, dst_idx);
      negative_tile_init();
      negative_tile(dst_idx);
      tile_regs_wait();
      cb_input_obj.pop_front(1);
      cb_ineg_obj.reserve_back(1);
      tile_regs_commit();
      pack_tile(dst_idx, cb_ineg);
      tile_regs_release();
      cb_ineg_obj.push_back(1);

      // Phase 2: Reduce negated tile, accumulate in cb_acc
      tile_regs_acquire();
      if (wt > 0):
        // Reload previous accumulator
        cb_acc_obj.wait_front(1);
        copy_tile_init(cb_acc);
        copy_tile(cb_acc, 0, dst_idx);
      cb_ineg_obj.wait_front(1);
      reduce_init<REDUCE_OP, REDUCE_DIM>(cb_ineg, cb_scaler, cb_acc);
      reduce_tile<REDUCE_OP, REDUCE_DIM>(cb_ineg, cb_scaler, 0, 0, dst_idx);
      reduce_uninit();
      tile_regs_wait();
      cb_ineg_obj.pop_front(1);
      if (wt > 0): cb_acc_obj.pop_front(1);
      cb_acc_obj.reserve_back(1);
      tile_regs_commit();
      pack_tile(dst_idx, cb_acc);
      tile_regs_release();
      cb_acc_obj.push_back(1);

    // Phase 3: Negate accumulated result -> output
    cb_acc_obj.wait_front(1);
    tile_regs_acquire();
    copy_tile_init(cb_acc);
    copy_tile(cb_acc, 0, dst_idx);
    negative_tile_init();
    negative_tile(dst_idx);
    tile_regs_wait();
    cb_acc_obj.pop_front(1);
    cb_output_obj.reserve_back(1);
    tile_regs_commit();
    pack_tile(dst_idx, cb_output);
    tile_regs_release();
    cb_output_obj.push_back(1);
```

### Key API Calls in Negation Path

| Call | Signature | Purpose |
|------|-----------|---------|
| `copy_tile_init(cbid)` | `void copy_tile_init(uint32_t cbid)` | Configure unpacker and math for copy (A2D datacopy mode). Reconfigures SRCA for the given CB's data format. |
| `copy_tile(cb, in_idx, dst_idx)` | `void copy_tile(uint32_t in_cb_id, uint32_t in_tile_index, uint32_t dst_tile_index)` | Unpack tile from CB into DST register. Calls `llk_unpack_A` + `llk_math_eltwise_unary_datacopy`. |
| `negative_tile_init()` | `void negative_tile_init()` | Initialize SFPU for negation operation (SFPU unary kernel init) |
| `negative_tile(idst)` | `void negative_tile(uint32_t idst)` | Negate all elements of DST[idst] in-place using SFPU. Operates on all 4 faces (VectorMode::RC). |
| `reduce_init<OP,DIM>(icb, icb_scaler, ocb)` | (see above) | Initialize reduce hardware. Called inside the inner loop because copy_tile_init corrupts unpacker config. |
| `reduce_tile<OP,DIM>(icb, icb_scaler, itile, itile_scaler, idst)` | (see above) | Perform one reduce tile operation, accumulating into DST[idst] |
| `reduce_uninit()` | (see above) | Reset packer edge masks after reduce |

### Data Format Reconfiguration Pattern

A critical implementation detail: the negation path alternates between **copy_tile** and **reduce_tile** operations, each requiring different unpacker configurations.

- `copy_tile_init(cb_input)` configures SRCA for the input CB's data format
- `copy_tile_init(cb_acc)` configures SRCA for the accumulator CB's data format
- `reduce_init(cb_ineg, cb_scaler, cb_acc)` configures SRCA for cb_ineg's format AND SRCB for cb_scaler's format, plus packer mask

Each `reduce_init`/`reduce_uninit` pair within the inner loop is necessary because:
1. `copy_tile_init` modifies unpacker registers
2. `reduce_tile` requires specific SRCA/SRCB configuration set by `reduce_init`
3. `reduce_uninit` clears the packer mask before the next `copy_tile` + `pack_tile`

This pattern of repeated init/uninit is expensive but necessary for correctness when mixing operation types in a single kernel.

---

## Binary Op Broadcast Pattern for Reduce

The `reduce_tile` operation uses a binary broadcast pattern at the hardware level:

1. **Operand A (SRCA)**: Input tile from `icb` (32x32 elements)
2. **Operand B (SRCB)**: Scaler tile from `icb_scaler` (only row 0 of each face is meaningful)
3. **Operation**: The reduce LLK multiplies each element of A by the corresponding scaler from B's row 0 (broadcast across rows), then accumulates across the specified dimension.

For **REDUCE_ROW** specifically:
- The hardware sums each row of the tile independently
- Row 0 of each face in the scaler tile provides the multiplicative weight
- After Wt calls with the same DST index, DST[0] contains the reduced result where each row holds `sum(A_row * scaler)` across all Wt input tiles
- The packer mask (set by `reduce_init`) ensures only column 0 of the output tile is valid

For the **matmul path** (SUM/AVG):
- The scaler tile has values only in the first column of left faces
- `matmul_tiles(A, B, ...)` computes `DST += A * B`
- With B structured as a column vector, this is equivalent to row-wise dot product, producing a sum in column 0

---

## Pipeline Pattern Summary

### Standard Path

| CB | Capacity | Block | Buffering | Overlap Potential |
|----|----------|-------|-----------|-------------------|
| c_0 (input) | 2 tiles | 1 tile | Double | Reader can prefetch next tile while compute processes current |
| c_2 (scaler) | 2 tiles | 1 tile | Double | N/A (written once, read many) |
| c_3 (output) | 2 tiles | 1 tile | Double | Writer can drain previous tile while compute produces next |

### Negation Path

| CB | Capacity | Block | Buffering | Overlap Potential |
|----|----------|-------|-----------|-------------------|
| c_0 (input) | 2 tiles | 1 tile | Double | Reader can prefetch |
| c_2 (scaler) | 2 tiles | 1 tile | Double | N/A (persistent) |
| c_3 (output) | 2 tiles | 1 tile | Double | Writer can drain |
| c_4 (accumulator) | 1 tile | 1 tile | Single | **No overlap** -- compute produces and consumes this CB itself |
| c_5 (neg intermediate) | 1 tile | 1 tile | Single | **No overlap** -- compute produces and consumes within same iteration |

---

## Index Calculations

### Reader: Tile-to-Memory Mapping

The reader kernel uses TensorAccessor for DRAM access. Tiles are read sequentially starting from `start_id`:
- `start_id = num_tiles_read` (sum of tiles assigned to all previous cores)
- Tiles per core: `num_rows_per_core * Wt`
- Linear tile index `i` maps to NCHW tile order (W-contiguous)

### Compute: No Index Math

The compute kernel does not perform index calculations. Input tiles arrive in W-contiguous order from the reader. The compute kernel simply processes Wt consecutive tiles per row.

### Writer: Output Tile Mapping

Output tiles are written sequentially:
- `start_id = num_tiles_read / Wt` (one output per Wt input tiles)
- `num_output_tiles = num_tiles_per_core / Wt`

---

## Memory Access Patterns

### Read Pattern
- **Sequential tile reads** from DRAM via TensorAccessor
- W-contiguous order: for each row, tiles T[h][0], T[h][1], ..., T[h][Wt-1] are read consecutively
- One tile at a time (single-tile granularity in reader loop)
- `noc_async_read` + `noc_async_read_barrier` per tile (fully blocking)

### Write Pattern
- **Sequential tile writes** to DRAM
- One output tile per Wt input tiles
- `noc_async_write_page` + `noc_async_writes_flushed` per tile

---

## Core Distribution Strategy

| Attribute | Value |
|-----------|-------|
| **Grid topology** | 1D (linearized from 2D grid) |
| **Grid dimensions** | Up to `compute_with_storage_grid_size` (device-dependent) |
| **Total cores** | `num_cores` (determined by `split_work_to_cores`) |
| **Work per core** | `num_rows_per_core` tile-rows (each = Wt input tiles -> 1 output tile) |
| **Load balancing** | Two core groups: group_1 gets `num_rows_per_core_group_1`, group_2 gets `num_rows_per_core_group_2` (differs by at most 1) |
| **Remainder handling** | `split_work_to_cores` distributes `NC*Ht` rows across cores, with group_2 handling the remainder (potentially fewer rows per core) |

The `split_work_to_cores` function divides `NC * Ht` total rows across available cores. If `sub_core_grids` is specified, it restricts to those cores; otherwise, it uses the full compute grid.

---

## Arguments

### Compile-Time Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | scaler_bits | uint32_t | Float scaler value bit-cast to uint32_t (e.g., 1.0f for SUM) |
| 1+ | tensor_accessor_args | uint32_t[] | TensorAccessorArgs for input buffer (bank info, page size, etc.) |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_cb_index | uint32_t | CB index for output (always c_3) |
| 1+ | tensor_accessor_args | uint32_t[] | TensorAccessorArgs for output buffer |

#### Compute Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Ht | uint32_t | Number of tile-rows per core (num_rows_per_core_group_N) |
| 1 | Wt | uint32_t | Number of tiles along W dimension (reduction dimension) |
| 2 | NC | uint32_t | Always 1 (NC is folded into Ht by the program factory) |

**Important**: The program factory passes `NC=1` to the compute kernel and folds the actual batch count into `Ht` (line 47: `num_rows = NC * Ht`). This means the compute kernel sees all assigned rows as a flat sequence, with no batch boundary awareness.

### Runtime Arguments

#### Reader Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Source buffer DRAM address |
| 1 | num_tensor_tiles_per_core | uint32_t | Total input tiles for this core (num_rows * Wt) |
| 2 | start_id | uint32_t | Starting tile index in the input tensor |

#### Writer Kernel

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Destination buffer DRAM address |
| 1 | num_output_tiles | uint32_t | Number of output tiles (num_tiles / Wt) |
| 2 | start_id | uint32_t | Starting tile index in the output tensor |

---

## Kernel Implementations

### Reader Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reader_unary_reduce_universal_start_id | RISCV_0 | NOC0 | DRAM (input tensor) | CB c_0 (input), CB c_2 (scaler) | Read tiles sequentially; generate scaler tile once |

- **File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/dataflow/reader_unary_reduce_universal_start_id.cpp`
- **Key Logic**: Generates scaler tile into c_2 at startup (format depends on REDUCE_ROW_SUM_VIA_MM define), then reads input tiles one at a time into c_0.

### Compute Kernel (Standard)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reduce_w | RISCV_2 (TRISC0/1/2) | N/A | CB c_0 (input), CB c_2 (scaler) | CB c_3 (output) | matmul_tiles or reduce_tile, pack_tile |

- **File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w.cpp`
- **Key Logic**: Two code paths based on REDUCE_ROW_SUM_VIA_MM. MM path uses acquire_dst/release_dst with matmul accumulation. Non-MM path delegates to `compute_kernel_lib::reduce<>()` helper.

### Compute Kernel (Negation)

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| reduce_w_neg | RISCV_2 (TRISC0/1/2) | N/A | CB c_0, c_2 | CB c_3, c_4 (intermediate), c_5 (intermediate) | copy_tile, negative_tile, reduce_tile, pack_tile |

- **File**: `ttnn/cpp/ttnn/operations/reduction/generic/device/kernels/compute/reduce_w_neg.cpp`
- **Key Logic**: Three-phase per-tile pipeline: (1) negate input into c_5, (2) reduce c_5 with accumulation in c_4, (3) negate final c_4 into c_3. Requires repeated reduce_init/reduce_uninit due to format switching between copy and reduce operations.

### Writer Kernel

| Kernel | Core | NOC | Input | Output | Operations |
|--------|------|-----|-------|--------|------------|
| writer_unary_interleaved_start_id | RISCV_1 | NOC1 | CB c_3 (output) | DRAM (output tensor) | Write tiles sequentially |

- **File**: `ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp`
- **Key Logic**: Simple sequential tile drain from c_3 to DRAM.

---

## Implementation Notes

### Precision Workaround: REDUCE_ROW_SUM_VIA_MM

The `REDUCE_ROW_SUM_VIA_MM` pattern is a known precision workaround. The standard `reduce_tile` LLK for SUM/AVG has precision issues, so SUM and AVG row reductions are implemented as matrix multiplications against a scaler column-vector tile. This is set automatically by `get_defines()` when `reduce_op` is SUM or AVG and `reduce_dim` is W. MAX reduction does not use this workaround.

### NC Folding

The program factory flattens `NC * Ht` into a single dimension for work distribution (line 47, 120-123). The compute kernel always receives `NC=1` and sees the folded row count as `Ht`. This simplifies the compute kernel but means it has no awareness of batch boundaries.

### Reduce Helper Library vs Direct API

The standard path (non-negate) uses two different approaches:
- **With REDUCE_ROW_SUM_VIA_MM**: Direct API calls (`mm_init`, `matmul_tiles`, `acquire_dst`, etc.)
- **Without REDUCE_ROW_SUM_VIA_MM**: The `compute_kernel_lib::reduce<>()` helper, which encapsulates all DST management, CB operations, and reduce_init/uninit.

For a new operation like layer_norm_rm, the `compute_kernel_lib::reduce<>()` helper is the recommended approach when applicable, as it handles DST register protocol correctly. Direct API calls should be used when the helper's policy options don't cover the required pattern (e.g., when mixing reduce with other operations like negation).

### Scaler CB Data Format

The scaler CB (c_2) uses `Float16_b` format regardless of the input/output data format (program factory line 39). This is hardcoded because tile creation in the reader generates bf16 values. The reduce LLK handles the mixed-format unpack internally (SRCA reads input format, SRCB reads scaler format).

### Relevance for Layer Norm RM

For a layer_norm_rm operation performing row-wise mean/variance computation:

1. **Row-wise mean**: Use `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>()` with scaler = 1/W, or use SUM with scaler=1.0 then multiply by 1/W separately. The `REDUCE_ROW_SUM_VIA_MM` path will be auto-activated for SUM.

2. **Centering (x - mean)**: Requires broadcasting the mean tile (1 column valid) back across the row. This is a different pattern from reduce -- it uses `sub_tiles_bcast` or similar binary broadcast operations.

3. **Variance computation**: Another REDUCE_ROW of (x-mean)^2, same pattern as mean.

4. **Multi-pass CB reuse**: The reduce_w_neg kernel demonstrates how to use intermediate CBs (c_4, c_5) for multi-pass computation within a single compute kernel. Layer norm will need similar patterns for centering and standardization, likely requiring the input tiles to persist across multiple passes (use `WaitUpfrontNoPop` or `NoWaitNoPop` policies).

5. **Post-reduce operations**: The `post_reduce_op` lambda parameter of `compute_kernel_lib::reduce<>()` can be used for operations like `recip_tile` (1/x) on the reduced result without needing an extra CB round-trip. This is useful for computing 1/sqrt(variance+eps).

---

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does reduce_tile work in the compute API? Specifically for REDUCE_ROW (W dimension reduction) - what does the scaler tile format need to look like, and how does the hardware accumulate results across multiple reduce_tile calls into the same DST register index?"
   **Reason**: Needed to understand the fundamental hardware mechanism behind row reduction and scaler tile interaction.
   **Key Findings**: reduce_tile calls `llk_unpack_AB_reduce` (unpacks input to SRCA and scaler to SRCB) and `llk_math_reduce` (performs reduction math). Multiple calls to the same DST index accumulate results. Scaler values go in row 0 of each face. For SUM, scaler=1.0; for AVG, scaler=1/N.

2. **Query**: "What is the REDUCE_ROW_SUM_VIA_MM pattern in tt-metal? When is matmul_tiles used instead of reduce_tile for row reduction?"
   **Reason**: Needed to understand why there are two code paths in reduce_w.cpp and when each is selected.
   **Key Findings**: This is a precision workaround. The reduce_tile LLK has precision issues for SUM/AVG, so row summation is implemented as matmul with a column-vector scaler tile. The `get_defines()` function automatically sets this for SUM/AVG W-dimension reductions.

3. **Query**: "How do tile_regs_acquire, tile_regs_commit, tile_regs_wait, and tile_regs_release work?"
   **Reason**: Needed to understand DST register synchronization protocol used in the reduce helper and negation path.
   **Key Findings**: These implement a half-sync double-buffered protocol. acquire zeros DST and claims it for math/unpack. commit transfers ownership to packer. wait blocks packer until data ready. release frees DST for next acquire. In half-sync mode (default), 8 tiles available (4 with fp32 accum).

4. **Query**: "How does the binary op broadcast pattern work for reduce operations in tt-metal?"
   **Reason**: Needed to understand the SRCB broadcast mechanism used by reduce_tile.
   **Key Findings**: For REDUCE_ROW, the scaler tile's row 0 values are broadcast across all rows of the input tile during unpack. The reduce math then sums/maxes each row independently. PoolType::SUM and PoolType::AVG differ only in the scaler value (1.0 vs 1/N); the hardware always performs a scaled sum.

### Documentation References

1. **Source**: `reduce_helpers_compute.hpp` and `reduce_helpers_compute.inl`
   **Reason**: Primary reference for the reduce helper library's template API, input policies, and accumulation support.
   **Key Information**: Full template signature, 4 input policies (WaitAndPopPerTile, BulkWaitBulkPop, WaitUpfrontNoPop, NoWaitNoPop), accumulation support, post_reduce_op callback, and data format reconfig modes.

2. **Source**: `reduce_helpers_dataflow.hpp` and `reduce_helpers_dataflow.inl`
   **Reason**: Understanding scaler tile generation (prepare_reduce_scaler vs calculate_and_prepare_reduce_scaler).
   **Key Information**: Scaler tile format: zeros the tile, then fills row 0 of each face with the scaler value. For bf16: packs two bf16 values per u32. For REDUCE_SCALAR with AVG, uses 1/sqrt(N) since LLK applies scaler twice.

3. **Source**: `tt_metal/hw/inc/api/compute/reduce.h`
   **Reason**: Authoritative API documentation for reduce_init, reduce_tile, reduce_uninit.
   **Key Information**: reduce_init configures unpacker, math, and packer (including edge masks). reduce_uninit clears packer masks. reduce_tile performs unpack + math in one call. enforce_fp32_accumulation template parameter available on all three.

4. **Source**: `tt_metal/hw/inc/api/compute/compute_kernel_hw_startup.h`
   **Reason**: Understanding hardware initialization requirements.
   **Key Information**: Must be called exactly once at kernel start. Configures unpacker (SRCA/SRCB), math engine, and packer. Performs MMIO writes requiring idle state. CB IDs must match next init call's CBs.

5. **Source**: `ttnn/cpp/ttnn/kernel_lib/dest_helpers.hpp`
   **Reason**: Understanding DEST_AUTO_LIMIT and DST register capacity.
   **Key Information**: DST capacity depends on sync mode and accumulation mode. Half-sync (default): 8 tiles (bf16) or 4 tiles (fp32). Full-sync: 16 tiles (bf16) or 8 tiles (fp32). DEST_AUTO_LIMIT is a constexpr computed from JIT-generated headers.

6. **Source**: `tt_metal/hw/inc/api/compute/matmul.h`
   **Reason**: Understanding mm_init and matmul_tiles signatures used in the REDUCE_ROW_SUM_VIA_MM path.
   **Key Information**: mm_init configures all three components (unpack, math, pack) for matmul. matmul_tiles accumulates: DST[idst] += A[i0] * B[i1]. No explicit zero -- acquire_dst/tile_regs_acquire handles that.
