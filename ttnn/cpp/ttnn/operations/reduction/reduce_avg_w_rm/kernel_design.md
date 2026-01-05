# Kernel Design: reduce_avg_w_rm

## Design Summary

| Kernel | Phases | Helpers Used | Raw Calls Needed |
|--------|--------|--------------|------------------|
| Reader | 2 | None (dataflow kernel) | `generate_reduce_scaler()`, `noc_async_read()`, CB ops |
| Compute | 3 | `compute_kernel_lib::tilize()`, `compute_kernel_lib::reduce()`, `compute_kernel_lib::untilize()` | None |
| Writer | 1 | None (dataflow kernel) | `noc_async_write()`, CB ops |

## Helper Library Analysis

### Available Helpers Reviewed
- [x] tilize_helpers.hpp - **Relevant**: `tilize()` for ROW_MAJOR to TILE_LAYOUT conversion
- [x] untilize_helpers.hpp - **Relevant**: `untilize()` for TILE_LAYOUT to ROW_MAJOR conversion
- [x] reduce_helpers.hpp - **Relevant**: `reduce()` with `REDUCE_ROW` for width reduction
- [x] dest_helpers.hpp - **Relevant**: Auto-detects DEST register limits (used internally by helpers)

### Helper Functions Applicable to This Operation

| Helper | Signature | Use Case in This Op |
|--------|-----------|---------------------|
| `compute_kernel_lib::tilize()` | `tilize<init, uninit, use_fast, use_dt, skip_wait>(icb, block_w, ocb, num_blocks, subblock_h, old_icb, input_count, total_rows)` | Tilize phase: Convert 32 row-major sticks to Wt tiles |
| `compute_kernel_lib::reduce()` | `reduce<PoolType, ReduceDim, ReduceInputMode, init, uninit>(icb, icb_scaler, ocb, Ht, Wt, num_batches, row_chunk, input_stride)` | Reduce phase: Sum Wt tiles across width with scaler 1/W |
| `compute_kernel_lib::untilize()` | `untilize<tile_width, icb_id, ocb_id, init, uninit, wait_upfront>(num_rows, block_rt_dim, total_tiles)` | Untilize phase: Convert 1 reduced tile to 32 row-major sticks |

## Reader Kernel Design

### Prerequisites
- Include: `#include "api/dataflow/dataflow_api.h"`
- Uses `generate_reduce_scaler()` from dataflow API
- Uses TensorAccessor for input stick addressing

### Phase 1: Scaler Generation (Once per kernel invocation)
- **Description**: Generate a single scaler tile containing value 1/W for averaging
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernels use dataflow API, not compute helpers)
  - **RAW CALLS**:
    - `cb_reserve_back(CB_scaler, 1)` - Reserve space for scaler tile
    - `generate_reduce_scaler(CB_scaler, packed_scaler_value)` - Generate tile filled with 1/W
    - `cb_push_back(CB_scaler, 1)` - Push scaler tile (persists for entire kernel)
- **CB Flow**:
  - CB_scaler (c_2): reserve(1) -> generate_reduce_scaler -> push(1)
  - This tile persists for the entire program lifetime

### Phase 2: Stick Reading (Per tile row)
- **Description**: Read 32 consecutive row-major sticks (one tile height) from DRAM into CB_rm_in
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernels don't use compute helpers)
  - **RAW CALLS**:
    - `cb_reserve_back(CB_rm_in, Wt)` - Reserve Wt tiles worth of space (32 sticks = Wt tiles)
    - Loop over 32 sticks:
      - Calculate stick address using TensorAccessor
      - `noc_async_read(noc_addr, l1_write_addr, stick_size)` - Read one stick
      - Advance l1_write_addr by stick_size
    - `noc_async_read_barrier()` - Wait for all reads to complete
    - `cb_push_back(CB_rm_in, Wt)` - Signal Wt tiles worth of data ready
- **CB Flow**:
  - CB_rm_in (c_0): Per tile row: reserve(Wt) -> read 32 sticks -> barrier -> push(Wt)

### Stick Addressing Logic
```
For each tile_row in 0..num_tile_rows:
    stick_base = start_tile_row * 32 + tile_row * 32
    For stick in 0..31:
        global_stick_id = stick_base + stick
        noc_addr = tensor_accessor.get_noc_addr(global_stick_id)
        // Each stick is W * element_size bytes
```

## Compute Kernel Design

### Prerequisites
- [x] Requires `compute_kernel_hw_startup()`: **Yes** (required before any helper calls)
- [x] Required defines:
  - `REDUCE_OP = PoolType::SUM` (for reduce helper)
  - `REDUCE_DIM = ReduceDim::REDUCE_ROW` (for width reduction)
  - `ENABLE_FP32_DEST_ACC = 1` when dtype is FLOAT32 (optional, for FP32 accumulation)

### Phase 1: Tilize
- **Description**: Convert Wt tiles worth of row-major stick data into Wt tiles in TILE_LAYOUT
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::tilize()`
  - **Parameters**: `tilize(CB_rm_in, Wt, CB_tilized, 1)` (per tile row iteration)
    - `icb = CB_rm_in` (c_0)
    - `block_w = Wt` (tiles per row)
    - `ocb = CB_tilized` (c_1)
    - `num_blocks = 1` (process one tile row at a time)
    - Default params: `subblock_h = 1`, `old_icb = 0`, `input_count = 0`, `total_rows = 0`
  - **CB Management**: Helper handles internally:
    - `cb_wait_front(CB_rm_in, Wt)`
    - `cb_reserve_back(CB_tilized, Wt)`
    - `tilize_block(CB_rm_in, Wt, CB_tilized)`
    - `cb_push_back(CB_tilized, Wt)`
    - `cb_pop_front(CB_rm_in, Wt)`
  - **Init/Uninit**: Use `init=true, uninit=true` for first/last tile row, or `init=true, uninit=false` inside loop to avoid repeated init/uninit overhead
- **CB Flow**: Helper level - wait(Wt) from CB_rm_in, push(Wt) to CB_tilized

### Phase 2: Reduce (Width Reduction with Streaming Input)
- **Description**: Sum all Wt tiles into 1 tile, multiplying by scaler 1/W to compute average
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW>()`
  - **Parameters**: `reduce<SUM, REDUCE_ROW>(CB_tilized, CB_scaler, CB_reduced, 1, Wt, 1)` (per tile row)
    - `icb = CB_tilized` (c_1)
    - `icb_scaler = CB_scaler` (c_2)
    - `ocb = CB_reduced` (c_3)
    - `Ht = 1` (processing one tile row at a time)
    - `Wt = Wt` (tiles across width to reduce)
    - `num_batches = 1` (one tile row per iteration)
  - **CB Management**: Helper handles internally (STREAMING mode):
    - `cb_wait_front(CB_scaler, 1)` (once at init)
    - For each tile in Wt:
      - `cb_wait_front(CB_tilized, 1)`
      - `reduce_tile<SUM, REDUCE_ROW>(...)` with accumulation in DST
      - `cb_pop_front(CB_tilized, 1)`
    - `cb_reserve_back(CB_reduced, 1)`
    - `pack_tile(0, CB_reduced)`
    - `cb_push_back(CB_reduced, 1)`
  - **DST Management**: Helper handles tile_regs_acquire/commit/wait/release
- **CB Flow**: Helper level - stream Wt tiles from CB_tilized, push 1 tile to CB_reduced

### Phase 3: Untilize
- **Description**: Convert 1 reduced tile back to row-major format (32 sticks of width 32)
- **Implementation Approach**:
  - **USE HELPER**: Yes
  - **Helper**: `compute_kernel_lib::untilize<>()`
  - **Parameters**: `untilize<1, CB_reduced, CB_rm_out>(1)` (per tile row)
    - `tile_width = 1` (1 tile to untilize)
    - `icb_id = CB_reduced` (c_3)
    - `ocb_id = CB_rm_out` (c_16)
    - `num_rows = 1` (1 tile row)
  - **CB Management**: Helper handles internally:
    - `cb_wait_front(CB_reduced, 1)`
    - `cb_reserve_back(CB_rm_out, 1)`
    - `pack_untilize_block<1, 1>(CB_reduced, 1, CB_rm_out, 0)` (or standard untilize)
    - `cb_pop_front(CB_reduced, 1)`
    - `cb_push_back(CB_rm_out, 1)`
- **CB Flow**: Helper level - wait(1) from CB_reduced, push(1) to CB_rm_out

### Compute Kernel Loop Structure

```cpp
// PSEUDO-CODE showing helper usage
compute_kernel_hw_startup(CB_rm_in, CB_tilized, CB_scaler, CB_reduced, CB_rm_out);

for (uint32_t tile_row = 0; tile_row < num_tile_rows; ++tile_row) {
    // Phase 1: Tilize (helper handles all CB ops)
    compute_kernel_lib::tilize(CB_rm_in, Wt, CB_tilized, 1);

    // Phase 2: Reduce (helper handles all CB ops and DST management)
    compute_kernel_lib::reduce<SUM, REDUCE_ROW>(CB_tilized, CB_scaler, CB_reduced, 1, Wt, 1);

    // Phase 3: Untilize (helper handles all CB ops)
    compute_kernel_lib::untilize<1, CB_reduced, CB_rm_out>(1);
}
```

**OPTIMIZATION NOTE**: The helpers have `init`/`uninit` template parameters. For better performance, the kernel writer MAY choose to:
1. Call init once before the loop
2. Call uninit once after the loop
3. Use `init=false, uninit=false` inside the loop

This requires careful sequencing of init calls since tilize, reduce, and untilize each have their own init/uninit.

## Writer Kernel Design

### Prerequisites
- Include: `#include "api/dataflow/dataflow_api.h"`
- Uses TensorAccessor for output stick addressing

### Phase 1: Stick Writing (Per tile row)
- **Description**: Write 32 output sticks (each width=32 elements) from CB_rm_out to DRAM
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernels don't use compute helpers)
  - **RAW CALLS**:
    - `cb_wait_front(CB_rm_out, 1)` - Wait for 1 tile worth of untilized data
    - `l1_read_addr = get_read_ptr(CB_rm_out)`
    - Loop over 32 sticks:
      - Calculate output stick address using TensorAccessor
      - `noc_async_write(l1_read_addr, noc_addr, output_stick_size)` - Write one stick
      - Advance l1_read_addr by output_stick_size
    - `noc_async_write_barrier()` - Wait for all writes to complete
    - `cb_pop_front(CB_rm_out, 1)` - Release the tile
- **CB Flow**:
  - CB_rm_out (c_16): Per tile row: wait(1) -> write 32 sticks -> barrier -> pop(1)

### Stick Addressing Logic
```
For each tile_row in 0..num_tile_rows:
    stick_base = start_tile_row * 32 + tile_row * 32
    For stick in 0..31:
        global_stick_id = stick_base + stick
        noc_addr = tensor_accessor.get_noc_addr(global_stick_id)
        // Each output stick is 32 * element_size bytes
```

## CB Synchronization Summary

| CB | ID | Producer | Consumer | Pages per Block | Sync Point | Lifetime |
|----|-----|----------|----------|-----------------|------------|----------|
| CB_rm_in | c_0 | Reader | Compute (tilize) | Wt | Reader pushes Wt, Compute waits Wt | Block |
| CB_tilized | c_1 | Compute (tilize) | Compute (reduce) | Wt (push) / 1 (pop streaming) | Tilize pushes Wt, Reduce streams 1 at a time | Block |
| CB_scaler | c_2 | Reader | Compute (reduce) | 1 | Reader pushes 1 (once), Reduce waits 1 | Program |
| CB_reduced | c_3 | Compute (reduce) | Compute (untilize) | 1 | Reduce pushes 1, Untilize waits 1 | Block |
| CB_rm_out | c_16 | Compute (untilize) | Writer | 1 | Untilize pushes 1, Writer waits 1 | Block |

### CB Flow Diagram (Per Tile Row)
```
Reader                    Compute                         Writer
-------                   -------                         ------

[Generate scaler]
  push(c_2, 1)  ------>   [reduce waits c_2, 1 once]

[Read 32 sticks]
  push(c_0, Wt) ------>   [tilize]
                            wait(c_0, Wt)
                            push(c_1, Wt) --\
                            pop(c_0, Wt)    |
                                            |
                          [reduce]          |
                            wait(c_1, 1) <--/  (streaming)
                            pop(c_1, 1)
                            ... repeat Wt times ...
                            push(c_3, 1) --\
                                            |
                          [untilize]        |
                            wait(c_3, 1) <--/
                            push(c_16, 1) ------>  [Write 32 sticks]
                            pop(c_3, 1)              wait(c_16, 1)
                                                     pop(c_16, 1)
```

## Helper Encapsulation Acknowledgment

For phases marked "USE HELPER", the following is encapsulated BY THE HELPER:

### `compute_kernel_lib::tilize()` encapsulates:
- `tilize_init()` / `tilize_uninit()` (when init/uninit=true)
- `cb_wait_front(icb, block_w)` / `cb_pop_front(icb, block_w)`
- `cb_reserve_back(ocb, block_w)` / `cb_push_back(ocb, block_w)`
- `tilize_block(icb, block_w, ocb)` hardware operation

### `compute_kernel_lib::reduce()` encapsulates:
- `reduce_init<type, dim>()` / `reduce_uninit()` (when init/uninit=true)
- `cb_wait_front(icb_scaler, 1)` for scaler
- `tile_regs_acquire()` / `tile_regs_commit()` / `tile_regs_wait()` / `tile_regs_release()`
- For STREAMING mode:
  - `cb_wait_front(icb, 1)` / `cb_pop_front(icb, 1)` per input tile
- `cb_reserve_back(ocb, 1)` / `cb_push_back(ocb, 1)` per output tile
- `reduce_tile<type, dim>(...)` hardware operation
- `pack_tile(dst_idx, ocb)` to write result

### `compute_kernel_lib::untilize()` encapsulates:
- `pack_untilize_init()` or `untilize_init()` (when init=true)
- `pack_untilize_uninit()` or `untilize_uninit()` (when uninit=true)
- `cb_wait_front(icb, tile_width)` / `cb_pop_front(icb, tile_width)`
- `cb_reserve_back(ocb, tile_width)` / `cb_push_back(ocb, tile_width)`
- `pack_untilize_block<>()` or `untilize_block()` hardware operation

**CRITICAL**: The kernel writer MUST NOT add redundant CB or DST operations around helper calls. The helper functions are self-contained.

## Special Considerations

### 1. CB_tilized Sizing Concern
The spec indicates CB_tilized should be double-buffered with 2 tiles. However, the tilize helper pushes Wt tiles at once, while reduce streams 1 tile at a time.

**Resolution**: The factory allocates CB_tilized with only 2 tiles (from program_factory.cpp line 70-71). This is insufficient if Wt > 2.

**Required Factory Fix**: CB_tilized should be sized to hold at least Wt tiles:
```cpp
uint32_t num_tilized_tiles = Wt;  // Or 2 * Wt for true double-buffering
```

Alternatively, the tilize phase can be modified to push 1 tile at a time to match reduce's streaming consumption. This would require `init=true, uninit=false` for first, `init=false, uninit=false` for middle, and `init=false, uninit=true` for last tile within a block.

### 2. Reduce Init/Uninit Overhead
Since reduce is called inside the main loop, using `init=true, uninit=true` every iteration has overhead. The kernel writer should consider:
- Calling `reduce_init<>()` once before the loop
- Using `reduce<..., false, false>()` inside the loop
- Calling `reduce_uninit()` once after the loop

### 3. Untilize Helper Template Requirements
The `untilize<>()` helper requires compile-time constant CB IDs due to template parameters:
```cpp
untilize<1, tt::CBIndex::c_3, tt::CBIndex::c_16>(1);
```

## Implementation Checklist for Kernel Writer

### Reader Kernel
- [ ] Include `"api/dataflow/dataflow_api.h"`
- [ ] Generate scaler tile once using `generate_reduce_scaler(CB_scaler, packed_scaler_value)`
- [ ] For each tile row: reserve Wt, read 32 sticks, barrier, push Wt
- [ ] Use TensorAccessor for stick addressing with stick_size page size

### Compute Kernel
- [ ] Include helpers: `tilize_helpers.hpp`, `reduce_helpers.hpp`, `untilize_helpers.hpp`
- [ ] Define `REDUCE_OP` and `REDUCE_DIM` before includes (or use explicit template args)
- [ ] Call `compute_kernel_hw_startup()` before any helper
- [ ] For each tile row, call in sequence:
  1. `compute_kernel_lib::tilize(c_0, Wt, c_1, 1)`
  2. `compute_kernel_lib::reduce<SUM, REDUCE_ROW>(c_1, c_2, c_3, 1, Wt, 1)`
  3. `compute_kernel_lib::untilize<1, c_3, c_16>(1)`
- [ ] DO NOT add cb_wait/pop/reserve/push around helper calls
- [ ] Consider init/uninit optimization for loop

### Writer Kernel
- [ ] Include `"api/dataflow/dataflow_api.h"`
- [ ] For each tile row: wait 1 tile, write 32 sticks, barrier, pop 1 tile
- [ ] Use TensorAccessor for stick addressing with output_stick_size page size
- [ ] Output stick width is 32 elements (not W)

### Verification
- [ ] CB push/pop counts match: Reader pushes Wt to c_0, Compute waits Wt from c_0
- [ ] Scaler is generated once and persists
- [ ] Factory CB sizes are sufficient (especially CB_tilized needs Wt capacity)
