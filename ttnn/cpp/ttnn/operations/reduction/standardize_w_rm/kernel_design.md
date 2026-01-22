# Kernel Design: standardize_w_rm

## Spec Validation Issues

### Issue 1: CB_16 Dual-Use Clarification
- **Spec says**: CB_16 is used for both multiply output (tiled) and untilize output (RM sticks)
- **Problem**: This is unusual - typically untilize needs separate input/output CBs
- **Resolution**: With 2*Wt double-buffered capacity, this works: multiply pushes Wt tiled tiles, untilize waits for Wt, reserves Wt more (into second half), processes, pushes Wt RM, pops Wt tiled. The helper handles this correctly.

### Issue 2: Phase 4 CB_4 Persistence Policy
- **Spec says**: CB_4 is PERSISTENT from Phase 3 through Phase 8
- **Problem**: The square operation in Phase 4 reads CB_4 but must NOT pop it (tiles needed for Phase 8)
- **Resolution**: Use custom CB policy `InputPolicy<WaitCallerManaged, PopNever>` (PreloadedNoPop) for Phase 4's input A. This ensures tiles remain for Phase 8.

### Issue 3: Phases 6-7 Can Be Combined
- **Spec says**: Phase 6 (add epsilon) outputs to CB_6, Phase 7 (rsqrt) reads CB_6 and outputs to CB_8
- **Problem**: Not an error, but inefficient - intermediate can stay in DST
- **Resolution**: Combine Phases 6-7 into a single DST-based sequence: copy variance to DST, add epsilon in DST, rsqrt in DST, pack to CB_8. This matches batch_norm pattern and eliminates CB_6 intermediate read/write.

## Data Semantics Model (MANDATORY)

### Buffer Content Analysis

| CB | Layout | Logical Shape | Tile Shape | Valid Region | Lifetime |
|----|--------|---------------|------------|--------------|----------|
| c_0 | RM | [32, W] | N/A (sticks) | All | Per tile-row (reader->tilize) |
| c_1 | TILE | [32, W] | 1 x Wt | All | PERSISTENT: Phase 1-3 |
| c_2 | TILE | [1] (scalar) | 1 x 1 | Row0 (bcast19) | Program lifetime |
| c_3 | TILE | [32, 1] (mean) | 1 x 1 | Col0 | Per tile-row (reduce->sub) |
| c_4 | TILE | [32, W] | 1 x Wt | All | PERSISTENT: Phase 3-8 |
| c_5 | TILE | [32, W] | 1 x Wt | All | Per tile-row (square->reduce) |
| c_6 | TILE | [32, 1] (var) | 1 x 1 | Col0 | Per tile-row (reduce->add_rsqrt) |
| c_7 | TILE | [1] (epsilon) | 1 x 1 | Row0 (bcast19) | Program lifetime |
| c_8 | TILE | [32, 1] (rsqrt) | 1 x 1 | Col0 | Per tile-row (rsqrt->mul) |
| c_16 | TILE->RM | [32, W] | 1 x Wt -> sticks | All | Per tile-row (mul->untilize->writer) |

### Binary Op Broadcast Verification (MANDATORY)

| Phase | Op | CB_A | CB_A Valid | CB_B | CB_B Valid | Broadcast |
|-------|-----|------|------------|------|------------|-----------|
| 3 | sub | c_1 | All | c_3 | Col0 | **COL** |
| 4 | square | c_4 | All | c_4 | All | NONE |
| 6 | add | c_6 | Col0 | c_7 | Row0 (scalar) | NONE (single tile) |
| 8 | mul | c_4 | All | c_8 | Col0 | **COL** |

**Broadcast verification**:
- Phase 3: Full tensor (All) - mean (Col0) requires **COL** broadcast to replicate mean across columns
- Phase 4: Same tensor (All) * same tensor (All) - **NONE** (self-multiply)
- Phase 6: Single tile (Col0) + single tile (scalar) - **NONE** works (element-wise on 1 tile)
- Phase 8: Full tensor (All) * rsqrt (Col0) requires **COL** broadcast to replicate rsqrt across columns

### Dataflow Graph

```
DRAM (RM sticks [H, W])
    |
    v [Reader: 32 sticks per tile-row]
c_0 (RM sticks, 2*Wt)
    |
    v [Phase 1: tilize]
c_1 (tiled, Wt) ──────────────────────────┐
    |                                      |
    v [Phase 2: reduce PERSISTENT]         |
c_3 (mean, 1 tile, Col0 valid)            |
    |                                      |
    v [Phase 3: sub COL] <─────────────────┘
c_4 (centralized, Wt, All valid) ─────────┬───────────────────────┐
    |                                      | (persist for Phase 8) |
    v [Phase 4: square, NO POP]            |                       |
c_5 (squared, Wt, All valid)              |                       |
    |                                      |                       |
    v [Phase 5: reduce STREAMING]          |                       |
c_6 (variance, 1 tile, Col0 valid)        |                       |
    |                                      |                       |
    v [Phase 6-7: add eps + rsqrt]         |                       |
c_8 (rsqrt, 1 tile, Col0 valid)           |                       |
    |                                      |                       |
    v [Phase 8: mul COL] <─────────────────┘                       |
c_16 (output tiled, Wt) <──────────────────────────────────────────┘
    |
    v [Phase 9: untilize]
c_16 (RM sticks, Wt tiles worth)
    |
    v [Writer: 32 sticks per tile-row]
DRAM (RM sticks [H, W])
```

### Persistence Analysis

| CB | Read Count | Last Reader | Can Release After | Persist? |
|----|------------|-------------|-------------------|----------|
| c_0 | 1 | Phase 1 | Phase 1 | No |
| c_1 | 2 | Phase 3 | Phase 3 | Yes (Phases 1-3) |
| c_2 | 2 | Phase 5 | Program end | Yes (Program) |
| c_3 | 1 | Phase 3 | Phase 3 | No |
| c_4 | 2 | Phase 8 | Phase 8 | Yes (Phases 3-8) |
| c_5 | 1 | Phase 5 | Phase 5 | No |
| c_6 | 1 | Phase 6-7 | Phase 6-7 | No |
| c_7 | 1 | Phase 6-7 | Program end | Yes (Program) |
| c_8 | 1 | Phase 8 | Phase 8 | No |
| c_16 | 2 | Writer | Writer | No |

## Design Summary

| Kernel | Phases | Helpers Used | Raw Calls Needed |
|--------|--------|--------------|------------------|
| Reader | 2 (scaler gen, epsilon gen, stick read) | generate_reduce_scaler, generate_bcast19_scalar | noc_async_read, TensorAccessor |
| Compute | 9 | tilize(), reduce(), sub(), square(), reduce(), untilize() | add_binary_tile, rsqrt_tile (Phases 6-7) |
| Writer | 1 (stick write) | None | noc_async_write, TensorAccessor |

## Helper Library Analysis

### Available Helpers Reviewed
- [x] tilize_helpers.hpp - YES, applicable for Phase 1
- [x] untilize_helpers.hpp - YES, applicable for Phase 9
- [x] reduce_helpers.hpp - YES, applicable for Phases 2 and 5
- [x] binary_op_helpers.hpp - YES, applicable for Phases 3, 4, 8
- [x] dest_helpers.hpp - YES, for DEST register limit awareness

### Helper Functions Applicable to This Operation

| Helper | Signature | Use Case in This Op |
|--------|-----------|---------------------|
| `compute_kernel_lib::tilize()` | `tilize(icb, block_w, ocb, num_blocks)` | Phase 1: RM to tiled |
| `compute_kernel_lib::reduce<SUM, REDUCE_ROW, PERSISTENT>()` | `reduce(icb, scaler_cb, ocb, TileShape::row(Wt))` | Phase 2: Mean |
| `compute_kernel_lib::sub<COL>()` | `sub<COL, PreloadedPopAtEnd, WaitUpfrontPopAtEnd>(...)` | Phase 3: Centralize |
| `compute_kernel_lib::binary_op<SQUARE>()` | `binary_op<SQUARE, NONE, PreloadedNoPop>(...)` | Phase 4: Square (no pop!) |
| `compute_kernel_lib::reduce<SUM, REDUCE_ROW, STREAMING>()` | `reduce(icb, scaler_cb, ocb, TileShape::row(Wt))` | Phase 5: Variance |
| `compute_kernel_lib::untilize<Wt>()` | `untilize<Wt, icb, ocb>(num_rows)` | Phase 9: Tiled to RM |

## Reader Kernel Design

### Phase 0: Initialization (Once at program start)

**Scaler Generation (1/W)**:
- **Description**: Generate reduce scaler tile containing 1/W for mean and variance computation
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernels don't use compute helpers)
  - **RAW CALLS**: Use `generate_reduce_scaler(cb_scaler, packed_scaler_value)`
    - cb_reserve_back(cb_scaler, 1)
    - Fill tile with zeros, write packed 1/W value
    - cb_push_back(cb_scaler, 1)
- **CB Flow**: Reserve 1, generate, push 1 to c_2 (persists for program)

**Epsilon Generation**:
- **Description**: Generate epsilon scalar tile for numerical stability
- **Implementation Approach**:
  - **USE HELPER**: No
  - **RAW CALLS**: Use `generate_bcast19_scalar(cb_epsilon, packed_epsilon_value)`
    - cb_reserve_back(cb_epsilon, 1)
    - Fill tile with epsilon value in bcast19 pattern
    - cb_push_back(cb_epsilon, 1)
- **CB Flow**: Reserve 1, generate, push 1 to c_7 (persists for program)

### Phase 1: Input Reading (Per tile-row)

- **Description**: Read 32 RM sticks from DRAM into c_0
- **Implementation Approach**:
  - **USE HELPER**: No
  - **RAW CALLS**:
    - `cb_reserve_back(c_0, Wt)` - reserve space for one tile-row
    - For each of 32 sticks:
      - `noc_addr = accessor.get_noc_addr(stick_id)`
      - `noc_async_read(noc_addr, l1_write_addr, input_stick_size)`
      - Advance l1_write_addr by input_stick_size
    - `noc_async_read_barrier()` - wait for all reads
    - `cb_push_back(c_0, Wt)` - signal data ready
- **CB Flow**: Reserve Wt pages, read 32 sticks, barrier, push Wt pages

## Compute Kernel Design

### Prerequisites
- [x] Requires `compute_kernel_hw_startup()`: YES
- [x] Template parameters for reduce helper:
  - `PoolType`: SUM (for both mean and variance)
  - `ReduceDim`: REDUCE_ROW (reduces width dimension)
  - `ReduceInputMode`: PERSISTENT (Phase 2), STREAMING (Phase 5)
  - `ReduceDataFormatReconfig`: BOTH (default)

### Custom CB Policies Needed

```cpp
// Phase 4: Tiles already in CB_4 from Phase 3, must NOT pop (needed for Phase 8)
using PreloadedNoPop = cb_policies::InputPolicy<cb_policies::WaitCallerManaged, cb_policies::PopNever>;

// Phase 3 Input A: Tiles already in CB_1 from Phase 2 PERSISTENT, pop at end (done with them)
using PreloadedPopAtEnd = cb_policies::InputPolicy<cb_policies::WaitCallerManaged, cb_policies::PopAtEnd>;

// Phase 3 Input B, Phase 8 Input B: Wait upfront for single tile, pop at end
using WaitUpfrontPopAtEnd = cb_policies::InputPolicy<cb_policies::WaitUpfront, cb_policies::PopAtEnd>;
```

### Phase 1: Tilize
- **Description**: Convert 32 RM sticks from c_0 to Wt tiled tiles in c_1
- **Implementation Approach**:
  - **USE HELPER**: YES
  - **Helper**: `compute_kernel_lib::tilize()`
  - **Parameters**: `(c_0, Wt, c_1, 1)` - block_w=Wt, num_blocks=1
  - **CB Management**: Helper handles internally (wait c_0, reserve c_1, tilize, push c_1, pop c_0)
- **CB Flow**: Encapsulated by helper

### Phase 2: Reduce (Mean) with PERSISTENT
- **Description**: Reduce Wt tiles along row to compute mean (tiles persist for Phase 3)
- **Implementation Approach**:
  - **USE HELPER**: YES
  - **Helper**: `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, ReduceInputMode::PERSISTENT>()`
  - **Parameters**: `(c_1, c_2, c_3, TileShape::row(Wt))`
  - **CB Management**: Helper waits for Wt tiles in c_1, does NOT pop them. Outputs 1 tile to c_3.
- **CB Flow**: Encapsulated by helper (c_1 tiles persist after reduce)

### Phase 3: Broadcast Subtract (Centralize)
- **Description**: Compute centralized = input - mean with COL broadcast
- **Implementation Approach**:
  - **USE HELPER**: YES
  - **Helper**: `compute_kernel_lib::sub<BroadcastDim::COL, PreloadedPopAtEnd, WaitUpfrontPopAtEnd>()`
  - **Parameters**: `(c_1, c_3, c_4, BinaryTileShape::row(Wt))`
  - **CB Management**:
    - Input A (c_1): Tiles already present from PERSISTENT reduce, pop all at end
    - Input B (c_3): Wait upfront for 1 mean tile, pop at end
    - Output (c_4): Per-tile reserve/push (default)
- **CB Flow**: Encapsulated by helper. After this phase: c_1 popped, c_3 popped, c_4 has Wt tiles (PERSISTS)

### Phase 4: Square
- **Description**: Compute squared = centralized^2 (element-wise self-multiply)
- **Implementation Approach**:
  - **USE HELPER**: YES
  - **Helper**: `compute_kernel_lib::binary_op<BinaryOpType::SQUARE, BroadcastDim::NONE, PreloadedNoPop>()`
  - **Parameters**: `(c_4, c_4, c_5, BinaryTileShape::row(Wt))`
  - **CB Management**:
    - Input A (c_4): Tiles already present from Phase 3, **DO NOT POP** (needed for Phase 8)
    - Output (c_5): Per-tile reserve/push
- **CB Flow**: Encapsulated by helper. **CRITICAL**: c_4 tiles remain for Phase 8!

### Phase 5: Reduce (Variance) with STREAMING
- **Description**: Reduce squared tiles to compute variance
- **Implementation Approach**:
  - **USE HELPER**: YES
  - **Helper**: `compute_kernel_lib::reduce<PoolType::SUM, ReduceDim::REDUCE_ROW, ReduceInputMode::STREAMING>()`
  - **Parameters**: `(c_5, c_2, c_6, TileShape::row(Wt))`
  - **CB Management**: Helper waits and pops tiles one at a time from c_5. Outputs 1 tile to c_6.
- **CB Flow**: Encapsulated by helper

### Phase 6-7: Add Epsilon + Rsqrt (Combined, NO HELPER)
- **Description**: Compute rsqrt(variance + epsilon) using DST registers directly
- **Implementation Approach**:
  - **USE HELPER**: NO (no helper for combined add+rsqrt pattern)
  - **Reason**: This is a specialized pattern; intermediate stays in DST
  - **RAW CALLS**:
    ```cpp
    // Wait for inputs
    cb_wait_front(c_6, 1);  // variance
    cb_wait_front(c_7, 1);  // epsilon (already present from program start)
    cb_reserve_back(c_8, 1);  // rsqrt output

    tile_regs_acquire();

    // Copy variance to DST[0]
    copy_tile_to_dst_init_short_with_dt(c_7, c_6);
    copy_tile(c_6, 0, 0);  // c_6 tile 0 -> DST[0]

    // Add epsilon from c_7
    add_binary_tile_init();
    copy_tile_to_dst_init_short_with_dt(c_6, c_7);
    copy_tile(c_7, 0, 1);  // c_7 tile 0 -> DST[1]
    add_binary_tile(0, 1, 0);  // DST[0] = DST[0] + DST[1]

    // Rsqrt
    rsqrt_tile_init();
    rsqrt_tile(0);  // DST[0] = rsqrt(DST[0])

    // Pack result
    tile_regs_commit();
    tile_regs_wait();
    pack_tile(0, c_8);
    tile_regs_release();

    cb_push_back(c_8, 1);
    cb_pop_front(c_6, 1);
    // Note: c_7 NOT popped (program lifetime)
    ```
- **CB Flow**: Wait c_6 and c_7, compute in DST, pack to c_8, pop c_6 only

### Phase 8: Broadcast Multiply (Standardize)
- **Description**: Compute output = centralized * rsqrt with COL broadcast
- **Implementation Approach**:
  - **USE HELPER**: YES
  - **Helper**: `compute_kernel_lib::mul<BroadcastDim::COL, PreloadedPopAtEnd, WaitUpfrontPopAtEnd>()`
  - **Parameters**: `(c_4, c_8, c_16, BinaryTileShape::row(Wt))`
  - **CB Management**:
    - Input A (c_4): Tiles still present from Phase 3, pop all at end (final use)
    - Input B (c_8): Wait upfront for 1 rsqrt tile, pop at end
    - Output (c_16): Per-tile reserve/push
- **CB Flow**: Encapsulated by helper. After this phase: c_4 popped, c_8 popped, c_16 has Wt tiled tiles

### Phase 9: Untilize
- **Description**: Convert Wt tiled tiles from c_16 to RM sticks
- **Implementation Approach**:
  - **USE HELPER**: YES
  - **Helper**: `compute_kernel_lib::untilize<Wt, c_16, c_16>()`
  - **Parameters**: `(1)` - num_rows=1
  - **CB Management**: Helper waits for Wt tiles, converts to RM, pushes Wt tiles worth of sticks
- **CB Flow**: Encapsulated by helper (uses same CB for input tiles and output sticks)

## Writer Kernel Design

### Phase 1: Output Writing (Per tile-row)

- **Description**: Write 32 RM sticks from c_16 to DRAM
- **Implementation Approach**:
  - **USE HELPER**: No (dataflow kernels don't use compute helpers)
  - **RAW CALLS**:
    - `cb_wait_front(c_16, Wt)` - wait for untilized data
    - For each of 32 sticks:
      - `noc_addr = accessor.get_noc_addr(stick_id)`
      - `noc_async_write(l1_read_addr, noc_addr, output_stick_size)`
      - Advance l1_read_addr by output_stick_size
    - `noc_async_write_barrier()` - wait for all writes
    - `cb_pop_front(c_16, Wt)` - release data
- **CB Flow**: Wait Wt pages, write 32 sticks (width W), barrier, pop Wt pages

## CB Synchronization Summary

| CB | Producer | Consumer | Pages per Block | Sync Point |
|----|----------|----------|-----------------|------------|
| c_0 | Reader | Compute (tilize) | Wt | Reader pushes Wt after 32 sticks read |
| c_1 | Compute (tilize) | Compute (reduce, sub) | Wt | PERSISTENT: reduce waits Wt, doesn't pop; sub pops at end |
| c_2 | Reader | Compute (reduce x2) | 1 | Reader pushes once at start; never popped |
| c_3 | Compute (reduce) | Compute (sub) | 1 | Reduce pushes 1, sub waits/pops 1 |
| c_4 | Compute (sub) | Compute (square, mul) | Wt | PERSISTENT: square NO POP; mul pops at end |
| c_5 | Compute (square) | Compute (reduce) | Wt | Square pushes per-tile; reduce pops STREAMING |
| c_6 | Compute (reduce) | Compute (add+rsqrt) | 1 | Reduce pushes 1, add+rsqrt waits/pops 1 |
| c_7 | Reader | Compute (add+rsqrt) | 1 | Reader pushes once at start; never popped |
| c_8 | Compute (rsqrt) | Compute (mul) | 1 | Rsqrt pushes 1, mul waits/pops 1 |
| c_16 | Compute (mul, untilize) | Writer | Wt | Mul pushes Wt tiled, untilize converts, writer waits/pops Wt |

## Helper Encapsulation Acknowledgment

For phases marked "USE HELPER", the following is encapsulated BY THE HELPER:
- CB wait/pop/reserve/push operations
- DST register management (acquire/commit/wait/release)
- Init/uninit sequences (tilize_init, reduce_init, etc.)

**CRITICAL**: The kernel writer MUST NOT add redundant CB or DST operations around helper calls. The helper functions are self-contained.

**EXCEPTION - Phase 6-7 (NO HELPER)**:
For the add_epsilon + rsqrt phase, the kernel writer MUST manually handle:
- CB operations (wait, reserve, push, pop)
- DST operations (tile_regs_acquire/commit/wait/release)
- Init sequences (copy_tile_to_dst_init_short_with_dt, add_binary_tile_init, rsqrt_tile_init)

## Implementation Checklist for Kernel Writer

- [ ] Reader: Generate scaler (1/W) once at start using generate_reduce_scaler
- [ ] Reader: Generate epsilon tile once at start using generate_bcast19_scalar
- [ ] Reader: Per tile-row: reserve Wt, read 32 sticks via TensorAccessor, barrier, push Wt
- [ ] Compute: Call compute_kernel_hw_startup() first
- [ ] Compute: Define custom policies (PreloadedNoPop, PreloadedPopAtEnd, WaitUpfrontPopAtEnd)
- [ ] Compute: Phase 1 tilize() helper
- [ ] Compute: Phase 2 reduce<PERSISTENT>() helper for mean
- [ ] Compute: Phase 3 sub<COL>() helper with PreloadedPopAtEnd for c_1
- [ ] Compute: Phase 4 binary_op<SQUARE>() with PreloadedNoPop for c_4 **CRITICAL**
- [ ] Compute: Phase 5 reduce<STREAMING>() helper for variance
- [ ] Compute: Phases 6-7 raw calls (copy, add_binary, rsqrt, pack)
- [ ] Compute: Phase 8 mul<COL>() helper with PreloadedPopAtEnd for c_4
- [ ] Compute: Phase 9 untilize<Wt>() helper
- [ ] Writer: Per tile-row: wait Wt, write 32 sticks (width W) via TensorAccessor, barrier, pop Wt
- [ ] Verify: CB push/pop counts match across kernels

## Include Requirements

### Reader Kernel
```cpp
#include "api/dataflow/dataflow_api.h"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_reduce_scaler.hpp"
#include "ttnn/deprecated/tt_dnn/kernels/dataflow/generate_bcast_scalar.hpp"
```

### Compute Kernel
```cpp
#include "compute_kernel_api/common.h"
#include "compute_kernel_api/eltwise_unary/rsqrt.h"
#include "compute_kernel_api/tile_move_copy.h"
#include "compute_kernel_api/eltwise_binary_sfpu.h"
#include "ttnn/cpp/ttnn/kernel_lib/tilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/untilize_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/cb_policies.hpp"
```

### Writer Kernel
```cpp
#include "api/dataflow/dataflow_api.h"
```

## CB Index Constants (for kernel code clarity)

```cpp
// Input/Output
constexpr uint32_t cb_in_rm = tt::CBIndex::c_0;           // RM input sticks
constexpr uint32_t cb_out_rm = tt::CBIndex::c_16;         // RM output sticks (also used for tiled intermediate)

// Scalers (program lifetime)
constexpr uint32_t cb_scaler = tt::CBIndex::c_2;          // 1/W scaler
constexpr uint32_t cb_epsilon = tt::CBIndex::c_7;         // epsilon scaler

// Tiled intermediates
constexpr uint32_t cb_in_tiled = tt::CBIndex::c_1;        // Tilized input (PERSISTENT phases 1-3)
constexpr uint32_t cb_mean_tiled = tt::CBIndex::c_3;      // Mean tile
constexpr uint32_t cb_centralized_tiled = tt::CBIndex::c_4; // Centralized tiles (PERSISTENT phases 3-8)
constexpr uint32_t cb_squared_tiled = tt::CBIndex::c_5;   // Squared tiles
constexpr uint32_t cb_variance_tiled = tt::CBIndex::c_6;  // Variance tile
constexpr uint32_t cb_rsqrt_tiled = tt::CBIndex::c_8;     // Rsqrt result tile
```
