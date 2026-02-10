# Kernel Design: layer_norm_rm

## Critical Spec Issues

### Issue: Step 6 add<SCALAR> + rsqrt as post_op may not work with binary_op helper

- **Problem**: The spec proposes `add<SCALAR, ..., true>(c_27, c_7, c_28, ..., [](uint32_t dst_idx){ rsqrt_tile_init(); rsqrt_tile(dst_idx); })` but the binary_op helper calls `post_op(dst_idx)` inside the tile loop, which means `rsqrt_tile_init()` is called per-tile. For a single-tile operation (BinaryInputBlockShape::single()) this is fine, but rsqrt_tile_init should only be called once. Since variance is 1 tile, this works but is wasteful if extended.
- **Fix**: Acceptable for single-tile variance. Keep as designed.

### Issue: Spec step 3 (sub) input policies are inconsistent

- **Problem**: Spec says `sub<COL, NoWaitNoPop, WaitUpfrontNoPop>` for c_2 (input_a) and c_24 (input_b). But c_2 was already waited on by reduce in step 2 (WaitUpfrontNoPop), so tiles are present. Using NoWaitNoPop for c_2 is correct. But then the spec says "Pop c_2 (consumed by sub's input_a_policy)" which contradicts NoWaitNoPop (NoPop means helper does NOT pop).
- **Fix**: Use `NoWaitPopAtEnd` for input_a (c_2) so the helper pops c_2 after sub completes. Use `WaitUpfrontNoPop` for input_b (c_24) then manually pop c_24 afterward.

### Issue: Step 4 (square) policy mismatch for centered reuse

- **Problem**: Spec says `square<WaitUpfrontNoPop>` for c_25 but c_25 was just produced by sub (step 3). The sub helper pushes to c_25 and the tiles are available. Using WaitUpfrontNoPop will wait for all Wt tiles upfront and NOT pop -- correct since c_25 is reused in step 7.
- **Fix**: No fix needed, WaitUpfrontNoPop is correct for square on c_25.

### Issue: Step 7 (mul centered * rstd) input_a policy

- **Problem**: Spec says `mul<COL, WaitUpfrontPopAtEnd, WaitUpfrontNoPop>`. c_25 (centered) was already waited on by square (WaitUpfrontNoPop) and NOT popped. Since tiles are still present in c_25, we should use `NoWaitPopAtEnd` for input_a (no need to re-wait).
- **Fix**: Use `NoWaitPopAtEnd` for input_a (c_25) instead of `WaitUpfrontPopAtEnd`.

## CB Allocation

| CB | Index | Pages | Page Size | Valid Region | Lifetime |
|----|-------|-------|-----------|--------------|----------|
| cb_input_rm | c_0 | Wt | tile_size | N/A (RM) | Per-row: reader->compute(tilize) |
| cb_reduce_scaler | c_1 | 1 | tile_size | Row0 (reduce scaler) | Program (persistent) |
| cb_input_tilized | c_2 | Wt | tile_size | All | Per-row: compute(tilize)->compute(norm) |
| cb_gamma_rm | c_3 | Wt | tile_size | N/A (RM) | One-shot: reader->compute(tilize) |
| cb_beta_rm | c_4 | Wt | tile_size | N/A (RM) | One-shot: reader->compute(tilize) |
| cb_gamma_tilized | c_5 | Wt | tile_size | All | Program (persistent, never popped) |
| cb_beta_tilized | c_6 | Wt | tile_size | All | Program (persistent, never popped) |
| cb_eps_scalar | c_7 | 1 | tile_size | [0,0] (scalar bcast) | Program (persistent) |
| cb_output_rm | c_16 | Wt | tile_size | N/A (RM) | Per-row: compute(untilize)->writer |
| cb_mean | c_24 | 1 | intermed_tile_size | Col0 (REDUCE_ROW out) | Per-row: step 2->step 3 |
| cb_centered | c_25 | Wt | intermed_tile_size | All | Per-row: step 3->step 4,7 |
| cb_squared | c_26 | Wt | intermed_tile_size | All | Per-row: step 4->step 5 |
| cb_var | c_27 | 1 | intermed_tile_size | Col0 (REDUCE_ROW out) | Per-row: step 5->step 6 |
| cb_rstd | c_28 | 1 | intermed_tile_size | Col0 (rsqrt of col0 val) | Per-row: step 6->step 7 |
| cb_normalized | c_29 | Wt | intermed_tile_size | All | Per-row: step 7->step 8 |
| cb_gamma_applied | c_30 | Wt | intermed_tile_size | All | Per-row: step 8->step 9 |
| cb_out_tilized | c_31 | Wt | intermed_tile_size | All | Per-row: step 9->step 10 |

## Binary Op Broadcast Verification

| Phase | Op | CB_A Valid | CB_B Valid | Broadcast | Correct? |
|-------|-----|------------|------------|-----------|----------|
| 3 (center) | SUB | All (c_2) | Col0 (c_24 mean) | COL | Yes |
| 4 (square) | SQUARE | All (c_25) | Self | NONE | Yes |
| 6 (add eps) | ADD | Col0 (c_27 var) | [0,0] (c_7 eps) | SCALAR | Yes |
| 7 (normalize) | MUL | All (c_25) | Col0 (c_28 rstd) | COL | Yes |
| 8 (gamma) | MUL | All (c_29) | All (c_5 gamma) | NONE | Yes |
| 9 (beta) | ADD | All (c_30) | All (c_6 beta) | NONE | Yes |

## Reader Kernel

**One-time setup:**
- `generate_reduce_scaler(c_1, reduce_scaler_packed)` -- scaler = 1/W in `(bf16 << 16 | bf16)` format
- `generate_bcast_scalar_bfloat16(c_7, eps_packed)` -- epsilon in `(bf16 << 16 | bf16)` format (or `generate_bcast_scalar` for float32)
- Read gamma: `cb_reserve_back(c_3, Wt)` -> read gamma row 32 times as sticks (replicate single row to fill tile-height) -> `noc_async_read_barrier()` -> `cb_push_back(c_3, Wt)`
- Read beta: same pattern into c_4

**Per tile-row (num_tile_rows iterations):**
- `cb_reserve_back(c_0, Wt)` -> read 32 input sticks via NOC (split-rows pattern, each stick = W * elem_size bytes) -> `noc_async_read_barrier()` -> `cb_push_back(c_0, Wt)`

**Gamma/beta stick replication**: Gamma is shape [1,...,1,W] = single row of W elements. Reader must write the same row 32 times into the CB to fill one tile-height. Each of the 32 "sticks" in the CB is an identical copy of the gamma row.

## Compute Kernel

**Startup**: `compute_kernel_hw_startup(c_0, c_1, c_16)`

**One-time gamma/beta tilize:**

```cpp
compute_kernel_lib::tilize<c_3, c_5>(Wt, 1);  // gamma RM -> tiles, 1 block
compute_kernel_lib::tilize<c_4, c_6>(Wt, 1);  // beta RM -> tiles, 1 block
// c_5, c_6 persist -- never popped
```

**Per tile-row loop** (num_tile_rows iterations, 10 phases):

### Phase 1: Tilize input
```cpp
compute_kernel_lib::tilize<c_0, c_2>(Wt, 1);
```
Tilize helper handles cb_wait_front(c_0, Wt), tilize_block, cb_pop_front(c_0, Wt), cb_push_back(c_2, Wt).

### Phase 2: Mean (reduce SUM row)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW, ReduceInputPolicy::WaitUpfrontNoPop>(
    c_2, c_1, c_24, ReduceInputBlockShape::row(Wt));
```
c_2 tiles stay (NoPop). c_24 gets 1 tile, valid in Col0.

### Phase 3: Center (x - mean)
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitPopAtEnd,     // c_2: already present, pop after
    BinaryInputPolicy::WaitUpfrontNoPop,   // c_24: wait+no pop (but only 1 tile, already present from step 2 push)
    BinaryOutputPolicy::Bulk>(
    c_2, c_24, c_25, BinaryInputBlockShape::row(Wt));
cb_pop_front(c_24, 1);  // manual pop: mean no longer needed
```
c_2 popped by PopAtEnd. c_24 manually popped after. c_25 has Wt tiles.

### Phase 4: Square centered values
```cpp
compute_kernel_lib::square<
    BinaryInputPolicy::WaitUpfrontNoPop,   // c_25: wait, don't pop (reused in step 7)
    BinaryOutputPolicy::Bulk>(
    c_25, c_26, BinaryInputBlockShape::row(Wt));
```
c_25 tiles stay. c_26 has Wt tiles of (x-mean)^2.

### Phase 5: Variance (reduce SUM row)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW, ReduceInputPolicy::WaitAndPopPerTile>(
    c_26, c_1, c_27, ReduceInputBlockShape::row(Wt));
```
c_26 consumed and popped. c_27 gets 1 tile with variance in Col0.

### Phase 6: Add epsilon + rsqrt
```cpp
compute_kernel_lib::add<BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerTile,  // c_27: wait, pop (consumed)
    BinaryInputPolicy::WaitUpfrontNoPop,   // c_7: epsilon persists
    BinaryOutputPolicy::PerTile,
    BinaryDataFormatReconfig::INPUT_AND_OUTPUT,
    true,                                   // init
    NoAccumulation,
    decltype(rsqrt_post_op)>(              // post_op type
    c_27, c_7, c_28, BinaryInputBlockShape::single(), {},
    NoAccumulation{},
    [](uint32_t dst_idx) { rsqrt_tile_init(); rsqrt_tile(dst_idx); });
```
c_27 consumed. c_7 stays. c_28 gets 1 tile with rstd = 1/sqrt(var+eps).

**Implementation note**: The rsqrt post_op lambda requires `#include "compute_kernel_api/eltwise_unary/rsqrt.h"`.

### Phase 7: Normalize (centered * rstd)
```cpp
compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitPopAtEnd,     // c_25: already present (from step 3), pop after
    BinaryInputPolicy::WaitUpfrontNoPop,   // c_28: rstd, don't pop (1 tile, already present)
    BinaryOutputPolicy::Bulk>(
    c_25, c_28, c_29, BinaryInputBlockShape::row(Wt));
cb_pop_front(c_28, 1);  // manual pop: rstd no longer needed
```
c_25 popped by PopAtEnd. c_28 manually popped. c_29 has Wt normalized tiles.

### Phase 8: Apply gamma (element-wise mul)
```cpp
compute_kernel_lib::mul<BroadcastDim::NONE,
    BinaryInputPolicy::WaitAndPopPerTile,  // c_29: stream and pop
    BinaryInputPolicy::NoWaitNoPop,        // c_5: gamma persists, already present
    BinaryOutputPolicy::PerTile>(
    c_29, c_5, c_30, BinaryInputBlockShape::row(Wt));
```
c_29 consumed. c_5 stays (persistent gamma). c_30 has Wt tiles.

### Phase 9: Apply beta (element-wise add)
```cpp
compute_kernel_lib::add<BroadcastDim::NONE,
    BinaryInputPolicy::WaitAndPopPerTile,  // c_30: stream and pop
    BinaryInputPolicy::NoWaitNoPop,        // c_6: beta persists, already present
    BinaryOutputPolicy::PerTile>(
    c_30, c_6, c_31, BinaryInputBlockShape::row(Wt));
```
c_30 consumed. c_6 stays (persistent beta). c_31 has Wt tiles.

### Phase 10: Untilize output
```cpp
compute_kernel_lib::untilize<Wt, c_31, c_16>(1);
```
Untilize helper handles cb_wait_front(c_31, Wt), untilize, cb_pop_front(c_31, Wt), cb_push_back(c_16, Wt).

## Writer Kernel

**Per tile-row** (num_tile_rows iterations):
1. `cb_wait_front(c_16, Wt)`
2. Get L1 read pointer
3. For each of 32 sticks: compute NOC addr via TensorAccessor, `noc_async_write` stick data, advance L1 pointer
4. `noc_async_write_barrier()`
5. `cb_pop_front(c_16, Wt)`

## Critical Notes

- **Manual pops required**: After Phase 3 (c_24 mean), after Phase 7 (c_28 rstd)
- **Persistent CBs (never popped in loop)**: c_1 (reduce scaler), c_5 (gamma), c_6 (beta), c_7 (epsilon)
- **NoWaitNoPop for persistent CBs**: c_5 and c_6 use NoWaitNoPop in phases 8-9 because they were pushed once during setup and tiles remain available for all iterations
- **Gamma/beta stick replication**: Reader must write the single gamma/beta row 32 times into the RM CB to create a valid tile-height block for tilize
- **rsqrt include**: Compute kernel needs `#include "compute_kernel_api/eltwise_unary/rsqrt.h"`
- **Scaler packing**: `reduce_scaler = (bf16_val << 16) | bf16_val` where bf16_val = bfloat16(1.0f/W). NOT IEEE float32.
- **Epsilon packing**: Same format: `eps_packed = (bf16_eps << 16) | bf16_eps`
- **Wt as compile-time arg**: untilize<Wt, ...> requires Wt as a compile-time template parameter. Pass as compile-time arg to compute kernel.

## Implementation Checklist

- [ ] Reader: `generate_reduce_scaler(c_1, scaler)`, `generate_bcast_scalar_bfloat16(c_7, eps)`
- [ ] Reader: Read gamma/beta with 32x stick replication into c_3, c_4
- [ ] Reader: Per-row loop reading 32 input sticks into c_0
- [ ] Compute: One-time `tilize<c_3, c_5>(Wt, 1)` and `tilize<c_4, c_6>(Wt, 1)`
- [ ] Compute: 10 phases per row with helpers: tilize, reduce, sub, square, reduce, add+rsqrt, mul, mul, add, untilize
- [ ] Compute: Manual pops after Phase 3 (c_24) and Phase 7 (c_28)
- [ ] Compute: `#include "compute_kernel_api/eltwise_unary/rsqrt.h"` for Phase 6 post_op
- [ ] Writer: Wt tiles per row, 32 sticks per tile-row via TensorAccessor
- [ ] Verify: CB push/pop balance across all phases
- [ ] Verify: Persistent CBs (c_1, c_5, c_6, c_7) never popped in loop
