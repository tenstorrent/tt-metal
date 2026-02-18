# Kernel Design: row_centralize

## Critical Spec Issues

### Issue: Phase 5 reduce input policy ambiguity
- **Problem**: Spec says "using `WaitAndPopPerTile` or `BulkWaitBulkPop`" for Phase 5 reduce of cb_squared without selecting one.
- **Fix**: Use `BulkWaitBulkPop` -- cb_squared has all Wt tiles ready before reduce starts, bulk is more efficient.

## CB Allocation

| CB | Pages | Layout | Valid Region | Lifetime | Notes |
|----|-------|--------|--------------|----------|-------|
| c_0 (cb_rm_in) | Wt | RM sticks | All | Per tile-row | Reader pushes Wt, compute tilize pops Wt |
| c_1 (cb_tilized) | Wt | TILE | All | Per tile-row | Tilize output. Not popped until after Phase 3 sub |
| c_2 (cb_mean) | 1 | TILE | Col0 | Per tile-row | REDUCE_ROW output (1 tile) |
| c_3 (cb_centered) | Wt | TILE | All | Per tile-row, persistent across Phase 4+8 | Not popped until after Phase 8 mul |
| c_24 (cb_squared) | Wt | TILE | All | Per tile-row | Square output, consumed by Phase 5 reduce |
| c_4 (cb_var) | 1 | TILE | Col0 | Per tile-row | REDUCE_ROW output (1 tile) |
| c_25 (cb_var_plus_eps) | 1 | TILE | Col0 | Per tile-row | var + eps, consumed by Phase 7 rsqrt |
| c_5 (cb_inv_std) | 1 | TILE | Col0 | Per tile-row | rsqrt output (1 tile) |
| c_6 (cb_result) | Wt | TILE | All | Per tile-row | Final standardized tiles |
| c_16 (cb_rm_out) | Wt | RM sticks | All | Per tile-row | Untilize output for writer |
| c_7 (cb_eps) | 1 | TILE | [0,0] | Program | Epsilon scalar tile, never popped |
| c_8 (cb_scaler) | 1 | TILE | Row0 | Program | 1/W reduce scaler, never popped |

## Binary Op Broadcast Verification

| Phase | Op | CB_A Valid | CB_B Valid | Broadcast | Shape |
|-------|-----|------------|------------|-----------|-------|
| 3 (sub) | SUB | All (c_1, Wt tiles) | Col0 (c_2, 1 tile) | COL | of(1, Wt) |
| 6 (add eps) | ADD | Col0 (c_4, 1 tile) | [0,0] (c_7, 1 tile) | SCALAR | single() |
| 8 (mul) | MUL | All (c_3, Wt tiles) | Col0 (c_5, 1 tile) | COL | of(1, Wt) |

All broadcasts match reduce output valid regions. Correct.

## Reader Kernel

**One-time setup**:
```cpp
dataflow_kernel_lib::generate_reduce_scaler(cb_scaler, packed_reduce_scaler);
dataflow_kernel_lib::generate_bcast_scalar_bfloat16(cb_eps, packed_eps);
```

**Per tile-row** (num_sticks/32 iterations):
- `cb_reserve_back(cb_rm_in, Wt)` -> read 32 sticks via NOC0 with TensorAccessor -> `noc_async_read_barrier()` -> `cb_push_back(cb_rm_in, Wt)`

## Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_rm_in, cb_rm_out, cb_tilized, cb_mean, cb_centered, cb_squared, cb_var, cb_var_plus_eps, cb_inv_std, cb_result, cb_eps, cb_scaler)`

Note: `compute_kernel_hw_startup` accepts the first and last CB IDs, not a list. Pass the min and max CB IDs used: `compute_kernel_hw_startup(cb_rm_in, cb_var_plus_eps)` -- or more precisely, use the two CBs that span the range. Check which CBs are min/max: c_0 (0) through c_25 (25). So: `compute_kernel_hw_startup(c_0, c_25)`.

**Main loop**: `for tile_row in 0..Ht_total`:

### Phase 1: Tilize (c_0 -> c_1)
```cpp
compute_kernel_lib::tilize<cb_rm_in, cb_tilized>(Wt, 1);
```
Standard tilize, defaults (InitAndUninit, WaitBlock) are correct. Processes 1 block of Wt tiles.

### Phase 2: Reduce mean (c_1 -> c_2)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW, ReduceInputPolicy::WaitUpfrontNoPop>(
    cb_tilized, cb_scaler, cb_mean,
    ReduceInputBlockShape::row(Wt));
```
WaitUpfrontNoPop: c_1 tiles persist for Phase 3 sub. Scaler CB (1/W) already has tile from reader startup.

### Phase 3: Subtract mean (c_1, c_2 -> c_3)
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitPopAtEnd,
    BinaryInputPolicy::WaitAndPopPerTile>(
    cb_tilized, cb_mean, cb_centered,
    BinaryInputBlockShape::of(1, Wt));
```
- Input A (c_1): `NoWaitPopAtEnd` -- already waited in Phase 2, pop all Wt tiles after processing.
- Input B (c_2): `WaitAndPopPerTile` -- wait for 1 mean tile, pop after row. COL broadcast pops B once per row.

### Phase 4: Square centered (c_3 -> c_24)
```cpp
compute_kernel_lib::square<BinaryInputPolicy::WaitUpfrontNoPop>(
    cb_centered, cb_squared,
    BinaryInputBlockShape::of(1, Wt));
```
WaitUpfrontNoPop: c_3 tiles persist for Phase 8 mul. No manual pop needed here.

### Phase 5: Reduce variance (c_24 -> c_4)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW, ReduceInputPolicy::BulkWaitBulkPop>(
    cb_squared, cb_scaler, cb_var,
    ReduceInputBlockShape::row(Wt));
```
BulkWaitBulkPop: waits for Wt tiles, processes all, pops all. Scaler CB reused (1/W, still has tile, never popped).

### Phase 6: Add epsilon (c_4 + c_7 -> c_25)
```cpp
compute_kernel_lib::add<BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop>(
    cb_var, cb_eps, cb_var_plus_eps,
    BinaryInputBlockShape::single());
```
- Input A (c_4): WaitAndPopPerTile -- wait for 1 tile, process, pop.
- Input B (c_7 eps): WaitUpfrontNoPop -- eps persists for all tile-rows. SCALAR broadcast waits upfront internally; NoPop ensures it is never freed.

### Phase 7: Rsqrt (c_25 -> c_5)
**NO HELPER** - raw implementation required:
```cpp
// Reconfig for unary op
rsqrt_tile_init();
cb_wait_front(cb_var_plus_eps, 1);
cb_reserve_back(cb_inv_std, 1);
tile_regs_acquire();
copy_tile(cb_var_plus_eps, 0, 0);   // unpack tile to DST[0]
rsqrt_tile(0);                       // compute rsqrt in-place in DST[0]
tile_regs_commit();
tile_regs_wait();
pack_tile(0, cb_inv_std);            // pack DST[0] to output CB
cb_push_back(cb_inv_std, 1);
tile_regs_release();
cb_pop_front(cb_var_plus_eps, 1);
```
Note: `copy_tile` requires `copy_tile_to_dst_init_short()` before first use. The kernel-writer should call this before the rsqrt sequence and then re-init for subsequent binary ops.

### Phase 8: Multiply by inv_std (c_3, c_5 -> c_6)
```cpp
compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitPopAtEnd,
    BinaryInputPolicy::WaitAndPopPerTile>(
    cb_centered, cb_inv_std, cb_result,
    BinaryInputBlockShape::of(1, Wt));
```
- Input A (c_3): `NoWaitPopAtEnd` -- already waited in Phase 4, pop all Wt tiles after processing. This finally frees cb_centered.
- Input B (c_5): `WaitAndPopPerTile` -- wait for 1 inv_std tile, pop after row.

### Phase 9: Untilize (c_6 -> c_16)
```cpp
compute_kernel_lib::untilize<Wt, cb_result, cb_rm_out>(1);
```
Standard untilize, defaults (InitAndUninit, WaitBlock). Processes 1 block (1 tile-row).

## Writer Kernel

**Per tile-row** (Ht_total iterations):
- `cb_wait_front(cb_rm_out, Wt)` -> write 32 sticks via NOC1 with TensorAccessor -> `noc_async_write_barrier()` -> `cb_pop_front(cb_rm_out, Wt)`

## Critical Notes

- **Persistent CBs (never popped)**: c_7 (eps) and c_8 (scaler) are pushed once by reader at startup and persist for the entire program. The reduce helper internally does `cb_wait_front` on the scaler CB but never pops it. The add helper with `WaitUpfrontNoPop` for B also never pops.
- **Two-pass CB c_3**: cb_centered is waited in Phase 4 (WaitUpfrontNoPop) and consumed in Phase 8 (NoWaitPopAtEnd). The Wt tiles sit in the CB across Phases 4-8. This is the most critical synchronization pattern.
- **Two-pass CB c_1**: cb_tilized is waited in Phase 2 (WaitUpfrontNoPop) and consumed in Phase 3 (NoWaitPopAtEnd). Similar pattern to c_3.
- **Rsqrt reconfig**: Phase 7 uses `copy_tile` which reconfigures the unpacker. The kernel-writer must re-init binary ops (or rely on the helper's `init=true` default) after Phase 7 for Phase 8's mul to work correctly.
- **Init/Uninit overhead**: Each phase calls init/uninit by default. This is correct for chaining different operation types (tilize, reduce, binary, unary, untilize) since each reconfigures the unpacker/packer differently.
- **Wt as compile-time arg**: `untilize` requires `block_width_tiles` as a template parameter. Wt must be a compile-time constant (passed via `get_compile_time_arg_val`).
- **compute_kernel_hw_startup CB range**: Pass the minimum and maximum CB IDs to cover all CBs. Since we use c_0 through c_25, pass e.g. `compute_kernel_hw_startup(c_0, c_25)`.

## Implementation Checklist

- [ ] Reader: generate_reduce_scaler(c_8) + generate_bcast_scalar_bfloat16(c_7), then loop reading 32 RM sticks per tile-row
- [ ] Compute: 9 phases using helpers: tilize, reduce(x2), sub, square, add, mul, untilize
- [ ] Compute: Phase 7 rsqrt is RAW (copy_tile + rsqrt_tile + pack_tile)
- [ ] Compute: No manual pops needed -- all handled by helper policies (PopAtEnd, WaitAndPopPerTile, BulkWaitBulkPop)
- [ ] Writer: loop writing 32 RM sticks per tile-row from cb_rm_out
- [ ] Verify: c_1 not popped until Phase 3 completes (NoWaitPopAtEnd)
- [ ] Verify: c_3 not popped until Phase 8 completes (NoWaitPopAtEnd)
- [ ] Verify: c_7 and c_8 never popped (program lifetime)
- [ ] Verify: Rsqrt reconfig does not corrupt subsequent Phase 8 mul (helper re-inits by default)
