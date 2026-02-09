# Kernel Design: row_standardize

## Critical Spec Issues

### Issue: cb_eps format for SCALAR broadcast
- **Problem**: Spec says "add_tiles_bcast_scalar" for adding epsilon to variance, but doesn't specify which scalar tile generator to use. For SCALAR broadcast, the tile must have the value at position [0][0] of face 0 only (not all faces). Using `generate_bcast_scalar_bfloat16` (for bf16) or `generate_bcast_scalar` (for fp32) is correct. Using `generate_bcast_col_scalar_*` or `generate_reduce_scaler` would be WRONG.
- **Fix**: Use `generate_bcast_scalar_bfloat16(cb_eps, packed_eps)` for bf16, `generate_bcast_scalar(cb_eps, eps_bits)` for fp32.

### Issue: Wt must be compile-time constant for untilize helper
- **Problem**: `untilize<block_width_tiles, ...>()` requires `block_width_tiles` as a template parameter. Spec lists Wt as compile-time arg index 0, but doesn't explicitly note this constraint.
- **Fix**: Kernel must use `constexpr uint32_t Wt = get_compile_time_arg_val(0);` to make it usable as template parameter.

## CB Allocation

| CB | Pages | Format | Valid Region | Lifetime | Purpose |
|----|-------|--------|--------------|----------|---------|
| c_0 (cb_rm_in) | Wt | dtype | N/A (RM sticks) | Per-block | Input RM sticks |
| c_1 (cb_scaler) | 1 | dtype | Row0 | Persistent | Reduce scaler 1/W |
| c_2 (cb_eps) | 1 | intermed_fmt | [0,0] | Persistent | Epsilon scalar |
| c_3 (cb_tilized) | Wt | intermed_fmt | All | Per-block | Tilized input |
| c_4 (cb_tilized_out) | Wt | intermed_fmt | All | Per-block | Normalized output |
| c_16 (cb_rm_out) | Wt | dtype | N/A (RM sticks) | Per-block | Output RM sticks |
| c_24 (cb_mean) | 1 | intermed_fmt | Col0 | Per-block | Row mean |
| c_25 (cb_xmm) | Wt | intermed_fmt | All | Per-block | x - mean (reused: Phase 4 + 7) |
| c_26 (cb_xmm_sq) | Wt | intermed_fmt | All | Per-block | (x - mean)^2 |
| c_27 (cb_var) | 1 | intermed_fmt | Col0 | Per-block | Row variance |
| c_28 (cb_invstd) | 1 | intermed_fmt | Col0 | Per-block | rsqrt(var + eps) |

## Binary Op Broadcast Verification

| Phase | Op | CB_A Valid | CB_B Valid | Broadcast | Correct? |
|-------|-----|------------|------------|-----------|----------|
| 3 (sub mean) | SUB | All (cb_tilized) | Col0 (cb_mean) | COL | Yes |
| 4 (square) | SQUARE | All (cb_xmm) | N/A (self) | NONE | Yes |
| 6 (add eps) | ADD | Col0 (cb_var) | [0,0] (cb_eps) | SCALAR | Yes |
| 7 (normalize) | MUL | All (cb_xmm) | Col0 (cb_invstd) | COL | Yes |

## Reader Kernel

**One-time setup:**
- Generate reduce scaler: `generate_reduce_scaler(cb_scaler, scaler_packed)` (bf16: `bf16<<16|bf16`, fp32: float bits)
- Generate epsilon tile: `generate_bcast_scalar_bfloat16(cb_eps, eps_packed)` (bf16) or `generate_bcast_scalar(cb_eps, eps_bits)` (fp32)
- Branching on `is_float32` compile-time arg for correct generator function

**Per-block loop** (nblocks iterations, 32 sticks per block):
1. `cb_reserve_back(cb_rm_in, Wt)`
2. Read 32 sticks via NOC: `noc_async_read(src_noc_addr, l1_write_addr, stick_size_bytes)` per stick
3. `noc_async_read_barrier()`
4. `cb_push_back(cb_rm_in, Wt)`

## Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_rm_in, cb_scaler, cb_rm_out)`

**Per-block loop** (nblocks iterations, 8 phases):

### Phase 1: Tilize (RM -> TILE)
```cpp
compute_kernel_lib::tilize<c_0, c_3>(Wt, 1);
```
Handles: wait cb_rm_in(Wt), tilize, push cb_tilized(Wt), pop cb_rm_in(Wt).

### Phase 2: Mean (SUM reduce * 1/W)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW,
    compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop,
    compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
    c_3, c_1, c_24,
    compute_kernel_lib::ReduceInputBlockShape::row(Wt));
```
cb_tilized tiles persist (NoPop) for Phase 3. Produces 1 tile in cb_mean.

### Phase 3: Subtract mean (x - mean)
```cpp
compute_kernel_lib::sub<
    compute_kernel_lib::BroadcastDim::COL,
    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::BinaryOutputPolicy::Bulk,
    compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    c_3, c_24, c_25,
    compute_kernel_lib::BinaryInputBlockShape::row(Wt));
```
- A policy `WaitUpfrontPopAtEnd`: waits for Wt tiles in cb_tilized (already there), pops all at end. Frees cb_tilized.
- B policy `WaitAndPopPerTile`: waits/pops cb_mean (1 tile). Frees cb_mean.
- Output `Bulk`: reserves Wt upfront, pushes Wt at end into cb_xmm.

### Phase 4: Square (x - mean)^2
```cpp
compute_kernel_lib::square<
    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop,
    compute_kernel_lib::BinaryOutputPolicy::Bulk,
    compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    c_25, c_26,
    compute_kernel_lib::BinaryInputBlockShape::row(Wt));
```
- Input `WaitUpfrontNoPop`: cb_xmm tiles persist for Phase 7.
- Output `Bulk`: pushes Wt tiles to cb_xmm_sq.

### Phase 5: Variance (SUM reduce * 1/W)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW,
    compute_kernel_lib::ReduceInputPolicy::BulkWaitBulkPop,
    compute_kernel_lib::ReduceDataFormatReconfigMode::INPUT_AND_OUTPUT>(
    c_26, c_1, c_27,
    compute_kernel_lib::ReduceInputBlockShape::row(Wt));
```
Waits for Wt tiles in cb_xmm_sq, processes, pops all. Produces 1 tile in cb_var.

### Phase 6: Add epsilon + rsqrt (NO HELPER - raw tile API)

No helper exists for fused add+rsqrt. The binary_op PostOp parameter is not invoked in the implementation.

```cpp
// Reconfig for add bcast
reconfig_data_format(c_27, c_2);
pack_reconfig_data_format(c_28);
add_bcast_scalar_init_short(c_27, c_2);

cb_wait_front(c_27, 1);   // variance
// cb_eps (c_2) is persistent, already available
cb_reserve_back(c_28, 1);

tile_regs_acquire();
add_tiles_bcast_scalar(c_27, c_2, 0, 0, 0);
rsqrt_tile_init();
rsqrt_tile(0);
tile_regs_commit();

tile_regs_wait();
pack_tile(0, c_28);
tile_regs_release();

cb_push_back(c_28, 1);
cb_pop_front(c_27, 1);    // free cb_var
```
Produces 1 tile in cb_invstd. Does NOT pop cb_eps (persistent).

### Phase 7: Normalize (x - mean) * inv_std
```cpp
compute_kernel_lib::mul<
    compute_kernel_lib::BroadcastDim::COL,
    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontPopAtEnd,
    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::BinaryOutputPolicy::Bulk,
    compute_kernel_lib::BinaryDataFormatReconfig::INPUT_AND_OUTPUT>(
    c_25, c_28, c_4,
    compute_kernel_lib::BinaryInputBlockShape::row(Wt));
```
- A policy `WaitUpfrontPopAtEnd`: waits for cb_xmm (already there from Phase 4 NoPop), pops Wt at end. Frees cb_xmm.
- B policy `WaitAndPopPerTile`: waits/pops cb_invstd (1 tile). Frees cb_invstd.
- Output `Bulk`: pushes Wt tiles to cb_tilized_out.

### Phase 8: Untilize (TILE -> RM)
```cpp
compute_kernel_lib::untilize<Wt, c_4, c_16>(1);
```
Handles: wait cb_tilized_out(Wt), untilize, push cb_rm_out(Wt), pop cb_tilized_out(Wt).

## Writer Kernel

**Per-block loop** (nblocks iterations):
1. `cb_wait_front(cb_rm_out, Wt)`
2. Get L1 base: `get_read_ptr(cb_rm_out)`
3. Write 32 sticks via NOC: `noc_async_write(l1_addr, dst_noc_addr, stick_size_bytes)` per stick, advancing l1_addr by stick_size_bytes
4. `noc_async_write_barrier()`
5. `cb_pop_front(cb_rm_out, Wt)`

## Critical Notes

- **cb_xmm (c_25) reuse**: Produced in Phase 3, read in Phase 4 (NoPop), consumed in Phase 7 (PopAtEnd). Must NOT be popped between Phase 4 and Phase 7.
- **cb_scaler (c_1) and cb_eps (c_2) are persistent**: Generated once by reader, never popped by compute. cb_scaler is used in Phase 2 and Phase 5; cb_eps in Phase 6.
- **Phase 6 raw calls require init**: `add_bcast_scalar_init_short` must be called before `add_tiles_bcast_scalar`. And `rsqrt_tile_init()` before `rsqrt_tile()`.
- **Wt as constexpr**: Must be `constexpr` for `untilize<Wt, ...>()` template parameter.
- **Scaler packing**: bf16: `(bf16_val << 16 | bf16_val)`. fp32: reinterpreted float bits as uint32.
- **Tilize/untilize init/uninit per block**: Each block calls tilize and untilize with default `InitAndUninit` mode. This is correct since other compute phases between them reconfigure hardware anyway.

## Implementation Checklist

- [ ] Reader: `generate_reduce_scaler(c_1, scaler)`, `generate_bcast_scalar_bfloat16/generate_bcast_scalar(c_2, eps)` (branch on is_float32)
- [ ] Reader: Per-block read 32 sticks, reserve/push Wt pages to c_0
- [ ] Compute: `compute_kernel_hw_startup(c_0, c_1, c_16)` at start
- [ ] Compute: Phase 1 tilize, Phase 2 reduce(SUM,ROW,WaitUpfrontNoPop), Phase 3 sub(COL)
- [ ] Compute: Phase 4 square(WaitUpfrontNoPop), Phase 5 reduce(SUM,ROW,BulkWaitBulkPop)
- [ ] Compute: Phase 6 raw add_tiles_bcast_scalar + rsqrt_tile (manual DST management)
- [ ] Compute: Phase 7 mul(COL), Phase 8 untilize
- [ ] Compute: cb_xmm NOT popped between Phase 4 and Phase 7
- [ ] Writer: Per-block wait Wt, write 32 sticks, pop Wt from c_16
- [ ] Verify: All CB push/pop balanced per block (all per-block CBs return to empty)
