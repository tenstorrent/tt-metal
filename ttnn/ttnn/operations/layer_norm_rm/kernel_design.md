# Kernel Design: layer_norm_rm

## Critical Spec Issues

### Issue: Epsilon scalar tile format depends on dtype
- **Problem**: Spec says cb_eps_scalar (c_7) uses "input dtype" but epsilon comes as a packed runtime arg. For bfloat16 inputs, the scalar generator expects `(bf16 << 16 | bf16)` packed format. For float32, it expects a raw 32-bit float reinterpreted as uint32_t.
- **Fix**: Reader must use `generate_bcast_scalar_bfloat16(c_7, packed_eps)` for bf16 or `generate_bcast_scalar(c_7, eps_bits)` for f32. Program factory packs epsilon appropriately per dtype.

### Issue: Untilize block_width_tiles must be compile-time constant
- **Problem**: The `untilize<Wt, cb_in, cb_out>()` helper requires `Wt` as a compile-time template parameter, but Wt varies per invocation.
- **Fix**: Pass Wt as a compile-time argument to the compute kernel. The kernel uses it as `constexpr uint32_t Wt = get_compile_time_arg_val(0);` then calls `untilize<Wt, c_8, c_16>(1)`.

### Issue: add<SCALAR> for var+eps writes to same CB it reads from
- **Problem**: Spec shows `add<SCALAR>(c_27, c_7, c_27, ...)` -- read-modify-write on c_27. Binary op helper pops input before pushing to same CB, which is safe as long as c_27 has capacity >= 1 tile (it does).
- **Fix**: No code fix needed, but kernel-writer must verify that the helper's internal pop-before-push handles the same-CB case correctly. Alternative: use a post_reduce_op lambda on the variance reduce step instead.

## CB Allocation

| CB | Pages | Layout | Valid Region | Lifetime | Data Format |
|----|-------|--------|--------------|----------|-------------|
| c_0 (cb_input_rm) | Wt | RM | All | Per row | input dtype |
| c_1 (cb_tilized_input) | Wt | TILE | All | Per row | input dtype |
| c_2 (cb_gamma_rm) | Wt | RM | All | One-shot | input dtype |
| c_3 (cb_gamma_tilized) | Wt | TILE | All (Row0 replicated 32x) | Program | input dtype |
| c_4 (cb_beta_rm) | Wt | RM | All | One-shot | input dtype |
| c_5 (cb_beta_tilized) | Wt | TILE | All (Row0 replicated 32x) | Program | input dtype |
| c_6 (cb_reduce_scaler) | 1 | TILE | Row0 | Program | **bfloat16 always** |
| c_7 (cb_eps_scalar) | 1 | TILE | [0,0] | Program | input dtype |
| c_8 (cb_output_tiles) | Wt | TILE | All | Per row | input dtype |
| c_16 (cb_output_rm) | Wt | RM | All | Per row | input dtype |
| c_24 (cb_mean) | 1 | TILE | Col0 | Per row | intermed dtype |
| c_25 (cb_centered) | Wt | TILE | All | Per row | intermed dtype |
| c_26 (cb_centered_sq) | Wt | TILE | All | Per row | intermed dtype |
| c_27 (cb_var) | 1 | TILE | Col0 | Per row | intermed dtype |
| c_28 (cb_rstd) | 1 | TILE | Col0 | Per row | intermed dtype |
| c_29 (cb_normed) | Wt | TILE | All | Per row | intermed dtype |
| c_30 (cb_gamma_applied) | Wt | TILE | All | Per row | intermed dtype |

## Binary Op Broadcast Verification

| Phase | Op | CB_A Valid | CB_B Valid | Broadcast | Correct? |
|-------|-----|------------|------------|-----------|----------|
| Mean reduce | SUM REDUCE_ROW | All | Row0 (scaler) | N/A (reduce) | Yes |
| Centering | SUB | All | Col0 (mean) | COL | Yes |
| Square | SQUARE | All | N/A (self) | NONE | Yes |
| Variance reduce | SUM REDUCE_ROW | All | Row0 (scaler) | N/A (reduce) | Yes |
| Add epsilon | ADD | Col0 (var) | [0,0] (eps) | SCALAR | Yes |
| Multiply rstd | MUL | All (centered) | Col0 (rstd) | COL | Yes |
| Apply gamma | MUL | All (normed) | All (gamma, Wt tiles) | NONE | Yes |
| Apply beta | ADD | All (gamma*normed) | All (beta, Wt tiles) | NONE | Yes |

## Reader Kernel

**Phase 0 -- One-time setup:**
- `generate_reduce_scaler(c_6, packed_1_over_W)` -- reduce scaler tile
- `generate_bcast_scalar_bfloat16(c_7, packed_eps)` (bf16) or `generate_bcast_scalar(c_7, eps_bits)` (f32)
- Read gamma: cb_reserve_back(c_2, Wt) -> read 1 gamma stick of W elements, memcpy 32 times to fill 32 rows -> noc_async_read_barrier -> cb_push_back(c_2, Wt)
- Read beta: same pattern into c_4

**Phase 1 -- Per tile-row (num_tile_rows iterations):**
- cb_reserve_back(c_0, Wt)
- For each of 32 sticks: noc_async_read from DRAM (stick_size = W * elem_size) using TensorAccessor
- noc_async_read_barrier()
- cb_push_back(c_0, Wt)

## Compute Kernel

**Startup**: `compute_kernel_hw_startup({c_0, c_1, c_2, c_3, c_4, c_5, c_6, c_7, c_8, c_24, c_25, c_26, c_27, c_28, c_29, c_30}, {c_16})`

**Compile-time args**: `Wt` (index 0), `num_tile_rows` (index 1), `has_gamma` (index 2), `has_beta` (index 3)

**Runtime arg**: `eps_scalar` (packed epsilon value) -- actually no, epsilon is in c_7 from reader. No runtime args for compute.

### Phase 0: Tilize gamma + beta (once, before main loop) -- USE HELPER
```cpp
// Tilize gamma
compute_kernel_lib::tilize<c_2, c_3>(Wt, 1);
// Tilize beta
compute_kernel_lib::tilize<c_4, c_5>(Wt, 1);
```
- After tilize: cb_pop_front(c_2, Wt) and cb_pop_front(c_4, Wt) -- tilize helper handles pop internally
- c_3 and c_5 are now persistent (never popped) for the rest of the program
- If has_gamma==0, skip gamma tilize. If has_beta==0, skip beta tilize.

### Phase 1: Tilize input (per row) -- USE HELPER
```cpp
compute_kernel_lib::tilize<c_0, c_1>(Wt, 1);
```

### Phase 2: Row-wise mean -- USE HELPER
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW,
    ReduceInputPolicy::WaitUpfrontNoPop>(
    c_1, c_6, c_24, ReduceInputBlockShape::row(Wt));
```
- c_1 tiles remain available (WaitUpfrontNoPop) for Phase 3
- Output: 1 tile in c_24 (Col0 valid = per-row means)

### Phase 3: Centering (x - mean) -- USE HELPER
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,
    BinaryInputPolicy::WaitAndPopPerTile>(
    c_1, c_24, c_25, BinaryInputBlockShape::row(Wt));
```
- c_1: NoWaitNoPop (already waited in Phase 2)
- c_24: WaitAndPopPerTile (1 tile, consumed after broadcast)
- **Manual pop required**: `cb_pop_front(c_1, Wt)` after this call (NoWaitNoPop on A)

### Phase 4: Square centered values -- USE HELPER
```cpp
compute_kernel_lib::square<BinaryInputPolicy::WaitAndPopPerTile>(
    c_25, c_26, BinaryInputBlockShape::row(Wt));
```
- Note: c_25 tiles are consumed here. But we need c_25 for Phase 7 (normed = centered * rstd).
- **CRITICAL**: Must use WaitUpfrontNoPop on c_25 instead, then manually pop after Phase 7.

**Corrected Phase 4:**
```cpp
compute_kernel_lib::square<BinaryInputPolicy::WaitUpfrontNoPop>(
    c_25, c_26, BinaryInputBlockShape::row(Wt));
```
- c_25 tiles remain for Phase 7

### Phase 5: Row-wise variance -- USE HELPER
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW,
    ReduceInputPolicy::WaitAndPopPerTile>(
    c_26, c_6, c_27, ReduceInputBlockShape::row(Wt));
```
- c_26 tiles consumed (popped per tile)
- Output: 1 tile in c_27 (Col0 valid = per-row variance)

### Phase 6: rstd = rsqrt(var + eps) -- NO HELPER (raw calls)
```cpp
// Step 6a: var + eps -> c_27 (read-modify-write)
compute_kernel_lib::add<BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::NoWaitNoPop>(
    c_27, c_7, c_27, BinaryInputBlockShape::single());
// c_7 (eps): NoWaitNoPop (persistent, never popped)

// Step 6b: rsqrt(var+eps) -> c_28 (raw SFPU call)
cb_wait_front(c_27, 1);
tile_regs_acquire();
copy_tile_to_dst_init_short(c_27);
copy_tile(c_27, 0, 0);  // Load var+eps into DST[0]
rsqrt_tile_init();
rsqrt_tile(0);           // DST[0] = rsqrt(DST[0])
tile_regs_commit();
tile_regs_wait();
cb_reserve_back(c_28, 1);
pack_tile(0, c_28);
cb_push_back(c_28, 1);
tile_regs_release();
cb_pop_front(c_27, 1);
```

### Phase 7: Standardize (centered * rstd) -- USE HELPER
```cpp
compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,
    BinaryInputPolicy::WaitAndPopPerTile>(
    c_25, c_28, c_29, BinaryInputBlockShape::row(Wt));
```
- c_25: NoWaitNoPop (already waited in Phase 4, still in CB)
- c_28: WaitAndPopPerTile (1 tile, consumed after broadcast)
- **Manual pop required**: `cb_pop_front(c_25, Wt)` after this call

### Phase 8: Apply gamma (normed * gamma) -- USE HELPER (conditional)
```cpp
if (has_gamma) {
    compute_kernel_lib::mul<BroadcastDim::NONE,
        BinaryInputPolicy::WaitAndPopPerTile,
        BinaryInputPolicy::NoWaitNoPop>(
        c_29, c_3, c_30, BinaryInputBlockShape::row(Wt));
    // c_3 (gamma): NoWaitNoPop (persistent, never popped)
}
```
- If no gamma: output of Phase 7 feeds directly to Phase 9 (or untilize)

### Phase 9: Apply beta (gamma*normed + beta) -- USE HELPER (conditional)
```cpp
if (has_beta) {
    uint32_t src_cb = has_gamma ? c_30 : c_29;
    compute_kernel_lib::add<BroadcastDim::NONE,
        BinaryInputPolicy::WaitAndPopPerTile,
        BinaryInputPolicy::NoWaitNoPop>(
        src_cb, c_5, c_8, BinaryInputBlockShape::row(Wt));
    // c_5 (beta): NoWaitNoPop (persistent, never popped)
}
```
- If no gamma and no beta: untilize from c_29
- If gamma only: untilize from c_30
- If both: untilize from c_8

### Phase 10: Untilize output -- USE HELPER
```cpp
uint32_t final_cb = has_beta ? c_8 : (has_gamma ? c_30 : c_29);
// Since untilize needs compile-time CB, may need to always route to c_8
// by copying or by having gamma/beta phases always write to c_8
compute_kernel_lib::untilize<Wt, c_8, c_16>(1);
```
- **Note**: Since untilize requires compile-time input_cb, the kernel should always route the final result to c_8. When no gamma/beta, copy directly from c_29 to c_8 (or make Phase 7 output to c_8 directly when no affine transform).

## Writer Kernel

**Per tile-row (num_tile_rows iterations):**
- cb_wait_front(c_16, Wt)
- For each of 32 sticks: noc_async_write to DRAM using TensorAccessor (stick_size = W * elem_size)
- noc_async_write_barrier()
- cb_pop_front(c_16, Wt)

## Critical Notes

- **NoWaitNoPop on c_1 (Phase 3)**: After `sub<COL>` with NoWaitNoPop on input A, must manually `cb_pop_front(c_1, Wt)`.
- **NoWaitNoPop on c_25 (Phase 7)**: After `mul<COL>` with NoWaitNoPop on input A, must manually `cb_pop_front(c_25, Wt)`.
- **Persistent CBs never popped**: c_3 (gamma), c_5 (beta), c_6 (scaler), c_7 (epsilon) are filled once and never popped. Use NoWaitNoPop for input_b policy.
- **Read-modify-write c_27**: The add<SCALAR> for epsilon writes back to c_27. The helper pops input A before pushing output (safe for same-CB case with 1-tile capacity).
- **Compile-time Wt**: Both tilize and untilize helpers require Wt as compile-time template param. Pass as compile-time arg index 0.
- **Gamma/beta tilize creates 32 identical rows**: Reader must write 32 copies of the gamma/beta row into c_2/c_4 so tilize produces correct tiles where every tile-row has the gamma/beta values.
- **Reduce scaler always bf16**: c_6 must be bf16 format even for f32 operations. Use `generate_reduce_scaler()` which writes packed bf16.
- **Data format reconfig**: After tilize, need INPUT_AND_OUTPUT reconfig for the first binary/reduce op. After rsqrt (Phase 6), need reconfig before mul in Phase 7 since copy_tile changes unpack config.
- **Final output CB routing**: When has_gamma=0 and has_beta=0, Phase 7 must write to c_8 instead of c_29 so untilize always reads from c_8. Simplest: always use c_8 as the final output CB, adjusting which phase writes there.

## Incremental Implementation Strategy

### Step 1: Passthrough (tilize + untilize)
- Reader: read 32 sticks -> push c_0
- Compute: tilize<c_0, c_1>(Wt, 1), then copy c_1 -> c_8, then untilize<Wt, c_8, c_16>(1)
- Writer: wait c_16 -> write 32 sticks
- Validates: data path, tilize/untilize correctness

### Step 2: Mean + subtraction
- Add: reduce<SUM, REDUCE_ROW>(c_1 -> c_24), sub<COL>(c_1, c_24 -> c_8), untilize
- Validates: reduce with 1/W scaler, COL broadcast subtract
- Expected: output = input - mean (should be roughly zero-mean per row)

### Step 3: Full normalization (variance + rstd)
- Add: square(c_25 -> c_26), reduce(c_26 -> c_27), add eps + rsqrt -> c_28, mul<COL>(c_25, c_28 -> c_8)
- Validates: complete standardization pipeline

### Step 4: Affine transform (gamma + beta)
- Add: mul<NONE>(c_29, c_3 -> c_30), add<NONE>(c_30, c_5 -> c_8)
- Validates: gamma/beta tilize-once-reuse, full layer norm

## Implementation Checklist

- [ ] Reader: generate scalers (c_6, c_7), read+fill gamma/beta (c_2, c_4), per-row read 32 sticks (c_0)
- [ ] Compute: 10 phases using helpers (tilize x3, reduce x2, sub, square, add, mul x3) + 1 raw phase (rsqrt)
- [ ] Writer: per-row write 32 sticks from c_16
- [ ] Verify: CB push/pop balance across all phases (especially NoWaitNoPop manual pops on c_1, c_25)
- [ ] Verify: Persistent CBs (c_3, c_5, c_6, c_7) never popped
- [ ] Verify: Final output always routes to c_8 for untilize
