# Operation Design: layer_norm_rm

## Overview
- **Operation Name**: layer_norm_rm
- **Category**: normalization
- **Planning Mode**: Derivative (from general RM-tilize-compute-untilize pattern)
- **Reference Operation(s)**: No single reference analysis; designed from requirements + helper library

## Mathematical Definition
```
mean[b,h] = sum(x[b,h,:]) / W
centered[b,h,w] = x[b,h,w] - mean[b,h]
var[b,h] = sum(centered[b,h,:]^2) / W
output[b,h,w] = centered[b,h,w] * rsqrt(var[b,h] + eps) * gamma[w] + beta[w]
```
Per-row layer normalization on RM interleaved tensors. Gamma/beta are optional affine parameters.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| epsilon | float | No | > 0 | 1e-5 | Numerical stability constant |
| gamma | Tensor | No | shape (1,1,1,W), bf16, RM | None | Scale parameter |
| beta | Tensor | No | shape (1,1,1,W), bf16, RM | None | Bias parameter |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| Dtype | bfloat16 | "Input must be bfloat16" |
| Layout | ROW_MAJOR_LAYOUT | "Input must be row-major" |
| Memory | Interleaved DRAM | "Input must be interleaved DRAM" |
| Rank | >= 2 | "Input must be at least 2D" |
| Last dim | divisible by 32 | "Width must be tile-aligned (div 32)" |
| Second-to-last dim | divisible by 32 | "Height must be tile-aligned (div 32)" |

### Output Tensor Specification
- **Shape**: same as input
- **Dtype**: bfloat16
- **Layout**: ROW_MAJOR_LAYOUT
- **Memory**: interleaved DRAM

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| gamma=None, beta=None | Skip affine, output = normalized |
| gamma provided, beta=None | output = normalized * gamma |
| gamma=None, beta provided | output = normalized + beta |
| W=32 (single tile width) | Wt=1, reduce processes 1 tile per row |

### Work Distribution
- **Work unit**: tile-row (32 rows x W elements = 32 x Wt tiles)
- **Grid**: single core (0,0)
- **Work per core**: all Ht tile-rows, processed sequentially
- **Remainder**: N/A (single core)

### Key Dimensions
- W = input width (last dim), H = product of all other dims
- Wt = W / 32 (tiles per row), Ht = H / 32 (tile-rows to process)
- stick_size = W * 2 bytes (bf16), tile_size = 32 * 32 * 2 = 2048 bytes

### Data Flow
Per tile-row: reader sends 32 RM sticks -> compute tilizes, performs multi-phase layer norm, untilizes -> writer sends RM sticks to DRAM. Gamma/beta are read once by the reader and persist in CBs for all tile-rows.

### Circular Buffer Requirements
| CB ID | Name | Purpose | Producer | Consumer | Pages | Page Size | Lifetime |
|-------|------|---------|----------|----------|-------|-----------|----------|
| 0 | cb_in | RM sticks for tilize | reader | compute | Wt | tile_size | per tile-row |
| 1 | cb_gamma | Tilized gamma tiles | compute(tilize) | compute(mul) | Wt | tile_size | entire kernel (if present) |
| 2 | cb_beta | Tilized beta tiles | compute(tilize) | compute(add) | Wt | tile_size | entire kernel (if present) |
| 3 | cb_gamma_rm | RM gamma sticks for tilize | reader | compute | Wt | tile_size | loaded once |
| 4 | cb_beta_rm | RM beta sticks for tilize | reader | compute | Wt | tile_size | loaded once |
| 8 | cb_scaler | Reduce scaler (1/W) | reader | compute | 1 | tile_size | entire kernel |
| 9 | cb_eps | Epsilon scalar tile | reader | compute | 1 | tile_size | entire kernel |
| 16 | cb_out | RM sticks from untilize | compute | writer | Wt | tile_size | per tile-row |
| 24 | cb_tilized | Tilized input tiles | compute(tilize) | compute(reduce,sub) | Wt | tile_size | persists: tilize -> sub |
| 25 | cb_mean | Row means (reduced) | compute(reduce) | compute(sub) | 1 | tile_size | persists: reduce -> sub |
| 26 | cb_centered | Centered tiles | compute(sub) | compute(square,mul) | Wt | tile_size | persists: sub -> mul(inv_std) |
| 27 | cb_sq | Squared centered tiles | compute(square) | compute(reduce) | Wt | tile_size | per phase |
| 28 | cb_var | Variance tile (reduced) | compute(reduce) | compute(add+rsqrt) | 1 | tile_size | per phase |
| 29 | cb_inv_std | 1/sqrt(var+eps) tile | compute(add+rsqrt) | compute(mul) | 1 | tile_size | persists: add -> mul |
| 30 | cb_norm | Normalized tiles | compute(mul) | compute(affine)/untilize | Wt | tile_size | per tile-row |

### Kernel Arguments

**Compile-time** (reader):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Wt | uint32_t | Tiles per row width |
| 1 | Ht | uint32_t | Total tile-rows to process |
| 2 | has_gamma | uint32_t | 0 or 1 |
| 3 | has_beta | uint32_t | 0 or 1 |
| 4+ | TensorAccessorArgs(input) | auto | Input tensor accessor |
| ... | TensorAccessorArgs(gamma) | auto | Gamma tensor accessor (slot always present) |
| ... | TensorAccessorArgs(beta) | auto | Beta tensor accessor (slot always present) |

**Compile-time** (compute):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Wt | uint32_t | Tiles per row width |
| 1 | Ht | uint32_t | Total tile-rows to process |
| 2 | has_gamma | uint32_t | 0 or 1 |
| 3 | has_beta | uint32_t | 0 or 1 |

**Compile-time** (writer):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Wt | uint32_t | Tiles per row width |
| 1 | Ht | uint32_t | Total tile-rows to process |
| 2+ | TensorAccessorArgs(output) | auto | Output tensor accessor |

**Runtime** (reader):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | input_addr | uint32_t | Input buffer base address |
| 1 | gamma_addr | uint32_t | Gamma buffer address (0 if absent) |
| 2 | beta_addr | uint32_t | Beta buffer address (0 if absent) |
| 3 | scaler_value | uint32_t | 1/W as packed bf16 (bf16<<16 | bf16) |
| 4 | eps_value | uint32_t | epsilon as packed bf16 |

**Runtime** (writer):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_addr | uint32_t | Output buffer base address |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count (validated per-CB below)
- [x] Reduce scaler CB is bfloat16 (cb_scaler=8, bf16 format)
- [x] DEST register holds max 8 tiles (bf16 half-sync) - all helpers manage DEST internally
- [x] RM CBs count pages in sticks, tile CBs count in tiles - cb_in uses tile-sized pages for tilize compatibility

### Test Criteria
- Output shape matches input shape
- Numerical accuracy vs `torch.nn.functional.layer_norm`

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile width | `(1, 1, 32, 32)` |
| Multi-tile | Tile iteration | `(1, 1, 64, 128)` |
| Non-square | W!=H | `(1, 1, 32, 256)` |
| Multi-batch | Batch handling | `(4, 2, 64, 64)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | Pages | Layout | Valid Region | Lifetime |
|----|-------|--------|--------------|----------|
| cb_in (0) | Wt | tile-sized RM data | All (32 sticks) | per tile-row, freed by tilize |
| cb_gamma (1) | Wt | TILE | All rows identical | entire kernel |
| cb_beta (2) | Wt | TILE | All rows identical | entire kernel |
| cb_gamma_rm (3) | Wt | tile-sized RM data | All (32 identical rows) | loaded once, freed by tilize |
| cb_beta_rm (4) | Wt | tile-sized RM data | All (32 identical rows) | loaded once, freed by tilize |
| cb_scaler (8) | 1 | TILE (bf16) | Row 0 of each face | entire kernel |
| cb_eps (9) | 1 | TILE (bf16) | Element [0,0] | entire kernel |
| cb_out (16) | Wt | tile-sized RM data | All | per tile-row |
| cb_tilized (24) | Wt | TILE | All | persists tilize -> sub |
| cb_mean (25) | 1 | TILE | Col 0 (reduce_row output) | persists reduce -> sub |
| cb_centered (26) | Wt | TILE | All | persists sub -> mul(inv_std) |
| cb_sq (27) | Wt | TILE | All | streaming square -> reduce |
| cb_var (28) | 1 | TILE | Col 0 (reduce_row output) | per phase |
| cb_inv_std (29) | 1 | TILE | Col 0 | persists add -> mul |
| cb_norm (30) | Wt | TILE | All | per tile-row |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| Sub mean | SUB | All (cb_tilized) | Col0 (cb_mean) | COL |
| Mul inv_std | MUL | All (cb_centered) | Col0 (cb_inv_std) | COL |
| Mul gamma | MUL | All (cb_norm) | All rows identical (cb_gamma) | NONE |
| Add beta | ADD | All | All rows identical (cb_beta) | NONE |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output | Phases |
|-------|------|-------------|-----------------|--------|
| 1 | data_pipeline | tilize + untilize | identity passthrough | 2 |
| 2 | centering | reduce(mean) + sub<COL> | x - mean | 2 |
| 3 | normalize | square + reduce(var) + add_eps_rsqrt + mul<COL> | layer_norm (no affine) | 4 |
| 4 | affine | mul(gamma) + add(beta) | full layer_norm | 2 |

Stage 3 has 4 new phases, exceeding the 3-phase guideline. Splitting is impractical because intermediate results (variance, inv_std) are reduced shapes requiring a different untilize block_width_tiles compile-time parameter.

### Stage 1: data_pipeline
- **Scope**: reader, compute (tilize + untilize), writer
- **Reference**: `return input_tensor`
- **Shapes**: `(1,1,32,32)`, `(1,1,64,128)`, `(1,1,32,256)`, `(4,2,64,64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: compute tilizes cb_in -> cb_tilized, immediately untilizes cb_tilized -> cb_out

### Stage 2: centering
- **Scope**: compute (add reduce + sub phases)
- **Reference**: `return input_tensor - input_tensor.to(torch.float32).mean(dim=-1, keepdim=True).to(torch.bfloat16)`
- **Shapes**: same
- **Tolerances**: rtol=0.02, atol=0.1
- **Delta from previous**: after tilize, insert reduce(mean) + sub(mean), then untilize cb_centered instead of cb_tilized

### Stage 3: normalize
- **Scope**: compute (add square + reduce_var + add_eps_rsqrt + mul_inv_std)
- **Reference**: `return torch.nn.functional.layer_norm(input_tensor.to(torch.float32), [input_tensor.shape[-1]], eps=1e-5).to(torch.bfloat16)`
- **Shapes**: same
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: after sub, insert square+reduce(var)+add_eps_rsqrt+mul(inv_std), untilize cb_norm instead of cb_centered

### Stage 4: affine
- **Scope**: reader (add gamma/beta loading), compute (add mul_gamma + add_beta)
- **Reference**: `return torch.nn.functional.layer_norm(input_tensor.to(torch.float32), [input_tensor.shape[-1]], weight=gamma.to(torch.float32), bias=beta.to(torch.float32), eps=1e-5).to(torch.bfloat16)`
- **Shapes**: same
- **Tolerances**: rtol=0.05, atol=0.2
- **Delta from previous**: reader loads gamma/beta, tilizes them; compute adds mul(gamma)+add(beta) after normalize

### Reader Kernel

Per tile-row iteration (Ht times):
1. Reserve cb_in(Wt), read 32 sticks from input DRAM via TensorAccessor, push cb_in(Wt)

Before main loop (once):
- Generate reduce scaler (1/W) into cb_scaler using `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(1.0f / W)`
- Generate epsilon scaler into cb_eps using `dataflow_kernel_lib::prepare_reduce_scaler<cb_eps>(epsilon_float)`
- If has_gamma: fill cb_gamma_rm with 32 copies of gamma stick per tile-width (replicate row for tilize), push Wt pages
- If has_beta: same for cb_beta_rm

Gamma/beta replication: for each of 32 rows, noc_async_read the same W-byte gamma stick into consecutive row offsets within the Wt tile-sized pages. This ensures tilize produces tiles with identical rows.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_in, cb_scaler, cb_out);`

Gamma/beta tilize (once, before main loop):
```cpp
if constexpr (has_gamma) {
    compute_kernel_lib::tilize<Wt, cb_gamma_rm, cb_gamma>(1);
}
if constexpr (has_beta) {
    compute_kernel_lib::tilize<Wt, cb_beta_rm, cb_beta>(1);
}
```

Main loop (Ht iterations):

#### Phase 1: Tilize
```cpp
compute_kernel_lib::tilize<Wt, cb_in, cb_tilized>(1);
```
- In: cb_in [Wt pages, waited+popped internally]
- Out: cb_tilized [Wt tiles pushed]

#### Phase 2: Reduce row (mean)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW, ReduceInputPolicy::WaitUpfrontNoPop>(
    cb_tilized, cb_scaler, cb_mean,
    ReduceInputBlockShape::of(1, Wt));
```
- A: cb_tilized [Wt tiles, WAITED upfront, NOT POPPED - persists for phase 3]
- Scaler: cb_scaler [1 tile, 1/W, persistent]
- Out: cb_mean [1 tile pushed, col 0 = row means]

#### Phase 3: Subtract mean (centering)
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitPopAtEnd,
    BinaryInputPolicy::WaitAndPopPerTile>(
    cb_tilized, cb_mean, cb_centered, BinaryInputBlockShape::of(1, Wt));
```
- A: cb_tilized [Wt tiles, ALREADY WAITED from phase 2, pop at end]
- B: cb_mean [1 tile, waited+popped per tile (1 tile for COL bcast)]
- Out: cb_centered [Wt tiles pushed]

**CB state after phase 3:**
| CB | Tiles | State |
|----|-------|-------|
| cb_tilized | 0 | freed (popped at end by sub) |
| cb_mean | 0 | freed (popped by sub) |
| cb_centered | Wt | pushed, persists for phases 4+6 |

#### Phase 4: Square centered values
```cpp
compute_kernel_lib::square<BinaryInputPolicy::WaitUpfrontNoPop>(
    cb_centered, cb_sq, BinaryInputBlockShape::of(1, Wt));
```
- A: cb_centered [Wt tiles, WAITED upfront, NOT POPPED - persists for phase 6]
- Out: cb_sq [Wt tiles pushed]

#### Phase 5: Reduce row (variance)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW>(
    cb_sq, cb_scaler, cb_var,
    ReduceInputBlockShape::of(1, Wt));
```
- A: cb_sq [Wt tiles, waited+popped per tile by default policy]
- Scaler: cb_scaler [1/W, persistent]
- Out: cb_var [1 tile pushed, col 0 = row variances]

#### Phase 6: Add epsilon + rsqrt (inv_std)
```cpp
compute_kernel_lib::add<BroadcastDim::SCALAR>(
    cb_var, cb_eps, cb_inv_std, BinaryInputBlockShape::of(1, 1),
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```
- A: cb_var [1 tile, waited+popped]
- B: cb_eps [1 tile, epsilon scalar, WaitAndPopPerTile - but only 1 tile exists. Need WaitUpfrontNoPop to keep eps persistent]

**Correction**: cb_eps must persist across all tile-rows. Use `WaitUpfrontNoPop` for B:
```cpp
compute_kernel_lib::add<BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop>(
    cb_var, cb_eps, cb_inv_std, BinaryInputBlockShape::of(1, 1),
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```
- A: cb_var [1 tile, waited+popped]
- B: cb_eps [1 tile, waited upfront first iteration, NOT popped - persists]
- Out: cb_inv_std [1 tile pushed]
- Post-op: rsqrt applied in-place in DEST before packing

#### Phase 7: Multiply by inv_std (normalize)
```cpp
compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitPopAtEnd,
    BinaryInputPolicy::WaitAndPopPerTile>(
    cb_centered, cb_inv_std, cb_norm, BinaryInputBlockShape::of(1, Wt));
```
- A: cb_centered [Wt tiles, ALREADY WAITED from phase 4 square, pop at end]
- B: cb_inv_std [1 tile, waited+popped per tile (1 tile for COL bcast)]
- Out: cb_norm [Wt tiles pushed]

**CB state after phase 7:**
| CB | Tiles | State |
|----|-------|-------|
| cb_centered | 0 | freed (popped at end by mul) |
| cb_inv_std | 0 | freed (popped by mul) |
| cb_norm | Wt | pushed |

#### Phase 8: Affine (conditional)
If has_gamma:
```cpp
compute_kernel_lib::mul<BroadcastDim::NONE,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop>(
    cb_norm, cb_gamma, cb_affine_out, BinaryInputBlockShape::of(1, Wt));
```
- A: cb_norm [Wt tiles, waited+popped per tile]
- B: cb_gamma [Wt tiles, waited upfront, NOT popped - persists across tile-rows]
- Out: cb_affine_out [Wt tiles]

If has_beta (after gamma, or directly from cb_norm if no gamma):
```cpp
compute_kernel_lib::add<BroadcastDim::NONE,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitUpfrontNoPop>(
    cb_prev, cb_beta, cb_final, BinaryInputBlockShape::of(1, Wt));
```
- B: cb_beta [Wt tiles, persistent]

When both gamma and beta: cb_norm -> mul(gamma) -> cb_affine_tmp -> add(beta) -> cb_final.
When only gamma: cb_norm -> mul(gamma) -> cb_final.
When only beta: cb_norm -> add(beta) -> cb_final.
When neither: cb_final = cb_norm (no affine phases).

The affine output CB feeds into untilize. To avoid conditional CB routing, use cb_norm (30) as compute output when no affine, or a dedicated cb (31) as affine temp. The final result always goes to a consistent CB for untilize.

**Implementation note**: Use `cb_norm (30)` as untilize input when no affine. When gamma only, mul outputs to a temp CB and that feeds untilize. When both, chain through temp. The kernel writer should use constexpr if to select the output CB for untilize based on has_gamma/has_beta.

#### Phase 9: Untilize
```cpp
compute_kernel_lib::untilize<Wt, cb_final, cb_out>(1);
```
- In: cb_final [Wt tiles, waited+popped internally]
- Out: cb_out [Wt pages of RM data pushed]

### Writer Kernel

Per tile-row iteration (Ht times):
1. Wait cb_out(Wt), read Wt tile-sized pages as RM sticks
2. For each tile (Wt tiles), extract 32 sticks and write to output DRAM via TensorAccessor
3. Pop cb_out(Wt)

The writer treats cb_out as containing RM sticks (post-untilize). Each tile-sized page contains 32 sticks of 32*elem_size bytes each. Writer iterates: for each of Wt pages, for each of 32 sticks within the page, noc_async_write the stick to the appropriate output page.

### Critical Notes

1. **cb_scaler and cb_eps persist** for the entire kernel. The reader pushes them once before the main loop. In compute, cb_scaler uses WaitUpfrontNoPop via the reduce helper. cb_eps uses WaitUpfrontNoPop in the add helper. Neither is ever popped.

2. **Gamma/beta replication**: Reader must write 32 identical copies of the gamma/beta stick into the CB before tilize. This ensures all 32 rows of the resulting tiles contain the same values, which is required for element-wise mul/add (BroadcastDim::NONE).

3. **Phase 4 (square) uses WaitUpfrontNoPop** for cb_centered, so centered tiles persist for phase 7 (mul inv_std). The square helper's NoWaitNoPop policy means compute does `cb_wait_front(cb_centered, Wt)` implicitly via WaitUpfrontNoPop, then does not pop.

4. **Phase 6 rsqrt post-op**: The `add` helper's post-op receives `dst_idx` and applies rsqrt in-place. Must call `rsqrt_tile_init()` inside the lambda. This combines add(var, eps) + rsqrt into a single DEST pass.

5. **cb_eps WaitUpfrontNoPop on first iteration**: The binary_op helper with WaitUpfrontNoPop for B will `cb_wait_front(cb_eps, 1)` on the first call. On subsequent tile-rows, the tile is already waited (still in CB, never popped). The helper must handle this — WaitUpfrontNoPop waits once and never pops, so repeated calls will find the tile already present.

6. **Scaler packing for reader**: 1/W as float -> bf16 -> packed as `(bf16 << 16) | bf16`. Use `dataflow_kernel_lib::prepare_reduce_scaler<cb_id>(float_value)` which handles format conversion.

### Implementation Checklist
- [ ] Reader: TensorAccessor reads, generate_reduce_scaler for 1/W and eps, gamma/beta replication+push
- [ ] Compute: 9 phases using helpers: tilize, reduce(mean), sub(COL), square, reduce(var), add(SCALAR)+rsqrt, mul(COL), mul/add(NONE for affine), untilize
- [ ] Writer: TensorAccessor writes RM sticks from untilized output
- [ ] CB push/pop balance verified per tile-row iteration
