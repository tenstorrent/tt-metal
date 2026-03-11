# Operation Design: layer_norm_rm

## Overview
- **Operation Name**: layer_norm_rm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input_stage), batch_norm (compute_core), untilize (output_stage)

## Mathematical Definition
```
mean[b,h] = (1/W) * sum_w(input[b,h,w])
var[b,h]  = (1/W) * sum_w((input[b,h,w] - mean[b,h])^2)
norm[b,h,w] = (input[b,h,w] - mean[b,h]) / sqrt(var[b,h] + eps)
output[b,h,w] = norm[b,h,w] * gamma[w] + beta[w]   (if gamma/beta provided)
```
Layer normalization normalizes across the last dimension (W) for each row independently. Gamma and beta are per-feature (per-column) affine parameters with shape `(1,1,1,W)`.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input | Tensor | Yes | >=2D, tile-aligned, RM, bf16, interleaved | - | Input tensor |
| gamma | Tensor | No | (1,1,1,W) RM bf16 interleaved | None | Per-feature scale |
| beta | Tensor | No | (1,1,1,W) RM bf16 interleaved | None | Per-feature shift |
| epsilon | float | No | >0 | 1e-5 | Numerical stability constant |

### Input Tensor Requirements
| Property | Requirement |
|----------|-------------|
| Layout | ROW_MAJOR_LAYOUT |
| Memory | Interleaved (DRAM) |
| Dtype | BFLOAT16 |
| Alignment | Last 2 dims divisible by 32 |
| Rank | >= 2 (outer dims flattened) |

### Output Tensor Specification
- **Shape**: Same as input
- **Dtype**: BFLOAT16
- **Layout**: ROW_MAJOR_LAYOUT
- **Memory**: Interleaved (DRAM)

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| gamma=None, beta=None | Pure normalization only |
| gamma provided, beta=None | Normalize then scale |
| Single tile width (W=32) | Wt=1, reduce_row produces 1 tile per row |
| epsilon=0 | Allowed but not recommended (div-by-zero risk) |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | tilize analysis | input_stage | Add gamma/beta RM stick reads, epsilon fill, reduce scaler generation |
| Compute | batch_norm analysis + NEW | compute_core | Replace batch_mean/var with self-computed mean/var via reduce; tilize input, untilize output |
| Writer | untilize analysis | output_stage | Standard RM stick writer from untilized output CB |

### Work Distribution
- **Work unit**: tile-row (one horizontal row of Wt tiles = 32 sticks of width W)
- **Grid**: Single core (0,0) for simplicity
- **Work per core**: All Ht tile-rows (Ht = total_height / 32)
- **Remainder**: N/A (single core)

### Data Flow

Reader reads 32 RM input sticks per tile-row into cb_in (also reads gamma/beta sticks into their CBs once at start, fills epsilon and reduce scaler CBs once). Compute tilizes cb_in, performs normalization math (reduce_row for mean, subtract mean, square, reduce_row for variance, add epsilon + rsqrt, multiply by inv_std, optional gamma/beta), then untilizes result to cb_out. Writer extracts 32 RM sticks per tile-row from cb_out and writes to DRAM.

### Circular Buffer Requirements

**Notation**: Wt = W/32 (tiles per row), Ht = total_height/32 (tile-rows). NC = product of all dims except last two, divided by 32 for height tiles.

| CB ID | Name | Purpose | Producer | Consumer | Pages | Page Size | Lifetime |
|-------|------|---------|----------|----------|-------|-----------|----------|
| c_0 | cb_in | RM sticks from reader | Reader | Compute (tilize) | Wt | tile_size | Per tile-row |
| c_1 | cb_tilized | Tilized input tiles | Compute (tilize) | Compute (reduce/sub) | Wt | tile_size | Per tile-row |
| c_2 | cb_mean | Row mean (reduce_row result) | Compute | Compute | 1 | tile_size | Per tile-row |
| c_3 | cb_centered | x - mean | Compute | Compute (square + mul) | Wt | tile_size | Per tile-row |
| c_4 | cb_sq | (x - mean)^2 | Compute | Compute (reduce) | Wt | tile_size | Per tile-row |
| c_5 | cb_var | Row variance | Compute | Compute | 1 | tile_size | Per tile-row |
| c_6 | cb_eps | Epsilon constant tile | Reader | Compute | 1 | tile_size | Program |
| c_7 | cb_inv_std | 1/sqrt(var+eps) | Compute | Compute | 1 | tile_size | Per tile-row |
| c_8 | cb_scaler | Reduce scaler (1/W) | Reader | Compute | 1 | tile_size | Program |
| c_9 | cb_gamma | Gamma tiles (tilized from RM) | Reader | Compute | Wt | tile_size | Program |
| c_10 | cb_beta | Beta tiles (tilized from RM) | Reader | Compute | Wt | tile_size | Program |
| c_24 | cb_normalized | Normalized output tiles | Compute | Compute (untilize) | Wt | tile_size | Per tile-row |
| c_16 | cb_out | Untilized RM output | Compute (untilize) | Writer | Wt | tile_size | Per tile-row |

### Kernel Arguments

**Compile-time** (Reader):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | W * 2 bytes (bf16) |
| 1 | Wt | uint32_t | Tiles per row (W/32) |
| 2 | Ht | uint32_t | Total tile-rows |
| 3 | W | uint32_t | Width in elements |
| 4 | has_gamma | uint32_t | 1 if gamma provided |
| 5 | has_beta | uint32_t | 1 if beta provided |
| 6+ | TensorAccessorArgs (input) | ... | Input tensor accessor |
| ... | TensorAccessorArgs (gamma) | ... | If has_gamma |
| ... | TensorAccessorArgs (beta) | ... | If has_beta |

**Runtime** (Reader):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer base address |
| 1 | gamma_addr | uint32_t | Gamma buffer address (0 if none) |
| 2 | beta_addr | uint32_t | Beta buffer address (0 if none) |
| 3 | packed_eps | uint32_t | Epsilon as packed bf16 pair |

**Compile-time** (Compute):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Wt | uint32_t | Tiles per row |
| 1 | Ht | uint32_t | Total tile-rows |
| 2 | has_gamma | uint32_t | 1 if gamma provided |
| 3 | has_beta | uint32_t | 1 if beta provided |

**Compile-time** (Writer):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | W * 2 bytes |
| 1 | Wt | uint32_t | Tiles per row |
| 2 | Ht | uint32_t | Total tile-rows |
| 3+ | TensorAccessorArgs (output) | ... | Output tensor accessor |

**Runtime** (Writer):

| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer base address |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count
- [x] Reduce scaler CB is bfloat16
- [x] DEST register holds max 8 tiles (bf16 half-sync) -- helpers handle sub-blocking
- [x] RM CBs count pages in sticks, tile CBs count in tiles (all CBs here use tile_size pages)

### Test Criteria
- Output shape matches input shape
- Numerical accuracy vs `torch.nn.functional.layer_norm` with rtol=0.02, atol=0.02

| Category | Purpose | Shape |
|----------|---------|-------|
| Minimal | Single tile | `(1, 1, 32, 32)` |
| Multi-tile W | Tile iteration in W | `(1, 1, 32, 64)` |
| Multi-tile HW | Multiple tile-rows | `(1, 1, 64, 64)` |
| Multi-batch | Batch handling | `(2, 1, 64, 64)` |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | Pages | Layout | Valid Region | Lifetime |
|----|-------|--------|--------------|----------|
| c_0 (cb_in) | Wt | tile | All (RM sticks, tilize input) | Per tile-row |
| c_1 (cb_tilized) | Wt | tile | All | Per tile-row |
| c_2 (cb_mean) | 1 | tile | Col0 (REDUCE_ROW output) | Per tile-row |
| c_3 (cb_centered) | Wt | tile | All | Per tile-row |
| c_4 (cb_sq) | Wt | tile | All | Per tile-row |
| c_5 (cb_var) | 1 | tile | Col0 (REDUCE_ROW output) | Per tile-row |
| c_6 (cb_eps) | 1 | tile | All (filled with eps) | Program |
| c_7 (cb_inv_std) | 1 | tile | Col0 (result of add+rsqrt on Col0 inputs) | Per tile-row |
| c_8 (cb_scaler) | 1 | tile | Row0 faces (reduce scaler format) | Program |
| c_9 (cb_gamma) | Wt | tile | Row0 (1D gamma tilized as single row) | Program |
| c_10 (cb_beta) | Wt | tile | Row0 (1D beta tilized as single row) | Program |
| c_24 (cb_normalized) | Wt | tile | All | Per tile-row |
| c_16 (cb_out) | Wt | tile | All (untilized RM sticks) | Per tile-row |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| subtract_mean | SUB | All (cb_tilized) | Col0 (cb_mean) | COL |
| square | MUL(self) | All (cb_centered) | All (cb_centered) | NONE |
| add_eps | ADD | Col0 (cb_var) | All (cb_eps) | SCALAR (both scalar-like) |
| mul_inv_std | MUL | All (cb_centered) | Col0 (cb_inv_std) | COL |
| mul_gamma | MUL | All (cb_normalized) | Row0 (cb_gamma) | ROW |
| add_beta | ADD | All (after gamma) | Row0 (cb_beta) | ROW |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output | Output Shape | Compare Slice |
|-------|------|-------------|-----------------|--------------|---------------|
| 1 | normalize | Full norm pipeline (tilize, mean, center, var, inv_std, normalize, untilize) | `layer_norm(input)` | Same as input | - |
| 2 | gamma | Add gamma multiply | `layer_norm(input) * gamma` | Same as input | - |
| 3 | affine | Add beta add | `layer_norm(input, gamma, beta)` | Same as input | - |

### Stage 1: normalize
- **Scope**: All 3 kernel files, compute phases 1-8 (tilize, reduce_mean, subtract_mean, square, reduce_var, add_eps+rsqrt, mul_inv_std, untilize)
- **Reference**: `torch.nn.functional.layer_norm(input, [input.shape[-1]])`
- **Shapes**: `(1,1,32,32)`, `(1,1,32,64)`, `(1,1,64,64)`, `(2,1,64,64)`
- **Tolerances**: rtol=0.02, atol=0.02
- **Notes**: Gamma/beta compile-time flags set to 0. Reader fills epsilon + scaler CBs, reads input sticks. Writer writes output sticks.

### Stage 2: gamma
- **Scope**: Reader adds gamma stick reads + tilize. Compute adds mul_gamma phase.
- **Reference**: `torch.nn.functional.layer_norm(input, [input.shape[-1]]) * gamma`
- **Delta from Stage 1**: has_gamma=1, reader reads gamma sticks and tilizes into cb_gamma once at start. Compute multiplies normalized result by gamma with ROW broadcast.

### Stage 3: affine
- **Scope**: Reader adds beta stick reads + tilize. Compute adds add_beta phase.
- **Reference**: `torch.nn.functional.layer_norm(input, [input.shape[-1]], weight=gamma, bias=beta)`
- **Delta from Stage 2**: has_beta=1, reader reads beta sticks similarly. Compute adds beta with ROW broadcast.

### Reader Kernel

Reads RM sticks from DRAM using TensorAccessor. Per tile-row: reserves cb_in for Wt pages, reads 32 sticks contiguously, pushes Wt pages. At program start: fills cb_eps with packed epsilon using `fill_with_val_bfloat16`, generates reduce scaler in cb_scaler using `generate_reduce_scaler`. If has_gamma: reads W/32 gamma sticks (each 64 bytes for W=32, i.e., stick_size bytes for gamma which has width W) into cb_gamma and pushes Wt pages. Similarly for beta into cb_beta. Gamma/beta are 1D tensors with shape (1,1,1,W) in RM, so there are 1 stick of W elements total (but we need to read it into a tile-format for the tilize helper -- actually, since gamma is (1,1,1,W) in RM, it has exactly 1 stick. We read that 1 stick 32 times to fill 32 rows of tile height, OR we read it once and use ROW broadcast in the binary op. Since binary_op helper with ROW broadcast expects Row0 valid, we only need the first row filled).

**Gamma/Beta read strategy**: Read the single gamma stick into cb_gamma_rm (a separate RM staging CB with 1 page of stick_size), then tilize it into cb_gamma. Since gamma has only 1 row, we handle this by reading 1 stick into position 0 of the CB (reserving Wt tile-pages but only writing 1 stick). The tilize helper with `total_input_pages=1` would handle single-row tilize. The resulting tiles will have valid data only in row 0, which is correct for ROW broadcast.

**Revised gamma/beta approach**: Since gamma and beta are (1,1,1,W) -- exactly 1 stick -- the reader writes this single stick into the first row position of a Wt-page CB. The rest of the tile is uninitialized but irrelevant since ROW broadcast only uses row 0. Use `tilize<cb_gamma_rm, cb_gamma>(Wt, 1, 1)` with total_input_pages=1 to tilize the single stick.

Actually, this creates a problem: tilize expects 32 sticks per block. With total_input_pages=1, the tilize helper waits for min(32,1)=1 page and tilizes. But the HW tilize unit needs 32 rows of data to produce correct tiles. With only 1 row, the remaining 31 rows contain garbage.

**Final gamma/beta approach**: Do NOT tilize gamma/beta. Instead, read gamma/beta as RM sticks directly into a CB, and use `fill_tile_with_first_element_bfloat16`-style broadcast, OR read the RM stick and manually replicate it. However, the simplest approach: read the gamma RM stick into a tile-sized CB with stick data at row 0, zero the rest. Since ROW broadcast in the binary_op helper replicates row 0 across all rows, this works perfectly. The reader reads 1 gamma stick into cb_gamma (Wt tile-pages pre-zeroed), positioning it at offset 0. But CB memory is not pre-zeroed.

**Simplest correct approach**: The reader reads the gamma stick (W * 2 bytes) into cb_gamma. The CB has Wt pages of tile_size each. We reserve Wt pages, zero the entire CB space first (using MEM_ZEROS like generate_reduce_scaler does), then write the single gamma stick at offset 0. This places gamma values in row 0 of each tile face, with all other rows zeroed. Binary ROW broadcast uses row 0 only. Same for beta.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_in, cb_scaler, cb_out)` -- 3-arg form since srcA (cb_in for tilize), srcB (cb_scaler for reduce), and output (cb_out for untilize) all differ.

Processing loop: for each of Ht tile-rows:

#### Phase 1: Tilize
```cpp
compute_kernel_lib::tilize<c_0, c_1,
    tilize_config::InitUninitMode::InitAndUninit,
    tilize_config::WaitMode::WaitBlock>(Wt, 1);
```
- In: c_0 [Wt pages, waited by helper, popped by helper]
- Out: c_1 [Wt tiles pushed by helper]

#### Phase 2: Reduce Mean (REDUCE_ROW with SUM scaler = 1/W)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW,
    compute_kernel_lib::ReduceInputPolicy::WaitUpfrontNoPop>(
    c_1, c_8, c_2,
    compute_kernel_lib::ReduceInputBlockShape::row(Wt));
```
- In: c_1 [Wt tiles, WAITED by helper, NOT POPPED -- persists for Phase 3]
- Scaler: c_8 [1 tile, waited by helper internally]
- Out: c_2 [1 tile pushed -- row mean, valid in Col0]

**CB state after Phase 2:**
| CB | Tiles | State |
|----|-------|-------|
| c_1 | Wt | waited, not popped -- persists for Phase 3 |
| c_2 | 1 | freshly pushed (mean) |

#### Phase 3: Subtract Mean (centered = input - mean, COL broadcast)
```cpp
compute_kernel_lib::sub<compute_kernel_lib::BroadcastDim::COL,
    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
    c_1, c_2, c_3, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
```
- A: c_1 [Wt tiles, ALREADY WAITED from Phase 2, NoWaitNoPop -- caller must pop after]
- B: c_2 [1 tile, WaitAndPopPerTile -- waited and popped by helper (COL bcast reuses same tile)]
- Out: c_3 [Wt tiles pushed]

Manual pop after Phase 3: `cb_pop_front(c_1, Wt)` -- releases tilized input.

**CB state after Phase 3 + manual pop:**
| CB | Tiles | State |
|----|-------|-------|
| c_1 | 0 | freed |
| c_2 | 0 | freed (popped by helper) |
| c_3 | Wt | freshly pushed (centered values) |

#### Phase 4: Square (sq = centered^2)
```cpp
compute_kernel_lib::square<
    compute_kernel_lib::BinaryInputPolicy::WaitUpfrontNoPop>(
    c_3, c_4, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
```
- In: c_3 [Wt tiles, WAITED by helper, NOT POPPED -- persists for Phase 7]
- Out: c_4 [Wt tiles pushed]

**CB state after Phase 4:**
| CB | Tiles | State |
|----|-------|-------|
| c_3 | Wt | waited, not popped -- persists for Phase 7 |
| c_4 | Wt | freshly pushed (squared) |

#### Phase 5: Reduce Variance (REDUCE_ROW on squared values)
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW>(
    c_4, c_8, c_5,
    compute_kernel_lib::ReduceInputBlockShape::row(Wt));
```
- In: c_4 [Wt tiles, waited and popped by helper]
- Scaler: c_8 [1 tile, reduce scaler = 1/W for AVG]
- Out: c_5 [1 tile pushed -- row variance, valid in Col0]

#### Phase 6: Inverse Std = rsqrt(var + eps)
```cpp
compute_kernel_lib::add<compute_kernel_lib::BroadcastDim::SCALAR>(
    c_5, c_6, c_7, compute_kernel_lib::BinaryInputBlockShape::single(),
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```
- A: c_5 [1 tile, waited and popped by helper]
- B: c_6 [1 tile, waited and popped per tile -- BUT eps is program-lifetime, so use WaitUpfrontNoPop for B]
- Out: c_7 [1 tile pushed -- inv_std, valid in Col0]

Corrected Phase 6 with proper epsilon persistence:
```cpp
compute_kernel_lib::add<compute_kernel_lib::BroadcastDim::SCALAR,
    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
    c_5, c_6, c_7, compute_kernel_lib::BinaryInputBlockShape::single(),
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```
- A: c_5 [1 tile, waited and popped]
- B: c_6 [1 tile, NoWaitNoPop -- epsilon persists, caller waited once at program start]

Note: The caller must `cb_wait_front(c_6, 1)` once before the main loop and `cb_pop_front(c_6, 1)` once after all tile-rows are processed.

#### Phase 7: Normalize (normalized = centered * inv_std, COL broadcast)
```cpp
compute_kernel_lib::mul<compute_kernel_lib::BroadcastDim::COL,
    compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop,
    compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile>(
    c_3, c_7, cb_target, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
```
Where `cb_target` = c_24 if has_gamma or has_beta, else c_24 (always c_24 since untilize reads from a different CB).

- A: c_3 [Wt tiles, ALREADY WAITED from Phase 4, NoWaitNoPop -- caller must pop]
- B: c_7 [1 tile, waited and popped by helper (COL bcast)]
- Out: cb_target [Wt tiles pushed]

Manual pop: `cb_pop_front(c_3, Wt)` -- releases centered values.

#### Phase 7.5: Multiply by Gamma (if has_gamma, ROW broadcast)
```cpp
if constexpr (has_gamma) {
    compute_kernel_lib::mul<compute_kernel_lib::BroadcastDim::ROW,
        compute_kernel_lib::BinaryInputPolicy::WaitAndPopPerTile,
        compute_kernel_lib::BinaryInputPolicy::NoWaitNoPop>(
        c_24, c_9, cb_target2, compute_kernel_lib::BinaryInputBlockShape::row(Wt));
}
```
Where `cb_target2` = c_24 if has_beta (reuse same CB since input is consumed), else c_24 again. Wait -- we need a different CB if input and output are the same. Since c_24 is consumed tile-by-tile (WaitAndPopPerTile for A), and output is pushed tile-by-tile, this should work if the CB has enough space. However, reading and writing to the same CB in the same phase is unsafe. We need a second intermediate CB.

**Resolution**: Add c_25 (cb_affine_tmp) as an intermediate for the gamma*normalize result when both gamma and beta are present.

Revised CB plan for affine:
- If has_gamma and has_beta: Phase 7 outputs to c_24, Phase 7.5 (gamma) reads c_24 writes c_25, Phase 7.6 (beta) reads c_25 writes c_24 (then untilize from c_24)
- If has_gamma only: Phase 7 outputs to c_24, Phase 7.5 reads c_24 writes c_25, untilize from c_25
- If neither: Phase 7 outputs to c_24, untilize from c_24

Actually, simpler: always output normalize to c_24. If gamma: mul c_24->c_25. If beta: add c_25->c_24 (or c_24->c_25 if no gamma). Then untilize from whichever CB holds the final result. This is getting complex with conditional CB routing.

**Simplest approach**: Use c_24 as normalize output, c_25 as gamma output, and back to c_24 for beta output. The untilize input CB is determined at compile time based on has_gamma/has_beta:
- Neither: untilize from c_24
- Gamma only: untilize from c_25
- Both: untilize from c_24 (beta writes back to c_24)

Add c_25 to CB table with Wt pages.

#### Phase 7.6: Add Beta (if has_beta, ROW broadcast)
Similar to gamma but with add instead of mul.

#### Phase 8: Untilize
```cpp
compute_kernel_lib::untilize<Wt, cb_final, c_16>(1);
```
Where `cb_final` is the CB holding the final result (c_24, c_25, or c_24 depending on gamma/beta).

### Updated CB Table (adding c_25)

| CB ID | Name | Purpose | Pages | Lifetime |
|-------|------|---------|-------|----------|
| c_25 | cb_affine_tmp | Intermediate for gamma result | Wt | Per tile-row (conditional) |

### Writer Kernel

Standard untilize writer pattern from the untilize analysis. Per tile-row: waits for Wt tiles in c_16, gets L1 base via `get_read_ptr(c_16)`, loops over 32 sticks writing each to DRAM via TensorAccessor with `noc_async_write`, then barrier + pop. Page ID = global_stick_index (row-major interleaved: one stick per page).

### Critical Notes

1. **Epsilon CB persistence**: cb_eps (c_6) is pushed once by reader. Compute must `cb_wait_front(c_6, 1)` before the main loop and use NoWaitNoPop in Phase 6. Pop once after all rows processed.
2. **Reduce scaler persistence**: cb_scaler (c_8) is pushed once by reader. The reduce helper waits internally. Since reduce is called twice per tile-row (mean and variance), the scaler must persist. Use the default reduce policy which waits for scaler internally -- the reduce helper does `cb_wait_front(scaler, 1)` each call but the scaler is never popped because the reduce helper doesn't pop the scaler CB. Actually, examining the reduce helper: it waits for scaler but does NOT pop it. So the scaler persists naturally.
3. **Gamma/Beta CB persistence**: cb_gamma (c_9) and cb_beta (c_10) are filled once by reader. Compute uses NoWaitNoPop for the B operand in the gamma/beta binary ops. Wait once before main loop, pop once after.
4. **Gamma/beta tilize**: Gamma/beta have shape (1,1,1,W) in RM = 1 stick. To create valid tiles for ROW broadcast, the reader zeros the CB space, writes the single stick at offset 0 of each tile-row (which places data in face row 0), and pushes Wt pages. The binary_op ROW broadcast uses row 0 of each tile.
5. **NoWaitNoPop manual pops**: After Phase 3, manually pop c_1 (Wt). After Phase 7, manually pop c_3 (Wt).
6. **Reduce scaler value**: For layer_norm, we reduce with SUM and scaler = 1/W. Use `dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<c_8, SUM, REDUCE_ROW, 32, W>()` in the reader to auto-compute the correct scaler.

### Implementation Checklist
- [ ] Reader: TensorAccessor for input RM sticks, gamma/beta stick reads with zeroing, epsilon fill, reduce scaler generation
- [ ] Compute: 8 phases using helpers: tilize, reduce (x2), sub (COL bcast), square, add (SCALAR bcast, post_op=rsqrt), mul (COL bcast), mul (ROW bcast), add (ROW bcast), untilize
- [ ] Writer: TensorAccessor for output RM sticks, standard untilize writer pattern
- [ ] CB push/pop balance verified per tile-row iteration
