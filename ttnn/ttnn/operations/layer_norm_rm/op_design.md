# Operation Design: layer_norm_rm

## Overview
- **Operation Name**: layer_norm_rm
- **Category**: normalization
- **Planning Mode**: Hybrid
- **Reference Operations**: tilize (input_stage), batch_norm (compute_core), untilize (output_stage)

## Mathematical Definition
```
mean[b,h] = (1/W) * sum_{w=0}^{W-1} x[b,h,w]
var[b,h]  = (1/W) * sum_{w=0}^{W-1} (x[b,h,w] - mean[b,h])^2
output[b,h,w] = gamma[w] * (x[b,h,w] - mean[b,h]) / sqrt(var[b,h] + eps) + beta[w]
```
Per-row normalization: each row of 32 sticks (one tile-row) is normalized independently across W. Gamma/beta are optional per-feature (per-column) affine parameters.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input | Tensor | Yes | 2D+, last 2 dims tile-aligned | - | Input tensor |
| gamma | Tensor | No | shape (1,1,1,W), bfloat16, RM | None | Per-feature scale |
| beta | Tensor | No | shape (1,1,1,W), bfloat16, RM | None | Per-feature shift |
| epsilon | float | No | > 0 | 1e-5 | Numerical stability |

### Input Tensor Requirements
| Property | Requirement |
|----------|-------------|
| dtype | BFLOAT16 |
| layout | ROW_MAJOR |
| memory | INTERLEAVED (DRAM) |
| alignment | Last 2 dims divisible by 32 |

### Output Tensor Specification
- **Shape**: Same as input
- **Dtype**: BFLOAT16
- **Layout**: ROW_MAJOR
- **Memory**: INTERLEAVED (DRAM)

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| W = 32 (single tile width) | Wt=1, reduce produces 1 tile per row |
| gamma=None, beta=None | Skip affine transform phases |
| gamma only (no beta) | Apply scale only |
| beta only (no gamma) | Apply shift only |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | tilize | input_stage | Add gamma/beta/epsilon reads; keep RM stick batching |
| Compute | batch_norm + NEW | compute_core | Replace pre-computed mean/var with reduce-based computation; tilize input, compute norm, untilize output |
| Writer | untilize | output_stage | Extract RM sticks from untilized output; same pattern |

### Work Distribution
- **Work unit**: tile-row (32 sticks x full width = Wt tiles)
- **Grid**: 1D linearized, up to `grid_size.x * grid_size.y` cores
- **Work per core**: `nblocks_per_core` tile-rows via `split_blocks_for_tilize`
- **Remainder**: Cliff core gets `nblocks % nblocks_per_core` tile-rows

### Data Flow

Each core processes complete tile-rows independently. Per tile-row:
```
Reader: read 32 RM sticks -> cb_in_rm (tile-sized pages)
Compute: tilize(cb_in_rm -> cb_tilized)
         reduce_sum_row(cb_tilized -> cb_mean)           [mean, tiles persist in cb_tilized]
         sub_col_bcast(cb_tilized - cb_mean -> cb_centered)
         square(cb_centered -> cb_var_input)
         reduce_sum_row(cb_var_input -> cb_var)           [variance]
         add_scalar(cb_var + cb_eps -> cb_var_eps)
         rsqrt via post_op on add -> cb_inv_std
         mul_col_bcast(cb_centered * cb_inv_std -> cb_normed)
         [optional] mul(cb_normed * cb_gamma -> cb_scaled)
         [optional] add(cb_scaled + cb_beta -> cb_affine_out)
         untilize(cb_final -> cb_out_rm)
Writer: extract 32 RM sticks from cb_out_rm -> DRAM
```

### Circular Buffer Requirements

Notation: Wt = input_width / 32, tile_size = 2048 bytes (bfloat16).

| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| 0 | cb_in_rm | RM sticks (tile-sized pages) | Reader | Compute (tilize) | Wt | block |
| 1 | cb_tilized | Tilized input tiles | Compute | Compute | Wt | block (persists for sub) |
| 2 | cb_mean | Row-reduced mean (1 tile/row) | Compute | Compute | 1 | block |
| 3 | cb_centered | x - mean | Compute | Compute | Wt | block (persists for mul_inv_std) |
| 4 | cb_var_input | (x - mean)^2 | Compute | Compute (reduce) | 1 | streaming |
| 5 | cb_var | Variance (1 tile/row) | Compute | Compute | 1 | block |
| 6 | cb_gamma | Gamma tiles (Wt per row) | Reader | Compute | Wt | block |
| 7 | cb_beta | Beta tiles (Wt per row) | Reader | Compute | Wt | block |
| 8 | cb_eps | Epsilon scalar tile | Compute | Compute | 1 | block |
| 9 | cb_scaler | Reduce scaler (1/W) | Reader | Compute | 1 | program |
| 16 | cb_normed | Normalized output (or final) | Compute | Compute | 1 | streaming |
| 17 | cb_out_rm | Untilized output (tile-sized pages) | Compute | Writer | Wt | block |
| 24 | cb_inv_std | 1/sqrt(var+eps) (1 tile/row) | Compute | Compute | 1 | block |

### Kernel Arguments

**Compile-time** (reader):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | stick_size | uint32_t | W * element_size bytes |
| 1+ | TensorAccessorArgs(input) | uint32_t[] | Input accessor |

**Compile-time** (compute):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | Wt | uint32_t | Tiles per row (W/32) |
| 1 | nblocks_per_core | uint32_t | Tile-rows this core processes |
| 2 | has_gamma | uint32_t | 1 if gamma present, 0 otherwise |
| 3 | has_beta | uint32_t | 1 if beta present, 0 otherwise |

**Compile-time** (writer):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | output_stick_size | uint32_t | W * element_size bytes |
| 1 | Wt | uint32_t | Tiles per row |
| 2+ | TensorAccessorArgs(output) | uint32_t[] | Output accessor |

**Runtime** (reader):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | src_addr | uint32_t | Input buffer address |
| 1 | start_stick_id | uint32_t | First stick for this core |
| 2 | num_sticks | uint32_t | nblocks * 32 |
| 3 | gamma_addr | uint32_t | Gamma buffer address (0 if absent) |
| 4 | beta_addr | uint32_t | Beta buffer address (0 if absent) |

**Runtime** (writer):
| Index | Name | Type | Description |
|-------|------|------|-------------|
| 0 | dst_addr | uint32_t | Output buffer address |
| 1 | start_stick_id | uint32_t | First output stick for this core |
| 2 | num_blocks | uint32_t | Tile-rows to write |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count (verified per-CB)
- [x] Reduce scaler CB (cb_scaler) is bfloat16
- [x] DEST register holds max 8 tiles (bf16 half-sync) -- reduce/binary helpers manage internally
- [x] RM CBs count pages in tiles (tile-sized pages for tilize compatibility)
- [x] cb_in_rm page_size = tile_size (NOT stick_size) per MEMORY.md requirement

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

| CB | Pages | Layout | Lifetime | Notes |
|----|-------|--------|----------|-------|
| 0 (cb_in_rm) | Wt | tile-sized (RM data) | block | tilize input; page_size MUST be tile_size |
| 1 (cb_tilized) | Wt | TILE | block | Persists: WaitUpfrontNoPop for sub, then popped |
| 2 (cb_mean) | 1 | TILE | block | REDUCE_ROW output: Col0 valid |
| 3 (cb_centered) | Wt | TILE | block | Persists for square+mul_inv_std |
| 4 (cb_var_input) | 1 | TILE | streaming | square output, consumed per-tile by reduce |
| 5 (cb_var) | 1 | TILE | block | add_eps output via post_op |
| 6 (cb_gamma) | Wt | tile-sized (RM data) | block | Reader reads RM gamma sticks |
| 7 (cb_beta) | Wt | tile-sized (RM data) | block | Reader reads RM beta sticks |
| 8 (cb_eps) | 1 | TILE | block | Filled by compute via fill_with_val |
| 9 (cb_scaler) | 1 | TILE | program | 1/W scaler, filled once by reader |
| 16 (cb_normed) | 1 | TILE | streaming | Normalized/affine output per tile |
| 17 (cb_out_rm) | Wt | tile-sized (RM data) | block | Untilize output |
| 24 (cb_inv_std) | 1 | TILE | block | rsqrt(var+eps), persists for mul across Wt tiles |

### Binary Op Broadcast Verification

| Phase | Op | CB_A Valid Region | CB_B Valid Region | Broadcast |
|-------|-----|-------------------|-------------------|-----------|
| sub (x - mean) | SUB | All | Col0 (REDUCE_ROW output) | COL |
| square (centered^2) | SQUARE | All | N/A (self) | N/A |
| add (var + eps) | ADD | Col0 (REDUCE_ROW output) | All (scalar tile) | SCALAR |
| mul (centered * inv_std) | MUL | All | Col0 (from add_scalar output) | COL |
| mul (normed * gamma) | MUL | All | All (gamma is Wt wide) | NONE |
| add (scaled + beta) | ADD | All | All (beta is Wt wide) | NONE |

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output |
|-------|------|-------------|-----------------|
| 1 | data_pipeline | Reader + tilize + untilize + writer (identity) | Input passthrough |
| 2 | reduce_mean | + reduce SUM_ROW, sub COL bcast | x - mean(x) |
| 3 | variance_normalize | + square, reduce variance, add_eps+rsqrt, mul | Full layer_norm (no affine) |
| 4 | affine_transform | + gamma mul, beta add | Full layer_norm with affine |

### Stage 1: data_pipeline
- **Scope**: reader kernel, compute (tilize + untilize only), writer kernel
- **Reference**: `torch.nn.functional.layer_norm(x, [x.shape[-1]], weight=None, bias=None, eps=1e-5)` is NOT the reference here. Identity passthrough: output = input.
- **Shapes**: `(1, 1, 32, 32)`, `(1, 1, 64, 128)`, `(1, 1, 32, 256)`, `(4, 2, 64, 64)`
- **Tolerances**: rtol=0.01, atol=0.01
- **CB bypass**: Compute does tilize(cb_in_rm -> cb_tilized), then immediately untilize(cb_tilized -> cb_out_rm). All normalization phases skipped.

### Stage 2: reduce_mean
- **Scope**: compute kernel adds reduce SUM_ROW for mean, sub COL broadcast
- **Reference**: `x - x.mean(dim=-1, keepdim=True)`
- **Delta from previous**: Add phases: reduce(SUM, REDUCE_ROW) with 1/W scaler, sub<COL>(tilized, mean, centered). Untilize now reads from cb_centered instead of cb_tilized.
- **Shapes**: `(1, 1, 32, 32)`, `(1, 1, 64, 128)`, `(1, 1, 32, 256)`, `(4, 2, 64, 64)`
- **Tolerances**: rtol=0.02, atol=0.1

### Stage 3: variance_normalize
- **Scope**: compute kernel adds square, reduce variance, add_eps+rsqrt, mul_inv_std
- **Reference**: `torch.nn.functional.layer_norm(x, [x.shape[-1]], weight=None, bias=None, eps=1e-5)`
- **Delta from previous**: Add phases: square(centered), reduce(SUM, REDUCE_ROW) for variance, add<SCALAR>(var, eps) with rsqrt post_op, mul<COL>(centered, inv_std). Untilize reads from cb_normed.
- **Shapes**: `(1, 1, 32, 32)`, `(1, 1, 64, 128)`, `(1, 1, 32, 256)`, `(4, 2, 64, 64)`
- **Tolerances**: rtol=0.05, atol=0.2

### Stage 4: affine_transform
- **Scope**: reader adds gamma/beta tile reads; compute adds mul(NONE) and add(NONE)
- **Reference**: `torch.nn.functional.layer_norm(x, [x.shape[-1]], weight=gamma_1d, bias=beta_1d, eps=1e-5)`
- **Delta from previous**: Reader reads gamma/beta RM sticks into cb_gamma/cb_beta (tilize them). Compute adds mul(normed, gamma) and add(scaled, beta).
- **Shapes**: `(1, 1, 32, 32)`, `(1, 1, 64, 128)`, `(1, 1, 32, 256)`, `(4, 2, 64, 64)`
- **Tolerances**: rtol=0.05, atol=0.2

### Reader Kernel

Reads RM input sticks in groups of 32, packing into tile-sized CB pages (same pattern as tilize reference reader). Additionally:
- Fills cb_scaler once at startup via `dataflow_kernel_lib::prepare_reduce_scaler<cb_scaler>(1.0f / W)` where W is the tensor width (number of elements, not tiles).
- For gamma/beta (Stage 4): reads Wt tiles of gamma/beta RM sticks into cb_gamma/cb_beta per tile-row using same batch-address pattern. Gamma/beta are (1,1,1,W) so the same Wt tiles are read for every tile-row.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_in_rm, cb_scaler, cb_out_rm)` (3-arg form: srcA=cb_in_rm, srcB=cb_scaler, ocb=cb_out_rm).

Note: `binary_op_init_common(cb_in_rm, cb_scaler, cb_out_rm)` is called by the first helper automatically via `init=true`.

The compute kernel loops over `nblocks_per_core` tile-rows. Per tile-row:

#### Phase 1: Tilize
```cpp
compute_kernel_lib::tilize<cb_in_rm, cb_tilized,
    InitUninitMode::InitAndUninit, WaitMode::WaitBlock>(Wt, 1);
```
- In: cb_in_rm [Wt tiles, FRESHLY PUSHED by reader]
- Out: cb_tilized [Wt tiles, pushed]

**CB state after Phase 1:**
| CB | Tiles | State |
|----|-------|-------|
| cb_in_rm | 0 | freed (popped by tilize helper) |
| cb_tilized | Wt | pushed, available for reduce and sub |

#### Phase 2: Reduce mean (SUM row with 1/W scaler)
```cpp
cb_wait_front(cb_tilized, Wt);  // explicit wait before NoWaitNoPop
compute_kernel_lib::reduce<SUM, REDUCE_ROW,
    ReduceInputPolicy::NoWaitNoPop>(
    cb_tilized, cb_scaler, cb_mean,
    ReduceInputBlockShape::row(Wt));
```
- A: cb_tilized [Wt tiles, ALREADY WAITED, not popped -- persists for Phase 3]
- B: cb_scaler [1 tile, program-lifetime, not popped]
- Out: cb_mean [1 tile, pushed]

NoWaitNoPop used because cb_tilized must persist for the subtraction in Phase 3.

#### Phase 3: Subtract mean (x - mean, COL broadcast)
```cpp
compute_kernel_lib::sub<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,
    BinaryInputPolicy::WaitAndPopPerTile>(
    cb_tilized, cb_mean, cb_centered,
    BinaryInputBlockShape::of(1, Wt));
cb_pop_front(cb_tilized, Wt);  // manual pop after NoWaitNoPop
```
- A: cb_tilized [Wt tiles, ALREADY WAITED from Phase 2, NoWaitNoPop -- manual pop after]
- B: cb_mean [1 tile, WaitAndPopPerTile -- waited and popped by helper]
- Out: cb_centered [Wt tiles, pushed]

**CB state after Phase 3:**
| CB | Tiles | State |
|----|-------|-------|
| cb_tilized | 0 | freed (manually popped) |
| cb_mean | 0 | freed (popped by helper) |
| cb_centered | Wt | pushed, persists for Phase 6 |

#### Phase 4: Square centered values
```cpp
cb_wait_front(cb_centered, Wt);  // explicit wait before NoWaitNoPop
compute_kernel_lib::square<BinaryInputPolicy::NoWaitNoPop>(
    cb_centered, cb_var_input,
    BinaryInputBlockShape::of(1, Wt));
```
- In: cb_centered [Wt tiles, ALREADY WAITED, not popped -- persists for Phase 6]
- Out: cb_var_input [Wt tiles, pushed streaming 1 at a time]

#### Phase 5: Reduce variance (SUM row with 1/W scaler) + add eps + rsqrt
```cpp
compute_kernel_lib::reduce<SUM, REDUCE_ROW,
    ReduceInputPolicy::WaitAndPopPerTile>(
    cb_var_input, cb_scaler, cb_var,
    ReduceInputBlockShape::row(Wt));

compute_kernel_lib::add<BroadcastDim::SCALAR,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitAndPopPerTile>(
    cb_var, cb_eps, cb_inv_std,
    BinaryInputBlockShape::single(),
    [](uint32_t dst_idx) {
        rsqrt_tile_init();
        rsqrt_tile(dst_idx);
    });
```
- Reduce: cb_var_input consumed per-tile, cb_var [1 tile] pushed
- Add+rsqrt: cb_var [1 tile] + cb_eps [1 tile] -> cb_inv_std [1 tile], rsqrt applied in-place via post_op

**CB state after Phase 5:**
| CB | Tiles | State |
|----|-------|-------|
| cb_var_input | 0 | freed |
| cb_var | 0 | freed |
| cb_inv_std | 1 | pushed, persists for Phase 6 |
| cb_centered | Wt | still waited from Phase 4 |

#### Phase 6: Multiply by inv_std (COL broadcast)
```cpp
compute_kernel_lib::mul<BroadcastDim::COL,
    BinaryInputPolicy::NoWaitNoPop,
    BinaryInputPolicy::WaitAndPopPerTile>(
    cb_centered, cb_inv_std, cb_normed,
    BinaryInputBlockShape::of(1, Wt));
cb_pop_front(cb_centered, Wt);  // manual pop after NoWaitNoPop
```
- A: cb_centered [Wt tiles, ALREADY WAITED from Phase 4, manual pop after]
- B: cb_inv_std [1 tile, popped by helper per-tile cycling]
- Out: cb_normed [Wt tiles, pushed streaming]

If no gamma and no beta, cb_normed is the final CB before untilize.

#### Phase 7: Optional affine transform (gamma * x_norm + beta)

When gamma is present:
```cpp
// cb_gamma contains tilized gamma tiles (Wt tiles, pushed by reader)
compute_kernel_lib::mul<BroadcastDim::NONE,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitAndPopPerTile>(
    cb_normed, cb_gamma, cb_affine_out,
    BinaryInputBlockShape::of(1, Wt));
```

When beta is present (cb_affine_out or cb_normed as input depending on gamma):
```cpp
compute_kernel_lib::add<BroadcastDim::NONE,
    BinaryInputPolicy::WaitAndPopPerTile,
    BinaryInputPolicy::WaitAndPopPerTile>(
    cb_prev, cb_beta, cb_final,
    BinaryInputBlockShape::of(1, Wt));
```

CB routing for conditional affine (same pattern as batch_norm):
- No gamma, no beta: untilize from cb_normed
- Gamma only: mul(cb_normed, cb_gamma -> cb_normed reuse or cb_16), untilize from output
- Gamma + beta: mul -> intermediate, add -> final, untilize from final
- Beta only: add(cb_normed, cb_beta -> final), untilize from final

The kernel writer should use compile-time `has_gamma`/`has_beta` flags to select the CB routing, writing the final result to a CB that the untilize phase reads from.

#### Phase 8: Untilize
```cpp
compute_kernel_lib::untilize<Wt, cb_final, cb_out_rm,
    InitUninitMode::InitAndUninit, WaitMode::NoWait>(1);
```
- In: cb_final [Wt tiles, already available from prior phase]
- Out: cb_out_rm [Wt tiles, pushed as tile-sized pages containing RM data]

NoWait used because data is already in the CB from the previous compute phase.

### Writer Kernel

Same pattern as untilize reference writer. Per tile-row block:
1. `cb_wait_front(cb_out_rm, Wt)` -- wait for full block
2. Extract 32 RM sticks: row `j` at offset `base_l1_read_addr + j * W * element_size`
3. Write each stick to DRAM via `TensorAccessor::get_noc_addr(stick_id)`
4. `noc_async_write_barrier()` after all 32 writes
5. `cb_pop_front(cb_out_rm, Wt)` -- release block

### Critical Notes

1. **cb_in_rm page_size = tile_size**: The tilize helper reads face/tile dimensions from CB metadata. Stick-sized pages cause only 16/32 rows to be processed (MEMORY.md).

2. **cb_tilized persists across phases 2-3**: Waited in Phase 2 (for reduce NoWaitNoPop), consumed in Phase 3 (sub NoWaitNoPop), manually popped after Phase 3. Both reduce and sub use NoWaitNoPop on this CB.

3. **cb_centered persists across phases 4-6**: Same pattern -- waited in Phase 4, consumed through Phase 6, manually popped after Phase 6.

4. **cb_eps filled per tile-row**: Unlike batch_norm's program-lifetime epsilon, here epsilon is produced and consumed within each tile-row iteration. The compute kernel fills it via `fill_with_val` at the start of each iteration (or once if moved to reader startup). The simplest approach: reader fills cb_eps once at startup (program lifetime), compute waits/pops per iteration -- but this requires re-pushing. Better: compute fills cb_eps locally each iteration using L1 write. Alternative: reader fills once, compute does WaitUpfrontNoPop. The kernel writer should use the simplest correct approach.

5. **Gamma/beta are RM tensors**: Shape (1,1,1,W). Reader tilizes them into cb_gamma/cb_beta the same way it tilizes the input. The same 32-stick batching into tile-sized pages, followed by a tilize in compute. Alternative: since gamma/beta are only 1 row repeated, the reader can read Wt tiles worth of RM data (just 1 stick repeated 32 times or read as tiles if pre-tilized on host). The kernel writer should decide the simplest approach.

6. **reduce scaler format**: The scaler CB must be bfloat16 format. `prepare_reduce_scaler<cb_scaler>(1.0f / W)` handles format-correct filling. W here is the number of elements (not tiles).

### Implementation Checklist
- [ ] Reader: RM stick batching (tilize pattern), epsilon fill, scaler fill, optional gamma/beta reads
- [ ] Compute: 8 phases using helpers: tilize, reduce(SUM, REDUCE_ROW), sub(COL), square, reduce(SUM, REDUCE_ROW), add(SCALAR)+rsqrt, mul(COL), optional mul(NONE)+add(NONE), untilize
- [ ] Writer: RM stick extraction from tile-sized CB pages (untilize pattern)
- [ ] CB push/pop balance verified per tile-row iteration
