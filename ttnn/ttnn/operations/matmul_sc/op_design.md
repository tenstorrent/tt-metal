# Operation Design: matmul_sc

## Overview
- **Operation Name**: matmul_sc
- **Category**: matmul
- **Planning Mode**: Derivative
- **Reference Operation(s)**: `ttnn/ttnn/operations/matmul_sc/matmul_multicore_analysis.md` (matmul multicore)

## Mathematical Definition
```
C[m, n] = sum_k(A[m, k] * B[k, n])   for m in [0, M), n in [0, N)
```
Single-core tiled matrix multiplication: C = A x B. All dimensions are multiples of 32.

---

## Part 1: Architecture

### Parameters
| Parameter | Type | Required | Valid Range | Default | Description |
|-----------|------|----------|-------------|---------|-------------|
| input_a | Tensor | Yes | rank-2, bf16, tiled, interleaved | - | Matrix A [M, K] |
| input_b | Tensor | Yes | rank-2, bf16, tiled, interleaved | - | Matrix B [K, N] |
| memory_config | MemoryConfig | No | DRAM interleaved | DRAM_MEMORY_CONFIG | Output memory config |

### Input Tensor Requirements
| Property | Requirement | Error Hint |
|----------|-------------|------------|
| Rank | 2 | "matmul_sc: inputs must be rank-2" |
| Dtype | bfloat16 | "matmul_sc: inputs must be bfloat16" |
| Layout | TILE_LAYOUT | "matmul_sc: inputs must be tiled" |
| Memory | Interleaved DRAM | "matmul_sc: inputs must be interleaved" |
| A.shape[-1] == B.shape[-2] | Inner dim match | "matmul_sc: inner dimensions must match" |
| All dims % 32 == 0 | Tile-aligned | "matmul_sc: dimensions must be multiples of 32" |

### Output Tensor Specification
- **Shape**: [M, N] where M = A.shape[0], N = B.shape[1]
- **Dtype**: bfloat16
- **Layout**: TILE_LAYOUT
- **Memory**: Interleaved DRAM

### Edge Cases
| Condition | Expected Behavior |
|-----------|-------------------|
| M=32, K=32, N=32 | Single tile matmul (1 output tile) |
| K very large | Accumulates many tiles in DEST; precision limited by bf16 accumulation |

### Component Sources
| Component | Source Reference | Role | Modifications |
|-----------|-----------------|------|---------------|
| Reader | matmul multicore | input_stage | Simplified: use `read_matmul_tiles` helper, single-core (batch=1) |
| Compute | matmul multicore | compute_core | Use `matmul_1d` helper instead of raw loops |
| Writer | matmul multicore | output_stage | Simplified: use `write_matmul_tiles` helper |

### Work Distribution
- **Work unit**: tile (32x32)
- **Grid**: single core (0, 0)
- **Work per core**: Mt * Nt output tiles (all work on one core)
- **Remainder**: N/A (single core)

### Data Flow
Reader pushes A and B tiles into in0_cb and in1_cb in batch x Mt x Nt x Kt order (one A tile then one B tile per Kt step). Compute consumes tiles in the same order via WaitPerTile, accumulating Kt tiles per output tile in DEST, then packs to out_cb. Writer pops output tiles from out_cb and writes to DRAM.

### Circular Buffer Requirements
| CB ID | Name | Purpose | Producer | Consumer | Pages | Lifetime |
|-------|------|---------|----------|----------|-------|----------|
| 0 | cb_in0 | A tile staging | Reader | Compute | 2 | Per Kt iteration |
| 1 | cb_in1 | B tile staging | Reader | Compute | 2 | Per Kt iteration |
| 16 | cb_out | C tile staging | Compute | Writer | 2 | Per output tile |

All CBs use tile-sized pages (bf16 tile size from `ttnn.tile_size(ttnn.bfloat16)`). 2 pages enables double-buffering for reader-compute and compute-writer overlap. WaitPerTile mode only requires 1 page minimum, but 2 pages improves throughput.

### Kernel Arguments

**Compile-time** (named, per kernel):
| Kernel | Name | Value | Description |
|--------|------|-------|-------------|
| Reader | cb_in0 | 0 | Input A CB index |
| Reader | cb_in1 | 1 | Input B CB index |
| Writer | cb_out | 16 | Output C CB index |
| Compute | cb_in0 | 0 | Input A CB index |
| Compute | cb_in1 | 1 | Input B CB index |
| Compute | cb_out | 16 | Output C CB index |

**Compile-time** (positional, per kernel):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0..N | TensorAccessorArgs(A) | uint32_t[] | Accessor args for A |
| Reader | N..M | TensorAccessorArgs(B) | uint32_t[] | Accessor args for B (chained after A) |
| Writer | 0..N | TensorAccessorArgs(C) | uint32_t[] | Accessor args for C |

**Runtime** (per kernel):
| Kernel | Index | Name | Type | Description |
|--------|-------|------|------|-------------|
| Reader | 0 | in0_addr | uint32_t | DRAM base address of A |
| Reader | 1 | in1_addr | uint32_t | DRAM base address of B |
| Reader | 2 | Mt | uint32_t | Tile rows of A/C |
| Reader | 3 | Kt | uint32_t | Inner dimension in tiles |
| Reader | 4 | Nt | uint32_t | Tile columns of B/C |
| Reader | 5 | batch | uint32_t | Always 1 for rank-2 |
| Writer | 0 | out_addr | uint32_t | DRAM base address of C |
| Writer | 1 | Mt | uint32_t | Tile rows of C |
| Writer | 2 | Nt | uint32_t | Tile columns of C |
| Writer | 3 | batch | uint32_t | Always 1 for rank-2 |
| Compute | 0 | Mt | uint32_t | Tile rows of C |
| Compute | 1 | Kt | uint32_t | Inner dimension in tiles |
| Compute | 2 | Nt | uint32_t | Tile columns of C |
| Compute | 3 | batch | uint32_t | Always 1 for rank-2 |

### Hardware Constraints Checklist
- [x] All `cb_wait_front` calls on same CB use same page count (1 for WaitPerTile)
- [x] DEST register holds 1 tile at a time (accumulate over Kt, pack, release)
- [x] Tile-sized CB pages with bf16 data format
- [x] dst_full_sync_en = true (full DST sync mode)
- [x] MathFidelity = HiFi4

### Test Criteria
- Output shape [M, N] matches A.shape[0] x B.shape[1]
- Numerical accuracy vs `torch.matmul(A, B)` (rtol=0.05, atol=0.2 for multi-step accumulation)
- Test shapes:

| Category | Purpose | A Shape | B Shape |
|----------|---------|---------|---------|
| Minimal | Single tile | (32, 32) | (32, 32) |
| Multi-tile | Tile iteration | (64, 64) | (64, 128) |
| Non-square | W!=H | (32, 128) | (128, 64) |
| Large-K | Accumulation stress | (32, 256) | (256, 32) |
| Multi-batch-dim | Outer dim handling | (128, 64) | (64, 64) |

---

## Part 2: Kernel Implementation

### CB Allocation (final, validated against helpers)

| CB | Pages | Layout | Lifetime |
|----|-------|--------|----------|
| 0 (cb_in0) | 2 | Tile bf16 | Per Kt step (WaitPerTile) |
| 1 (cb_in1) | 2 | Tile bf16 | Per Kt step (WaitPerTile) |
| 16 (cb_out) | 2 | Tile bf16 | Per output tile |

Helper requirements: WaitPerTile mode needs >= 1 page per input CB, >= 1 page for output CB. 2 pages satisfies this and enables double-buffering.

### TDD Stage Plan

| Stage | Name | What's Added | Expected Output |
|-------|------|-------------|-----------------|
| 1 | data_pipeline | Reader reads A tiles to cb_in0, compute copies cb_in0 to cb_out (identity), writer writes cb_out to DRAM | First Mt*Nt tiles of A (reshaped to [M, N] by taking first N columns per row) |
| 2 | matmul_compute | Full matmul_1d compute with all three helpers | `torch.matmul(A, B)` |

### Stage 1: data_pipeline
- **Scope**: Reader kernel (simplified: read only A tiles into cb_in0), compute kernel (tile copy: wait cb_in0, pack to cb_out), writer kernel (full write_matmul_tiles helper)
- **Reference**: Reader pushes Mt*Nt tiles from A (first Nt tiles per row, ignoring K dimension). Compute does identity copy (cb_wait_front(cb_in0,1), pack_tile(0, cb_out), cb_pop_front). Writer uses write_matmul_tiles helper.
- **Shapes**: (32,32)x(32,32), (64,128)x(128,64), (32,128)x(128,64), (128,64)x(64,64)
- **Tolerances**: rtol=0.01, atol=0.01 (passthrough)
- **CB bypass**: in1_cb unused. Compute reads from cb_in0 only. Reader writes only A tiles to cb_in0 (Mt*Nt tiles, one per output position). No matmul accumulation.
- **Note**: The reference for this stage reads the first Nt tile-columns of each of Mt tile-rows of A. The expected output is `A_tiled[:Mt*32, :Nt*32]` where Mt and Nt are derived from the output shape. Since A has shape [M, K] and output has shape [M, N], and K >= N is not guaranteed, we use a simpler approach: the reader sends Mt*Nt tiles sequentially from A (tile indices 0 through Mt*Nt-1), and the reference is the first M*N elements of A reshaped. However, this only works if Mt*Nt <= Mt*Kt. For safety, the reference expression tiles A and takes the appropriate slice.

### Stage 2: matmul_compute
- **Scope**: All three kernel files use helpers: read_matmul_tiles, matmul_1d, write_matmul_tiles
- **Reference**: `torch.matmul(A, B)`
- **Delta from previous**: Replace identity reader+compute with full helper-based implementation. Reader now reads both A and B using read_matmul_tiles. Compute uses matmul_1d. All three helpers engaged.
- **Shapes**: (32,32)x(32,32), (64,64)x(64,128), (32,128)x(128,64), (128,64)x(64,64), (32,256)x(256,32)
- **Tolerances**: rtol=0.05, atol=0.2

### Reader Kernel
Uses `dataflow_kernel_lib::read_matmul_tiles<cb_in0, cb_in1>(in0_addr, in1_addr, Mt, Nt, Kt, batch)`. TensorAccessor compile-time args are chained: `TensorAccessorArgs<0>()` for A, `TensorAccessorArgs<s0_args.next_compile_time_args_offset()>()` for B.

### Compute Kernel

**Startup**: `compute_kernel_hw_startup(cb_in0, cb_in1, cb_out)` (three-arg form required since srcA != srcB)

#### Phase 1: Matrix multiplication
```cpp
compute_kernel_lib::matmul_1d<cb_in0, cb_in1, cb_out>(Mt, Nt, Kt, batch);
```
Default template params: InitAndUninit, WaitPerTile, UnpackAndPackReconfigure. Single phase -- the helper handles all CB synchronization, DST acquire/release, mm_init, matmul_tiles, and pack_tile internally.

### Writer Kernel
Uses `dataflow_kernel_lib::write_matmul_tiles<cb_out>(out_addr, Mt, Nt, batch)`. Single TensorAccessor at compile-time offset 0.

### Critical Notes
- `compute_kernel_hw_startup` MUST use the 3-arg form `(cb_in0, cb_in1, cb_out)` because srcA and srcB are different CBs.
- ComputeConfigDescriptor must set `math_fidelity=HiFi4` and `dst_full_sync_en=true`.
- The matmul_1d helper uses `tile_regs_acquire/commit/wait/release` (not `acquire_dst/release_dst`). This is handled internally by the helper.
- Reader runtime args order: in0_addr, in1_addr, Mt, Kt, Nt, batch (Kt before Nt).
- Writer runtime args order: out_addr, Mt, Nt, batch.
- Compute runtime args order: Mt, Kt, Nt, batch.

### Implementation Checklist
- [ ] Reader: `read_matmul_tiles` helper with chained TensorAccessorArgs
- [ ] Compute: 1 phase using `matmul_1d` helper (after `compute_kernel_hw_startup`)
- [ ] Writer: `write_matmul_tiles` helper with TensorAccessorArgs
- [ ] CB push/pop balance: all managed by helpers
- [ ] Program descriptor: 3 CBs (0, 1, 16) with 2 tile-sized pages each
- [ ] Named compile-time args: cb_in0=0, cb_in1=1, cb_out=16
- [ ] ComputeConfigDescriptor: math_fidelity=HiFi4, dst_full_sync_en=true
