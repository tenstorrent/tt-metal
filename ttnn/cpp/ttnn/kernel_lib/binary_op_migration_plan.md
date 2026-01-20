# Binary Op Helpers Migration Plan

This document outlines a comprehensive plan to migrate compute kernels from direct `add_tiles`, `sub_tiles`, `mul_tiles` (and their broadcast variants) to the unified `binary_op_helpers.hpp` API.

## Table of Contents
1. [Executive Summary](#executive-summary)
2. [Pattern Categories](#pattern-categories)
3. [Prioritized Migration List](#prioritized-migration-list)
4. [Detailed Kernel Analysis](#detailed-kernel-analysis)
5. [Testing Strategy](#testing-strategy)
6. [Migration Cookbook](#migration-cookbook)
7. [Blockers and Missing Features](#blockers-and-missing-features)

---

## Executive Summary

### Current State
- **72 compute kernels** in `ttnn/cpp/ttnn/operations/` use binary operations
- **4 kernels** already migrated: `moreh_dot`, `rotary_embedding`, `rotary_embedding_llama`, `ssm_prefix_scan`
- **~68 kernels** are candidates for migration

### Benefits of Migration
- **Code reduction**: Average 50-70% fewer lines per kernel
- **Consistency**: Unified CB management patterns
- **Maintainability**: Single source of truth for binary op implementation
- **Policy flexibility**: Easy switching between streaming/preloaded/persistent modes

### Effort Estimate
| Difficulty | Count | Time/Kernel | Total |
|------------|-------|-------------|-------|
| Easy       | 8     | 30 min      | 4 hrs |
| Medium     | 25    | 1-2 hrs     | 40 hrs |
| Hard       | 35    | 2-4 hrs     | 100 hrs |

---

## Pattern Categories

### Category A: Simple Element-wise (NONE broadcast)
**Pattern signature:**
```cpp
mul_tiles_init(cb_a, cb_b);
for (uint32_t i = 0; i < N; i++) {
    mul_tiles(cb_a, cb_b, i, i, i);  // Same index for both inputs
}
```

**Replacement:**
```cpp
compute_kernel_lib::mul(cb_a, cb_b, cb_out, BinaryTileShape::row(N));
```

**Typical use cases:** Variance calculation `(x-E[x])²`, element-wise multiply/add

---

### Category B: ROW Broadcast (γ*x + β pattern)
**Pattern signature:**
```cpp
mul_bcast_rows_init_short(cb_fusion, cb_gamma);
for (auto i : block.local()) {
    mul_tiles_bcast_rows(cb_fusion, cb_gamma, i, block.to_global(i), i);
}
```

**Replacement:**
```cpp
compute_kernel_lib::mul<BroadcastDim::ROW, cb_policies::Streaming, cb_policies::Persistent>(
    cb_fusion, cb_gamma, cb_out, BinaryTileShape::col(block_h));
```

**Typical use cases:** Layernorm/groupnorm affine transforms, bias addition

---

### Category C: COL Broadcast (x - E[x] pattern)
**Pattern signature:**
```cpp
sub_bcast_cols_init_short(cb_x, cb_ex);
for (auto i : block.local()) {
    sub_tiles_bcast_cols(cb_x, cb_ex, i, 0, i);  // Second index always 0
}
```

**Replacement:**
```cpp
compute_kernel_lib::sub<BroadcastDim::COL>(
    cb_x, cb_ex, cb_out, BinaryTileShape::row(Wt));
```

**Typical use cases:** Centering (subtract mean), scaling by row-wise values

---

### Category D: SCALAR Broadcast
**Pattern signature:**
```cpp
mul_tiles_bcast_scalar_init_short(cb_x, cb_scalar);
cb_wait_front(cb_scalar, 1);
for (uint32_t i = 0; i < N; i++) {
    mul_tiles_bcast_scalar(cb_x, cb_scalar, i, 0, i);
}
```

**Replacement:**
```cpp
compute_kernel_lib::mul<BroadcastDim::SCALAR>(
    cb_x, cb_scalar, cb_out, BinaryTileShape::row(N));
```

**Typical use cases:** Scale by epsilon, temperature scaling

---

### Category E: Inplace Operations (BLOCKER)
**Pattern signature:**
```cpp
mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);
// OR
cb_pop_front(in0_cb, 1);
cb_reserve_back(in0_cb, 1);  // Same CB for input and output
pack_tile(0, in0_cb);
cb_push_back(in0_cb, 1);
```

**Status:** NOT SUPPORTED - Requires new `_inplace` API variants

---

### Category F: Fused with Other Operations
**Pattern signature:**
```cpp
// Binary op immediately followed by unary (e.g., exp, rsqrt)
sub_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);
exp_tile<true>(j);  // Fused in same acquire/release block
```

**Status:** PARTIAL SUPPORT - Can migrate binary part, fused op stays separate

---

## Prioritized Migration List

### Tier 1: Already Migrated (Validated)
| Kernel | Test File | Status |
|--------|-----------|--------|
| `moreh/moreh_dot/device/kernels/moreh_dot.cpp` | `tests/ttnn/nightly/unit_tests/operations/moreh/test_moreh_dot.py` | ✅ Done |
| `experimental/transformer/rotary_embedding/device/kernels/compute/rotary_embedding.cpp` | `tests/ttnn/integration_tests/falcon7b/test_falcon_rotary_embeddings.py` | ✅ Done |
| `experimental/transformer/rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama.cpp` | Same as above | ✅ Done |
| `experimental/ssm/prefix_scan/device/kernels/ssm_prefix_scan.cpp` | `tests/ttnn/nightly/unit_tests/operations/ssm/test_ssm_prefix_scan.py` | ✅ Done |

### Tier 2: Easy Targets (1-2 ops, simple CB pattern)
| # | Kernel | Broadcast | Ops | Test File | Est. Time |
|---|--------|-----------|-----|-----------|-----------|
| 1 | `reduction/sampling/device/kernels/compute/sampling.cpp` | SCALAR, COL | mul, sub | `tests/ttnn/unit_tests/operations/eltwise/test_sampling.py` | 30 min |
| 2 | `normalization/batch_norm/device/kernels/compute/batch_norm_kernel.cpp` | NONE | sub | `tests/ttnn/unit_tests/operations/fused/test_batch_norm.py` | 30 min |
| 3 | `eltwise/ternary/addcmul/device/kernels/compute/*.cpp` | NONE | mul, add | Unit tests in eltwise | 30 min |

### Tier 3: Medium - Normalization Kernels
| # | Kernel | Broadcast | Ops | Test File | Est. Time |
|---|--------|-----------|-----|-----------|-----------|
| 4 | `normalization/layernorm/device/kernels/compute/layernorm.cpp` | NONE, ROW, COL | add, mul, sub | `tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm.py` | 2 hrs |
| 5 | `normalization/layernorm/device/kernels/compute/layernorm_welford.cpp` | NONE, ROW, COL | add, mul, sub | Same | 2 hrs |
| 6 | `normalization/layernorm/device/kernels/compute/layernorm_sharded.cpp` | NONE, ROW, COL | add, mul, sub | Same | 2 hrs |
| 7 | `normalization/layernorm/device/kernels/compute/layernorm_large_tensor.cpp` | NONE, ROW, COL | add, mul, sub | Same | 2 hrs |
| 8 | `reduction/moe/device/kernels/compute/moe.cpp` | ROW, COL | add, mul, sub | `tests/ttnn/unit_tests/operations/reduce/test_moe.py` | 2 hrs |

### Tier 4: Medium - Layernorm Variants
| # | Kernel | Notes | Test File |
|---|--------|-------|-----------|
| 9 | `layernorm_sharded_welford.cpp` | Similar to layernorm_welford | Same tests |
| 10 | `layernorm_large_tensor_welford.cpp` | Similar to layernorm_welford | Same tests |
| 11 | `layernorm_sharded_pre_allgather.cpp` | Distributed variant | Same tests |
| 12 | `layernorm_sharded_post_allgather.cpp` | Distributed variant | Same tests |

### Tier 5: Medium - RMSNorm and Distributed Variants
| # | Kernel | Notes | Test File |
|---|--------|-------|-----------|
| 13 | `rmsnorm_distributed/device/kernels/compute/rmsnorm_post_allgather.cpp` | ROW, COL | TBD |
| 14 | `layernorm_distributed/device/kernels/compute/layernorm_post_allgather.cpp` | Function pointers | TBD |
| 15 | `layernorm_distributed/device/kernels/compute/layernorm_post_allgather_welford.cpp` | Function pointers | TBD |
| 16 | `layernorm_distributed/device/kernels/compute/layernorm_pre_allgather.cpp` | NONE | TBD |

### Tier 6: Hard - GroupNorm
| # | Kernel | Complexity | Test File |
|---|--------|------------|-----------|
| 17 | `groupnorm/device/kernels/compute/groupnorm.cpp` | Triple-nested loops, SCALAR+ROW+COL | Manual testing |
| 18 | `groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp` | Subblock indexing | Manual testing |
| 19 | `groupnorm/device/kernels/compute/welford_groupnorm.cpp` | Welford + complex CB | Manual testing |
| 20 | `groupnorm/device/kernels/compute/welford_groupnorm_sharded_v2.cpp` | Most complex | Manual testing |

### Tier 7: Hard - SDPA (Blocked by Inplace API)
| # | Kernel | Blocker | Test File |
|---|--------|---------|-----------|
| 21 | `transformer/sdpa/device/kernels/compute/sdpa.cpp` | `_inplace` API needed | `tests/tt_eager/.../test_scaled_dot_product_attention.py` |
| 22 | `transformer/sdpa/device/kernels/compute/joint_sdpa.cpp` | `_inplace` API needed | Same |
| 23 | `transformer/sdpa/device/kernels/compute/ring_joint_sdpa.cpp` | `_inplace` API needed | Same |
| 24 | `transformer/sdpa_windowed/device/kernels/compute/sdpa_windowed.cpp` | `_inplace` API needed | Same |

### Tier 8: Hard - Fused Matmul Variants
| # | Kernel | Blocker | Notes |
|---|--------|---------|-------|
| 25 | `matmul/device/kernels/compute/bmm_large_block_zm_fused_bias_activation.cpp` | Tightly coupled to matmul loop | Not recommended |

---

## Detailed Kernel Analysis

### 1. sampling.cpp (EASY)

**Location:** `reduction/sampling/device/kernels/compute/sampling.cpp`

**Current Binary Ops:**
```cpp
// Line 52-83: sub_exp_block_bcast_cols_inplace - COL broadcast with fused exp
sub_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);
exp_tile<true>(j);

// Line 85-103: add_block_inplace - NONE broadcast
add_tiles(in0_cb, in1_cb, 0, i, 0);

// Line 105-127: mul_block_bcast_cols - COL broadcast
mul_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);

// Line 327-355: mul_block_bcast_scalar_inplace - SCALAR broadcast
mul_tiles_bcast_scalar(in0_cb, in1_scalar_cb, i, 0, i);
```

**Migration Notes:**
- `add_block_inplace` → Easy, use `Streaming` policy with same CB
- `mul_block_bcast_cols` → Easy, standard COL pattern
- `mul_block_bcast_scalar_inplace` → Easy, SCALAR pattern
- `sub_exp_block_bcast_cols_inplace` → Partial, binary part only (exp stays separate)

**Test:** `pytest tests/ttnn/unit_tests/operations/eltwise/test_sampling.py -v`

---

### 2. layernorm.cpp (MEDIUM)

**Location:** `normalization/layernorm/device/kernels/compute/layernorm.cpp`

**Current Binary Ops:**
```cpp
// Line 96-116: Pre-add (optional) - NONE broadcast
add_tiles(cb_in, cb_inb, i, i, i);

// Line 139-150: x - E[x] - COL broadcast
sub_tiles_bcast_cols(cb_x, cb_ex, i, 0, i);

// Line 160-176: (x-E[x])² - NONE broadcast (square pattern)
mul_tiles(cb_xmm, cb_xmm, global_i, global_i, i);

// Line 190-191: Var + eps - NONE broadcast (single tile)
add_tiles(cb_ex2, cb_eps, 0, 0, dst0);

// Line 216-219: Normalize - COL broadcast
mul_tiles_bcast_cols(cb_xmm, cb_ex2pe, block.to_global(i), 0, i);

// Line 239-246: Apply gamma - ROW broadcast
mul_tiles_bcast_rows(cb_fusion, cb_gamma, i, block.to_global(i), i);

// Line 262-269: Apply beta - ROW broadcast
add_tiles_bcast_rows(cb_fusion, cb_beta, i, block.to_global(i), i);
```

**Migration Strategy:**
1. Pre-add → `add(cb_in, cb_inb, cb_x, BinaryTileShape::row(block_size))`
2. x - E[x] → `sub<BroadcastDim::COL, Streaming, Persistent>(...)` (E[x] persists)
3. Square → `square(cb_xmm, cb_xmm2, shape)` (uses new SQUARE op type)
4. Var + eps → Single tile, may keep as-is
5. Normalize → `mul<BroadcastDim::COL, Preloaded, Persistent>(...)`
6. Gamma → `mul<BroadcastDim::ROW, Streaming, Persistent>(...)`
7. Beta → `add<BroadcastDim::ROW, Streaming, Persistent>(...)`

**Complexity:** Uses `generic::blocks()` iterator and conditional gamma/beta - requires careful policy selection.

**Test:** `pytest tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm.py -v`

---

### 3. moe.cpp (MEDIUM)

**Location:** `reduction/moe/device/kernels/compute/moe.cpp`

**Current Binary Ops:**
```cpp
// Line 23-54: sub_exp_block_bcast_cols_inplace
sub_tiles_bcast_cols(in0_cb, in1_cb, j, i, j);
exp_tile<true>(j);

// Line 56-83: add_block_bcast_rows_inplace - ROW broadcast
add_tiles_bcast_rows(in0_cb, in1_cb, 0, j, 0);

// Line 84-102: mul_block_inplace - NONE broadcast
mul_tiles(in0_cb, in1_cb, 0, i, 0);

// Line 103-125: mul_block_bcast_cols_inplace - COL broadcast
mul_tiles_bcast_cols(in0_cb, in1_cb, 0, i, 0);
```

**Migration Notes:**
- Similar structure to sampling.cpp
- Uses inplace patterns extensively
- Has both ROW and COL broadcasts

**Test:** `pytest tests/ttnn/unit_tests/operations/reduce/test_moe.py -v`

---

### 4. groupnorm.cpp (HARD)

**Location:** `normalization/groupnorm/device/kernels/compute/groupnorm.cpp`

**Current Binary Ops (simplified):**
```cpp
// Input masking - NONE
mul_tiles(cb_in0, cb_input_mask, index, index_mask, w);

// x - E[x] - SCALAR broadcast
sub_tiles_bcast_scalar(cb_in0, cb_ex_global, index, 0, w);

// (x-E[x])² - NONE (square)
mul_tiles(cb_x, cb_x, index, index, w);

// Normalize - SCALAR broadcast
mul_tiles_bcast_scalar(cb_x, cb_ex2pe, index, 0, w);

// Optional gamma - ROW broadcast (conditional)
if (apply_gamma_beta[j]) {
    mul_tiles_bcast_rows(cb_reread_write_out, cb_gamma, index, index_gamma, dst0);
} else {
    copy_tile(cb_reread_write_out, index, dst0);
}

// Optional beta - ROW broadcast (conditional)
if (apply_gamma_beta[j]) {
    add_tiles_bcast_rows(cb_inbeta, cb_beta, index, index_beta, dst0);
} else {
    copy_tile(cb_inbeta, index, dst0);
}
```

**Complexity Factors:**
1. Triple-nested loops: batch × group × out_block
2. Subblock iteration pattern
3. Conditional gamma/beta application per-tile
4. Complex index management (`index_g_offset`, `group_reset_index`, etc.)
5. Uses `apply_gamma_beta[]` array to track per-tile conditions

**Migration Challenge:** The conditional `if (apply_gamma_beta[j])` pattern cannot be directly expressed with current `binary_op` API - would need per-tile control flow.

---

### 5. sdpa.cpp (BLOCKED)

**Location:** `transformer/sdpa/device/kernels/compute/sdpa.cpp`

**Blocking Pattern:**
```cpp
mul_tiles_bcast_cols_inplace(alias_prev_sum, cb_exp_max_diff, Sq_chunk_t);
```

**Why Blocked:**
- Uses `_inplace` variants not available in `binary_op_helpers.hpp`
- Inplace means output CB == input CB
- Would require new API: `mul_inplace<BroadcastDim::COL>(...)`

---

## Testing Strategy

### Pre-Migration Test
Before modifying any kernel:
```bash
pkill -9 -f pytest || true
tt-smi -r 0
timeout 120 pytest <test_file> -v
```

### Post-Migration Validation
After each kernel migration:
```bash
# 1. Clean device state
pkill -9 -f pytest || true
tt-smi -r 0

# 2. Run specific test with timeout
timeout 120 pytest <test_file> -v

# 3. If tests pass, run broader sweep
timeout 300 pytest <test_directory> -v
```

### Test File Mapping

| Kernel Category | Primary Test |
|-----------------|--------------|
| Layernorm variants | `tests/ttnn/unit_tests/operations/fused/test_distributed_layernorm.py` |
| Groupnorm variants | Manual testing (no dedicated unit test found) |
| MOE | `tests/ttnn/unit_tests/operations/reduce/test_moe.py` |
| Sampling | `tests/ttnn/unit_tests/operations/eltwise/test_sampling.py` |
| Batch norm | `tests/ttnn/unit_tests/operations/fused/test_batch_norm.py` |
| SDPA | `tests/tt_eager/.../test_scaled_dot_product_attention.py` |

---

## Migration Cookbook

### Recipe 1: Simple Element-wise

**Before:**
```cpp
mul_tiles_init(cb_a, cb_b);
cb_wait_front(cb_a, N);
cb_wait_front(cb_b, N);
for (uint32_t i = 0; i < N; i++) {
    acquire_dst();
    mul_tiles(cb_a, cb_b, i, i, i);
    pack_tile(i, cb_out);
    release_dst();
}
cb_pop_front(cb_a, N);
cb_pop_front(cb_b, N);
cb_reserve_back(cb_out, N);
cb_push_back(cb_out, N);
```

**After:**
```cpp
compute_kernel_lib::mul(cb_a, cb_b, cb_out, BinaryTileShape::row(N));
```

---

### Recipe 2: ROW Broadcast with Persistent B

**Before:**
```cpp
mul_bcast_rows_init_short(cb_x, cb_gamma);
cb_wait_front(cb_x, block_size);
cb_wait_front(cb_gamma, Wt);  // Gamma persists across rows
for (auto i : block.local()) {
    acquire_dst();
    mul_tiles_bcast_rows(cb_x, cb_gamma, i, block.to_global(i), i);
    pack_tile(i, cb_out);
    release_dst();
}
cb_pop_front(cb_x, block_size);
// Note: cb_gamma is NOT popped - reused for next batch
```

**After:**
```cpp
compute_kernel_lib::mul<
    BroadcastDim::ROW,
    cb_policies::Streaming,      // A: wait/pop per tile
    cb_policies::Persistent,     // B: wait upfront, never pop
    cb_policies::OutputPerTile   // Out: reserve/push per tile
>(cb_x, cb_gamma, cb_out, BinaryTileShape::col(block_size));
```

---

### Recipe 3: COL Broadcast (Centering)

**Before:**
```cpp
sub_bcast_cols_init_short(cb_x, cb_mean);
cb_wait_front(cb_x, Wt);
cb_wait_front(cb_mean, 1);
for (uint32_t i = 0; i < Wt; i++) {
    acquire_dst();
    sub_tiles_bcast_cols(cb_x, cb_mean, i, 0, i);
    pack_tile(i, cb_xmm);
    release_dst();
}
cb_pop_front(cb_x, Wt);
cb_pop_front(cb_mean, 1);
```

**After:**
```cpp
compute_kernel_lib::sub<BroadcastDim::COL>(
    cb_x, cb_mean, cb_xmm, BinaryTileShape::row(Wt));
```

---

### Recipe 4: SCALAR Broadcast

**Before:**
```cpp
mul_tiles_bcast_scalar_init_short(cb_x, cb_scale);
cb_wait_front(cb_x, N);
cb_wait_front(cb_scale, 1);
for (uint32_t i = 0; i < N; i++) {
    acquire_dst();
    mul_tiles_bcast_scalar(cb_x, cb_scale, i, 0, i);
    pack_tile(i, cb_out);
    release_dst();
}
cb_pop_front(cb_x, N);
cb_pop_front(cb_scale, 1);
```

**After:**
```cpp
compute_kernel_lib::mul<BroadcastDim::SCALAR>(
    cb_x, cb_scale, cb_out, BinaryTileShape::row(N));
```

---

### Recipe 5: Square Operation

**Before:**
```cpp
mul_tiles_init(cb_x, cb_x);
cb_wait_front(cb_x, N);
for (uint32_t i = 0; i < N; i++) {
    acquire_dst();
    mul_tiles(cb_x, cb_x, i, i, i);
    pack_tile(i, cb_x2);
    release_dst();
}
```

**After:**
```cpp
compute_kernel_lib::square(cb_x, cb_x2, BinaryTileShape::row(N));
```

---

## Blockers and Missing Features

### 1. Inplace API (HIGH PRIORITY)
**Needed for:** SDPA, MOE, Sampling (partial)

**Proposal:**
```cpp
template <BroadcastDim bcast_dim = BroadcastDim::NONE, ...>
ALWI void mul_inplace(uint32_t icb, uint32_t icb_b, BinaryTileShape shape);
// Uses icb as both input and output
```

### 2. Fused Binary + Unary
**Needed for:** sub + exp pattern in softmax

**Current workaround:** Migrate binary part, keep exp_tile separate

**Future proposal:**
```cpp
template <BinaryOpType op, UnaryOpType post_op, ...>
ALWI void binary_unary_fused(...);
```

### 3. Conditional Per-Tile Operations
**Needed for:** Groupnorm gamma/beta application

**Status:** Not supported - keep manual loop with condition

### 4. Function Pointer Interface
**Needed for:** Distributed layernorm variants

**Status:** Would require significant API redesign

---

## Appendix: Full File List

### Kernels Using Binary Ops (72 total)

```
ttnn/cpp/ttnn/operations/
├── normalization/
│   ├── layernorm/device/kernels/compute/
│   │   ├── layernorm.cpp                    [MEDIUM]
│   │   ├── layernorm_welford.cpp            [MEDIUM]
│   │   ├── layernorm_sharded.cpp            [MEDIUM]
│   │   ├── layernorm_sharded_welford.cpp    [MEDIUM]
│   │   ├── layernorm_large_tensor.cpp       [MEDIUM]
│   │   ├── layernorm_large_tensor_welford.cpp [MEDIUM]
│   │   ├── layernorm_sharded_pre_allgather.cpp [MEDIUM]
│   │   └── layernorm_sharded_post_allgather.cpp [MEDIUM]
│   ├── layernorm_distributed/device/kernels/compute/
│   │   ├── layernorm_post_allgather.cpp     [HARD - fn ptrs]
│   │   ├── layernorm_post_allgather_welford.cpp [HARD - fn ptrs]
│   │   └── layernorm_pre_allgather.cpp      [MEDIUM]
│   ├── groupnorm/device/kernels/compute/
│   │   ├── groupnorm.cpp                    [HARD]
│   │   ├── groupnorm_sharded_v2.cpp         [HARD]
│   │   ├── welford_groupnorm.cpp            [HARD]
│   │   └── welford_groupnorm_sharded_v2.cpp [HARD]
│   ├── batch_norm/device/kernels/compute/
│   │   └── batch_norm_kernel.cpp            [EASY]
│   └── rmsnorm_distributed/device/kernels/compute/
│       └── rmsnorm_post_allgather.cpp       [MEDIUM]
├── reduction/
│   ├── sampling/device/kernels/compute/
│   │   └── sampling.cpp                     [EASY]
│   └── moe/device/kernels/compute/
│       └── moe.cpp                          [MEDIUM]
├── transformer/
│   ├── sdpa/device/kernels/compute/
│   │   ├── sdpa.cpp                         [BLOCKED - inplace]
│   │   ├── joint_sdpa.cpp                   [BLOCKED - inplace]
│   │   └── ring_joint_sdpa.cpp              [BLOCKED - inplace]
│   └── sdpa_windowed/device/kernels/compute/
│       └── sdpa_windowed.cpp                [BLOCKED - inplace]
├── matmul/device/kernels/compute/
│   └── bmm_large_block_zm_fused_bias_activation.cpp [NOT RECOMMENDED]
├── moreh/moreh_dot/device/kernels/
│   └── moreh_dot.cpp                        [DONE ✅]
└── experimental/
    ├── transformer/
    │   ├── rotary_embedding/device/kernels/compute/
    │   │   └── rotary_embedding.cpp         [DONE ✅]
    │   ├── rotary_embedding_llama/device/kernels/compute/
    │   │   └── rotary_embedding_llama.cpp   [DONE ✅]
    │   ├── fused_distributed_rmsnorm/device/kernels/compute/
    │   │   └── rmsnorm_post_allgather.cpp   [MEDIUM]
    │   └── dit_layernorm_post_all_gather/device/kernels/compute/
    │       └── layernorm_post_allgather_welford.cpp [MEDIUM]
    ├── ssm/prefix_scan/device/kernels/
    │   └── ssm_prefix_scan.cpp              [DONE ✅]
    └── ccl/rms_allgather/device/kernels/compute/
        └── rms_compute.cpp                  [MEDIUM]
```

---

## Recommended Migration Order

### Phase 1: Validate and Document (Current)
- [x] Document all patterns
- [x] Identify blockers
- [x] Create test mapping

### Phase 2: Easy Wins (Week 1)
1. `sampling.cpp` - Good variety of patterns
2. `batch_norm_kernel.cpp` - Simple NONE broadcast

### Phase 3: Layernorm Family (Week 2-3)
3. `layernorm.cpp` - Reference implementation
4. Apply same pattern to all layernorm variants (8 files)

### Phase 4: Other Normalizations (Week 4)
5. `rmsnorm_post_allgather.cpp`
6. MOE kernel

### Phase 5: Implement Inplace API (Week 5)
7. Add `mul_inplace`, `add_inplace`, `sub_inplace` variants
8. Migrate SDPA kernels

### Phase 6: GroupNorm (Week 6+)
9. Attempt groupnorm migration with hybrid approach
10. Document any remaining gaps

---

## Success Metrics

| Metric | Target |
|--------|--------|
| Kernels migrated | 50+ |
| Average line reduction | 50% |
| Test pass rate | 100% |
| Performance regression | < 1% |
