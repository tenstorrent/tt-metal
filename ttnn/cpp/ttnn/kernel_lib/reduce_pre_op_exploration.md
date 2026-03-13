# Reduce Helper Pre-Op Exploration: Findings & Feasibility

## Table of Contents
1. [Current Reduce Helper Capabilities](#1-current-reduce-helper-capabilities)
2. [What is a "Pre-Op" and Why Do We Want It?](#2-what-is-a-pre-op-and-why-do-we-want-it)
3. [Real-World Pre-Op Patterns Found in the Codebase](#3-real-world-pre-op-patterns-found-in-the-codebase)
4. [Hardware Architecture: Data Flow Through the Compute Core](#4-hardware-architecture-data-flow-through-the-compute-core)
5. [Implementation Approaches for Pre-Op](#5-implementation-approaches-for-pre-op)
6. [EltwiseBinaryReuseDestType and D2A/D2B: The DEST Reuse Mechanism](#6-eltwisebinaryreusedesttype-and-d2ad2b-the-dest-reuse-mechanism)
7. [SFPU-Based Reduce: The All-in-DEST Path](#7-sfpu-based-reduce-the-all-in-dest-path)
8. [FPU + SFPU Fusion: What's Possible Today](#8-fpu--sfpu-fusion-whats-possible-today)
9. [What Would Need Additional LLK Support](#9-what-would-need-additional-llk-support)
10. [Recommendations for the Reduce Helper Library](#10-recommendations-for-the-reduce-helper-library)

---

## 1. Current Reduce Helper Capabilities

The reduce helper (`reduce_helpers_compute.hpp`) currently provides:

- **Post-reduce op** via `PostReduceOp` template parameter — a callback invoked on the DEST tile *after* reduce completes. Used for softmax's `recip_tile` pattern (reduce_sum → recip gives 1/sum).
- **Accumulation** via `Accumulate` type — for block-wise reduction across multiple `reduce()` calls, with CB-based save/reload of partial results.
- **Input policies** — control CB wait/pop behavior for different data reuse patterns.
- **Data format reconfiguration** — controls unpacker/packer reconfig between operations.

**What it does NOT have**: A `pre_reduce_op` — a callback that transforms each tile *before* it enters the FPU reduce pipeline.

---

## 2. What is a "Pre-Op" and Why Do We Want It?

Many operations follow a pattern:

```
for each tile:
    transform(tile)     ← "pre-op" (exp, square, negate, etc.)
    reduce(transformed) ← FPU reduce (SUM, MAX, AVG)
```

Currently, this requires an **intermediate circular buffer**:
1. Unpack tile from input CB
2. Apply transform (SFPU or FPU) → writes to DEST
3. **Pack to intermediate CB** ← CB round-trip OUT
4. **Unpack from intermediate CB** to SRCA for reduce ← CB round-trip IN
5. Reduce tile (GAPOOL/GMPOOL) → accumulates in DEST

The intermediate CB wastes:
- **L1 memory** (typically 1-2 tile pages of SRAM)
- **Latency** (pack → wait → unpack cycle)
- **Code complexity** (extra CB configuration, separate pack/unpack loops)

A `pre_reduce_op` that fuses step 2-4 would eliminate this overhead.

---

## 3. Real-World Pre-Op Patterns Found in the Codebase

### Pattern A: `exp` → `reduce<SUM>` (Softmax, 7+ kernels)

| File | Lines | Notes |
|------|-------|-------|
| `ttnn/.../softmax/device/kernels/attention/compute/softmax.cpp` | 253-292 | exp → pack to `cb_exps` → reduce |
| `ttnn/.../softmax/device/kernels/attention/compute/softmax_sharded.cpp` | 196-238 | Same pattern, sharded variant |
| `ttnn/.../softmax/device/kernels/attention/compute/softmax_large_tensor.cpp` | 205-295 | `exp_cb()` → `reduce_cb()` with separate functions |
| `tests/tt_metal/tt_metal/test_kernels/compute/softmax.cpp` | 105-133 | Test kernel, explicit loop |
| `tt-train/.../softmax/device/kernels/compute/softmax_kernel.cpp` | 198-301 | Manual DEST accumulation variant |
| `tt-train/.../cross_entropy_fw/.../cross_entropy_fw_kernel.cpp` | 190-339 | sub_max → exp → reduce_sum → log (log-sum-exp) |
| `tt-train/.../cross_entropy_bw/.../cross_entropy_bw_kernel.cpp` | 196-360 | sub_max → exp → accumulate → "reduce" via matmul |

**Fusion value**: Eliminating the `cb_exps` intermediate CB in softmax alone would save 1 CB + significant latency in the most performance-critical kernel.

### Pattern B: `square` → `reduce<SUM/AVG>` (LayerNorm/RMSNorm variance, 11+ kernels)

| File | Lines | Notes |
|------|-------|-------|
| `ttnn/.../layernorm/device/kernels/compute/layernorm.cpp` | 197-220 | mul_tiles(x,x) → `cb_xmm2` → reduce |
| `ttnn/.../layernorm/device/kernels/compute/layernorm_large_tensor.cpp` | 149-186 | square_tile → `cb_xmm2` → reduce |
| `ttnn/.../layernorm/device/kernels/compute/layernorm_sharded.cpp` | 242-285 | mul_tiles(x,x) → `cb_xmm2` → reduce |
| `ttnn/.../layernorm/device/kernels/compute/layernorm_sharded_pre_allgather.cpp` | 149-182 | mul_tiles → `cb_x2` → reduce |
| `ttnn/.../rmsnorm_distributed/device/kernels/compute/rmsnorm_pre_allgather.cpp` | 57-75 | mul_tiles → `cb_x2` → reduce |
| `ttnn/.../experimental/ccl/rms_allgather/device/kernels/compute/rms_compute.cpp` | 112-141 | mul_tiles → `cb_x2` → reduce |
| `ttnn/.../normalization/groupnorm/device/kernels/compute/groupnorm_sharded_v2.cpp` | 316-343 | mul_tiles → DEST accum → reduce_scalar |
| `tests/tt_metal/tt_metal/test_kernels/compute/layernorm.cpp` | 136-167 | mul_tiles(x,x) → `cb_xmm2` → reduce |
| `tests/tt_metal/tt_metal/test_kernels/compute/rmsnorm.cpp` | 95-123 | mul_tiles(x,x) → `cb_x2` → reduce |
| `tt-train/.../rmsnorm_fw/.../rmsnorm_fw_kernel.cpp` | 52-305 | Square in DEST → manual accumulation → reduce |
| `tt-train/.../layernorm_fw/.../layernorm_fw_kernel.cpp` | 174-304 | sub → square → manual DEST accumulation |

**Fusion value**: Eliminating `cb_xmm2`/`cb_x2` in normalization ops saves 1 CB per kernel. These kernels are used in every transformer layer.

### Pattern C: `negative` → `reduce<SUM>` (Neg-reduce, 3 kernels)

| File | Lines | Notes |
|------|-------|-------|
| `ttnn/.../reduction/generic/device/kernels/compute/reduce_w_neg.cpp` | 45-80 | copy → negate → `cb_ineg` → reduce |
| `ttnn/.../reduction/generic/device/kernels/compute/reduce_h_neg.cpp` | 56-113 | copy → negate → `cb_ineg` → reduce |
| `ttnn/.../reduction/generic/device/kernels/compute/reduce_hw_neg.cpp` | — | Same pattern for HW reduction |

### Pattern D: `mul(y, grad)` → `reduce<SUM>` (Softmax/RMSNorm backward, 2+ kernels)

| File | Lines | Notes |
|------|-------|-------|
| `tt-train/.../softmax_backward/.../softmax_backward_kernel.cpp` | 132-140 | elementwise_mul → CB → reduce |
| `tt-train/.../rmsnorm_bw/.../rmsnorm_bw_kernel.cpp` | 159-304 | mul → DEST accumulation → matmul-reduce |

### Pattern E: Compound `sub_max` → `exp` → `reduce<SUM>` (Numeric-stable softmax)

This is effectively pattern A composed with a broadcast-subtract. All softmax kernels with `NUMERIC_STABLE` defined use this compound pattern. The `calc_numeric_stable()` helper in softmax kernels chains: sub_tiles_bcast_cols → exp_tile → pack to CB.

### Summary Table

| Pre-Op | Reduce Type | Intermediate CB | Kernel Count | Operations |
|--------|-------------|-----------------|-------------|------------|
| `exp_tile` | SUM | Required (`cb_exps`) | 7+ | Softmax, cross-entropy |
| `square`/`mul(x,x)` | SUM/AVG | Required (`cb_xmm2`, `cb_x2`) | 11+ | LayerNorm, RMSNorm, GroupNorm |
| `negative_tile` | SUM | Required (`cb_ineg`) | 3 | reduce_neg variants |
| `mul_tiles(a,b)` | SUM | Required | 2+ | Backward passes |
| `sub_bcast + exp` | SUM | Required | 6+ | Numeric-stable softmax |

---

## 4. Hardware Architecture: Data Flow Through the Compute Core

Understanding the hardware is essential to evaluate pre-op feasibility.

### Register Files and Execution Units

```
                     L1 (Circular Buffers)
                      |              |
               [Unpacker 0]    [Unpacker 1]
                 /       \          |
              SRCA      DEST     SRCB
                \        |       /
                 [FPU: GAPOOL/GMPOOL/ELWMUL/ELWADD]
                         |
                       DEST
                      /     \
               [SFPU]      [Packer]
              (LREG0-7)      |
                |            L1
              DEST
```

### Key Constraints

1. **FPU reduce (GAPOOL/GMPOOL)** reads from **SRCA + SRCB**, accumulates into **DEST**
   - SRCA = input tile (from CB via Unpacker 0)
   - SRCB = scaler tile (from CB via Unpacker 1)
   - Cannot read input from DEST directly

2. **SFPU** reads/writes **DEST only** (load/store architecture: SFPLOAD from DEST → LREGs → compute → SFPSTORE to DEST)
   - Cannot read from SRCA or SRCB
   - Cannot directly feed data into GAPOOL/GMPOOL

3. **Hardware MOV instructions** can shuffle data between register files:
   - `MOVD2A` — DEST → SRCA (4 rows/instruction on WH/BH, 8 rows/instruction on Quasar)
   - `MOVD2B` — DEST → SRCB
   - `MOVA2D` — SRCA → DEST
   - `MOVB2D` — SRCB → DEST
   - `MOVB2A` — SRCB → SRCA

4. **FPU and SFPU share the math thread (TRISC1)** but are separate execution units with asynchronous pipelines. Hardware stalls (`STALL_SFPU, MATH` and `STALL_CFG, WAIT_SFPU`) synchronize them automatically within LLK calls.

### Implication for Pre-Op

A naive "SFPU pre-op then FPU reduce" requires:
```
Unpack tile → DEST (via unpack-to-dest)
SFPU(pre_op) on DEST          ← pre-op in DEST
MOVD2A: DEST → SRCA           ← 4-16 MOV instructions (face-by-face)
FPU reduce: SRCA + SRCB → DEST  ← reduce accumulates into DEST
```

This avoids the CB round-trip but pays for MOV instruction overhead. Whether this is faster than a CB round-trip depends on the tile size and pipeline overlap.

---

## 5. Implementation Approaches for Pre-Op

### Approach 1: Software Pre-Op (Current State — CB Round-Trip)

```
tile_regs_acquire();
unpack(input_cb → SRCA);
SFPU(exp/square/neg);      // operates on DEST after datacopy
pack_tile(dst, intermediate_cb);
tile_regs_release();

tile_regs_acquire();
unpack(intermediate_cb → SRCA);
reduce_tile(intermediate_cb, scaler_cb, ...);
pack_tile(dst, output_cb);
tile_regs_release();
```

**Cost**: 1 extra CB (L1 memory) + pack/unpack latency per tile
**Benefit**: Works today, no LLK changes needed

### Approach 2: D2A Bridge (MOVD2A Path)

```
tile_regs_acquire();
unpack_to_dest(input_cb → DEST);    // Need unpack-to-dest support
SFPU(exp/square/neg) on DEST;        // Pre-op
MOVD2A(DEST → SRCA);                 // 4-16 MOV instructions
unpack_B(scaler_cb → SRCB);          // Scaler still from CB
GAPOOL(SRCA + SRCB → DEST);          // FPU reduce
pack_tile(DEST → output_cb);
tile_regs_release();
```

**Cost**: ~4-16 MOVD2A instructions (one face at a time) + SRCA precision limited to ~19 bits
**Benefit**: No intermediate CB, no pack/unpack latency
**Feasibility**: Partially possible today. The binary eltwise LLK already implements this pattern via `EltwiseBinaryReuseDestType::DEST_TO_SRCA`. However, the *reduce* LLK does not have a "reuse dest" mode — it always expects the unpacker to populate SRCA. **Would need new LLK support** to add a `DEST_TO_SRCA` path to `_llk_math_reduce_()`.

**Critical issue**: reduce_tile internally uses MOVD2B for transpose operations (REDUCE_ROW path). If DEST already contains the pre-op result, the D2B operations in reduce would corrupt it. The timing would need careful orchestration.

### Approach 3: SFPU-Based Reduce (All-in-DEST)

```
tile_regs_acquire();
unpack_to_dest(input_cb → DEST);     // Load tile into DEST
SFPU(exp/square/neg) on DEST;         // Pre-op — still in DEST
SFPU_reduce(DEST);                     // SFPU-based reduce — still in DEST
pack_tile(DEST → output_cb);
tile_regs_release();
```

**Cost**: SFPU reduce is slower than FPU reduce (more instructions, SFPSHFT2 latency)
**Benefit**: No CB round-trip, no MOV instructions, all operations stay in DEST
**Feasibility**: The SFPU reduce already exists in `ckernel_sfpu_reduce.h` (see Section 7). However, it's currently only used in experimental LLK tests and the SDPA pack-thread path — not exposed as a general compute API.

### Approach 4: Pre-Op as Lambda in Reduce Helper (Pragmatic)

Add a `pre_reduce_op` callback to the reduce helper that runs between unpack and reduce:

```cpp
// Inside reduce helper, for each tile:
if constexpr (waits_per_tile(input_policy)) {
    cb_wait_front(input_cb, onetile);
}

// --- NEW: Pre-reduce op ---
// Apply transform, pack to scratch CB, pop input
tile_regs_acquire();
copy_tile(input_cb, tile_idx, dst_idx);
pre_reduce_op(dst_idx);                    // exp_tile, square_tile, etc.
pack_tile(dst_idx, scratch_cb);
tile_regs_commit(); tile_regs_wait(); tile_regs_release();
if constexpr (should_pop(input_policy)) { cb_pop_front(input_cb, onetile); }

// --- Then reduce from scratch CB ---
tile_regs_acquire();
reduce_tile(scratch_cb, scaler_cb, 0, 0, reduce_dst);
// ...
```

**Cost**: Still uses a CB round-trip, but the helper manages it automatically — the user doesn't need to create and manage the intermediate CB.
**Benefit**: Works today, simplifies user code significantly, hides the intermediate CB inside the library.
**Feasibility**: Fully possible today, no LLK changes needed. This is the **pragmatic near-term solution**.

---

## 6. EltwiseBinaryReuseDestType and D2A/D2B: The DEST Reuse Mechanism

### Enum Definition

```cpp
// llk_defs.h (identical across WH, BH, Quasar)
enum class EltwiseBinaryReuseDestType {
    NONE         = 0,   // Standard: both operands from CBs
    DEST_TO_SRCA = 1,   // DEST contents become SRCA input (other from CB via SRCB)
    DEST_TO_SRCB = 2,   // DEST contents become SRCB input (other from CB via SRCA)
};
```

### How It Works at the LLK Level

When `DEST_TO_SRCA`:
1. Unpacker zeros SRCA and sets dvalid (since math thread will fill it)
2. Before each face of the binary op, `move_d2a_fixed_face()` copies DEST → SRCA
3. Unpacker loads the *other* operand into SRCB from a CB
4. FPU performs binary op (SRCA op SRCB → DEST)

When `DEST_TO_SRCB`:
1. Unpacker zeros SRCB and sets dvalid
2. Before each face, `move_d2b_fixed_face()` copies DEST → SRCB
3. Unpacker loads the other operand into SRCA from a CB
4. FPU performs binary op (SRCA op SRCB → DEST)

### Who Uses It Today (13+ kernel files)

| Kernel | Reuse Type | Operation | Purpose |
|--------|-----------|-----------|---------|
| `mish_kernel.cpp` | DEST_TO_SRCA | MUL | tanh(softplus(x)) * x |
| `tanhshrink_kernel.cpp` | DEST_TO_SRCB | SUB | x - tanh(x) |
| `hardshrink_kernel.cpp` | Both | MUL, ADD, SUB | Complex conditional chain |
| `hardswish_kernel.cpp` | DEST_TO_SRCA | MUL | hardswish activation |
| `ternary_addc_ops_fpu.cpp` | DEST_TO_SRCA | ADD | addcmul: (b*c)*value + a |
| `layernorm_large_tensor.cpp` | DEST_TO_SRCB | ADD | Accumulation across tiles |
| `softmax_large_tensor.cpp` | DEST_TO_SRCB | ADD | Accumulation across tiles |
| `batch_norm_kernel.cpp` | DEST_TO_SRCA | MUL | Scale + bias |
| `where_kernel.cpp` | DEST_TO_SRCA | MUL, ADD | Conditional selection |
| `deepseek/mla/matmul_wo/compute_collector.cpp` | DEST_TO_SRCA | ADD | Multi-source accumulation |
| `deepseek/moe/moe_gate_mm/compute.cpp` | DEST_TO_SRCA | ADD | Gate accumulation |
| `softmax_backward_kernel.cpp` | DEST_TO_SRCA | MUL | Backward pass |
| `deepseek_v3_b1/unified_kernels/rmsnorm.hpp` | DEST_TO_SRCA | MUL | Gamma multiplication |

### What binary_op_helpers.inl Hardcodes

In `binary_op_helpers.inl:121`, `binary_exec` hardcodes `EltwiseBinaryReuseDestType::NONE`:
```cpp
MATH((llk_math_eltwise_binary<elt_type, bcast_type, DST_ACCUM_MODE, MATH_FIDELITY,
      EltwiseBinaryReuseDestType::NONE>(icb_a, icb_b, idst, true)));
```

Adding DEST reuse to the binary_op helper would enable patterns like:
- SFPU(exp) in DEST → binary_dest_reuse(MUL, DEST_TO_SRCA, scaler_cb) — multiply exp result with a scale factor without CB round-trip
- reduce result in DEST → binary_dest_reuse(MUL, DEST_TO_SRCA, cb) — post-reduce multiply

### Relevance to Reduce Pre-Op

The DEST reuse mechanism proves that the hardware **can** feed data from DEST into FPU operations. However:

1. **The reduce LLK does NOT support `EltwiseBinaryReuseDestType`** — there is no "reduce from DEST" variant of `_llk_math_reduce_()`. Adding one would require LLK-level changes.

2. **Reduce internally uses MOVD2B** for transpose operations (REDUCE_ROW path), which would conflict with a DEST_TO_SRCB pre-op.

3. The experimental `llk_math_mul_reduce_scalar.h` (Blackhole only) demonstrates a fused multiply+reduce that uses MOVD2A to feed DEST data into the GAPOOL pipeline. This proves the concept is feasible but is not generalized.

---

## 7. SFPU-Based Reduce: The All-in-DEST Path

### Existing Implementation

File: `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_reduce.h`

A complete SFPU-based reduce implementation exists that operates **entirely within DEST registers**:

- **SUM/AVG column reduction** (`calculate_reduce_sum_avg`): Uses SFPLOAD/SFPSTORE + SFPTRANSP + replay buffers for tree reduction. Processes face-by-face, accumulating sums in LREGs, then stores back to DEST.

- **SUM row reduction** (`perform_reduce_row_sum`): Uses SFPSHFT2 for cross-column reduction (horizontal reduce), processes 8 column slices in parallel.

- **MAX/MIN column reduction** (`calculate_reduce_max_min`): Uses SFPSWAP + LOADMACRO sequences for compare-and-swap operations.

### Supported Operations and Data Formats

| Pool Type | Reduce Dim | Supported Formats |
|-----------|-----------|-------------------|
| SUM | REDUCE_COL | Int32, UInt32, UInt16, Float32, Float16_b |
| AVG | REDUCE_COL | Int32, UInt32, UInt16, Float32, Float16_b |
| SUM | REDUCE_ROW | Same as above |
| MAX | REDUCE_COL | Same as above |
| MIN | REDUCE_COL | Same as above |

### Current Usage Status

- **LLK tests**: Used in `sfpu_reduce_sdpa_test.cpp` — runs on the **pack thread** (TRISC2), enabling overlap with FPU math
- **SDPA**: Used in attention score computation for pack-thread reduce
- **NOT exposed** as a general compute kernel API (`reduce_tile` equivalent)

### Fusion Potential

With SFPU reduce, a fully-fused pre-op+reduce path becomes possible:

```
unpack_to_dest(input_cb → DEST)
SFPU(exp_tile on DEST)           // pre-op: stays in DEST
SFPU(reduce_sum on DEST)         // reduce: stays in DEST, no SRCA/SRCB needed
pack_tile(DEST → output_cb)
```

**No intermediate CB, no MOV instructions, no unpacker involvement for reduce.**

### Limitations

1. **SFPU reduce is slower** than FPU reduce (GAPOOL/GMPOOL are dedicated hardware; SFPU reduce uses software-emulated tree reduction)
2. **Row reduction** needs SFPSHFT2 for cross-column communication — only 3 reduction stages per shift, with 2-cycle latency each
3. **REDUCE_SCALAR** would need both row and column reduction, requiring the most SFPU instructions
4. **Precision**: SFPU operates in FP32 internally (LREGs are 32-bit), which is actually *better* than the FPU path for non-FP32 formats
5. **Accumulation across tiles**: SFPU reduce within a tile is fine, but accumulating across tiles needs either SFPADD in DEST or a CB round-trip

---

## 8. FPU + SFPU Fusion: What's Possible Today

### Proven Fusion Patterns (Production Kernels)

| Pattern | Example Kernel | Status |
|---------|---------------|--------|
| FPU(binary) → SFPU(unary) | `sub_tiles_bcast_cols` → `exp_tile` | **Working** in SDPA, softmax |
| FPU(reduce) → SFPU(unary) | `reduce_tile` → `recip_tile` | **Working** via `post_reduce_op` |
| FPU(binary) → SFPU → SFPU → SFPU | `sub` → `exp` → `add_unary(1)` → `recip` | **Working** in SDPA (fused sigmoid) |
| FPU(copy) → SFPU(custom) | `copy_tile` → `fused_max_sub_exp_add` | **Working** in SDPA |
| Multiple SFPU chained | `exp` → `add_unary` → `recip` | **Working** (sequential in same block) |
| FPU(mul) → SFPU(unary) → FPU(binary_dest_reuse) | `mul_tiles` → `mul_unary_tile` → `binary_dest_reuse<ADD>` | **Working** in ternary ops |

### The Stalling Protocol

Every SFPU tile operation automatically inserts stalls:
```cpp
TTI_STALLWAIT(p_stall::STALL_SFPU, p_stall::MATH);    // Before SFPU: wait for FPU done
// ... SFPU computation ...
TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU); // After SFPU: wait for SFPU done
```

This means within a single `tile_regs_acquire/release` block:
- **FPU → SFPU**: Safe (stall waits for FPU to finish writing DEST)
- **SFPU → FPU**: Safe (stall waits for SFPU to finish writing DEST)
- **SFPU → SFPU**: Safe (each operation is self-contained with stalls)

### Init/Uninit Requirements for Switching

| Transition | Re-init Required? | Notes |
|-----------|-------------------|-------|
| reduce → SFPU(any) | `*_tile_init()` only | SFPU uses ADDR_MOD_7, doesn't conflict with reduce's ADDR_MOD_0-3 |
| SFPU(any) → reduce | `reduce_init` or `reduce_init_short_with_dt` | Restore addr modes, MOP, CLR_DVALID settings |
| binary → SFPU(any) | `*_tile_init()` only | Same as reduce → SFPU |
| SFPU(any) → binary | `binary_init<op,bcast>(cb_a, cb_b)` | Restore eltwise binary config |
| binary → reduce | `reduce_init` | Different hardware configs |

---

## 9. What Would Need Additional LLK Support

### For D2A-Based Pre-Op in FPU Reduce

1. **New `_llk_math_reduce_with_dest_reuse_()` variant** — analogous to `_llk_math_eltwise_binary_with_dest_reuse_()` but for GAPOOL/GMPOOL. Would need to:
   - Call `move_d2a_fixed_face()` before each face pair
   - Handle the conflict with reduce's internal MOVD2B for transpose (REDUCE_ROW)
   - Handle SRCA counter management differently (reduce expects auto-increment from unpacker)

2. **Corresponding `_llk_unpack_AB_reduce_with_dest_reuse_()` variant** — that zeros SRCA and sets dvalid (like the binary reuse unpack path), while still unpacking SRCB for the scaler

3. **Address mode reconciliation** — reduce uses ADDR_MOD_0-3 extensively. A DEST_TO_SRCA path would need compatible address mode configuration.

**Difficulty**: Medium-high. The reduce math implementation is complex (different for REDUCE_ROW/COL/SCALAR) with many internal register shuffles. Adding DEST reuse without breaking existing behavior would require careful validation.

### For Exposing SFPU Reduce as Compute API

1. **New compute API function** — `sfpu_reduce_tile<pool_type, reduce_dim, format>(dst_idx)` that wraps `_init_reduce_<pool_type, format>()` and `_calculate_reduce_<pool_type, reduce_dim, format>()`

2. **Multi-tile accumulation** — the current SFPU reduce operates on tiles already in DEST. For cross-tile accumulation, need to define how partial results are combined (SFPADD in DEST, or pack-reload).

3. **Integration with reduce helper** — would need a new dispatch path that uses SFPU reduce instead of FPU reduce when a pre-op is present.

**Difficulty**: Medium. The SFPU reduce code exists and works; the main effort is API design, integration, and testing across hardware targets.

### For `mul_reduce_scalar` Generalization

The experimental `llk_math_mul_reduce_scalar.h` (Blackhole only) already implements:
- MOVD2A to feed DEST data into SRCA
- GAPOOL to reduce the SRCA data
- D2B + transpose for final scalar accumulation

This could be generalized to:
1. Support more pre-ops beyond multiply (exp, square, negate)
2. Support all reduce dimensions (not just scalar)
3. Port to Wormhole B0 and Quasar

**Difficulty**: Medium. The Blackhole implementation is a working proof-of-concept.

---

## 10. Recommendations for the Reduce Helper Library

### Near-Term: Pragmatic Pre-Op via Lambda + Internal Scratch CB

Add a `pre_reduce_op` parameter to the reduce helper that uses an **internally-managed scratch CB** for the transformation:

```cpp
template <PoolType reduce_type, ReduceDim reduce_dim, ...>
void reduce(
    uint32_t input_cb, uint32_t scaler_cb, uint32_t output_cb,
    ReduceInputBlockShape shape,
    ...,
    PreReduceOp pre_reduce_op = NoOp{},    // NEW
    uint32_t scratch_cb = 0                 // NEW: CB for pre-op intermediates
);
```

Usage:
```cpp
// Softmax: exp → reduce_sum → recip
reduce<SUM, REDUCE_ROW>(cb_in, cb_scaler, cb_out, shape,
    ...,
    [](uint32_t dst_idx) { exp_tile_init(); exp_tile(dst_idx); },  // pre-op
    cb_scratch,                                                      // scratch CB
    NoAccumulation{},
    [](uint32_t dst_idx) { recip_tile_init(); recip_tile(dst_idx); } // post-op
);

// LayerNorm variance: square → reduce_avg
reduce<AVG, REDUCE_ROW>(cb_in, cb_scaler, cb_out, shape,
    ...,
    [](uint32_t dst_idx) { square_tile_init(); square_tile(dst_idx); }, // pre-op
    cb_scratch);
```

**Benefits**:
- Works on all hardware today, no LLK changes
- User doesn't manage the intermediate CB
- Eliminates boilerplate code (separate loops, CB wait/pop management)
- Easy to optimize later (swap CB round-trip for D2A or SFPU reduce when available)

### Medium-Term: D2A Pre-Op for Binary Helper

Add `EltwiseBinaryReuseDestType` support to `binary_op_helpers.hpp`:

```cpp
template <BinaryOpType op_type, BroadcastDim bcast_dim,
          EltwiseBinaryReuseDestType reuse_dest = EltwiseBinaryReuseDestType::NONE, ...>
void binary_op(uint32_t icb_a, uint32_t icb_b, uint32_t ocb, ...);
```

This enables efficient post-reduce binary operations (e.g., `reduce → multiply by scale`) without CB round-trip. This is already done by 13+ kernels using the raw `binary_dest_reuse_tiles` API; the helper would just make it ergonomic.

### Long-Term: SFPU Reduce Integration

Once SFPU reduce is exposed as a compute API:

```cpp
// Full fusion: unpack → pre-op → sfpu_reduce → post-op → pack
// All in DEST, zero intermediate CBs
reduce<SUM, REDUCE_ROW, ReduceEngine::SFPU>(cb_in, cb_scaler, cb_out, shape,
    ...,
    [](uint32_t dst_idx) { exp_tile(dst_idx); },    // pre-op: fused!
    NoAccumulation{},
    [](uint32_t dst_idx) { recip_tile(dst_idx); });  // post-op: fused!
```

This would be the ultimate optimization for softmax and normalization kernels, eliminating all intermediate CBs and register shuffles.

### Hardware-Specific Considerations

| Hardware | D2A Cost (per face) | SFPU Reduce Status | Recommendation |
|----------|--------------------|--------------------|----------------|
| **Wormhole B0** | 4x MOV_4_ROWS | Exists, not exposed | Focus on scratch-CB approach |
| **Blackhole** | 4x MOV_4_ROWS | Exists + mul_reduce_scalar experimental | Can prototype D2A approach |
| **Quasar** | 2x MOV_8_ROWS (faster!) | Exists | Better candidate for D2A path |

---

## Appendix: Key Source File References

### Reduce Helper Library
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.hpp` — Public API
- `ttnn/cpp/ttnn/kernel_lib/reduce_helpers_compute.inl` — Implementation
- `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.hpp` — Binary op helper (for comparison)
- `ttnn/cpp/ttnn/kernel_lib/binary_op_helpers.inl` — Binary op implementation

### LLK Reduce Implementation
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_reduce.h` — FPU reduce core (WH)
- `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/llk_math_reduce.h` — FPU reduce core (BH)
- `tt_metal/third_party/tt_llk/tt_llk_quasar/llk_lib/llk_math_reduce.h` — FPU reduce core (Quasar)
- `tt_metal/hw/inc/api/compute/reduce.h` — Compute API for reduce_tile

### SFPU Reduce
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_reduce.h` — SFPU reduce implementation
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_reduce_custom.h` — Custom SFPU reduce max col

### D2A/D2B and DEST Reuse
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_defs.h:60` — EltwiseBinaryReuseDestType enum
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/cmath_common.h:43` — move_d2a/d2b helpers
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/llk_lib/llk_math_eltwise_binary.h:285` — Binary DEST reuse impl
- `tt_metal/hw/inc/api/compute/eltwise_binary.h:248` — binary_dest_reuse_tiles API

### Experimental Fused Mul-Reduce
- `tt_metal/third_party/tt_llk/tt_llk_blackhole/llk_lib/experimental/llk_math_mul_reduce_scalar.h` — Blackhole-only fused mul+reduce

### Hardware Instructions
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/instructions/assembly.yaml:273` — MOVD2A
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/instructions/assembly.yaml:316` — MOVD2B
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/instructions/assembly.yaml:937` — GAPOOL
- `tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/instructions/assembly.yaml:970` — GMPOOL
