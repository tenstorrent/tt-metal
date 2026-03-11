## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the floating-point MUL operation via the legacy SFPU binary path.

**Important context**: The MUL operation on this SFPU path has **two distinct code paths** depending on data type:
1. **Floating-point MUL** (BFloat16, Float32): Uses `mul_binary_tile()` which calls `calculate_sfpu_binary_mul` -- a specialized MUL kernel with BF16 rounding and zero-handling. This is the primary focus of this analysis.
2. **Integer MUL** (INT32, UINT32, UINT16): Uses `mul_int_tile<DataFormat>()` which follows a completely different SFPU path through `llk_math_eltwise_binary_sfpu_mul_int`. Not analyzed here.

For floating-point inputs, the define generation in `get_defines_fp32()` produces:
- `BINOP_INIT` = `mul_binary_tile_init();`
- `BINARY_SFPU_OP` = `mul_binary_tile(i*2, i*2+1, i*2);`

This means input A occupies DST[i*2], input B occupies DST[i*2+1], and the output overwrites DST[i*2] (same location as input A).

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` (metal-level, contains `calculate_sfpu_binary_mul`) and `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_binary.h` (tt_llk-level, contains `_calculate_sfpu_binary_` and `_sfpu_binary_init_`) |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. **Compute kernel** calls `mul_binary_tile_init()` (once via `BINOP_INIT` define) and `mul_binary_tile(i*2, i*2+1, i*2)` (per tile via `BINARY_SFPU_OP` define).
2. **`mul_binary_tile_init()`** (in `eltwise_binary_sfpu.h`) calls `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::MUL>()` which calls `llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROX>()` with `sfpu_binary_init<APPROX, BinaryOp::MUL>` as the init callback. The init function calls `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` which configures SFPU config registers, sets up `ADDR_MOD_7`, and resets counters, then invokes `_sfpu_binary_init_<APPROX, BinaryOp::MUL>()` which is a no-op for MUL (only DIV/POW/XLOGY need special init).
3. **`mul_binary_tile(idst0, idst1, odst)`** (in `eltwise_binary_sfpu.h`) calls `llk_math_eltwise_binary_sfpu_binop_mul<APPROX, BinaryOp::MUL, DST_ACCUM_MODE>(idst0, idst1, odst)`.
4. **`llk_math_eltwise_binary_sfpu_binop_mul`** (in `llk_math_eltwise_binary_sfpu_binop.h`) calls `_llk_math_eltwise_binary_sfpu_params_<APPROX>()` passing `calculate_sfpu_binary_mul<APPROX, BinaryOp::MUL, 8, is_fp32_dest_acc_en>` as the SFPU function callback.
5. **`_llk_math_eltwise_binary_sfpu_params_`** (in `llk_math_eltwise_binary_sfpu_params.h`) sets the DST write address, stalls until math is ready, then iterates over 4 tile faces (in `VectorMode::RC` mode), calling `calculate_sfpu_binary_mul()` once per face with 8 iterations each, advancing the DST address by 16 rows between faces via `TTI_SETRWC`.
6. **`calculate_sfpu_binary_mul`** (in metal-level `ckernel_sfpu_binary.h`) is the core SFPU function that loads two vectors from DST, multiplies them, optionally rounds to BF16 and handles zero-propagation, then writes back to DST.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h
// (Identical for both Wormhole B0 and Blackhole)

// Helper: Convert float32 to bfloat16 using IEEE 754 Round-to-Nearest-Even (RNE)
sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);   // SFPCAST: reinterpret float bits as uint
    sfpi::vUInt lsb = (bits >> 16) & 1;                      // SFPSHFT + SFPAND: extract BF16 mantissa LSB (bit 16)
    // Implementation notes, see the original file for more details
    bits = bits + 0x7fffU + lsb;                             // SFPIADD sequence: add rounding bias + tie-breaker
    bits = bits & 0xFFFF0000U;                               // SFPAND: clear lower 16 bits to produce BF16-in-FP32
    return sfpi::reinterpret<sfpi::vFloat>(bits);            // SFPCAST: reinterpret back as float
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_mul(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // APPROXIMATION_MODE=true (APPROX define), BINOP=BinaryOp::MUL, ITERATIONS=8, is_fp32_dest_acc_en=false (typical BF16 case)
    constexpr uint dst_tile_size_sfpi = 32; // each tile face = 32 rows in SFPI addressing (64 / SFP_DESTREG_STRIDE)
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // SFPLOAD: load 64-element vector from DST tile A
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // SFPLOAD: load 64-element vector from DST tile B

        sfpi::vFloat result = in0 * in1; // SFPMUL: element-wise FP32 multiply

        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result); // software RNE rounding to BF16 precision

            // To match FPU behaviour for bfloat16 multiplication, 0 * x = 0 and x * 0 = 0
            v_if(in0 == 0 || in1 == 0) { result = 0.0f; } // SFPSETCC + SFPCOMPC + SFPMOV: zero-propagation
            v_endif;                                        // SFPPOPC: restore condition code stack
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // SFPSTORE: write result back to DST tile A
        sfpi::dst_reg++;                                            // INCRWC: advance DST pointer by SFP_DESTREG_STRIDE
    }
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void sfpu_binary_init() {
    _sfpu_binary_init_<APPROXIMATION_MODE, BINOP>(); // For MUL: no-op (no reciprocal/log init needed)
}
```

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_binary.h
// (Contains the _sfpu_binary_init_ called during initialization)

template <bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW)
    {
        _init_sfpu_reciprocal_<false>();
    }
    else if constexpr (BINOP == BinaryOp::XLOGY)
    {
        _init_log_<APPROXIMATION_MODE>();
    }
    // For BinaryOp::MUL: no initialization needed -- falls through with no action
}
```

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_binary_sfpu_params.h
// (Face iteration and DST address management -- shown for VectorMode::RC path only)

template <bool APPROXIMATE, typename Callable, typename... Args>
inline void _llk_math_eltwise_binary_sfpu_params_(
    Callable&& sfpu_func,
    std::uint32_t dst_index_in0,
    std::uint32_t dst_index_in1,
    std::uint32_t dst_index_out,
    int vector_mode = static_cast<int>(VectorMode::RC),
    Args&&... args)
{
    _llk_math_eltwise_binary_sfpu_start_<DST_SYNC_MODE>(0); // set DST write addr, stall SFPU until math done

    // VectorMode::RC: process all 4 faces of the 32x32 tile
    // Each face = 16x16 = 8 SFPU iterations of 32-wide vectors (8 rows * 2 via stride)
    for (int face = 0; face < 4; face++)
    {
        sfpu_func(dst_index_in0, dst_index_in1, dst_index_out, std::forward<Args>(args)...);
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D); // advance DST by 8 rows
        TTI_SETRWC(p_setrwc::CLR_NONE, p_setrwc::CR_D, 8, 0, 0, p_setrwc::SET_D); // advance DST by 8 more rows (16 total per face)
    }

    _llk_math_eltwise_binary_sfpu_done_(); // clear DST addr
}
```

### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|------------------------|-------------|
| `SFPLOAD` (`dst_reg[idx]` read) | Loads a 64-element vector from the specified DEST register row into an SFPU local register (LREG). Uses `__builtin_rvtt_sfpload` with `ADDR_MODE_NOINC`. |
| `SFPSTORE` (`dst_reg[idx]` write) | Stores a 64-element vector from an SFPU local register back to the specified DEST register row. Uses `__builtin_rvtt_sfpstore` with `ADDR_MODE_NOINC`. |
| `SFPMUL` (`in0 * in1`) | Element-wise floating-point multiplication of two SFPU vector registers. Maps to `__builtin_rvtt_sfpmul`. This is the core operation of the MUL kernel. |
| `SFPCAST` / `reinterpret` | Reinterprets the bit pattern of a vFloat as vUInt (or vice versa) without changing bits. Used in the BF16 RNE rounding helper. Maps to `__builtin_rvtt_sfpcast`. |
| `SFPSHFT` (`bits >> 16`) | Logical right shift of vector integer elements. Used to extract the BF16 mantissa LSB during rounding. |
| `SFPAND` (`bits & mask`) | Bitwise AND on vector integer elements. Used twice in RNE rounding: once to isolate the LSB, once to clear lower 16 bits. |
| `SFPIADD` (integer add sequence) | Integer addition on vector elements (`bits + 0x7fffU + lsb`). Implements the RNE rounding bias. |
| `SFPSETCC` / `SFPCOMPC` (condition codes via `v_if`) | Sets condition codes based on comparison (`in0 == 0`, `in1 == 0`). The `\|\|` is implemented by testing `in0 == 0`, complementing the CC, testing `in1 == 0`, then combining. Controls which SIMD lanes execute the zero-propagation store. |
| `SFPMOV` (conditional assignment `result = 0.0f`) | Conditionally moves a constant (0.0f) into the result register for lanes where the condition code is true. |
| `SFPPOPC` (`v_endif`) | Pops the condition code stack, restoring the previous condition code state after the `v_if` block. |
| `INCRWC` / `TTI_SETRWC` (`dst_reg++`) | Increments the DEST register write cursor by `SFP_DESTREG_STRIDE` (2 rows), advancing to the next row-pair within a face. Also used at the params level to advance by 16 rows between faces. |
| `TTI_STALLWAIT` | Stalls the SFPU until the math pipeline (FPU) has completed, ensuring DEST register data is ready before SFPU reads. Used in `_llk_math_eltwise_binary_sfpu_start_`. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **DEST[dst_index_in0 * 32 + offset]** | Input tensor A tile data. Read by SFPLOAD. For the standard call pattern `mul_binary_tile(i*2, i*2+1, i*2)`, this is `DEST[i*64 + face_offset]`. Also serves as the output location (overwritten by SFPSTORE). |
| **DEST[dst_index_in1 * 32 + offset]** | Input tensor B tile data. Read by SFPLOAD. For the standard call, this is `DEST[(i*2+1)*32 + face_offset]`. |
| **LREG (in0)** | SFPU local register holding the loaded input A vector (64 elements). |
| **LREG (in1)** | SFPU local register holding the loaded input B vector (64 elements). |
| **LREG (result)** | SFPU local register holding the multiplication result. In the BF16 path, this register is further manipulated by the RNE rounding logic (reinterpreted as integer, shifted, masked, then reinterpreted back). |
| **LREG (bits, lsb)** | Temporary SFPU local registers used within `float32_to_bf16_rne()` for integer bit manipulation during rounding. These are the same physical register file entries reused via compiler register allocation. |
| **Condition Code Stack** | Used by `v_if(in0 == 0 \|\| in1 == 0)` to conditionally zero-out the result. The CC stack is pushed/popped to handle the OR logic (two comparisons combined). |

### Address Mode Configuration

The init path calls `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` which invokes `eltwise_binary_sfpu_configure_addrmod<SfpuType::unused>()`.

For the floating-point MUL operation (`SfpuType::unused`), only **ADDR_MOD_7** is configured:

| Field | ADDR_MOD_7 Value |
|-------|-----------------|
| `srca.incr` | 0 |
| `srcb.incr` | 0 |
| `dest.incr` | 0 |

The `ADDR_MOD_6` with `dest.incr = 2` is **not** configured for floating-point MUL -- it is only set for integer MUL (`SfpuType::mul_int32`, `SfpuType::mul_uint16`) and min/max operations.

This configuration is **identical between Wormhole B0 and Blackhole** -- the `eltwise_binary_sfpu_configure_addrmod` template function has the same implementation in both architectures.

**Note on Wormhole B0 vs Blackhole differences**: The `_llk_math_eltwise_binary_sfpu_start_` function has a minor difference -- Wormhole B0 additionally calls `math::set_addr_mod_base()` before the stall, and `_llk_math_eltwise_binary_sfpu_done_` on Wormhole B0 includes `TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU)` and `math::clear_addr_mod_base()` which are absent on Blackhole. This reflects architectural differences in how the two chips manage SFPU-math synchronization.

**Note on the BF16 rounding path**: The `is_fp32_dest_acc_en` template parameter is set to `false` when the output data format is not Float32/Int32/UInt32 (line 169-171 of the program factory). When `false`, the kernel performs software Round-to-Nearest-Even (RNE) truncation to BF16 precision after the FP32 multiply, plus explicit zero-propagation (`0 * x = 0`). When `true` (FP32 output), neither rounding nor zero-propagation is needed, and the raw FP32 multiply result is stored directly. This is because the SFPU always computes in FP32 internally, but BF16 outputs need explicit rounding to match FPU behavior.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the binary SFPU compute kernel work for operations like MUL? What defines like SFPU_OP_CHAIN_0, SFPU_OP_INIT_0 control the SFPU dispatch? Trace how get_defines_fp32 produces these defines for BinaryOpType::MUL."
   **Reason**: Needed to understand the macro-driven dispatch mechanism from the compute kernel to the SFPU functions.
   **Key Findings**: For floating-point MUL, `get_defines_fp32` produces `BINOP_INIT` with `mul_binary_tile_init()` and `BINARY_SFPU_OP` with `mul_binary_tile(i*2, i*2+1, i*2)`. The compute kernel uses `#ifdef BINOP_INIT` / `#ifdef BINARY_SFPU_OP` to dispatch these.

2. **Query**: "Show me the implementation of _llk_math_eltwise_binary_sfpu_params_ and _llk_math_eltwise_binary_sfpu_init_ and _sfpu_binary_init_" (to tenstorrent/tt-llk)
   **Reason**: These functions are in the tt_llk submodule and handle face iteration, ADDR_MOD configuration, and operation-specific SFPU initialization.
   **Key Findings**: `_llk_math_eltwise_binary_sfpu_params_` iterates over 4 faces in RC mode, calling the SFPU function 4 times with 8 iterations each (32 total iterations = 32 rows = full tile). `_sfpu_binary_init_` is a no-op for MUL. `ADDR_MOD_7` is configured with all-zero increments.

3. **Query**: "Explain how sfpi::dst_reg works for loading and storing SFPU data from DEST registers" (to tenstorrent/sfpi)
   **Reason**: Needed to understand how the SFPI C++ constructs map to actual SFPU instructions.
   **Key Findings**: `dst_reg[index]` maps to `SFPLOAD`/`SFPSTORE` instructions. `dst_reg++` maps to `INCRWC` (increment write cursor by `SFP_DESTREG_STRIDE`). `vFloat` multiplication maps to `SFPMUL`. `v_if`/`v_endif` maps to condition code manipulation (`SFPSETCC`, `SFPCOMPC`, `SFPPOPC`).

### Confluence References
Not consulted for this analysis -- DeepWiki and source code provided sufficient detail for the MUL SFPU kernel instructions.

### Glean References
Not consulted for this analysis.
