## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the floating-point MUL operation in binary_ng.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. **Compute kernel** (`eltwise_binary_sfpu_no_bcast.cpp`): The `BINARY_SFPU_OP(i*2, i*2+1, i*2)` macro expands to `mul_binary_tile(i*2, i*2+1, i*2)` via the define `BINARY_SFPU_OP = mul_binary_tile`. Similarly, `BINARY_SFPU_INIT` expands to `mul_binary_tile_init();`.

2. **API header** (`eltwise_binary_sfpu.h`): `mul_binary_tile(idst0, idst1, odst)` calls `llk_math_eltwise_binary_sfpu_binop_mul<APPROX, BinaryOp::MUL, DST_ACCUM_MODE>(idst0, idst1, odst)` via the `MATH()` macro, routing execution to the math RISC-V core.

3. **LLK dispatch** (`llk_math_eltwise_binary_sfpu_binop.h`): `llk_math_eltwise_binary_sfpu_binop_mul` calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_sfpu_binary_mul<APPROX, BinaryOp::MUL, 8, is_fp32_dest_acc_en>, dst_index0, dst_index1, odst, VectorMode::RC)`.

4. **Parameters dispatch** (`llk_math_eltwise_binary_sfpu_params.h`): `_llk_math_eltwise_binary_sfpu_params_` sets up the SFPU start sequence, then in RC mode iterates over 4 faces calling `calculate_sfpu_binary_mul` once per face with `TTI_SETRWC` to advance the DEST register pointer between faces.

5. **Core SFPU** (`ckernel_sfpu_binary.h`): `calculate_sfpu_binary_mul` executes the actual element-wise multiply on 8 rows per face using SFPI vector instructions.

6. **Init path**: `mul_binary_tile_init()` calls `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::MUL>()`, which calls `llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROX>(sfpu_binary_init<APPROX, BinaryOp::MUL>)`. This initializes the SFPU config register, configures ADDR_MOD_7 (all increments = 0), resets counters, and then calls `_sfpu_binary_init_<APPROX, BinaryOp::MUL>()` which is a no-op for MUL (no reciprocal or log initialization needed).

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary.h

sfpi_inline sfpi::vFloat float32_to_bf16_rne(sfpi::vFloat in) {
    sfpi::vUInt bits = sfpi::reinterpret<sfpi::vUInt>(in);
    sfpi::vUInt lsb = (bits >> 16) & 1;   // extract bit 16 (bf16 mantissa LSB) for tie-breaking
    bits = bits + 0x7fffU + lsb;           // RNE: add rounding bias + LSB for even tie-break
    bits = bits & 0xFFFF0000U;             // truncate lower 16 bits to produce bf16-in-fp32
    return sfpi::reinterpret<sfpi::vFloat>(bits);
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS, bool is_fp32_dest_acc_en>
inline void calculate_sfpu_binary_mul(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // APPROXIMATION_MODE=true (from APPROX), BINOP=BinaryOp::MUL, ITERATIONS=8, is_fp32_dest_acc_en=DST_ACCUM_MODE
    constexpr uint dst_tile_size_sfpi = 32; // 64 rows / SFP_DESTREG_STRIDE(2) = 32 sfpi rows per tile
    for (int d = 0; d < ITERATIONS; d++) {
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // SFPLOAD from DEST at input0 tile offset
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // SFPLOAD from DEST at input1 tile offset

        sfpi::vFloat result = in0 * in1; // SFPMUL: element-wise fp32 multiply

        if constexpr (!is_fp32_dest_acc_en) {
            result = float32_to_bf16_rne(result); // software bf16 RNE rounding
            // To match FPU behaviour for bfloat16 multiplication, 0 * x = 0 and x * 0 = 0
            v_if(in0 == 0 || in1 == 0) { result = 0.0f; } // SFPSETCC + SFPPUSHC/SFPPOPC for conditional
            v_endif;
        }

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // SFPSTORE back to DEST at output tile offset
        sfpi::dst_reg++; // advance DEST row pointer by SFP_DESTREG_STRIDE (next row)
    }
}

template <bool APPROXIMATION_MODE, BinaryOp BINOP>
inline void _sfpu_binary_init_() {
    // For MUL: no initialization needed (DIV/POW need reciprocal init, XLOGY needs log init)
}
```

### SFPU Instructions Used

| Instruction / Intrinsic | Description |
|------------------------|-------------|
| `SFPLOAD` (`dst_reg[idx]` read) | Loads a vector of elements from DEST register file into an SFPU LREG. The index is computed as `dst_index * 32` to address the correct tile, with auto-increment per iteration. |
| `SFPMUL` (`in0 * in1`) | Performs element-wise floating-point multiplication of two SFPU vector registers. This is the core operation. |
| `SFPSTORE` (`dst_reg[idx] = result`) | Stores a vector of elements from an SFPU LREG back to the DEST register file at the specified offset. |
| `SFPSHFT` (via `bits >> 16`) | Right-shifts a vector register by an immediate amount. Used in `float32_to_bf16_rne` to extract bit 16. Only executed when `!is_fp32_dest_acc_en`. |
| `SFPAND` (via `& 1`, `& 0xFFFF0000U`) | Bitwise AND on vector registers with immediate. Used in bf16 rounding logic. Only executed when `!is_fp32_dest_acc_en`. |
| `SFPIADD` (via `bits + 0x7fffU + lsb`) | Integer add on vector registers. Used in bf16 RNE rounding bias computation. Only executed when `!is_fp32_dest_acc_en`. |
| `SFPSETCC` (via `v_if(in0 == 0 \|\| in1 == 0)`) | Sets the condition code register based on comparison result (equality to zero). Used for the zero-input special case. Only executed when `!is_fp32_dest_acc_en`. |
| `SFPPUSHC` / `SFPPOPC` (via `v_if`/`v_endif`) | Push/pop condition code stack for predicated execution blocks. Only executed when `!is_fp32_dest_acc_en`. |
| `SFPCOMPC` (via `\|\|` in condition) | Complements condition code for combining multiple conditions in the OR expression. Only executed when `!is_fp32_dest_acc_en`. |
| `SFPLOADI` (via `result = 0.0f`) | Loads an immediate value into an SFPU register. Used to set result to zero in the zero-input special case. Only executed when `!is_fp32_dest_acc_en`. |
| `TTI_SETRWC` | Not an SFPU instruction but a Tensix instruction used by the params dispatch layer to advance the DEST register read/write counter by 8 rows between SFPU calls (face transitions). |
| `TTI_STALLWAIT` | Stalls the math pipeline until SFPU is ready (at start) or waits for SFPU completion (at done). |

### SFPU Register Usage

- **DEST register file**: The primary data source and sink. Two input tiles are loaded into DEST at indices `dst_index_in0` and `dst_index_in1` (via `copy_tile` in the compute kernel before the SFPU runs). The output is written back to DEST at `dst_index_out`. For the no-broadcast case, `dst_index_in0 = i*2`, `dst_index_in1 = i*2+1`, `dst_index_out = i*2`, so the output overwrites the first input's slot.
- **LREG (Local Registers)**: The SFPU has 8 local vector registers (LREG0-LREG7). The SFPI compiler allocates these automatically:
  - `in0`, `in1`, `result` each occupy one LREG for the main computation path.
  - When `!is_fp32_dest_acc_en`, additional LREGs are used for `bits`, `lsb` in the `float32_to_bf16_rne` helper.
- **DEST row pointer**: Auto-incremented by `dst_reg++` (advances by `SFP_DESTREG_STRIDE = 2` rows in DEST) each iteration, processing 8 rows per SFPU call across the 8 ITERATIONS.
- **Condition code stack**: Used for the `v_if(in0 == 0 || in1 == 0)` predicated block when bf16 rounding is active.

### Address Mode Configuration

The address mode is configured in `eltwise_binary_sfpu_configure_addrmod()` during init. For floating-point MUL (where `SfpuType = SfpuType::unused`), only `ADDR_MOD_7` is set:

| Field | Value | Description |
|-------|-------|-------------|
| `srca.incr` | 0 | No auto-increment for source A addressing |
| `srcb.incr` | 0 | No auto-increment for source B addressing |
| `dest.incr` | 0 | No auto-increment for DEST addressing |

The `ADDR_MOD_6` with `dest.incr = 2` is only configured for integer multiplication (`SfpuType::mul_int32`, `SfpuType::mul_uint16`) and min/max operations, not for floating-point MUL.

This configuration is **identical** between Wormhole B0 and Blackhole -- both use the same `eltwise_binary_sfpu_configure_addrmod` template with the same `addr_mod_t` values.

**Note on Wormhole vs Blackhole differences in the params/done functions**: The Wormhole `_llk_math_eltwise_binary_sfpu_start_` calls `math::set_addr_mod_base()` and `_llk_math_eltwise_binary_sfpu_done_` calls `TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU)` followed by `math::clear_addr_mod_base()`. The Blackhole versions omit these addr_mod_base and stall calls, reflecting architectural differences in how SFPU synchronization is handled.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the binary_ng SFPU compute kernel work for multiplication? What compute kernel file does it use, and how does OpConfig determine whether to use SFPU vs FPU path?"
   **Reason**: Needed to understand the dispatch mechanism from program factory to compute kernel files and how MUL is routed to the SFPU path.
   **Key Findings**: OpConfig selects SFPU vs FPU based on `is_sfpu` flag. For SFPU MUL, defines map to `mul_binary_tile_init()` and `mul_binary_tile`. The compute kernel is `eltwise_binary_sfpu_no_bcast.cpp` for the no-broadcast case.

2. **Query**: "How does the SFPU binary multiplication kernel work in the LLK layer? What is the call chain from the compute API through llk_math to the ckernel_sfpu implementation for binary multiply?"
   **Reason**: Needed to trace the full abstraction layer chain from API to core SFPU implementation.
   **Key Findings**: The call chain goes through `llk_math_eltwise_binary_sfpu_binop_mul` -> `_llk_math_eltwise_binary_sfpu_params_` -> `calculate_sfpu_binary_mul`. The MUL-specific variant (`calculate_sfpu_binary_mul`) includes bf16 RNE rounding and zero-input handling that the generic `_calculate_sfpu_binary_` does not.

3. **Query**: "How does the vFloat multiply operator work in SFPI? What SFPU instruction does `in0 * in1` compile to?"
   **Reason**: Needed to map SFPI C++ operators to actual SFPU hardware instructions.
   **Key Findings**: `vFloat * vFloat` compiles to `__builtin_rvtt_sfpmul` which maps to the `SFPMUL` instruction. Conditional execution (`v_if`/`v_endif`) uses `SFPPUSHC`/`SFPSETCC`/`SFPCOMPC`/`SFPPOPC`. `reinterpret` is a zero-cost C++ type cast with no hardware instruction.

### Confluence References
Not consulted for this analysis -- DeepWiki provided sufficient detail on the SFPU instructions used.

### Glean References
Not consulted for this analysis.
