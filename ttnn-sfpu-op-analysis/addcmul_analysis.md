## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the ADDCMUL operation.

The ADDCMUL operation computes: `output = input_a + (value * input_b * input_c)`, where `value` is a scalar constant passed as a runtime argument. Three input tiles are loaded into DST registers, and the SFPU kernel performs the fused multiply-add computation entirely in SFPU local registers.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/addcmul.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_ternary_sfpu_addcmul.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_addcmul.h` |
| **Parameters Dispatch** | `tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_ternary_sfpu_params.h` (in tt_llk submodule) |

### Call Chain

1. **Compute kernel** calls `TERNARY_SFPU_OP_INIT()` which resolves to `addcmul_tile_init()`, and `TERNARY_SFPU_OP_FUNC(0, 1, 2, 0, scalar_arg)` which resolves to `addcmul_tile<DataFormat::Float16_b>(0, 1, 2, 0, scalar_arg)` (or `Float32` variant). These macros are defined via preprocessor defines set in `ternary_op_utils.cpp`.

2. **API Header** (`addcmul.h`): `addcmul_tile<data_format>()` wraps `llk_math_eltwise_ternary_sfpu_addcmul<APPROX, DST_ACCUM_MODE, data_format>()` inside a `MATH()` guard ensuring it only runs on the math RISC-V. `addcmul_tile_init()` wraps `llk_math_eltwise_ternary_sfpu_addcmul_init<APPROX>()`.

3. **LLK Dispatch** (`llk_math_eltwise_ternary_sfpu_addcmul.h`): `llk_math_eltwise_ternary_sfpu_addcmul()` calls `_llk_math_eltwise_ternary_sfpu_params_<APPROXIMATE>()`, passing `sfpu::calculate_addcmul` as the callback along with the four DST indices, vector mode (default `VectorMode::RC`), and the scalar `value`.

4. **Parameters Dispatch** (`llk_math_eltwise_ternary_sfpu_params.h` in tt_llk submodule): `_llk_math_eltwise_ternary_sfpu_params_()` validates DST indices, calls `_llk_math_eltwise_ternary_sfpu_start_()` to configure the SFPU, then invokes the callback 4 times (once per tile face in `VectorMode::RC`), and finally calls `_llk_math_eltwise_ternary_sfpu_done_()`. The init function `_llk_math_eltwise_ternary_sfpu_init_<SfpuType::addcmul>()` calls `eltwise_ternary_sfpu_configure_addrmod<SfpuType::addcmul>()` to set up address modifier registers.

5. **Core SFPU Implementation** (`ckernel_sfpu_addcmul.h`): `calculate_addcmul()` executes the actual SFPU instruction sequence -- loading the scalar, iterating over 8 sub-rows per face, performing multiply and multiply-add, optionally rounding, and storing the result.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_addcmul.h
// (Blackhole version is identical except uses ADDR_MOD_7 instead of ADDR_MOD_3)

namespace ckernel::sfpu {

template <bool APPROXIMATION_MODE, bool is_fp32_dest_acc_en, DataFormat data_format, int ITERATIONS>
inline void calculate_addcmul( // APPROXIMATION_MODE not used (no approximations in this kernel), ITERATIONS=8
    const uint dst_index_in0,  // input_a
    const uint dst_index_in1,  // input_b
    const uint dst_index_in2,  // input_c
    const uint dst_index_out,  // output
    const uint value) {        // scalar value to multiply with input_b
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b || data_format == DataFormat::Bfp8_b,
        "Unsupported data format for calculate_addcmul(). Only Float32, Float16_b (BFloat16), and Bfp8_b (BFloat8B) "
        "are allowed.");

    constexpr InstrModLoadStore mod0 =
        (data_format == DataFormat::Float32) ? InstrModLoadStore::FP32 : InstrModLoadStore::DEFAULT; // FP32 loads 32b as-is; DEFAULT resolves to FP16B based on ALU config
    constexpr uint dst_tile_size = 64; // each tile occupies 64 rows in Dest register file
    // addcmul = input_a + ((value * input_b) * input_c)
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_LOWER, value & 0xFFFF); // Load lower 16 bits of scalar into LREG3
    TT_SFPLOADI(p_sfpu::LREG3, sfpi::SFPLOADI_MOD0_UPPER, value >> 16);    // Load upper 16 bits, completing 32-bit scalar in LREG3
#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_3, dst_index_in1 * dst_tile_size); // LREG1 = input_b[row] from Dest
        TTI_SFPMUL(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LCONST_0, p_sfpu::LREG4, 0); // LREG4 = LREG1 * LREG3 = input_b * value
        TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_3, dst_index_in0 * dst_tile_size); // LREG0 = input_a[row] from Dest
        TT_SFPLOAD(p_sfpu::LREG2, mod0, ADDR_MOD_3, dst_index_in2 * dst_tile_size); // LREG2 = input_c[row] from Dest
        TTI_SFPMAD(p_sfpu::LREG2, p_sfpu::LREG4, p_sfpu::LREG0, p_sfpu::LREG5, 0); // LREG5 = LREG2 * LREG4 + LREG0 = input_c * (input_b * value) + input_a
        TTI_SFPNOP; // Pipeline stall: required on Wormhole to avoid SFPMAD result hazard
        if constexpr (!is_fp32_dest_acc_en) {
            TTI_SFP_STOCH_RND( // Round FP32 result down to FP16A precision before storing to FP16 Dest
                sfpi::SFPSTOCHRND_RND_EVEN,            // Round-to-nearest-even mode
                sfpi::SFPSTOCHRND_MOD1_FP32_TO_FP16A,  // Convert FP32 to FP16A (BFloat16-compatible) format
                0,
                p_sfpu::LREG5,  // source
                p_sfpu::LREG5,  // destination (in-place)
                InstrModLoadStore::FP16A); // Mod0: store format is FP16A
        }
        TT_SFPSTORE(p_sfpu::LREG5, mod0, ADDR_MOD_3, dst_index_out * dst_tile_size); // Store result back to Dest at output tile location
        sfpi::dst_reg++; // Advance Dest row pointer by SFP_DESTREG_STRIDE (2 rows) for next iteration
    }
}
}  // namespace ckernel::sfpu
```

### SFPU Instructions Used

| Instruction | Count per Iteration | Description |
|-------------|-------------------|-------------|
| `SFPLOADI` | 2 (once before loop) | Loads a 16-bit immediate value into an SFPU local register. Used twice to construct the full 32-bit scalar `value` in LREG3 (lower 16 bits then upper 16 bits). |
| `SFPLOAD` | 3 | Moves 32 datums (one row-pair) from the Dest register file into an SFPU local register, with optional FP32/FP16 format conversion controlled by `mod0`. |
| `SFPMUL` | 1 | Lanewise floating-point multiplication: `VD = VA * VB`. Computes `input_b * value`. The `LCONST_0` operand in the VD position of the macro is a placeholder (unused third operand in the hardware encoding). |
| `SFPMAD` | 1 | Lanewise floating-point multiply-add: `VD = VA * VB + VC`. Computes `input_c * (input_b * value) + input_a` in a single instruction. |
| `SFPNOP` | 1 | No-operation pipeline bubble. Required on Wormhole to avoid read-after-write hazards on the SFPMAD result. Blackhole handles this automatically but the NOP is still present in the code. |
| `SFP_STOCH_RND` | 0 or 1 (conditional) | Reduces FP32 mantissa precision to FP16A format using round-to-nearest-even. Only emitted when `is_fp32_dest_acc_en` is false (i.e., Dest is in FP16 mode). Prevents precision loss during the FP32-to-FP16 store. |
| `SFPSTORE` | 1 | Moves 32 datums from an SFPU local register back to the Dest register file, with format conversion controlled by `mod0`. |
| `INCRWC` (via `dst_reg++`) | 1 | Increments the Dest register write counter by `SFP_DESTREG_STRIDE` (2), advancing the row pointer so the next iteration processes the next pair of rows. |

### SFPU Register Usage

| Register | Role | Lifetime |
|----------|------|----------|
| `LREG0` (p_sfpu::LREG0) | Holds `input_a` loaded from Dest | Loaded mid-iteration, consumed by SFPMAD as addend (VC) |
| `LREG1` (p_sfpu::LREG1) | Holds `input_b` loaded from Dest | Loaded at start of iteration, consumed by SFPMUL |
| `LREG2` (p_sfpu::LREG2) | Holds `input_c` loaded from Dest | Loaded mid-iteration, consumed by SFPMAD as first multiplicand (VA) |
| `LREG3` (p_sfpu::LREG3) | Holds the scalar `value` (32-bit float) | Loaded once before the loop via two SFPLOADI instructions; persists across all iterations |
| `LREG4` (p_sfpu::LREG4) | Intermediate: `input_b * value` | Written by SFPMUL, consumed by SFPMAD as second multiplicand (VB) |
| `LREG5` (p_sfpu::LREG5) | Final result: `input_c * (input_b * value) + input_a` | Written by SFPMAD, optionally rounded by SFP_STOCH_RND, then stored to Dest |
| `LCONST_0` | Hardware constant zero register | Used as unused operand placeholder in SFPMUL encoding |
| `dst_reg` (Dest row pointer) | Hardware write counter tracking current row position in Dest | Incremented by 2 each iteration via `dst_reg++`; 8 iterations cover 16 rows = one tile face |

**Dest Register File Layout**: Each tile occupies 64 rows in Dest. The four input/output tiles are addressed at offsets `dst_index * 64`. Within each tile face (16 rows), the SFPU processes 2 rows per iteration (32 elements per row-pair across 32 lanes), so 8 iterations cover one complete face. The params dispatch layer calls `calculate_addcmul` 4 times (once per face) for `VectorMode::RC`.

### Address Mode Configuration

The address mode is configured during initialization by `eltwise_ternary_sfpu_configure_addrmod<SfpuType::addcmul>()`, which is called from `_llk_math_eltwise_ternary_sfpu_init_()`.

**Both Wormhole and Blackhole** configure the same logical address mode with identical field values:

```cpp
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}.set(ADDR_MOD_7);  // No auto-increment for any register
```

All increment fields are zero because the SFPU kernel manages Dest addressing explicitly via `dst_reg++` (which emits `INCRWC` instructions) rather than relying on hardware auto-increment through address modifiers.

**Hardware generation difference in ADDR_MOD index used in instructions**:
- **Wormhole**: The SFPLOAD/SFPSTORE instructions in `ckernel_sfpu_addcmul.h` reference `ADDR_MOD_3`.
- **Blackhole**: The SFPLOAD/SFPSTORE instructions reference `ADDR_MOD_7`.

This difference reflects different address modifier slot conventions between the two architectures. Both are configured with zero increments, so the functional behavior is identical -- no automatic Dest pointer advancement occurs from the address modifier itself.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "Where is `_llk_math_eltwise_ternary_sfpu_params_` defined? What does it do?"
   **Reason**: Needed to understand the params dispatch layer that bridges LLK to the core SFPU function.
   **Key Findings**: Defined in `tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_eltwise_ternary_sfpu_params.h`. It validates DST indices, calls start/done helpers, and invokes the SFPU callback 4 times for VectorMode::RC (once per tile face).

2. **Query**: "Where is `_llk_math_eltwise_ternary_sfpu_init_` defined for SfpuType::addcmul? What ADDR_MOD configuration does it set up?"
   **Reason**: Needed to document the address mode configuration for the operation.
   **Key Findings**: Both architectures configure ADDR_MOD_7 with zero increments for srca, srcb, and dest. The `SfpuType::where` operation additionally configures ADDR_MOD_6 with dest.incr=2, but addcmul does not.

3. **Query**: "Does Wormhole use ADDR_MOD_3 while Blackhole uses ADDR_MOD_7 in ckernel_sfpu_addcmul.h?"
   **Reason**: Noticed a discrepancy between the Wormhole SFPLOAD/SFPSTORE instructions (ADDR_MOD_3) and DeepWiki's claim that both use ADDR_MOD_7 in the init.
   **Key Findings**: DeepWiki could not confirm (file not in its index). Direct source code reading confirmed: Wormhole uses ADDR_MOD_3, Blackhole uses ADDR_MOD_7 in the instruction operands.

4. **Query**: "Explain SFPU instructions: SFPLOADI, SFPLOAD, SFPMUL, SFPMAD, SFPNOP, SFP_STOCH_RND, SFPSTORE"
   **Reason**: Needed detailed instruction semantics for the annotation.
   **Key Findings**: SFPLOADI loads 16-bit immediates into LRegs. SFPLOAD/SFPSTORE move data between Dest and LRegs with format conversion. SFPMUL does lanewise FP multiply. SFPMAD does fused multiply-add. SFPNOP is a pipeline stall. SFP_STOCH_RND reduces mantissa precision with configurable rounding mode.

5. **Query**: "How does dst_reg++ work in SFPI?"
   **Reason**: Needed to understand the Dest row pointer advancement mechanism.
   **Key Findings**: `dst_reg++` emits `__builtin_rvtt_ttincrwc(0, SFP_DESTREG_STRIDE, 0, 0)` which advances the hardware Dest write counter by stride 2, moving to the next row-pair for processing.

### Confluence References
No Confluence references were needed for this analysis. The DeepWiki queries provided sufficient detail on all SFPU instructions used.

### Glean References
No Glean references were needed for this analysis.
