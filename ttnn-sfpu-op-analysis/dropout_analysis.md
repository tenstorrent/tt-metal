## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to. The dropout operation implements element-wise stochastic zeroing with inverted scaling: each element is either zeroed out based on a probability threshold against a hardware PRNG, or multiplied by a scale factor (typically `1 / (1 - p)`).

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/dropout.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` (macro `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS` at line 99, `SFPU_ONE_PARAM_KERNEL_INIT` at line 18) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_dropout.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` (function `_llk_math_eltwise_unary_sfpu_params_`) |

### Call Chain

1. **Compute kernel** calls `dropout_kernel_init(seed)` and `dropout_tile(0, int_probability, int_scale_factor)` from `api/compute/eltwise_unary/dropout.h`.
2. **`dropout_kernel_init(seed)`** expands via the `SFPU_ONE_PARAM_KERNEL_INIT` macro to `llk_math_eltwise_unary_sfpu_init<SfpuType::dropout, APPROX>(sfpu::dropout_init<APPROX>, seed)`, which calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::dropout>()` (configuring ADDR_MOD_7 and resetting counters), then calls `sfpu::dropout_init<APPROX>(seed)` which calls `_init_dropout_(seed)` -> `init_prng_seed(seed)`.
3. **`dropout_tile(0, probability, scale_factor)`** expands via the `SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS` macro to `_llk_math_eltwise_unary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_dropout<APPROX>, 0, (int)VectorMode::RC, probability, scale_factor)`.
4. **`_llk_math_eltwise_unary_sfpu_params_`** sets the DST write address, stalls until SFPU is ready, then calls `calculate_dropout(probability, scale)` once per face (4 times total for `VectorMode::RC`), incrementing the DST face address between calls.
5. **`calculate_dropout`** (in `ckernel_sfpu_dropout.h` at the arch-specific llk_api level) delegates to `_calculate_dropout_<APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, probability, scale)`.
6. **`_calculate_dropout_`** (in `sfpu/ckernel_sfpu_dropout.h`) is the core SFPU microcode that loads parameters into LREGs, then iterates over 8 rows per face: loading each DEST element, scaling it, generating a PRNG random number, comparing against the probability, and conditionally zeroing the result before storing back.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_dropout.h
// (Identical for both Wormhole B0 and Blackhole)

// probability should be between 0 - INT_MAX (signed)
// scale should be binary representation of a float32
template <bool APPROXIMATION_MODE, int ITERATIONS>
inline void _calculate_dropout_(const int iterations, std::uint32_t probability, std::uint32_t scale)
{   // APPROXIMATION_MODE=false, ITERATIONS=8
    // SFPU microcode

    TT_SFPLOADI(p_sfpu::LREG1, 10, scale & 0xFFFF);     // Load low 16 bits of scale into LREG1 (InstrMod=10=LO16_ONLY: preserves upper bits)
    TT_SFPLOADI(p_sfpu::LREG1, 8, scale >> 16);          // Load high 16 bits of scale into LREG1 (InstrMod=8=HI16_ONLY: preserves lower bits)
    TT_SFPLOADI(p_sfpu::LREG2, 10, probability & 0xFFFF); // Load low 16 bits of probability into LREG2
    TT_SFPLOADI(p_sfpu::LREG2, 8, probability >> 16);     // Load high 16 bits of probability into LREG2
#pragma GCC unroll 0
    for (int d = 0; d < iterations; d++)
    {
        // Scale samples: dst_reg[0] = dst_reg[0] * scale
        TTI_SFPLOAD(p_sfpu::LREG0, 0, 3, 0);   // Load current DEST element into LREG0 (InstrMod=0=IMPLIED format, AddrMod=3)
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG1, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // LREG0 = LREG0 * LREG1 (element * scale)

        // Implementation notes, see the original file for more details
        TTI_SFPMOV(0, 9, p_sfpu::LREG3, 8);    // InstrMod=8 reads from RS[VC]; VC=9 is PRNG source -> LREG3 = AdvancePRNG()
        TTI_SFPSETSGN(0, p_sfpu::LREG3, p_sfpu::LREG3, 1); // InstrMod=1: sign bit = Imm12[0]=0, clearing sign bit for unsigned comparison

        // Drop samples: if (rand < probability) then zero the element
        TTI_SFPIADD(0, p_sfpu::LREG2, p_sfpu::LREG3, 10); // InstrMod=10=0b1010: bit[1:0]=2 -> subtract (LREG2 - LREG3), bit[2]=0 -> update CC, bit[3]=1 -> invert CC; CC.Res = !(probability - rand < 0) = (rand < probability)
        TTI_SFPMOV(0, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // Conditionally move 0 into LREG0 (only for lanes where CC says rand < probability)
        TTI_SFPENCC(0, 0, 0, 0);                // InstrMod=0: CC.Res=1 (unconditional), CC.En unchanged (keep=0) -> disables predication
        TTI_SFPSTORE(0, 0, 3, 0);               // Store LREG0 back to DEST (InstrMod=0=IMPLIED format, AddrMod=3)

        sfpi::dst_reg++;  // Advance DEST pointer by SFP_DESTREG_STRIDE=2 rows
    }
}

inline void _init_dropout_(const std::uint32_t seed)
{
    init_prng_seed(seed);  // Writes seed to PRNG_SEED config register, then waits 600 NOPs for PRNG to stabilize
}
```

### SFPU Instructions Used

| Instruction | TTI Macro Used | Description |
|-------------|---------------|-------------|
| **SFPLOADI** | `TT_SFPLOADI` | Loads a 16-bit immediate value into an LREG. Used with InstrMod=10 (LO16_ONLY) and InstrMod=8 (HI16_ONLY) to construct full 32-bit values for `scale` and `probability` in LREG1 and LREG2 respectively. |
| **SFPLOAD** | `TTI_SFPLOAD` | Loads a datum from the DEST register file into LREG0. InstrMod=0 (IMPLIED) uses the default data format. AddrMod=3 selects address mode 3 for DEST addressing. |
| **SFPMUL** | `TTI_SFPMUL` | Lanewise floating-point multiplication. Computes `LREG0 = LREG0 * LREG1` to scale the input element by the dropout scale factor. |
| **SFPMOV** | `TTI_SFPMOV` | With InstrMod=8, reads from the RS (Read-Side) register view. When VC=9 (PRNG source), the hardware PRNG is advanced and a new pseudo-random 32-bit integer is generated per lane and written to LREG3. With InstrMod=0, performs a conditional register copy (used here to zero out LREG0 for dropped lanes). |
| **SFPSETSGN** | `TTI_SFPSETSGN` | Sets the sign bit of the value. With InstrMod=1 and Imm12=0, forces the sign bit to 0, effectively making the PRNG output non-negative for unsigned comparison against probability. |
| **SFPIADD** | `TTI_SFPIADD` | Integer addition/subtraction with CC flag update. With InstrMod=10 (0b1010): performs `LREG2 - LREG3` (probability - rand), updates CC based on result sign, then inverts CC. Net effect: CC.Res=true when `rand < probability`. |
| **SFPENCC** | `TTI_SFPENCC` | Directly manipulates the Condition Code state. With all-zero args (InstrMod=0, Imm12=0): sets CC.Res=1, keeps CC.En unchanged. This effectively disables predicated execution so the subsequent SFPSTORE writes unconditionally. |
| **SFPSTORE** | `TTI_SFPSTORE` | Stores LREG0 back to the DEST register file. InstrMod=0 (IMPLIED) format. AddrMod=3 for DEST addressing. |
| **SFPNOP** | `TTI_SFPNOP` | No-operation. Used 600 times in `init_prng_seed` to wait for the PRNG seed to propagate through the hardware before use. |

### SFPU Register Usage

| Register | Role |
|----------|------|
| **LREG0** | Working register. Holds the current DEST element loaded via SFPLOAD, then holds the scaled result (after SFPMUL), and finally holds either the scaled value or zero (after conditional SFPMOV). Stored back to DEST via SFPSTORE. |
| **LREG1** | Holds the `scale` parameter (float32, bit-cast from uint32). Loaded once before the loop via two SFPLOADI instructions (low 16 + high 16 bits). Read-only during the iteration loop. |
| **LREG2** | Holds the `probability` parameter (uint32, range 0 to INT_MAX). Loaded once before the loop via two SFPLOADI instructions. Used as the comparison threshold in SFPIADD. |
| **LREG3** | Holds the PRNG output. Written by SFPMOV (InstrMod=8, VC=9), then sign-cleared by SFPSETSGN, then consumed by SFPIADD for comparison. Overwritten each iteration. |
| **LCONST_0** | Constant zero register. Used as the source for conditional zeroing (SFPMOV into LREG0) and as the unused operand in SFPMUL (the `lreg_c` operand for the MAD instruction's addend, effectively making it a pure multiply). |
| **DEST registers** | The tile data in the DEST register file. Each iteration of the inner loop processes one row (2 DEST addresses). The `dst_reg++` at the end of each iteration advances by stride 2. Over 8 iterations per face and 4 faces per tile, all 64 rows (32x32 tile = 4 faces x 16 rows, but SFPU processes 32 lanes wide so 8 iterations x 4 faces = 32 row-pairs) are processed. |
| **PRNG_SEED config register** | Hardware configuration register at `PRNG_SEED_Seed_Val_ADDR32`. Written once during `init_prng_seed(seed)` to seed the per-lane PRNG. |
| **CC (Condition Code)** | Per-lane condition code with CC.En (enable) and CC.Res (result) bits. SFPIADD sets CC.Res per lane based on `rand < probability`. This enables predicated execution of SFPMOV (zeroing). SFPENCC then disables predication for the unconditional SFPSTORE. |

### Address Mode Configuration

The dropout operation uses `SfpuType::dropout`, which does **not** match any of the special-case `constexpr if` branches in `eltwise_unary_sfpu_configure_addrmod`. Therefore, only the default `ADDR_MOD_7` is configured:

```cpp
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
    .set(ADDR_MOD_7);
```

All increment fields are zero. This means the ADDR_MOD does not auto-increment any register addresses between SFPU iterations. Instead, DEST pointer advancement is handled explicitly by `sfpi::dst_reg++` (which increments by `SFP_DESTREG_STRIDE=2`) at the end of each loop iteration, and by `_llk_math_eltwise_unary_sfpu_inc_dst_face_addr_()` (which calls `math::inc_dst_addr<8>()` twice, advancing by 16) between faces.

The SFPLOAD and SFPSTORE instructions in the kernel use `AddrMod=3`, which references the address mode register used for DEST addressing during those instructions.

This configuration is **identical for both Wormhole B0 and Blackhole** architectures. The only minor difference between the two architectures' `_llk_math_eltwise_unary_sfpu_start_` functions is that Wormhole B0 additionally calls `math::set_addr_mod_base()` and its `_done_` function calls `math::clear_addr_mod_base()` with a STALL_CFG/WAIT_SFPU stall, whereas Blackhole omits these calls.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the dropout SFPU kernel work? Where is the dropout_tile and dropout_kernel_init API defined?"
   **Reason**: Needed to trace the full call chain from the compute API through LLK to the core SFPU implementation.
   **Key Findings**: Identified the API header (`dropout.h`), the macro expansion path (`SFPU_UNARY_PARAMS_KERNEL_EXTRA_ARGS` and `SFPU_ONE_PARAM_KERNEL_INIT`), and the core implementation in `sfpu/ckernel_sfpu_dropout.h`. Confirmed that `_calculate_dropout_` implements the PRNG-based stochastic zeroing with scaling.

2. **Query**: "Explain the SFPU instructions SFPLOADI, SFPLOAD, SFPMUL, SFPMOV (especially with instr_mod1=8 for PRNG), SFPSETSGN, SFPIADD (with instr_mod1=10 for comparison), SFPENCC, SFPSTORE"
   **Reason**: Needed precise semantics for each SFPU instruction used in the dropout kernel.
   **Key Findings**: SFPMOV with InstrMod=8 reads from the RS (Read-Side) register view, and when VC=9, it triggers PRNG advancement. SFPIADD with InstrMod bits controls addition/subtraction, CC update, and CC inversion. SFPENCC directly manipulates CC state.

3. **Query**: "How does dst_reg++ work in SFPI? What does it do to the DEST register pointer?"
   **Reason**: Needed to understand the DEST pointer advancement mechanism used in the SFPU kernel loop.
   **Key Findings**: `dst_reg++` increments the DEST write counter by `SFP_DESTREG_STRIDE=2` via the `__builtin_rvtt_ttincrwc` intrinsic.

### Confluence References
- **Page**: Tensix SFPU Instruction Set Architecture (Page ID: 1170505767)
- **Sections consulted**:
  - **SFPMOV**: Confirmed algorithmic implementation showing that InstrMod=8 reads from RS[VC], and when VC=9 it calls `AdvancePrng()` before the read.
  - **SFPIADD**: Confirmed InstrMod bit field layout: `[3]=CC inversion`, `[2]=CC update control`, `[1:0]=operation select (0=add, 1=imm add, 2=subtract)`. InstrMod=10=0b1010 means subtract with CC inversion.
  - **SFPENCC**: Confirmed that InstrMod=0 with Imm12=0 sets CC.Res=1 and keeps CC.En unchanged.
  - **SFPSETSGN**: Confirmed that InstrMod[0]=1 uses Imm12[0] as the new sign bit. With Imm12=0, this clears the sign bit.
  - **SFPLOADI**: Confirmed InstrMod=10 (LO16_ONLY) preserves upper 16 bits while writing lower 16, and InstrMod=8 (HI16_ONLY) preserves lower 16 bits while writing upper 16.
  - **SFPSTORE**: Confirmed InstrMod=0 (IMPLIED) uses the default data format for the store.

### Glean References
No Glean queries were needed for this analysis. All required information was available through DeepWiki and the Confluence SFPU ISA page.
