## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the GCD (greatest common divisor) binary SFPU operation.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/gcd.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_gcd.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. **Compute kernel** (`eltwise_binary_sfpu_kernel.cpp`): For each tile, the `BINOP_INIT` define expands to `gcd_tile_init();` (called once per tile), and `BINARY_SFPU_OP` expands to `gcd_tile(i * 2, i * 2 + 1, i * 2);` where operand A is at DST index `i*2`, operand B at `i*2+1`, and the output overwrites `i*2`.
2. **API Header** (`gcd.h`): `gcd_tile(idst0, idst1, odst)` calls `llk_math_eltwise_binary_sfpu_gcd<APPROX>(idst0, idst1, odst)` within a `MATH(...)` guard. `gcd_tile_init()` calls `llk_math_eltwise_binary_sfpu_gcd_init<APPROX>()`.
3. **LLK Dispatch** (`llk_math_eltwise_binary_sfpu_gcd.h`): The init function calls `llk_math_eltwise_binary_sfpu_init<SfpuType::gcd, APPROXIMATE>(sfpu::calculate_sfpu_gcd_init)`, which first runs `_llk_math_eltwise_binary_sfpu_init_<SfpuType::gcd>()` (configures SFPU config register, ADDR_MOD, resets counters) and then calls the `calculate_sfpu_gcd_init` function to record the replay buffer. The operation function calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(sfpu::calculate_sfpu_gcd, dst_index0, dst_index1, odst, VectorMode::RC)`.
4. **Parameters Dispatch** (`llk_math_eltwise_binary_sfpu_params.h`): Iterates over 4 faces (in RC mode), calling `calculate_sfpu_gcd(dst_index_in0, dst_index_in1, dst_index_out)` for each face, with `TTI_SETRWC` to advance the DEST register write pointer by 16 rows between faces.
5. **Core SFPU Implementation** (`ckernel_sfpu_gcd.h`): `calculate_sfpu_gcd` loads operands from DEST into LREG0/LREG1, calls `calculate_sfpu_gcd_body<31>()` which implements the binary GCD algorithm using 30 iterations of a 7-instruction replay loop, then stores the result (LREG1) back to DEST.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h
// NOTE: Blackhole and Wormhole B0 implementations are identical.

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <int max_input_bits = 31>
inline void calculate_sfpu_gcd_body() {
    // --- Preamble: compute trailing zero count of (a | b) ---
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0); // c = a
    TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG2, 0); // c |= b

    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG3, 0); // d = c
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // d = -d (two's complement negate)
    TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG3, 0); // d &= c (isolate lowest set bit)
    TTI_SFPLZ(0, p_sfpu::LREG3, p_sfpu::LREG3, 0); // d = clz(d) => gives bit position of common factor of 2

    // --- Ensure b is odd: if b's LSB is zero, swap a and b ---
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG2, SFPSHFT2_MOD1_SHFT_LREG); // c = b << d (shift left by clz, tests if b has trailing zeros matching d)
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, 6); // set CC if c == 0 (meaning b is even, has all its lowest bits as zero); mode 6 = set CC based on VC == 0
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 0); // conditional swap(a, b) — only executes in lanes where CC is set
    TTI_SFPENCC(0, 0, 0, 0); // disable conditional execution (clear lane enable flags)
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 0); // a = abs(a)
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, 0); // b = abs(b)

    // --- Prepare for replay loop: negate a and d ---
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // a = -a (negate for subtraction in loop)
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // d = -d (negate for right-shift in loop)

    // --- Replay the recorded 7-instruction loop body ---
    int iterations = max_input_bits - 1; // 30 for 31-bit inputs

    #pragma GCC unroll 7
    while (iterations / 4 > 0) {
        TTI_REPLAY(0, 7 * 4, 0, 0); // replay 4 iterations (28 instructions) from replay buffer, load_mode=0 (execute)
        iterations -= 4;
    }

    // Implementation notes, see the original file for more details
    TTI_REPLAY(0, 7 * iterations - 1, 0, 0); // replay remaining 2 iterations minus 1 instruction (13 instructions)

    TTI_SFPENCC(0, 0, 0, 0); // disable conditional execution
}

template <int ITERATIONS = 8>
inline void calculate_sfpu_gcd(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Binary GCD algorithm. ITERATIONS=8 processes 8 rows per face (32 rows / 4 rows per SFPU pass = 8).
    for (int d = 0; d < ITERATIONS; d++) {
        constexpr uint dst_tile_size = 64; // each tile occupies 64 rows in DEST

        TT_SFPLOAD(p_sfpu::LREG0, 4, 3, dst_index_in0 * dst_tile_size);  // a: load from DEST as SMAG32 (InstrMod=4), AddrMod=3
        TT_SFPLOAD(p_sfpu::LREG1, 4, 3, dst_index_in1 * dst_tile_size);  // b: load from DEST as SMAG32 (InstrMod=4), AddrMod=3

        calculate_sfpu_gcd_body<31>();

        TT_SFPSTORE(p_sfpu::LREG1, 4, 3, dst_index_out * dst_tile_size); // store result (b holds GCD) to DEST as SMAG32
        dst_reg++; // advance DEST row pointer for next SFPU pass
    }
}

inline void calculate_sfpu_gcd_init() {
    // Record the 7-instruction loop body into the replay buffer.
    // TTI_REPLAY with load_mode=1 starts recording subsequent instructions.
    TTI_REPLAY(0, 7 * 4, 0, 1); // start_idx=0, len=28, execute_while_loading=0, load_mode=1 (record)
    #pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
        // Each iteration of the binary GCD loop:
        // Given: LREG0 = -a (negative), LREG1 = b (positive, odd), LREG3 = -d (negative shift count)
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG2, 0); // LREG2 = abs(-a) = +a
        TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG0, 0); // LREG0 = (-a) & (+a) = isolate LSB of a (lowest set bit)
        TTI_SFPLZ(0, p_sfpu::LREG0, p_sfpu::LREG0, SFPLZ_MOD1_CC_NE0); // LREG0 = clz(LSB), also disables lanes where a == 0 (GCD already found)
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE); // LREG0 += d (adjust shift count: clz(LSB) + (-d) gives total right-shift to strip trailing zeros)
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG0, SFPSHFT2_MOD1_SHFT_LREG); // LREG0 = a >> -LREG0, making a odd
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, SFPSWAP_MOD1_VEC_MIN_MAX); // ensure b <= a: LREG1 = min(a, b), LREG0 = max(a, b)
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // a = b - a (result is even since both are odd, so it becomes the new a to strip factors of 2)
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `SFPMOV` | Copies a value from one LREG to another: `VD = VC`. Used to copy operands into temporary registers. |
| `SFPOR` | Bitwise OR: `VD = VB \| VC`. Used to compute `a \| b` to find common trailing zeros. |
| `SFPAND` | Bitwise AND: `VD = VB & VC`. Used to isolate the lowest set bit via `x & (-x)`. |
| `SFPIADD` | Two's complement integer add/subtract. With `SFPIADD_MOD1_ARG_2SCOMP_LREG_DST`, it negates the destination before adding (`VD = VC - VD`). With `SFPIADD_MOD1_CC_NONE`, it does not modify condition codes. Used for negation (adding 0 with 2's complement flag) and for computing `b - a`. |
| `SFPLZ` | Count leading zeros: `VD = clz(VC)`. With `SFPLZ_MOD1_CC_NE0`, it also sets per-lane CC flags to disable lanes where `VC == 0` (i.e., GCD computation is complete for that lane). |
| `SFPSHFT2` | Bitwise shift with `SFPSHFT2_MOD1_SHFT_LREG` mode: if shift amount (VC) is non-negative, left-shifts VB; if negative, right-shifts VB. Used to strip trailing zeros from operands. |
| `SFPSETCC` | Sets per-lane condition code flags. Mode 6 tests `VC == 0` (among other conditions based on Mod1). Used to conditionally swap operands when b is even. |
| `SFPSWAP` | Without modifier: conditional swap of two registers (only in lanes where CC is enabled). With `SFPSWAP_MOD1_VEC_MIN_MAX`: unconditional min/max assignment: `VD = min(VC, VD)`, `VC = max(VC, VD)`. |
| `SFPENCC` | Enables/disables per-lane conditional execution. Called with 0 to disable conditional execution (re-enable all lanes). |
| `SFPABS` | Absolute value: `VD = abs(VC)`. Used to ensure operands are positive and to recover `+a` from `-a` in the loop body. |
| `SFPLOAD` | Loads a value from the DEST register file into an LREG. InstrMod=4 (SMAG32) loads as signed-magnitude 32-bit integer. AddrMod=3 controls RWC-based address calculation. |
| `SFPSTORE` | Stores a value from an LREG back to the DEST register file. Same format and addressing as SFPLOAD. |
| `REPLAY` | Hardware instruction replay mechanism. With `load_mode=1`, records subsequent instructions into a replay buffer. With `load_mode=0`, replays the recorded instructions. This avoids repeated instruction fetch overhead for the tight inner loop. |

### SFPU Register Usage

| Register | Role |
|----------|------|
| **LREG0** | Holds operand `a` (first input). During the loop, holds `-a` (negated for efficient subtraction). After stripping trailing zeros, holds the odd part of `a`. |
| **LREG1** | Holds operand `b` (second input). After the algorithm completes, LREG1 contains the GCD result (the surviving non-zero value). |
| **LREG2** | Temporary register. Used for intermediate computations: `c = a \| b`, then `abs(a)` in the loop body. |
| **LREG3** | Holds `d`, the count of common trailing zeros (negated as `-d` for use as a right-shift amount). Also used as a temporary for leading zero count. |
| **LCONST_0** | Constant register holding 0. Used with `SFPIADD` to perform negation (`0 - VD`). |
| **DEST** | Source/destination register file. Tiles are loaded from DEST into LREGs at the start of each face iteration and stored back at the end. Each tile occupies 64 rows in DEST. Two input tiles are interleaved at `dst_index * 64` offsets. |
| **dst_reg** | Internal SFPU address counter, incremented by `dst_reg++` after each of the 8 iterations to advance through the 4 rows processed per SFPU pass. |
| **CC (Condition Codes)** | Per-lane flags used for two purposes: (1) in the preamble, `SFPSETCC` enables conditional swap to ensure b is odd; (2) in the loop, `SFPLZ` with `CC_NE0` disables lanes where a has become 0, meaning GCD is found for that lane. |

### Address Mode Configuration

The GCD operation uses `SfpuType::gcd` which falls through to the default ADDR_MOD configuration in `eltwise_binary_sfpu_configure_addrmod` (it is not in the special-cased list for `mul_int32`, `max`, `min`, etc.).

**ADDR_MOD_7** is configured with all-zero increments:

```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
    .set(ADDR_MOD_7);
```

This means no automatic DEST address increment happens between SFPU instructions via ADDR_MOD. Instead, the DEST address advancement is handled explicitly:
- Within `calculate_sfpu_gcd`, `dst_reg++` advances the row pointer after each of the 8 SFPU passes per face.
- Between faces, `_llk_math_eltwise_binary_sfpu_params_` issues `TTI_SETRWC` instructions to advance the DEST pointer by 16 rows (two increments of 8).

The SFPLOAD/SFPSTORE instructions use `AddrMod=3` (ADDR_MOD_3), which is the standard SFPU addressing mode for direct DEST access with the base address provided in the instruction operand plus the RWC offset.

**Wormhole B0 vs Blackhole**: The ADDR_MOD configuration is identical between architectures. The only difference in the binary SFPU infrastructure is that Wormhole B0's `_llk_math_eltwise_binary_sfpu_start_` calls `math::set_addr_mod_base()` and `_llk_math_eltwise_binary_sfpu_done_` calls `math::clear_addr_mod_base()` with an additional `TTI_STALLWAIT` for SFPU completion, while Blackhole omits these calls.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How is the GCD (greatest common divisor) SFPU binary operation implemented?"
   **Reason**: To locate the file paths and understand the overall structure of the GCD SFPU kernel implementation.
   **Key Findings**: GCD is implemented as a binary GCD algorithm using SFPU instructions. The implementation lives in `ckernel_sfpu_gcd.h` with LLK wrappers in `llk_math_eltwise_binary_sfpu_gcd.h`. The LCM operation reuses the GCD body with `calculate_sfpu_gcd_body<15>()`.

2. **Query**: "What do the SFPU instructions SFPMOV, SFPOR, SFPAND, SFPIADD, SFPLZ, SFPSHFT2, SFPSETCC, SFPSWAP, SFPENCC, SFPABS do?"
   **Reason**: To document the precise semantics of each SFPU instruction used in the GCD kernel.
   **Key Findings**: Detailed descriptions of all instructions obtained. Key findings: SFPSWAP with VEC_MIN_MAX performs simultaneous min/max assignment; SFPLZ counts leading zeros; SFPSETCC mode 6 tests equality to zero; SFPSHFT2 with SHFT_LREG performs variable-amount shifts where negative amounts become right shifts.

### Confluence References
- **Tensix SFPU Instruction Set Architecture** (Page ID: 1170505767): Consulted for SFPLOAD InstrMod field descriptions. Confirmed InstrMod=4 corresponds to SMAG32 (signed-magnitude 32-bit integer format), which loads/stores 32-bit values from DEST without floating-point conversion. Also referenced for AddrMod and RWC (Register Word Counter) behavior.

### Glean References
No Glean queries were necessary for this analysis. The DeepWiki and Confluence sources provided sufficient detail for all SFPU instructions and register behaviors used in the GCD kernel.
