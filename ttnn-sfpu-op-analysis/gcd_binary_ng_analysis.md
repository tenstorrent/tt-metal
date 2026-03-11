## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the GCD (Greatest Common Divisor) binary operation. The GCD operation requires INT32 inputs and implements the binary GCD algorithm entirely in SFPU integer arithmetic, making heavy use of bitwise operations, conditional swaps, and the REPLAY hardware mechanism for loop unrolling.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/gcd.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_gcd.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h` |
| **Parameters Dispatch** | `llk_math_eltwise_binary_sfpu.h` (in tt_llk submodule) -- provides `_llk_math_eltwise_binary_sfpu_params_` and `_llk_math_eltwise_binary_sfpu_init_` |

### Call Chain

1. The compute kernel (`eltwise_binary_sfpu.cpp`) invokes `BINARY_SFPU_INIT` which expands to `gcd_tile_init();` and `BINARY_SFPU_OP(i*2, i*2+1, i*2)` which expands to `gcd_tile(i*2, i*2+1, i*2)`. These defines are injected at compile time from `binary_ng_utils.cpp`.

2. `gcd_tile_init()` (in `api/compute/gcd.h`) calls `llk_math_eltwise_binary_sfpu_gcd_init<APPROX>()`, which calls `llk_math_eltwise_binary_sfpu_init<SfpuType::gcd, APPROXIMATE>(sfpu::calculate_sfpu_gcd_init)`. This runs `_llk_math_eltwise_binary_sfpu_init_<SfpuType::gcd>()` (configures ADDR_MOD_7 and resets counters) and then calls `calculate_sfpu_gcd_init()` to record the 7-instruction replay buffer.

3. `gcd_tile(idst0, idst1, odst)` (in `api/compute/gcd.h`) calls `llk_math_eltwise_binary_sfpu_gcd<APPROX>(idst0, idst1, odst)`, which calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(sfpu::calculate_sfpu_gcd, idst0, idst1, odst, VectorMode::RC)`. This starts the SFPU, iterates over all 4 tile faces (each 8 rows of 32 lanes), calling `calculate_sfpu_gcd(dst_index_in0, dst_index_in1, dst_index_out)` per face, then finalizes.

4. `calculate_sfpu_gcd` loads two tiles from DEST into LREG0 and LREG1, calls `calculate_sfpu_gcd_body<31>()` which executes the binary GCD algorithm, then stores LREG1 (result) back to DEST.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_gcd.h

template <int max_input_bits = 31>
inline void calculate_sfpu_gcd_body() {
    TTI_SFPMOV(0, p_sfpu::LREG0, p_sfpu::LREG2, 0); // c = a
    TTI_SFPOR(0, p_sfpu::LREG1, p_sfpu::LREG2, 0); // c |= b

    TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG3, 0); // d = c
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // d = -d (two's complement negate)
    TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG3, 0); // d &= c (isolate lowest set bit via d & (-d))
    TTI_SFPLZ(0, p_sfpu::LREG3, p_sfpu::LREG3, 0); // d = clz(d), count leading zeros to get bit position

    // Ensure that b is odd: if LSB is zero, then swap with a.
    TTI_SFPSHFT2(p_sfpu::LREG1, p_sfpu::LREG3, p_sfpu::LREG2, SFPSHFT2_MOD1_SHFT_LREG); // c = b << d (shift by LREG amount)
    TTI_SFPSETCC(0, p_sfpu::LREG2, 0, 6); // set CC if c == 0 (mod1=6: two's complement int == 0)
    TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, 0); // conditionally swap(a, b) on active CC lanes (mod1=0: plain swap)
    TTI_SFPENCC(0, 0, 0, 0); // clear all condition codes, re-enable all lanes
    TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG0, 0); // a = abs(a)
    TTI_SFPABS(0, p_sfpu::LREG1, p_sfpu::LREG1, 0); // b = abs(b)

    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // a = -a (negate for subtraction in replay loop)
    TTI_SFPIADD(0, p_sfpu::LCONST_0, p_sfpu::LREG3, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // d = -d (negate shift amount for right-shift in replay)

    int iterations = max_input_bits - 1; // 30 iterations for 31-bit inputs

    #pragma GCC unroll 7
    while (iterations / 4 > 0) {
        TTI_REPLAY(0, 7 * 4, 0, 0); // replay 4 iterations of the 7-instruction recorded sequence
        iterations -= 4;
    }

    // Implementation notes, see the original file for more details
    TTI_REPLAY(0, 7 * iterations - 1, 0, 0); // replay remaining 2 iterations minus 1 instruction (last op only affects a)

    TTI_SFPENCC(0, 0, 0, 0); // clear all condition codes
}

template <int ITERATIONS = 8>
inline void calculate_sfpu_gcd(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // Binary GCD algorithm.
    for (int d = 0; d < ITERATIONS; d++) {
        // size of each tile in Dest is 64 rows
        constexpr uint dst_tile_size = 64;

        TT_SFPLOAD(p_sfpu::LREG0, 4, 3, dst_index_in0 * dst_tile_size);  // a: load from DEST[in0] as int32 (addr_mode=4, instr_mod=3)
        TT_SFPLOAD(p_sfpu::LREG1, 4, 3, dst_index_in1 * dst_tile_size);  // b: load from DEST[in1] as int32

        calculate_sfpu_gcd_body<31>();

        TT_SFPSTORE(p_sfpu::LREG1, 4, 3, dst_index_out * dst_tile_size); // store result (b holds GCD) to DEST[out] as int32
        dst_reg++; // advance to next row block within the face
    }
}

inline void calculate_sfpu_gcd_init() {
    TTI_REPLAY(0, 7 * 4, 0, 1); // record mode: capture next 28 instructions into replay buffer
    #pragma GCC unroll 4
    for (int i = 0; i < 4; ++i) {
        // Each iteration of the replay body is 7 instructions:
        // We store {-a, a} in {LREG0, LREG2} for LSB isolation
        TTI_SFPABS(0, p_sfpu::LREG0, p_sfpu::LREG2, 0); // LREG2 = +a (abs of negated a)
        TTI_SFPAND(0, p_sfpu::LREG2, p_sfpu::LREG0, 0); // LREG0 = (-a) & (+a) = isolate LSB of a
        TTI_SFPLZ(0, p_sfpu::LREG0, p_sfpu::LREG0, SFPLZ_MOD1_CC_NE0); // LREG0 = clz(LSB), set CC where a != 0 (mod1=2: disable lanes where a==0)
        TTI_SFPIADD(0, p_sfpu::LREG3, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE); // LREG0 += d (add negated shift to clz result)
        TTI_SFPSHFT2(p_sfpu::LREG2, p_sfpu::LREG0, p_sfpu::LREG0, SFPSHFT2_MOD1_SHFT_LREG); // LREG0 = a >> -LREG0, making a odd
        TTI_SFPSWAP(0, p_sfpu::LREG0, p_sfpu::LREG1, SFPSWAP_MOD1_VEC_MIN_MAX); // ensure b <= a (mod1=1: VD=min, VC=max)
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, SFPIADD_MOD1_CC_NONE | SFPIADD_MOD1_ARG_2SCOMP_LREG_DST); // a = b - a (result is even since both were odd)
    }
}
```

### SFPU Instructions Used

| Instruction | Description |
|---|---|
| **SFPLOAD** (`TT_SFPLOAD`) | Loads a 32-element vector from DEST register file into an SFPU local register (LREG). Parameters: addr_mode=4 and instr_mod=3 select raw int32 format. |
| **SFPSTORE** (`TT_SFPSTORE`) | Stores a 32-element vector from an SFPU local register back to DEST register file. Same addr_mode/instr_mod as SFPLOAD for int32. |
| **SFPMOV** (`TTI_SFPMOV`) | Copies the contents of one LREG to another. When mod1=0, performs a plain copy VD = VC. |
| **SFPOR** (`TTI_SFPOR`) | Bitwise OR: VD = VB \| VC across all 32 lanes. Used to compute `a | b` for shared factor extraction. |
| **SFPAND** (`TTI_SFPAND`) | Bitwise AND: VD = VB & VC across all 32 lanes. Used to isolate the lowest set bit via `x & (-x)`. |
| **SFPIADD** (`TTI_SFPIADD`) | Integer addition/subtraction on SFPU registers. With `SFPIADD_MOD1_ARG_2SCOMP_LREG_DST`, it negates the destination before adding (effectively: VD = VC - VD). With `SFPIADD_MOD1_CC_NONE`, condition codes are not modified. When both flags are set with `LCONST_0`, it computes VD = 0 - VD (negate). |
| **SFPLZ** (`TTI_SFPLZ`) | Count Leading Zeros: VD = CLZ(VC). With `SFPLZ_MOD1_CC_NE0` (mod1=2), it additionally sets per-lane condition codes to disable lanes where the input is zero. |
| **SFPSHFT2** (`TTI_SFPSHFT2`) | Bitwise shift with `SFPSHFT2_MOD1_SHFT_LREG` mode: VD = VB << VC (if VC positive) or VD = VB >> (-VC) (if VC negative). The shift amount comes from LREG rather than an immediate. |
| **SFPSETCC** (`TTI_SFPSETCC`) | Sets per-lane condition codes based on a comparison. With mod1=6 (two's complement integer equality check): sets CC for lanes where VC == 0. |
| **SFPSWAP** (`TTI_SFPSWAP`) | With mod1=0: conditionally swaps VD and VC on lanes where CC is set. With `SFPSWAP_MOD1_VEC_MIN_MAX` (mod1=1): unconditionally sets VD=min(VD,VC) and VC=max(VD,VC) using sign-magnitude comparison. |
| **SFPENCC** (`TTI_SFPENCC`) | Clears all per-lane condition codes, re-enabling all 32 lanes for subsequent operations. |
| **SFPABS** (`TTI_SFPABS`) | Computes absolute value: VD = abs(VC) across all 32 lanes. |
| **REPLAY** (`TTI_REPLAY`) | Hardware instruction replay mechanism. With last parameter=1: records the next N instructions into a replay buffer. With last parameter=0: replays N instructions from the buffer. The second parameter specifies the count of instructions to record or replay. |

### SFPU Register Usage

| Register | Role in GCD Kernel |
|---|---|
| **LREG0** | Holds operand `a` throughout the algorithm. Loaded from `DEST[dst_index_in0]`. In the replay loop body, holds `-a` (negated) for the subtraction step. |
| **LREG1** | Holds operand `b` throughout the algorithm. Loaded from `DEST[dst_index_in1]`. At algorithm completion, holds the GCD result. Stored back to `DEST[dst_index_out]`. |
| **LREG2** | Temporary register (`c`). Used for intermediate calculations: `a | b`, absolute value of `a`, and shifted values for evenness checks. |
| **LREG3** | Temporary register (`d`). Stores the shift count derived from the common trailing zeros of `a | b`. Negated for use as a right-shift amount in the replay loop. |
| **LCONST_0** | Constant zero register. Used as the source operand in SFPIADD to perform negation (0 - VD). |
| **DEST** | Tile destination register file (64 rows per tile). Tiles at indices `dst_index_in0`, `dst_index_in1` are read; result is written to `dst_index_out`. The `dst_reg` counter advances through 8 rows per SFPU iteration (ITERATIONS=8 covers all 64 rows of a face via 8 passes of 8 rows). |
| **CC (Condition Codes)** | Per-lane flags used to conditionally execute SFPSWAP. Set by SFPSETCC (lanes where b is even) and SFPLZ with CC_NE0 flag (lanes where a != 0). Cleared by SFPENCC. |

### Address Mode Configuration

The `_llk_math_eltwise_binary_sfpu_init_` function calls `eltwise_binary_sfpu_configure_addrmod<SfpuType::gcd>()`. Since GCD is not in the special list of operations that configure ADDR_MOD_6 (that list includes only mul_int32, mul_uint16, max, min, max_int32, min_int32, max_uint32, min_uint32), only **ADDR_MOD_7** is configured:

| Address Mode | Field | Value |
|---|---|---|
| **ADDR_MOD_7** | `srca.incr` | 0 |
| **ADDR_MOD_7** | `srcb.incr` | 0 |
| **ADDR_MOD_7** | `dest.incr` | 0 |

All increments are zero because the SFPU kernel manages DEST addressing explicitly through `TT_SFPLOAD`/`TT_SFPSTORE` with computed offsets (`dst_index * dst_tile_size`) and the `dst_reg++` counter that advances through rows within each face. The `_llk_math_eltwise_binary_sfpu_params_` function handles face-to-face advancement using `TTI_SETRWC` instructions (advancing by 8 rows per face transition).

This configuration is identical for Wormhole B0 and Blackhole -- the `eltwise_binary_sfpu_configure_addrmod` template produces the same ADDR_MOD_7 settings for both architectures, and the GCD ckernel source code is identical across both.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "What does _llk_math_eltwise_binary_sfpu_params_ do? Show its full implementation."
   **Reason**: Needed to understand the dispatch layer between the LLK API and the core SFPU function, including how tile faces are iterated and how dst indices are passed.
   **Key Findings**: The function starts SFPU, iterates 4 faces for VectorMode::RC, calls sfpu_func with dst indices per face, uses TTI_SETRWC to advance by 8 rows between faces, then finalizes SFPU.

2. **Query**: "Explain SFPU instructions: SFPMOV, SFPOR, SFPAND, SFPIADD, SFPLZ, SFPSHFT2, SFPSETCC, SFPSWAP, SFPENCC, SFPABS, SFPLOAD, SFPSTORE, REPLAY"
   **Reason**: Needed detailed semantics of each SFPU instruction used in the GCD kernel.
   **Key Findings**: Comprehensive descriptions of each instruction's operation, operand roles, and modifier flags. SFPSWAP has plain swap (mod1=0) and min/max (mod1=1) modes. SFPSETCC mod1=6 checks two's complement integer equality to zero. SFPLZ computes CLZ with optional CC setting.

3. **Query**: "Explain SFPSWAP instruction in detail, mod1=0, SFPSWAP_MOD1_VEC_MIN_MAX"
   **Reason**: The GCD kernel uses both plain swap (mod1=0, conditional on CC) and min/max swap (mod1=1) modes, needed to understand the distinction.
   **Key Findings**: mod1=0 performs unconditional swap of VD and VC. However, when condition codes are active (set by prior SFPSETCC), the swap only executes on enabled lanes. mod1=1 (VEC_MIN_MAX) sets VD=min, VC=max unconditionally using sign-magnitude comparison.

4. **Query**: "Explain the REPLAY instruction (TTI_REPLAY)"
   **Reason**: The GCD kernel makes critical use of REPLAY for its main loop, needed to understand recording vs playback.
   **Key Findings**: DeepWiki did not have detailed documentation. From code analysis: last parameter=1 starts recording the next N instructions; last parameter=0 replays N instructions from the buffer. The second parameter is the instruction count.

5. **Query**: "What is SFPSWAP_MOD1_VEC_MIN_MAX, SFPSWAP_MOD1_SWAP, SFPLZ_MOD1_CC_NE0 numeric values?"
   **Reason**: Needed exact constant values for accurate instruction annotation.
   **Key Findings**: SFPSWAP_MOD1_SWAP=0, SFPSWAP_MOD1_VEC_MIN_MAX=1, SFPLZ_MOD1_CC_NE0=2. Defined in sfpi_hw.h for both Wormhole and Blackhole.

6. **Query**: "Show the full implementation of eltwise_binary_sfpu_configure_addrmod"
   **Reason**: Needed to document ADDR_MOD configuration for the GCD operation.
   **Key Findings**: ADDR_MOD_7 is always set with all-zero increments. ADDR_MOD_6 (dest.incr=2) is only set for mul_int32, mul_uint16, max, min, max_int32, min_int32, max_uint32, min_uint32 -- GCD is not in this list.

### Confluence References
The Tensix SFPU Instruction Set Architecture page (Page ID: 1170505767) was retrieved but contained only an MCP deprecation notice rather than ISA content. No usable instruction details were obtained from Confluence for this analysis.

### Glean References
No Glean searches were performed for this analysis. The DeepWiki sources provided sufficient detail on the SFPU instructions and modifier flags used in the GCD kernel.
