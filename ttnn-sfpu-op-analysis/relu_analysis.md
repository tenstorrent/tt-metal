## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the RELU operation.

Standard RELU is implemented as `relu_min` with a threshold of 0, computing `max(x, 0)`. The TTNN RELU operation maps to `relu_tile(idst)` which internally calls `_relu_min_<sfpi::vFloat, APPROX, 8, uint32_t>(0)`. The threshold of 0 means: for each element, if the value is less than 0, replace it with 0 -- the classic rectified linear unit.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_macros.h` (macro-based, no dedicated LLK file for relu) |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_relu.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

Note: `{arch}` is `blackhole` or `wormhole_b0`. There is also a legacy overlay in `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_relu.h` which wraps the tt_llk implementation for backward compatibility, but the authoritative source is in the tt_llk submodule.

### Call Chain

1. **Compute kernel** calls `relu_tile(idst)` (defined in `tt_metal/hw/inc/api/compute/eltwise_unary/relu.h`).
2. `relu_tile` expands macro `SFPU_UNARY_ONE_PARAM_KERNEL_FN_FLOAT(_relu_min_, RC, APPROX, idst, 0)` which resolves to `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::_relu_min_<sfpi::vFloat, APPROXIMATE, 8, uint32_t>, idst, (int)VectorMode::RC, 0)`.
3. `_llk_math_eltwise_unary_sfpu_params_` (in `llk_math_eltwise_unary_sfpu_params.h`) sets the destination write address, stalls until SFPU is ready, then calls the SFPU function once per face (4 faces for VectorMode::RC), incrementing the destination face address between calls.
4. Each face invocation calls `_relu_min_<sfpi::vFloat, APPROX, 8, uint32_t>(0)` which converts the threshold from `uint32_t` (bit pattern for 0.0f) to a `vFloat`, loads it into LREG2 (Wormhole) or directly as a SFPI variable (Blackhole), then calls `_relu_min_impl_` to process 8 rows of the face.
5. `_relu_min_impl_` iterates 8 times (one per row of 4 elements), loading each element from DEST, comparing with the threshold, and storing the maximum of the two back to DEST.

The init path is: `relu_tile_init()` -> `SFPU_UNARY_KERNEL_INIT(relu_min, APPROX)` -> `llk_math_eltwise_unary_sfpu_init<SfpuType::relu_min, APPROXIMATE>()` -> `_llk_math_eltwise_unary_sfpu_init_<SfpuType::relu_min>()` which initializes the SFPU config register, configures address modifiers, and resets counters.

### Annotated SFPU Kernel Source

The Blackhole and Wormhole implementations differ significantly for `_relu_min_impl_`. Blackhole uses high-level SFPI constructs (`v_if`/`v_endif` with `dst_reg`), while Wormhole uses explicit TTI instructions (`SFPLOAD`, `SFPMOV`, `SFPSWAP`, `SFPSTORE`). Both share the same wrapper and `_relu_max_` functions.

#### Blackhole Implementation

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_relu.h

template <bool APPROXIMATION_MODE>
inline void _calculate_lrelu_(const int iterations, std::uint32_t slope)
{
    TT_SFPLOADI(p_sfpu::LREG2, 10, slope & 0xFFFF);   // Load lower 16 bits of slope into LREG2
    TT_SFPLOADI(p_sfpu::LREG2, 8, slope >> 16);        // Load upper 16 bits of slope into LREG2
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);        // load from dest into lreg[0]
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);                                         // condition - if value in LREG0 is negative
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // Multiply LREG0 * LREG2 (x * slope)
        TTI_SFPENCC(0, 0, 0, 0);                                                      // clear cc result reg
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_7, 0);       // store from lreg0 into dest register
        sfpi::dst_reg++;
    }
}

sfpi_inline sfpi::vFloat _relu_max_body_(sfpi::vFloat val, sfpi::vFloat threshold)
{
    sfpi::vFloat result = val;
    v_if (result > threshold)
    {
        result = threshold;
    }
    v_endif;
    v_if (result < 0.0f)
    {
        result = 0.0f;
    }
    v_endif;
    return result;
}

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_max_impl_(const int iterations, VecType threshold)
{
    for (int d = 0; d < iterations; d++)
    {
        VecType result = sfpi::dst_reg[0];
        v_if (result > threshold)
        {
            result = threshold;
        }
        v_endif;
        v_if (result < 0)
        {
            result = 0;
        }
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_max_(T threshold) // VectorType=sfpi::vFloat for float path, ITERATIONS=8
{
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>, "VectorType must be sfpi::vFloat or sfpi::vInt");

    VectorType v_threshold;
    if constexpr (std::is_same_v<T, float>)
    {
        v_threshold = threshold;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            v_threshold = static_cast<int>(Converter::as_float(threshold));
        }
        else
        {
            v_threshold = Converter::as_float(threshold); // Reinterpret uint32 bit pattern as float
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_max_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold);
}

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_impl_(const int iterations, VecType threshold) // Blackhole: uses SFPI v_if/v_endif
{
    for (int d = 0; d < iterations; d++)
    {
        VecType a = sfpi::dst_reg[0];       // SFPLOAD: load element from DEST register
        v_if (a < threshold)                 // SFPSETCC: set CC if a < threshold (computes a - threshold, checks sign)
        {
            sfpi::dst_reg[0] = threshold;    // SFPSTORE: conditionally write threshold to DEST (only active lanes)
        }
        v_endif;                             // SFPPOPC: restore previous CC state
        sfpi::dst_reg++;                     // Advance DEST register pointer by 1 row
    }
}

template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_min_(T threshold) // VectorType=sfpi::vFloat for float path, ITERATIONS=8
{
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>, "VectorType must be sfpi::vFloat or sfpi::vInt");

    VectorType v_threshold;
    if constexpr (std::is_same_v<T, float>)
    {
        v_threshold = threshold;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            v_threshold = static_cast<int>(threshold);
        }
        else
        {
            v_threshold = Converter::as_float(threshold); // For RELU: threshold=0 -> 0.0f
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_min_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold);
}
```

#### Wormhole B0 Implementation

The Wormhole B0 implementation differs in `_relu_min_impl_` -- it uses explicit TTI instructions and the SFPSWAP instruction for an optimized min/max comparison, rather than the SFPI `v_if`/`v_endif` conditional execution pattern used by Blackhole.

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_relu.h

template <bool APPROXIMATION_MODE>
inline void _calculate_lrelu_(const int iterations, std::uint32_t slope)
{
    TT_SFPLOADI(p_sfpu::LREG2, 10, slope & 0xFFFF);
    TT_SFPLOADI(p_sfpu::LREG2, 8, slope >> 16);
#pragma GCC unroll 8
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);        // load from dest into lreg[0]; ADDR_MOD_3 maps to HW slot 7 via addr_mod_base
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, 0);                                         // condition - if value in LREG0 is negative
        TTI_SFPMUL(p_sfpu::LREG0, p_sfpu::LREG2, p_sfpu::LCONST_0, p_sfpu::LREG0, 0); // Multiply LREG0 * LREG2 (x * slope)
        TTI_SFPENCC(0, 0, 0, 0);                                                      // clear cc result reg
        TTI_SFPSTORE(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, 0);       // store from lreg0 into dest register
        sfpi::dst_reg++;
    }
}

sfpi_inline sfpi::vFloat _relu_max_body_(sfpi::vFloat val, sfpi::vFloat threshold)
{
    sfpi::vFloat result = val;
    v_if (result > threshold)
    {
        result = threshold;
    }
    v_endif;
    v_if (result < 0.0f)
    {
        result = 0.0f;
    }
    v_endif;
    return result;
}

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_max_impl_(const int iterations, VecType threshold)
{
    for (int d = 0; d < iterations; d++)
    {
        VecType result = sfpi::dst_reg[0];
        v_if (result > threshold)
        {
            result = threshold;
        }
        v_endif;
        v_if (result < 0)
        {
            result = 0;
        }
        v_endif;
        sfpi::dst_reg[0] = result;
        sfpi::dst_reg++;
    }
}

template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_max_(T threshold) // VectorType=sfpi::vFloat for float path, ITERATIONS=8
{
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>, "VectorType must be sfpi::vFloat or sfpi::vInt");

    VectorType v_threshold;
    if constexpr (std::is_same_v<T, float>)
    {
        v_threshold = threshold;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            v_threshold = static_cast<int>(Converter::as_float(threshold));
        }
        else
        {
            v_threshold = Converter::as_float(threshold);
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_max_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold);
}

template <typename VecType, bool APPROXIMATION_MODE, int ITERATIONS>
inline void _relu_min_impl_(const int iterations, [[maybe_unused]] VecType threshold, int sfpload_instr_mod)
{
    for (int d = 0; d < iterations; d++)
    {
        TTI_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, 0);  // Load input from DEST into LREG0; sfpload_instr_mod=DEFAULT(0) for float
        TTI_SFPMOV(0, p_sfpu::LREG2, p_sfpu::LREG1, 0);               // Copy threshold from LREG2 to LREG1
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, 1);              // Swap: LREG1=max(LREG0,LREG1), LREG0=min(LREG0,LREG1); Mod1=1 enables DEST_INDEX
        TTI_SFPSTORE(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_3, 0); // Store max value (LREG1) back to DEST
        sfpi::dst_reg++;
    }
}

template <typename VectorType, bool APPROXIMATION_MODE, int ITERATIONS, typename T>
inline void _relu_min_(T threshold) // VectorType=sfpi::vFloat for float path, ITERATIONS=8
{
    static_assert(std::is_same_v<VectorType, sfpi::vFloat> || std::is_same_v<VectorType, sfpi::vInt>, "VectorType must be sfpi::vFloat or sfpi::vInt");

    VectorType v_threshold;
    int scalar = threshold;
    if (scalar < 0)
    { // To convert from 2's complement to sign+magnitude
        scalar  = -scalar;
        int res = 0x80000000 | (scalar & 0x7FFFFFFF);
        scalar  = res;
    }
    int sfpload_instr_mod = DEFAULT;                          // DEFAULT=0 for standard float load/store mode
    if constexpr (std::is_same_v<T, float>)
    {
        v_threshold = threshold;
    }
    else if constexpr (std::is_same_v<T, std::uint32_t>)
    {
        if constexpr (std::is_same_v<VectorType, sfpi::vInt>)
        {
            _sfpu_load_imm32_(p_sfpu::LREG2, scalar);        // Load sign+magnitude int threshold into LREG2
            sfpload_instr_mod = INT32_2S_COMP;                // Use 2's complement load/store mode for integers
        }
        else
        {
            _sfpu_load_imm32_(p_sfpu::LREG2, threshold);     // Load float bit pattern (0x00000000 for RELU) into LREG2
        }
    }
    else
    {
        static_assert(std::is_same_v<T, float> || std::is_same_v<T, std::uint32_t>, "Threshold type must be float or uint32_t");
    }

    _relu_min_impl_<VectorType, APPROXIMATION_MODE, ITERATIONS>(ITERATIONS, v_threshold, sfpload_instr_mod);
}
```

### SFPU Instructions Used

**Blackhole `_relu_min_impl_` (SFPI-based):**

The Blackhole implementation uses SFPI high-level constructs that compile to the following SFPU instructions:

| Instruction | Description |
|-------------|-------------|
| `SFPLOAD` (via `dst_reg[0]` read) | Loads a row of 4 elements from the DEST register file into an SFPU local register (LREG). |
| `SFPMAD` / `SFPMOV` (via `v_if (a < threshold)`) | The comparison `a < threshold` compiles to: negate threshold via `SFPMOV` with COMPSIGN, compute `a - threshold` via `SFPMAD`, then check sign. |
| `SFPSETCC` (via `v_if`) | Sets the condition code register based on the comparison result. Lanes where the condition is true become "active". |
| `SFPPUSHC` (via `v_if`) | Pushes the current condition code state onto the CC stack, enabling nested conditionals. |
| `SFPSTORE` (via `dst_reg[0] = threshold`) | Conditionally stores the threshold value back to DEST for active lanes only (those where `a < threshold`). |
| `SFPPOPC` (via `v_endif`) | Pops the CC stack, restoring the previous condition code state. |

**Wormhole B0 `_relu_min_impl_` (TTI instruction-based):**

| Instruction | Description |
|-------------|-------------|
| `TTI_SFPLOAD` | Loads a row of 4 elements from DEST into LREG0. The `InstrModLoadStore` field selects float (DEFAULT=0) or integer (INT32_2S_COMP) format. |
| `TTI_SFPMOV` | Copies the threshold value from LREG2 to LREG1 (non-destructive copy for reuse across iterations). |
| `TTI_SFPSWAP` | Compares LREG0 (input) and LREG1 (threshold) and simultaneously assigns max to LREG1 and min to LREG0. With Mod1=1, this enables the DEST_INDEX feature. This single instruction implements `max(input, threshold)`. |
| `TTI_SFPSTORE` | Stores the result (max value in LREG1) back to the DEST register. |

**Shared instructions (threshold loading in Wormhole `_relu_min_` wrapper):**

| Instruction | Description |
|-------------|-------------|
| `TT_SFPLOADI` (via `_sfpu_load_imm32_`) | Loads a 32-bit immediate value into an LREG in two steps: lower 16 bits with insmod=10, upper 16 bits with insmod=8. Used to load the threshold (0 for standard RELU) into LREG2. |

**Leaky RELU instructions (`_calculate_lrelu_`, shared between architectures):**

| Instruction | Description |
|-------------|-------------|
| `TT_SFPLOADI` | Loads the slope immediate into LREG2 (lower + upper 16 bits). |
| `TTI_SFPLOAD` | Loads input from DEST into LREG0. |
| `TTI_SFPSETCC` | Sets CC based on whether LREG0 is negative (sign bit check). |
| `TTI_SFPMUL` | Multiplies LREG0 * LREG2 (input * slope). Result written to LREG0. Only active for lanes where CC is set (negative values). |
| `TTI_SFPENCC` | Clears the condition code result register, ending conditional execution. |
| `TTI_SFPSTORE` | Stores result from LREG0 back to DEST. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Working register -- holds the input value loaded from DEST. In Wormhole `_relu_min_impl_`, after SFPSWAP it holds `min(input, threshold)` (discarded). In leaky RELU, holds the result after conditional multiply. |
| **LREG1** | Wormhole `_relu_min_impl_` only: receives a copy of the threshold from LREG2 via SFPMOV. After SFPSWAP, holds `max(input, threshold)` which is the output. |
| **LREG2** | Holds the threshold value. For standard RELU, this is 0.0f (loaded as bit pattern 0x00000000). For leaky RELU, holds the slope value. Persists across all 8 iterations within a face. |
| **LREG3** | Not used by relu operations. |
| **DEST register file** | Source and destination for tile data. Each face is 16x16 elements; the SFPU processes 4 elements per row, 8 rows per invocation (covering half a face). The `_llk_math_eltwise_unary_sfpu_params_` function advances the DEST face pointer between the 4 face invocations. |
| **CC register (condition code)** | Used by Blackhole `v_if`/`v_endif` for predicated execution. Also used by leaky RELU's `SFPSETCC`/`SFPENCC` sequence for conditional multiply. Not used in Wormhole `_relu_min_impl_` which uses SFPSWAP instead. |
| **CC stack** | Used by Blackhole's `v_if`/`v_endif` to push/pop condition code state (SFPPUSHC/SFPPOPC). |

### Address Mode Configuration

The address mode for RELU (and all standard unary SFPU operations using `SfpuType::relu_min`) is configured in `eltwise_unary_sfpu_configure_addrmod<SfpuType::relu_min>()` during initialization. Since `relu_min` does not match any of the special-cased SfpuTypes (topk_local_sort, reciprocal, typecast, unary_max, unary_min, etc.), only the default address modifier is configured:

**Both Wormhole B0 and Blackhole configure the same logical address mode:**

```cpp
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
    .set(ADDR_MOD_7);
```

All three increment fields are 0, meaning the address modifier does not auto-increment any register addresses between SFPU instruction executions. Instead, destination register advancement is handled explicitly by `sfpi::dst_reg++` (which compiles to a SETRWC instruction incrementing the DEST write pointer).

**Hardware slot mapping differs:**
- **Blackhole**: Uses `ADDR_MOD_7` directly in TTI instructions. There is no addr_mod_base offset mechanism.
- **Wormhole B0**: Sets `addr_mod_base = 1` via `TTI_SETC16(ADDR_MOD_SET_Base_ADDR32, 1)` at the start of SFPU execution. This causes TTI instructions referencing `ADDR_MOD_3` (slot 3) to actually access hardware slot `3 + 4 = 7`. This is why Wormhole TTI instructions use `ADDR_MOD_3` while Blackhole uses `ADDR_MOD_7` -- they both resolve to the same hardware address modifier slot 7 with `dest.incr = 0`.

The reason for using slot 7 (or equivalent) is documented in the code: "this kernel is typically used in conjunction with A2D, which is using ADDR_MOD_0 and ADDR_MOD_2, so use one that doesn't conflict."

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How is the RELU operation implemented at the SFPU kernel level? Trace from the compute kernel API through the LLK layer down to the ckernel SFPU implementation."
   **Reason**: Needed to identify the full call chain and file locations for the RELU SFPU implementation.
   **Key Findings**: RELU is implemented as `relu_min` with threshold 0. The call chain goes: `relu_tile()` -> macro expansion -> `_llk_math_eltwise_unary_sfpu_params_` -> `_relu_min_`. The core implementation is in `ckernel_sfpu_relu.h` within the tt_llk submodule.

2. **Query**: "How is relu implemented in the LLK/ckernel layer? What SFPU instructions does it use?"
   **Reason**: Needed to understand the specific SFPU instructions and architectural differences between Wormhole and Blackhole.
   **Key Findings**: Wormhole uses explicit TTI instructions (SFPLOAD, SFPMOV, SFPSWAP, SFPSTORE) for `_relu_min_impl_`, while Blackhole uses SFPI constructs (v_if/v_endif). Leaky RELU uses SFPSETCC/SFPMUL/SFPENCC for conditional multiply. The addr_mod differs: ADDR_MOD_7 on Blackhole, ADDR_MOD_3 (mapped to slot 7 via base offset) on Wormhole.

3. **Query**: "What does the SFPSWAP instruction do? What does mode 1 do?"
   **Reason**: The Wormhole `_relu_min_impl_` uses `TTI_SFPSWAP` with Mod1=1, needed to understand the exact semantics.
   **Key Findings**: SFPSWAP compares two LREGs and simultaneously assigns Min to VD and Max to VC. With Mod1=1, it enables the DEST_INDEX bit in the LaneConfig register. The instruction has 2-cycle latency and can perform argmin/argmax when DEST_INDEX is enabled.

4. **Query**: "How do v_if, v_endif, conditional comparisons work in SFPI? What SFPU instructions do they compile to?"
   **Reason**: The Blackhole implementation uses SFPI conditionals; needed to understand what hardware instructions these map to.
   **Key Findings**: `v_if(a < threshold)` compiles to: SFPMOV (negate threshold), SFPMAD (compute a - threshold), SFPSETCC (check sign), SFPPUSHC (push CC state). `v_endif` compiles to SFPPOPC. All code paths execute but the CC register masks which lanes are active for writes.

### Confluence References
No Confluence references were needed for this analysis. The SFPU instructions used by RELU (SFPLOAD, SFPSTORE, SFPMOV, SFPSWAP, SFPSETCC, SFPMUL, SFPENCC, SFPPUSHC, SFPPOPC) were sufficiently documented via DeepWiki queries.

### Glean References
No Glean references were needed for this analysis.
