# ADD (binary_ng) SFPU Analysis

## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the ADD operation in the binary_ng framework.

The ADD operation in binary_ng supports two distinct SFPU paths depending on data type:
1. **Float path** (`add_binary_tile`): For BFLOAT16 and FLOAT32 data types, uses SFPI vector intrinsics (`vFloat` addition).
2. **Integer path** (`add_int_tile`): For INT32, UINT32, and UINT16 data types, uses explicit SFPU load/add/store instructions (`SFPLOAD`, `SFPIADD`, `SFPSTORE`).

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header (float)** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **API Header (int)** | `tt_metal/hw/inc/api/compute/add_int_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation (float)** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_binary.h` |
| **Core SFPU Implementation (int)** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_add_int.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |
| **Init / ADDR_MOD Config** | `tt_metal/third_party/tt_llk/tt_llk_{arch}/llk_lib/llk_math_eltwise_binary_sfpu.h` |

### Call Chain

#### Float ADD Path (BFLOAT16 / FLOAT32)

1. **Compute kernel** invokes macro `BINARY_SFPU_OP(i*2, i*2+1, i*2)` which resolves to `add_binary_tile(i*2, i*2+1, i*2)` (defined via `get_sfpu_init_fn` returning `"add_binary_tile"`).
2. **API header** `add_binary_tile()` calls `MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, ckernel::BinaryOp::ADD>(idst0, idst1, odst)))`.
3. **LLK dispatch** `llk_math_eltwise_binary_sfpu_binop()` calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>()` passing the SFPU function `calculate_sfpu_binary<APPROXIMATE, BinaryOp::ADD, 8, false>`.
4. **Params dispatch** `_llk_math_eltwise_binary_sfpu_params_()` sets up DST write address, stalls until SFPU is ready, then iterates over 4 faces (in RC mode), calling the SFPU function once per face and advancing the DEST read/write counter by 16 rows (2x `SETRWC +8`) between faces.
5. **Core SFPU function** `calculate_sfpu_binary()` delegates to `_calculate_sfpu_binary_<APPROXIMATE, BinaryOp::ADD, 8>()`.
6. **Inner loop** `_calculate_sfpu_binary_()` iterates 8 times per face, loading `in0` and `in1` from `dst_reg[]`, computing `result = in0 + in1`, storing back to `dst_reg[]`, and incrementing `dst_reg++`.

#### Integer ADD Path (INT32 / UINT32 / UINT16)

1. **Compute kernel** invokes macro `BINARY_SFPU_OP(i*2, i*2+1, i*2)` which resolves to `add_int_tile<DataFormat::Int32>(i*2, i*2+1, i*2)`.
2. **API header** `add_int_tile()` calls `MATH((llk_math_eltwise_binary_sfpu_add_int<APPROX, 8, DataFormat::Int32, false>(idst0, idst1, odst)))`.
3. **LLK dispatch** `llk_math_eltwise_binary_sfpu_add_int()` calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>()` passing `_add_int_<APPROXIMATE, 8, InstrModLoadStore::INT32, false>`.
4. **Params dispatch** same face iteration as float path.
5. **Core SFPU function** `_add_int_()` uses explicit `SFPLOAD`, `SFPIADD`, `SFPSTORE` instructions.

#### Init Path

1. `BINARY_SFPU_INIT` resolves to `add_binary_tile_init()` (float) or `add_int_tile_init()` (int).
2. Float init calls `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::ADD>()`, which calls `llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROXIMATE>(sfpu_binary_init<APPROXIMATE, BinaryOp::ADD>)`.
3. `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` initializes the SFPU config register, configures ADDR_MOD_7, and resets counters.
4. `_sfpu_binary_init_<APPROXIMATE, BinaryOp::ADD>()` is a no-op for ADD (only DIV/POW/XLOGY need special init).

### Annotated SFPU Kernel Source

#### Float ADD: Core Implementation

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_binary.h
// (Wormhole and Blackhole implementations are identical for _calculate_sfpu_binary_ and _sfpu_binary_init_)

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{   // For ADD: APPROXIMATION_MODE=true (APPROX), BINOP=BinaryOp::ADD, ITERATIONS=8
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        // size of each tile in Dest is 64/SFP_DESTREG_STRIDE = 32 rows when using sfpi to load/store
        constexpr std::uint32_t dst_tile_size_sfpi = 32;
        sfpi::vFloat in0 = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // SFPLOAD from DEST tile 0
        sfpi::vFloat in1 = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // SFPLOAD from DEST tile 1

        sfpi::vFloat result = 0.0f;

        if constexpr (BINOP == BinaryOp::ADD)
        {
            result = in0 + in1; // Emits SFPADD instruction
        }
        // SUB, MUL, DIV, RSUB, POW, XLOGY branches omitted (not taken for ADD)

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // SFPSTORE to DEST tile 0 (output)
        sfpi::dst_reg++; // Advance DEST row pointer by 1 (SFP_DESTREG_STRIDE)
    }
}

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{   // For ADD: no-op — ADD requires no special SFPU initialization
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW)
    {
        _init_sfpu_reciprocal_<false>();
    }
    else if constexpr (BINOP == BinaryOp::XLOGY)
    {
        _init_log_<APPROXIMATION_MODE>();
    }
    // ADD, SUB, MUL, RSUB: no initialization needed
}
```

#### Integer ADD: Core Implementation (Wormhole B0)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_add_int.h

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void _add_int_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{   // For Int32: INSTRUCTION_MODE=InstrModLoadStore::INT32, SIGN_MAGNITUDE_FORMAT=false
    static_assert(is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    // sfpload_instr_mod: INT32 (=4) for 2's complement int32; INT32_2S_COMP (=12) for sign-magnitude conversion
    constexpr int sfpload_instr_mod = SIGN_MAGNITUDE_FORMAT ? INT32_2S_COMP : to_underlying(INSTRUCTION_MODE);

    constexpr std::uint32_t dst_tile_size = 64; // DEST tile stride (64 rows per tile in raw DEST addressing)

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, dst_index_in0 * dst_tile_size); // Load operand A into LREG0
        TT_SFPLOAD(p_sfpu::LREG1, sfpload_instr_mod, ADDR_MOD_3, dst_index_in1 * dst_tile_size); // Load operand B into LREG1
        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4); // LREG0 = LREG0 + LREG1; imod=4 selects 2's complement integer add
        TT_SFPSTORE(p_sfpu::LREG0, sfpload_instr_mod, ADDR_MOD_3, dst_index_out * dst_tile_size); // Store result from LREG0
        sfpi::dst_reg++;
    }
}
```

#### Integer ADD: Core Implementation (Blackhole)

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_add_int.h

template <bool APPROXIMATION_MODE, int ITERATIONS, InstrModLoadStore INSTRUCTION_MODE, bool SIGN_MAGNITUDE_FORMAT>
inline void _add_int_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
{   // For Int32: INSTRUCTION_MODE=InstrModLoadStore::INT32, SIGN_MAGNITUDE_FORMAT=false
    static_assert(is_valid_instruction_mode(INSTRUCTION_MODE), "INSTRUCTION_MODE must be one of: INT32_2S_COMP, INT32, LO16.");

    // Implementation notes, see the original file for more details
    constexpr auto INSTR_MOD_CAST = InstrModCast::INT_SIGN_MAGN_TO_INT32_2S_COMP;

    constexpr std::uint32_t dst_tile_size = 64;

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        TT_SFPLOAD(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, dst_index_in0 * dst_tile_size); // Load operand A
        if constexpr (SIGN_MAGNITUDE_FORMAT) // Not taken for standard INT32 ADD
        {
            apply_sign_magnitude_conversion(p_sfpu::LREG0, p_sfpu::LREG2, INSTR_MOD_CAST);
        }

        TT_SFPLOAD(p_sfpu::LREG1, INSTRUCTION_MODE, ADDR_MOD_7, dst_index_in1 * dst_tile_size); // Load operand B
        if constexpr (SIGN_MAGNITUDE_FORMAT)
        {
            apply_sign_magnitude_conversion(p_sfpu::LREG1, p_sfpu::LREG2, INSTR_MOD_CAST);
        }

        TTI_SFPIADD(0, p_sfpu::LREG1, p_sfpu::LREG0, 4); // imod=4: integer add in 2's complement

        if constexpr (SIGN_MAGNITUDE_FORMAT)
        {
            apply_sign_magnitude_conversion(p_sfpu::LREG0, p_sfpu::LREG1, INSTR_MOD_CAST);
        }
        TT_SFPSTORE(p_sfpu::LREG0, INSTRUCTION_MODE, ADDR_MOD_7, dst_index_out * dst_tile_size); // Store result
        sfpi::dst_reg++;
    }
}
```

### SFPU Instructions Used

#### Float ADD Path

| Instruction | Description |
|---|---|
| **SFPLOAD** | Loads a vector of 32 elements from a DEST register row into an SFPU local register (LREG). Emitted by `sfpi::dst_reg[index]` reads. Two loads per iteration: one for each operand. |
| **SFPADD** | Performs element-wise floating-point addition of two SFPU local registers. Emitted by `in0 + in1`. This is the core computation for the float ADD operation. The `mod1` parameter is 0 for standard addition. |
| **SFPSTORE** | Stores a vector from an SFPU local register back to a DEST register row. Emitted by `sfpi::dst_reg[index] = result`. |
| **SFPLOADI** | Loads an immediate constant (0.0f) into a local register. Emitted by the `result = 0.0f` initialization (though this is overwritten immediately by the ADD result). |

#### Integer ADD Path

| Instruction | Description |
|---|---|
| **SFPLOAD** | Loads integer data from DEST into LREG0/LREG1. The `instr_mod` parameter selects the data format: `INT32` (4) for 2's complement, `LO16` for uint16. |
| **SFPIADD** | Integer addition: `LREG_dest = LREG_dest + LREG_c`. With `imm=0` and `imod=4`, this performs a full 32-bit integer add between two LREGs. LREG0 is both source and destination. |
| **SFPSTORE** | Stores the integer result from LREG0 back to DEST. Same `instr_mod` as the load to maintain consistent data format interpretation. |

### SFPU Register Usage

#### Float ADD Path

| Register | Usage |
|---|---|
| **LREG0** | Implicitly used by SFPI compiler for `in0` (loaded from `dst_reg[dst_index_in0 * 32]`) |
| **LREG1** | Implicitly used by SFPI compiler for `in1` (loaded from `dst_reg[dst_index_in1 * 32]`) |
| **LREG2-3** | Available as scratch; the SFPI compiler may use them for intermediate results |
| **DEST[idst0 * 32 + d]** | Source tile for operand A (tile index 0 in typical no_bcast case). Each iteration `d` reads the next row. |
| **DEST[idst1 * 32 + d]** | Source tile for operand B (tile index 1 in typical no_bcast case). |
| **DEST[odst * 32 + d]** | Output tile (tile index 0, same as operand A). Result overwrites operand A's DEST location. |

#### Integer ADD Path

| Register | Usage |
|---|---|
| **LREG0 (p_sfpu::LREG0)** | Loaded with operand A, receives the addition result, and is stored back to DEST |
| **LREG1 (p_sfpu::LREG1)** | Loaded with operand B, used as source in SFPIADD |
| **LREG2 (p_sfpu::LREG2)** | Used as scratch only in Blackhole sign-magnitude conversion path (not used for standard INT32) |
| **DEST[in0 * 64 + d]** | Source for operand A (note: integer path uses `dst_tile_size=64` vs float's `32` due to different DEST stride) |
| **DEST[in1 * 64 + d]** | Source for operand B |
| **DEST[out * 64 + d]** | Output destination |

### Address Mode Configuration

The ADDR_MOD configuration differs between architectures and between the float and integer paths.

#### ADDR_MOD_7 (Both Wormhole B0 and Blackhole, configured during init)

Set by `eltwise_binary_sfpu_configure_addrmod<SfpuType::unused>()` called from `_llk_math_eltwise_binary_sfpu_init_()`:

```
ADDR_MOD_7: { srca.incr = 0, srcb.incr = 0, dest.incr = 0 }
```

All increments are zero. This is because the SFPU manages its own DEST addressing via `dst_reg++` (which increments by `SFP_DESTREG_STRIDE=2`), rather than relying on hardware auto-increment through ADDR_MOD. The ADDR_MOD_7 slot is chosen to avoid conflicts with ADDR_MOD_0 and ADDR_MOD_2 used by the A2D (Accumulate-to-DEST) pipeline that runs the `copy_tile` operations prior to SFPU execution.

**Note**: `ADDR_MOD_6` is additionally configured (with `dest.incr = 2`) only for certain operations like `mul_int32`, `max`, `min` -- not for ADD.

#### ADDR_MOD_3 (Wormhole B0 integer path only)

The Wormhole integer ADD path uses `ADDR_MOD_3` in its `SFPLOAD`/`SFPSTORE` instructions. This is a pre-configured address mode (not set by the binary SFPU init), presumably configured by the broader LLK framework.

#### ADDR_MOD_7 (Blackhole integer path)

The Blackhole integer ADD path uses `ADDR_MOD_7` in its `SFPLOAD`/`SFPSTORE` instructions, consistent with the init configuration above.

#### Key Differences Between Architectures

- **Wormhole B0**: The `_llk_math_eltwise_binary_sfpu_start_` function calls `math::set_addr_mod_base()` to set the address mode base. `_llk_math_eltwise_binary_sfpu_done_` calls `math::clear_addr_mod_base()`. These calls are absent in the Blackhole version.
- **Wormhole B0**: The `_done_` function also issues `TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU)` to wait for SFPU completion, while Blackhole does not.
- **Integer path**: Wormhole uses `ADDR_MOD_3` for SFPLOAD/SFPSTORE; Blackhole uses `ADDR_MOD_7`. The Blackhole integer path also adds optional `apply_sign_magnitude_conversion()` calls for sign-magnitude format support, which is a no-op for standard INT32 operations (SIGN_MAGNITUDE_FORMAT=false).

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the binary_ng SFPU compute kernel work? What is the compute kernel file path for binary_ng SFPU operations, and how does it dispatch to SFPU functions like add?"
   **Reason**: Needed to locate compute kernel file paths and understand the dispatch mechanism from the compute kernel through LLK to ckernel.
   **Key Findings**: Identified the multiple compute kernel variants (no_bcast, bcast, scalar, row_bcast, row_col_bcast) and the `BINARY_SFPU_OP` / `BINARY_SFPU_INIT` macro dispatch pattern. Confirmed the call chain from API -> LLK -> ckernel.

2. **Query**: "Where is the function `_calculate_sfpu_binary_` defined?" (tt-llk repo)
   **Reason**: The function is called from `ckernel_sfpu_binary.h` in `hw/ckernels/` but was not defined there. Needed to locate the actual implementation.
   **Key Findings**: Found it in `tt_llk_{arch}/common/inc/sfpu/ckernel_sfpu_binary.h`. For ADD, it performs `result = in0 + in1` using SFPI vector floats. The init function is a no-op for ADD.

3. **Query**: "How does the SFPI vFloat addition operator (in0 + in1) map to SFPU hardware instructions?" (sfpi repo)
   **Reason**: Needed to understand which hardware instruction the `+` operator on `vFloat` compiles to.
   **Key Findings**: `vFloat + vFloat` maps to the `SFPADD` hardware instruction via `__builtin_rvtt_sfpadd`. `dst_reg[index]` reads emit `SFPLOAD` and writes emit `SFPSTORE`.

### Confluence References
Not consulted for this analysis. The SFPU instructions used (SFPLOAD, SFPADD, SFPSTORE, SFPIADD) are well-documented through DeepWiki and the source code itself.

### Glean References
Not consulted for this analysis.
