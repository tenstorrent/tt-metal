## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the **ADD (legacy SFPU)** binary operation.

The legacy SFPU binary ADD path is used when the program factory `element_wise_multi_core_sfpu_pgm_factory.cpp` is selected. For floating-point types, the operation is dispatched through `add_binary_tile()` which calls the generic `_calculate_sfpu_binary_<APPROX, BinaryOp::ADD, 8>()`. For integer types (INT32, UINT32, UINT16), a separate `add_int_tile` path is used -- this analysis focuses on the floating-point SFPU path.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_binary_sfpu.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_binop.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/common/inc/sfpu/ckernel_sfpu_binary.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_binary_sfpu_params.h` |

### Call Chain

1. **Compute kernel** (`eltwise_binary_sfpu_kernel.cpp`): The preprocessor define `BINARY_SFPU_OP` expands to `add_binary_tile(i*2, i*2+1, i*2);` and `BINOP_INIT` expands to `add_binary_tile_init();`. The compute kernel copies input A to DST[i*2] and input B to DST[i*2+1], then calls the SFPU op which reads both and writes the result to DST[i*2].

2. **API header** (`eltwise_binary_sfpu.h`): `add_binary_tile(idst0, idst1, odst)` calls `MATH((llk_math_eltwise_binary_sfpu_binop<APPROX, ckernel::BinaryOp::ADD>(idst0, idst1, odst)))`. The `MATH()` macro ensures the call runs only on the math RISC-V processor. The init function `add_binary_tile_init()` calls `llk_math_eltwise_binary_sfpu_binop_init<APPROX, BinaryOp::ADD>()`.

3. **LLK dispatch** (`llk_math_eltwise_binary_sfpu_binop.h`): `llk_math_eltwise_binary_sfpu_binop<APPROX, BinaryOp::ADD>()` calls `_llk_math_eltwise_binary_sfpu_params_<APPROX>(ckernel::sfpu::calculate_sfpu_binary<APPROX, BinaryOp::ADD, 8, false>, dst_index0, dst_index1, odst, VectorMode::RC)`. The init function calls `llk_math_eltwise_binary_sfpu_init<SfpuType::unused, APPROX>(ckernel::sfpu::sfpu_binary_init<APPROX, BinaryOp::ADD>)`.

4. **Parameters dispatch** (`llk_math_eltwise_binary_sfpu_params.h`): `_llk_math_eltwise_binary_sfpu_params_` sets DST write address, stalls until SFPU is ready, then in `VectorMode::RC` iterates over all 4 tile faces, calling `calculate_sfpu_binary()` once per face. Between faces, it issues `TTI_SETRWC` to advance the DST row counter by 16 rows (two increments of 8).

5. **Core SFPU** (`ckernel_sfpu_binary.h`): `calculate_sfpu_binary<APPROX, BinaryOp::ADD, 8, false>()` delegates to `_calculate_sfpu_binary_<APPROX, BinaryOp::ADD, 8>()`, which loops 8 iterations per face. Each iteration loads one row from each input tile in DST, performs `in0 + in1` (compiling to SFPADD), stores the result, and increments `dst_reg` to advance to the next row.

6. **Init path** (`ckernel_sfpu_binary.h`): `_sfpu_binary_init_<APPROX, BinaryOp::ADD>()` is a no-op for ADD -- it only initializes state for DIV, POW, and XLOGY. The SFPU hardware init (`_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()`) configures `ADDR_MOD_7` with all-zero increments, initializes the SFPU config register, and resets math counters.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_wormhole_b0/common/inc/sfpu/ckernel_sfpu_binary.h
// (Blackhole version is identical)

template <bool APPROXIMATION_MODE, BinaryOp BINOP, int ITERATIONS = 8>
inline void _calculate_sfpu_binary_(const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_out)
    // For ADD: APPROXIMATION_MODE=true (typical), BINOP=BinaryOp::ADD, ITERATIONS=8
{
    static constexpr float nan = std::numeric_limits<float>::quiet_NaN();
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++)
    {
        constexpr std::uint32_t dst_tile_size_sfpi = 32; // each tile occupies 32 SFPI-addressable rows in DEST
        sfpi::vFloat in0                           = sfpi::dst_reg[dst_index_in0 * dst_tile_size_sfpi]; // SFPLOAD from DEST tile 0
        sfpi::vFloat in1                           = sfpi::dst_reg[dst_index_in1 * dst_tile_size_sfpi]; // SFPLOAD from DEST tile 1
        sfpi::vFloat result                        = 0.0f; // SFPLOADI: load immediate 0.0 into LReg

        if constexpr (BINOP == BinaryOp::ADD) // compile-time selected for ADD
        {
            result = in0 + in1; // SFPADD: lanewise FP32 addition
        }
        // Other branches (SUB, MUL, DIV, RSUB, POW, XLOGY) eliminated at compile time

        sfpi::dst_reg[dst_index_out * dst_tile_size_sfpi] = result; // SFPSTORE: write result back to DEST
        sfpi::dst_reg++; // advance DEST row pointer by SFP_DESTREG_STRIDE (=2), moving to next row-pair
    }
}

template <bool APPROXIMATION_MODE /*unused*/, BinaryOp BINOP>
inline void _sfpu_binary_init_()
{
    // For ADD: no initialization needed -- this branch is only entered for DIV, POW, or XLOGY
    if constexpr (BINOP == BinaryOp::DIV || BINOP == BinaryOp::POW)
    {
        _init_sfpu_reciprocal_<false>();
    }
    else if constexpr (BINOP == BinaryOp::XLOGY)
    {
        _init_log_<APPROXIMATION_MODE>();
    }
}
```

### SFPU Instructions Used

| Instruction | SFPI Expression | Description |
|-------------|-----------------|-------------|
| **SFPLOAD** | `sfpi::dst_reg[index]` (read) | Loads a 32-element vector from the specified DEST register row into an SFPU LReg. Used twice per iteration to load `in0` and `in1` from their respective tile positions. IPC=1, latency=1 cycle. |
| **SFPLOADI** | `result = 0.0f` | Loads a BF16 immediate value (converted to FP32) into an LReg. Used to initialize `result` to zero. IPC=1, latency=1 cycle. In practice, the compiler may optimize this away since `result` is immediately overwritten by the ADD. |
| **SFPADD** | `in0 + in1` | Performs lanewise FP32 addition across all 32 SIMD lanes: `VD = VB + VC`. This is the core arithmetic instruction for the ADD operation. IPC=1, latency=2 cycles. |
| **SFPSTORE** | `sfpi::dst_reg[index] = result` (write) | Stores a 32-element vector from an SFPU LReg back to the specified DEST register row. Used to write the addition result. IPC=1, latency=1 cycle. |
| **TTI_SETRWC** | (in params dispatch) | Sets the read/write counter to advance the DEST face address between the 4 faces of a tile. Issued by `_llk_math_eltwise_binary_sfpu_params_` between face iterations. |
| **TTI_STALLWAIT** | (in start/done) | Stalls until the SFPU is ready (at start) or until the SFPU completes (at done, Wormhole only). Ensures synchronization between the math pipeline and SFPU execution. |

### SFPU Register Usage

- **DEST register file**: The operation reads from two tile positions in DEST (`dst_index_in0 * 32` and `dst_index_in1 * 32`) and writes back to `dst_index_out * 32`. In the typical ADD case from the compute kernel, `dst_index_in0 = i*2`, `dst_index_in1 = i*2+1`, and `dst_index_out = i*2`, so the result overwrites input A's tile slot. Each tile occupies 32 SFPI-addressable rows in DEST (physical size is 64 rows / SFP_DESTREG_STRIDE=2).
- **LRegs (L0-L7)**: The SFPI compiler allocates local registers for `in0`, `in1`, and `result`. The ADD path requires at most 3 LRegs. Since the operation is a simple add, register pressure is very low.
- **dst_reg pointer**: The `sfpi::dst_reg++` statement advances the internal DEST row pointer by `SFP_DESTREG_STRIDE` (value 2) each iteration. Over 8 iterations this covers 16 physical rows, which is one face (16 rows x 16 columns = 256 elements, but stored as 16 rows x 32 lanes since the SFPU processes 32 lanes per row). The params dispatch then advances to the next face via `TTI_SETRWC`.

### Address Mode Configuration

The init function `_llk_math_eltwise_binary_sfpu_init_<SfpuType::unused>()` configures address mode 7:

```
ADDR_MOD_7:
  .srca = {.incr = 0}   -- no auto-increment for SrcA
  .srcb = {.incr = 0}   -- no auto-increment for SrcB
  .dest = {.incr = 0}   -- no auto-increment for DEST
```

This is intentional because the SFPU binary kernel manages DEST addressing explicitly through `dst_reg[index]` absolute addressing (with tile offsets computed as `dst_index * 32`) and `dst_reg++` for row-by-row advancement. The address mode is set to ADDR_MOD_7 specifically to avoid conflicts with ADDR_MOD_0 and ADDR_MOD_2, which are used by the A2D (Accumulator-to-DEST) pipeline that runs alongside SFPU operations.

For certain other binary operations (mul_int32, mul_uint16, max, min, and their int32/uint32 variants), an additional `ADDR_MOD_6` with `.dest = {.incr = 2}` is configured, but this is **not** used for the ADD operation.

The address mode configuration is **identical between Wormhole and Blackhole** for this operation. The only architectural difference in the LLK layer between the two is that Wormhole's `_llk_math_eltwise_binary_sfpu_start_` calls `math::set_addr_mod_base()` and `_llk_math_eltwise_binary_sfpu_done_` calls `math::clear_addr_mod_base()` plus an additional `TTI_STALLWAIT(p_stall::STALL_CFG, p_stall::WAIT_SFPU)`, while Blackhole omits these.

## External Knowledge Sources

### DeepWiki Queries

1. **Query**: "How does the SFPU binary eltwise operation work for ADD? What defines get set, and how does the binary SFPU compute kernel dispatch to the underlying SFPU add implementation?"
   **Reason**: To understand the full call chain from program factory through compute kernel defines to LLK and ckernel layers.
   **Key Findings**: Confirmed that `BINOP_INIT` is set to `add_binary_tile_init()` and `BINARY_SFPU_OP` is set to `add_binary_tile(i*2, i*2+1, i*2)` for floating-point ADD. The call chain flows through `llk_math_eltwise_binary_sfpu_binop` to `_llk_math_eltwise_binary_sfpu_params_` to `calculate_sfpu_binary<APPROX, BinaryOp::ADD, 8, false>`.

2. **Query**: "What SFPU instruction does the vFloat + operator compile to? What instructions do dst_reg loads and stores compile to?"
   **Reason**: To determine the exact SFPU instructions emitted for the core ADD operation.
   **Key Findings**: `vFloat + vFloat` compiles to the `SFPADD` instruction via `__builtin_rvtt_sfpadd`. `dst_reg[index]` reads compile to `SFPLOAD`, writes compile to `SFPSTORE`.

3. **Query**: "What are the SFPU instructions SFPIADD, SFPADDI, SFPADD? Describe SFPLOADI and SFPSTORE."
   **Reason**: To get precise instruction semantics including latency and IPC.
   **Key Findings**: `SFPADD` performs `VD = VB + VC` with IPC=1, latency=2 cycles. `SFPLOADI` loads a BF16 immediate into an LReg (IPC=1, latency=1). `SFPSTORE` moves data from LReg to DEST (IPC=1, latency=1). `SFPLOAD` loads from DEST to LReg (IPC=1, latency=1).

### Confluence References

Not consulted -- DeepWiki provided sufficient detail on the SFPU instructions used by this simple ADD operation.

### Glean References

Not consulted -- no confidential hardware specs were needed beyond what DeepWiki provided.
