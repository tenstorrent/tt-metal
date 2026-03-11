## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the ABS operation.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/compute_kernel_api.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/llk_math_eltwise_unary_sfpu_abs.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{wormhole_b0,blackhole}/llk_lib/llk_math_eltwise_unary_sfpu_params.h` |

### Call Chain

1. The compute kernel calls `abs_tile_init()` and `abs_tile(idst)` from `compute_kernel_api.h`. These are thin wrappers using the `MATH()` macro.
2. `abs_tile(idst)` expands to `llk_math_eltwise_unary_sfpu_abs<APPROX>(idst)` in `llk_math_eltwise_unary_sfpu_abs.h`.
3. That function calls `_llk_math_eltwise_unary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_abs<APPROXIMATE>, dst_index, VectorMode::RC)` in `llk_math_eltwise_unary_sfpu_params.h`.
4. `_llk_math_eltwise_unary_sfpu_params_` sets the DST write address, stalls until SFPU is ready, then invokes the `calculate_abs` functor once per face (4 times for VectorMode::RC), advancing the DST face pointer between calls.
5. Inside `calculate_abs`, the SFPU microcode loop iterates 8 times (ITERATIONS=8, one row per iteration within a 16x16 face), loading each row from `dst_reg[0]`, computing `sfpi::abs(v)`, and storing back.

For initialization: `abs_tile_init()` calls `llk_math_eltwise_unary_sfpu_abs_init<APPROX>()`, which calls `llk_math_eltwise_unary_sfpu_init<SfpuType::abs, APPROXIMATE>()`, which calls `_llk_math_eltwise_unary_sfpu_init_<SfpuType::abs>()`. This initializes the SFPU config register, configures ADDR_MOD_7, and resets math counters.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/{wormhole_b0,blackhole}/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h
// NOTE: Wormhole and Blackhole implementations are identical for the floating-point path.
// The int32 path differs only in SFPLOAD/SFPSTORE Mod0 encoding (see below).

namespace ckernel {
namespace sfpu {

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs() { // APPROXIMATION_MODE is unused; ITERATIONS=8 (one per row in a 16x16 face)
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        vFloat v = dst_reg[0];           // SFPLOAD from DEST register at current row offset
        dst_reg[0] = sfpi::abs(v);       // SFPABS with SFPABS_MOD1_FLOAT: clears sign bit (bit 31), then SFPSTORE back
        dst_reg++;                        // Advance DEST row pointer by 1 row (auto-increment via ADDR_MOD_7)
    }
}

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() { // APPROXIMATION_MODE is unused; ITERATIONS=8
    // SFPU microcode
    for (int d = 0; d < ITERATIONS; d++) {
        // Wormhole: TT_SFPLOAD(1, 4, 3, 0)  -- Mod0=4 is INT32 format on WH
        // Blackhole: TT_SFPLOAD(1, 12, ADDR_MOD_7, 0) -- Mod0=12 is INT32 format on BH
        TT_SFPLOAD(1, /*Mod0*/, /*AddrMod*/, 0);   // Load INT32 from DEST into LReg[1]
        TTI_SFPABS(0, 1, 0, 0);                     // LReg[0] = abs(LReg[1]), Mod1=0 => SFPABS_MOD1_INT (two's complement)
        TTI_SFPSTORE(0, /*Mod0*/, /*AddrMod*/, 0);  // Store LReg[0] back to DEST as INT32
        dst_reg++;                                    // Advance DEST row pointer
    }
}

}  // namespace sfpu
}  // namespace ckernel
```

**Wormhole-specific int32 variant** (actual source):
```cpp
// File: tt_metal/hw/ckernels/wormhole_b0/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() {
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(1, 4, 3, 0);       // VD=LReg[1], Mod0=4 (INT32), AddrMod=3, Imm10=0
        TTI_SFPABS(0, 1, 0, 0);       // VD=LReg[0], VC=LReg[1], Mod1=0 (INT mode)
        TTI_SFPSTORE(0, 4, 3, 0);     // VD=LReg[0], Mod0=4 (INT32), AddrMod=3, Imm10=0
        dst_reg++;
    }
}
```

**Blackhole-specific int32 variant** (actual source):
```cpp
// File: tt_metal/hw/ckernels/blackhole/metal/llk_api/llk_sfpu/ckernel_sfpu_abs.h

template <bool APPROXIMATION_MODE, int ITERATIONS = 8>
inline void calculate_abs_int32() {
    for (int d = 0; d < ITERATIONS; d++) {
        TT_SFPLOAD(1, 12, ADDR_MOD_7, 0);   // VD=LReg[1], Mod0=12 (INT32), AddrMod=ADDR_MOD_7, Imm10=0
        TTI_SFPABS(0, 1, 0, 0);              // VD=LReg[0], VC=LReg[1], Mod1=0 (INT mode)
        TTI_SFPSTORE(0, 12, ADDR_MOD_7, 0);  // VD=LReg[0], Mod0=12 (INT32), AddrMod=ADDR_MOD_7, Imm10=0
        dst_reg++;
    }
}
```

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| **SFPLOAD** (implicit via `dst_reg[0]` read) | Loads a row vector from the DEST register file into an SFPU local register (LReg). In the floating-point path, the SFPI compiler emits this automatically when reading `dst_reg[0]`. In the int32 path, it is called explicitly with INT32 format mode. |
| **SFPABS** (`sfpi::abs` / `TTI_SFPABS`) | Computes element-wise absolute value. With `SFPABS_MOD1_FLOAT` (used by `sfpi::abs(vFloat)`): clears the sign bit (bit 31) of each FP32 element; -NaN stays -NaN. With `SFPABS_MOD1_INT` (Mod1=0, used by `TTI_SFPABS`): performs two's complement negation for negative integers; -2147483648 is unchanged. |
| **SFPSTORE** (implicit via `dst_reg[0]` write) | Stores a row vector from an SFPU local register back to the DEST register file. In the floating-point path, emitted automatically by the compiler. In the int32 path, called explicitly with INT32 format mode. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **dst_reg[0]** (DEST register file) | Source and destination for each row. The SFPU reads from and writes back to the same DEST row position. |
| **LReg[0]** | In the int32 path, used as the destination of SFPABS and source for SFPSTORE. In the float path, the compiler allocates LRegs automatically. |
| **LReg[1]** | In the int32 path, used as the load target for SFPLOAD and source operand for SFPABS. |
| **dst_reg pointer** | Auto-incremented via `dst_reg++` after each iteration to advance to the next row within the face. Each `++` advances by one row (32 elements). |

The floating-point path is simple: only one intermediate LReg is needed since `sfpi::abs` is a single-instruction operation that clears the sign bit. The compiler typically uses LReg[0] for both load and store.

### Address Mode Configuration

The ABS operation uses **ADDR_MOD_7** configured during `_llk_math_eltwise_unary_sfpu_init_<SfpuType::abs>()`:

```
addr_mod_t {
    .srca = {.incr = 0},
    .srcb = {.incr = 0},
    .dest = {.incr = 0},
}
    .set(ADDR_MOD_7);
```

All increment fields are zero. This means the SFPU does not auto-increment the DEST address between instructions within a single iteration. Instead, the `dst_reg++` in the SFPU microcode loop and the `TTI_SETRWC` calls in `_llk_math_eltwise_unary_sfpu_params_` handle address advancement explicitly.

The ABS operation does **not** configure ADDR_MOD_6 (it does not match any of the special-cased `SfpuType` values like `topk_local_sort`, `typecast`, `unary_max`, etc.).

**Wormhole vs Blackhole**: The ADDR_MOD_7 configuration is identical across both architectures for the ABS operation. The only architectural difference is in the int32 path where the Mod0 encoding for SFPLOAD/SFPSTORE differs: Wormhole uses `Mod0=4` with `AddrMod=3`, while Blackhole uses `Mod0=12` with `ADDR_MOD_7`. This reflects different INT32 format encodings in the two ISA variants.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How does the ABS (absolute value) unary SFPU operation work? Trace the call chain from the compute kernel API through the LLK layer down to the ckernel SFPU implementation."
   **Reason**: Needed to identify all files in the ABS call chain and understand the abstraction layers.
   **Key Findings**: Identified the four-layer call chain: compute_kernel_api.h -> llk_math_eltwise_unary_sfpu_abs.h -> ckernel_sfpu_abs.h, with params dispatch via llk_math_eltwise_unary_sfpu_params.h. Confirmed two variants exist: floating-point (using sfpi::abs) and int32 (using raw TTI_SFPABS).

2. **Query**: "How is the abs SFPU operation implemented in the LLK/ckernel layer?" (tt-llk repo)
   **Reason**: Needed LLK-level details on initialization, address mode configuration, and the params dispatch function.
   **Key Findings**: Confirmed ADDR_MOD_7 is used with zero increments. The init sequence calls _init_sfpu_config_reg(), configures address modes, and resets counters. The params function handles face iteration and DST pointer management.

3. **Query**: "What does the SFPABS instruction do? What are its operands, encoding fields, and behavior?" (tt-isa-documentation)
   **Reason**: Needed precise ISA-level documentation for the SFPABS instruction.
   **Key Findings**: SFPABS opcode is 0x7D, operates lanewise on LReg[VC] to LReg[VD]. Float mode clears sign bit (except -NaN). Int mode performs two's complement negation (except INT_MIN). Also documented SFPLOAD (opcode 0x70) and SFPSTORE (opcode 0x72) INT32 modes.

4. **Query**: "How is sfpi::abs(vFloat) implemented?" (sfpi repo)
   **Reason**: Needed to confirm the SFPI library implementation maps to the SFPABS hardware instruction.
   **Key Findings**: `sfpi::abs(vFloat)` calls `__builtin_rvtt_sfpabs(v.get(), SFPABS_MOD1_FLOAT)` which compiles to the SFPABS instruction with float mode. The `vInt` overload uses `SFPABS_MOD1_INT`.

### Confluence References
Not consulted. DeepWiki and source code provided sufficient detail for this simple operation.

### Glean References
Not consulted. The ABS operation uses well-documented SFPU instructions with no ambiguity requiring confidential hardware specifications.
