## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the MAXIMUM binary SFPU operation. The MAXIMUM operation computes `y = max(x0, x1)` element-wise and has three variants depending on data type: float (default), int32, and uint32.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/binary_max_min.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_max_min.h` |
| **Core SFPU Implementation** | `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h` |
| **Parameters Dispatch** | `llk_math_eltwise_binary_sfpu_params.h` (in tt_llk submodule, not directly in-tree) and `tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/llk_math_eltwise_binary_sfpu_init.h` |

### Call Chain

1. **Compute kernel** (`eltwise_binary_sfpu_kernel.cpp`): For each tile `i`, the `BINOP_INIT` macro expands to `binary_max_tile_init();` (called once per inner iteration), and `BINARY_SFPU_OP` expands to `binary_max_tile(i*2, i*2+1, i*2);`. Input A is at DST slot `i*2`, input B at `i*2+1`, and the result overwrites `i*2`.

2. **API Header** (`binary_max_min.h`): `binary_max_tile(idst0, idst1, odst)` calls `MATH((llk_math_eltwise_binary_sfpu_binary_max<APPROX>(idst0, idst1, odst, VectorMode::RC)))`.

3. **LLK Dispatch** (`llk_math_eltwise_binary_sfpu_max_min.h`): `llk_math_eltwise_binary_sfpu_binary_max<APPROX>()` calls `_llk_math_eltwise_binary_sfpu_params_<APPROXIMATE>(ckernel::sfpu::calculate_binary_max_min<true>, dst_index0, dst_index1, odst, vector_mode)`.

4. **Parameters Dispatch** (`llk_math_eltwise_binary_sfpu_params.h` in tt_llk): This function validates dst indices, calls `_llk_math_eltwise_binary_sfpu_start_` to prepare the SFPU, then invokes `calculate_binary_max_min<true>` four times (once per tile face in `VectorMode::RC`), advancing the DEST read/write pointer via `TTI_SETRWC` after each face.

5. **Core SFPU** (`ckernel_sfpu_binary_max_min.h`): `calculate_binary_max_min<IS_MAX_OP=true>` loads two values from DEST, performs `SFPSWAP` with min/max mode, and stores the maximum back.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/hw/ckernels/{arch}/metal/llk_api/llk_sfpu/ckernel_sfpu_binary_max_min.h
// NOTE: The Wormhole and Blackhole implementations are structurally identical. The only
// difference is the ADDR_MOD indices used (WH: ADDR_MOD_3/ADDR_MOD_2, BH: ADDR_MOD_7/ADDR_MOD_6)
// and the replay buffer API (WH: lltt::record/lltt::replay, BH: load_replay_buf/lltt::replay).
// The annotated source below uses the Wormhole variant for reference.

using namespace sfpi;

namespace ckernel {
namespace sfpu {

template <bool IS_MAX_OP = true, int ITERATIONS = 8>
inline void calculate_binary_max_min(const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // IS_MAX_OP=true for MAXIMUM, ITERATIONS=8 (one per row-pair across 4 faces)
    uint offset0 = (dst_index_in0 * 32) << 1;  // byte offset into DEST for input A tile
    uint offset1 = (dst_index_in1 * 32) << 1;  // byte offset into DEST for input B tile
    uint offset2 = (dst_index_out * 32) << 1;  // byte offset into DEST for output tile

#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Swap and store maximum in lreg1, minimum in lreg0
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_3, offset0);  // load A row from DEST
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::DEFAULT, ADDR_MOD_3, offset1);  // load B row from DEST
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);  // mod1=1: VD=min, VC=max
        TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0, InstrModLoadStore::DEFAULT, ADDR_MOD_2, offset2);
        // For IS_MAX_OP=true: store LREG1 (max) since SFPSWAP puts max in VC (LREG1)
        // ADDR_MOD_2 has dest.incr=2, auto-advancing DEST pointer by 2 rows per store
    }
#else
    // Implementation notes, see the original file for more details
    // Uses SFPLOADMACRO for 3-cycle-per-row throughput pipeline scheduling.

    constexpr int b = p_sfpu::LREG2;
    constexpr int c = p_sfpu::LREG3;

#pragma GCC unroll 8
    for (int i = 0; i < ITERATIONS; ++i) {
        int a = i & 1;  // alternate between p_sfpu::LREG0 and p_sfpu::LREG1
        TT_SFPLOADMACRO((0 << 2) | (a & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_3, offset0 | (a >> 2));
        // Macro 0: loads A row into LREG[a], schedules SFPSWAP (simple) and round-to-L16
        TT_SFPLOAD(b, InstrModLoadStore::DEFAULT, ADDR_MOD_3, offset1);
        // Standard load of B row into LREG2
        TT_SFPLOADMACRO((1 << 2) | (c & 3), InstrModLoadStore::DEFAULT, ADDR_MOD_2, offset2 | (c >> 2));
        // Macro 1: loads next A row into LREG[c] for pipelining, schedules store from L16
    }

    TTI_SFPNOP;  // pipeline drain NOPs
    TTI_SFPNOP;
    TTI_SFPNOP;
#endif
}

template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false, int ITERATIONS = 8>
inline void calculate_binary_max_min_int32(
    const uint dst_index_in0, const uint dst_index_in1, const uint dst_index_out) {
    // IS_MAX_OP=true, IS_UNSIGNED=false for int32 max; IS_UNSIGNED=true for uint32 max
    uint offset0 = (dst_index_in0 * 32) << 1;
    uint offset1 = (dst_index_in1 * 32) << 1;
    uint offset2 = (dst_index_out * 32) << 1;

#ifdef DISABLE_SFPLOADMACRO
#pragma GCC unroll 0
    for (int d = 0; d < ITERATIONS; d++) {
        // Swap and store maximum in lreg1, minimum in lreg0 (or reversed if unsigned)
        TT_SFPLOAD(p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_3, offset0);  // INT32 load mode
        TT_SFPLOAD(p_sfpu::LREG1, InstrModLoadStore::INT32, ADDR_MOD_3, offset1);
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
        // mod1=1 (signed): VD=min, VC=max using sign-magnitude compare
        // mod1=9 (unsigned): VD=max, VC=min (inverted default for unsigned correction)

        // Conditionally swap again to fix cases where SFPSWAP got the result backwards
        // SFPSWAP compares as sign-magnitude floats, so int32 values need correction
        TTI_SFPSETCC(0, p_sfpu::LREG0, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        // Sets condition code per-lane: for signed, CC=1 where LREG0 < 0; for unsigned, CC=1 where LREG0 >= 0
        TTI_SFPSETCC(0, p_sfpu::LREG1, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
        // AND with second register's condition
        TTI_SFPSWAP(0, p_sfpu::LREG1, p_sfpu::LREG0, sfpi::SFPSWAP_MOD1_SWAP);
        // mod1=0: unconditional swap, but only executes in lanes where CC is set
        TTI_SFPENCC(0, 0, 0, 0);  // disable conditional execution, restore all lanes active

        TT_SFPSTORE(IS_MAX_OP ? p_sfpu::LREG1 : p_sfpu::LREG0, InstrModLoadStore::INT32, ADDR_MOD_2, offset2);
    }
#else
    // Implementation notes, see the original file for more details
    // Uses SFPLOADMACRO + replay buffer for 5-cycle-per-row throughput.

    constexpr int a0 = p_sfpu::LREG0;
    constexpr int b0 = p_sfpu::LREG1;
    constexpr int a1 = p_sfpu::LREG2;
    constexpr int b1 = p_sfpu::LREG3;
    constexpr int c = p_sfpu::LREG7;

    lltt::record<lltt::NoExec>(0, 10);  // record 10 instructions into replay buffer slot 0

    // first iteration, with a0, b0, c
    TT_SFPLOADMACRO((0 << 2) | (a0 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset0 | (a0 >> 2));
    TT_SFPLOADMACRO((2 << 2) | (b0 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset1 | (b0 >> 2));
    TTI_SFPSETCC(0, a1, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPENCC(0, 0, 0, 0);
    TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_2, offset2 | (c >> 2));

    // second iteration, with a1, b1, c
    TT_SFPLOADMACRO((1 << 2) | (a1 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset0 | (a1 >> 2));
    TT_SFPLOADMACRO((2 << 2) | (b1 & 3), InstrModLoadStore::INT32, ADDR_MOD_3, offset1 | (b1 >> 2));
    TTI_SFPSETCC(0, a0, 0, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);
    TTI_SFPENCC(0, 0, 0, 0);
    TT_SFPLOADMACRO((3 << 2) | (c & 3), InstrModLoadStore::INT32, ADDR_MOD_2, offset2 | (c >> 2));

#pragma GCC unroll 4
    for (int i = 0; i < ITERATIONS / 2; ++i) {
        lltt::replay(0, 10);  // replay the 10-instruction sequence
    }

    if constexpr (ITERATIONS & 1) {
        lltt::replay(0, 5);
        TTI_SFPNOP;
        TTI_SFPNOP;
        lltt::replay(5 + 2, 2);
    } else {
        TTI_SFPNOP;
        TTI_SFPNOP;
        lltt::replay(2, 2);  // drain pipeline: replay SFPENCC + store
    }

    TTI_SFPNOP;
#endif
}

template <bool IS_MAX_OP = true>
inline void binary_max_min_init() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int b = p_sfpu::LREG2;

    // InstructionTemplate[0]: SFPSWAP for min/max
    TTI_SFPSWAP(0, b, 12, IS_MAX_OP ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    // mod1=9 for MAX: VD=max, VC=min; mod1=1 for MIN: VD=min, VC=max
    // VD=12 is a template placeholder, replaced by SFPLOADMACRO's VD at execution time

    // InstructionTemplate[1]: SFPSHFT2 identity shift (used as round-stage NOP with VD override)
    TTI_SFPSHFT2(0, 0, 13, 6);  // SFPSHFT2_MOD1_SHFT_IMM, template slot 13

    // Macro 0: schedule SFPSWAP in simple stage, round-to-L16 in round stage
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (1 << 3) | 4;
        // 0x80=enable, template_idx=1, delay=4 cycles
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (3 << 3) | 5;
        // 0x80=enable, 0x40=use L16 as VD, template_idx=3, delay=5
        constexpr uint store_bits = 0;

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);  // write to LoadMacroConfig.Sequence[0]
    }

    // Macro 1: schedule store from L16 to DEST
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (2 << 3) | 3;
        // 0x40=read from L16, template_idx=2, delay=3

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);  // write to LoadMacroConfig.Sequence[1]
    }

    // Misc config: StoreMod0=DEFAULT, UsesLoadMod0ForStore={1,1}, UnitDelayKind={1,1}
    TTI_SFPCONFIG(0x330, 8, 1);
#endif
}

template <bool IS_MAX_OP = true, bool IS_UNSIGNED = false>
inline void binary_max_min_int32_init() {
#ifndef DISABLE_SFPLOADMACRO
    constexpr int b0 = p_sfpu::LREG1;
    constexpr int b1 = p_sfpu::LREG3;

    // InstructionTemplate[0]: SFPSWAP for int32 min/max (first register pair)
    TTI_SFPSWAP(
        0, b0, 12, IS_MAX_OP ^ IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);
    // XOR logic: for signed max (true^false=true) => mod1=9 (VD=max);
    //            for unsigned max (true^true=false) => mod1=1 (VD=min), then corrected by SFPSETCC+SFPSWAP

    // InstructionTemplate[1]: SFPSWAP for second register pair
    TTI_SFPSWAP(
        0, b1, 13, IS_MAX_OP ^ IS_UNSIGNED ? 9 : sfpi::SFPSWAP_MOD1_VEC_MIN_MAX);

    // InstructionTemplate[2]: SFPSETCC for conditional correction
    TTI_SFPSETCC(0, 0, 14, IS_UNSIGNED ? sfpi::SFPSETCC_MOD1_LREG_GTE0 : sfpi::SFPSETCC_MOD1_LREG_LT0);

    // InstructionTemplate[3]: SFPSHFT2 identity shift (round-stage placeholder)
    TTI_SFPSHFT2(0, 0, 15, 6);  // SFPSHFT2_MOD1_SHFT_IMM

    // Macro 0-3: 5-cycle pipeline for int32 max with correction
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (3 << 3) | 4;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (5 << 3) | 7;
        constexpr uint store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 0, 0);
    }
    {
        constexpr uint simple_bits = 0x80 | 0x00 | (3 << 3) | 5;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (5 << 3) | 7;
        constexpr uint store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 1, 0);
    }
    {
        constexpr uint simple_bits = 0x00 | 0x00 | (4 << 3) | 6;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0x80 | 0x40 | (6 << 3) | 7;
        constexpr uint store_bits = 0;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 2, 0);
    }
    {
        constexpr uint simple_bits = 0;
        constexpr uint mad_bits = 0;
        constexpr uint round_bits = 0;
        constexpr uint store_bits = 0x00 | 0x40 | (4 << 3) | 3;
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits);
        TTI_SFPCONFIG(0, 4 + 3, 0);
    }

    // Misc: StoreMod0=DEFAULT, UsesLoadMod0ForStore={1,1,1,1}, UnitDelayKind={1,1,1,1}
    TTI_SFPCONFIG(0xff0, 8, 1);
#endif
}

}  // namespace sfpu
}  // namespace ckernel
```

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| **SFPLOAD** (`TT_SFPLOAD`) | Loads a 32-element vector from a DEST register row into an SFPU local register (LREG). Supports `DEFAULT` (float) and `INT32` load modes via `InstrModLoadStore`. |
| **SFPSTORE** (`TT_SFPSTORE`) | Stores a 32-element vector from an SFPU local register back to a DEST register row. Uses ADDR_MOD_2/6 with `dest.incr=2` to auto-advance the DEST pointer. |
| **SFPSWAP** (`TTI_SFPSWAP`) | Performs lanewise conditional swap between two LREGs. With `mod1=SFPSWAP_MOD1_VEC_MIN_MAX` (1): VD gets min, VC gets max. With `mod1=9` (any-other): VD gets max, VC gets min. With `mod1=SFPSWAP_MOD1_SWAP` (0): unconditional swap (used under conditional execution for int32 correction). Comparison uses sign-magnitude ordering. |
| **SFPSETCC** (`TTI_SFPSETCC`) | Sets the per-lane condition code based on a register test. `SFPSETCC_MOD1_LREG_LT0`: CC=1 where LREG < 0 (for signed int32). `SFPSETCC_MOD1_LREG_GTE0`: CC=1 where LREG >= 0 (for unsigned int32). Used to identify lanes where SFPSWAP's sign-magnitude comparison produced incorrect results for integer comparison. |
| **SFPENCC** (`TTI_SFPENCC`) | Disables conditional execution, restoring all lanes to active. Called after conditional correction swap to return to normal operation. |
| **SFPLOADMACRO** (`TT_SFPLOADMACRO`) | Combined load + multi-stage instruction scheduler. Performs an SFPLOAD and simultaneously schedules pre-configured instructions on the simple, MAD, round, and store sub-units. Enables 3-cycle-per-row throughput for float max and 5-cycle-per-row for int32 max. |
| **SFPLOADI** (`TTI_SFPLOADI`) | Loads an immediate value into LREG0 (lower or upper 16 bits). Used during init to configure `LoadMacroConfig.Sequence` bits for SFPLOADMACRO scheduling. |
| **SFPCONFIG** (`TTI_SFPCONFIG`) | Writes to SFPU configuration registers. Used to program `LoadMacroConfig` instruction templates and sequence configurations for SFPLOADMACRO. |
| **SFPSHFT2** (`TTI_SFPSHFT2`) | Barrel shift instruction. Used as an identity operation (shift by immediate 0) in the round stage of the SFPLOADMACRO pipeline to facilitate data movement through L16. |
| **SFPNOP** (`TTI_SFPNOP`) | No-operation. Used to drain the SFPLOADMACRO pipeline at the end of the loop. |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Holds input A row (float path: alternates with LREG1 in SFPLOADMACRO path; int32 path: first register pair input A) |
| **LREG1** | Holds input B row (float path, DISABLE_SFPLOADMACRO); also holds the maximum after SFPSWAP with mod1=1 since SFPSWAP puts max in VC. In int32 path: first register pair input B. |
| **LREG2** | Float SFPLOADMACRO path: holds input B row. Int32 path: second register pair input A (`a1`). |
| **LREG3** | Float SFPLOADMACRO path: pipeline staging register (`c`). Int32 path: second register pair input B (`b1`). |
| **LREG7** | Int32 SFPLOADMACRO path only: output staging register (`c`) for pipelined store. |
| **L16** | Bonus register only accessible via SFPLOADMACRO. Used as intermediate storage in the round stage to bridge the result from the simple stage (SFPSWAP) to the store stage. |
| **DEST[i*2]** | Source for input A tile and destination for the output (max result). |
| **DEST[i*2+1]** | Source for input B tile. |

### Address Mode Configuration

The address modes control DEST register auto-increment between SFPU iterations.

**Wormhole B0:**
- **ADDR_MOD_3**: `srca.incr=0, srcb.incr=0, dest.incr=0` -- No auto-increment. Used for SFPLOAD operations where the offset is managed explicitly by the loop.
- **ADDR_MOD_2**: `srca.incr=0, srcb.incr=0, dest.incr=2` -- Auto-increments DEST address by 2 rows after each SFPSTORE. This advances through the 8 row-pairs of a tile face across iterations.

**Blackhole:**
- **ADDR_MOD_7**: `srca.incr=0, srcb.incr=0, dest.incr=0` -- Equivalent to Wormhole's ADDR_MOD_3.
- **ADDR_MOD_6**: `srca.incr=0, srcb.incr=0, dest.incr=2` -- Equivalent to Wormhole's ADDR_MOD_2.

The address mode indices differ between architectures because the binary SFPU init function (`_llk_math_eltwise_binary_sfpu_init_`) configures different ADDR_MOD slots per architecture, but the functional behavior is identical. Both architectures use `eltwise_binary_sfpu_configure_addrmod` which sets ADDR_MOD_7/ADDR_MOD_6 (BH) or ADDR_MOD_3/ADDR_MOD_2 (WH) to avoid conflicting with address modes used by non-SFPU binary operations.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "What is the implementation of `_llk_math_eltwise_binary_sfpu_params_` function?"
   **Reason**: Needed to understand how the LLK params dispatcher invokes the core SFPU function with dst indices and manages tile face iteration.
   **Key Findings**: The function validates dst indices, calls `_llk_math_eltwise_binary_sfpu_start_` to prepare SFPU, invokes the compute function 4 times (for VectorMode::RC) with TTI_SETRWC to advance between faces, and calls `_llk_math_eltwise_binary_sfpu_done_` to finalize.

2. **Query**: "What is the implementation of `_llk_math_eltwise_binary_sfpu_init_`? What ADDR_MOD configurations does it set up?"
   **Reason**: Needed to determine the ADDR_MOD field values for the address modes used in the max/min kernel.
   **Key Findings**: ADDR_MOD_7 (BH) / ADDR_MOD_3 (WH) has all increments=0. ADDR_MOD_6 (BH) / ADDR_MOD_2 (WH) has dest.incr=2, others=0. These are configured via `eltwise_binary_sfpu_configure_addrmod`.

3. **Query**: "What does the SFPSWAP instruction do with mod1=SFPSWAP_MOD1_VEC_MIN_MAX?"
   **Reason**: SFPSWAP is the core instruction performing the max/min operation and understanding its mod1 modes is critical.
   **Key Findings**: mod1=1 (SFPSWAP_MOD1_VEC_MIN_MAX) puts min in VD and max in VC. mod1=9 (any undefined value) puts max in VD and min in VC. mod1=0 (SFPSWAP_MOD1_SWAP) performs unconditional swap. Comparison uses sign-magnitude ordering, which is correct for IEEE floats but requires correction for two's complement integers.

4. **Query**: "What is the SFPLOADMACRO instruction?"
   **Reason**: The optimized (non-DISABLE_SFPLOADMACRO) path relies heavily on SFPLOADMACRO for pipeline scheduling.
   **Key Findings**: SFPLOADMACRO combines a DEST-to-LREG load with scheduling of up to 4 additional instructions across simple, MAD, round, and store sub-units. Configuration is pre-set via SFPCONFIG writing to LoadMacroConfig. VD encoding splits across VDHi and VDLo fields. The 0x40 bit in sequence configuration routes through the L16 bonus register.

### Confluence References
Not consulted for this analysis -- DeepWiki and ISA documentation provided sufficient detail.

### Glean References
Not consulted for this analysis.
