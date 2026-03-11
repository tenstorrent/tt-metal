## SFPU Kernel Implementation
This section provides a dedicated deep dive into the underlying SFPU kernel function that the compute kernel dispatches to for the WHERE ternary operation.

### SFPU Abstraction Layers

| Layer | File Path |
|-------|-----------|
| **API Header** | `tt_metal/hw/inc/api/compute/eltwise_unary/where.h` |
| **LLK Dispatch** | `tt_metal/hw/ckernels/{blackhole,wormhole_b0}/metal/llk_api/llk_sfpu/llk_math_eltwise_ternary_sfpu_where.h` |
| **Core SFPU Implementation** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/common/inc/sfpu/ckernel_sfpu_where.h` |
| **Parameters Dispatch** | `tt_metal/third_party/tt_llk/tt_llk_{blackhole,wormhole_b0}/llk_lib/llk_math_eltwise_ternary_sfpu_params.h` |

### Call Chain

1. The compute kernel (`ternary_sfpu_no_bcast_ttt.cpp`) calls `TERNARY_SFPU_OP_INIT()` which is macro-defined to `where_tile_init()`, and `TERNARY_SFPU_OP_FUNC(0, 1, 2, 0)` which is macro-defined to `where_tile<DataFormat::*>(0, 1, 2, 0)`.
2. `where_tile_init()` (in `where.h`) calls `llk_math_eltwise_ternary_sfpu_where_init<APPROX>()`, which calls `_llk_math_eltwise_ternary_sfpu_init_<SfpuType::where>()` to configure addr mods, then `ckernel::sfpu::_init_where_<APPROX>()` to set up SFPLOADMACRO instruction templates and macro configurations.
3. `where_tile<data_format>(0, 1, 2, 0)` calls `llk_math_eltwise_ternary_sfpu_where<APPROX, data_format>(0, 1, 2, 0)`, which calls `_llk_math_eltwise_ternary_sfpu_params_<APPROX>(ckernel::sfpu::_calculate_where_<APPROX, data_format, 8>, 0, 1, 2, 0, VectorMode::RC)`.
4. `_llk_math_eltwise_ternary_sfpu_params_` calls `_llk_math_eltwise_ternary_sfpu_start_` (sets DST write addr, stalls SFPU), then iterates over 4 faces in RC mode calling `_calculate_where_<APPROX, data_format, 8>(0, 1, 2, 0)` per face with `TTI_SETRWC` to advance the face pointer, and finally calls `_llk_math_eltwise_ternary_sfpu_done_()`.

### Annotated SFPU Kernel Source

```cpp
// File: tt_metal/third_party/tt_llk/tt_llk_blackhole/common/inc/sfpu/ckernel_sfpu_where.h

template <bool APPROXIMATION_MODE, DataFormat data_format, int ITERATIONS>
inline void _calculate_where_( // APPROXIMATION_MODE=false (unused), ITERATIONS=8
    const std::uint32_t dst_index_in0, const std::uint32_t dst_index_in1, const std::uint32_t dst_index_in2, const std::uint32_t dst_index_out)
{
    static_assert(
        data_format == DataFormat::Float32 || data_format == DataFormat::Float16_b || data_format == DataFormat::Int32 || data_format == DataFormat::UInt32,
        "Unsupported data format for _calculate_where_(). Only Float32, Int32, UInt32, and Float16_b are allowed.");

    // Compute byte offsets into DEST register space; each tile index is 32 rows, <<1 for 16-bit word addressing
    int offset0 = (dst_index_in0 * 32) << 1; // predicate tile offset
    int offset1 = (dst_index_in1 * 32) << 1; // true-value tile offset
    int offset2 = (dst_index_in2 * 32) << 1; // false-value tile offset

    // mod0 selects data width: LO16 for bfloat16, INT32 for 32-bit formats
    constexpr std::uint32_t mod0 = data_format == DataFormat::Float16_b ? InstrModLoadStore::LO16 : InstrModLoadStore::INT32;

#ifdef DISABLE_SFPLOADMACRO
    int offset3 = (dst_index_out * 32) << 1;

    lltt::record(0, 6); // Record 6 instructions into replay buffer 0
    TT_SFPLOAD(p_sfpu::LREG0, mod0, ADDR_MOD_7, offset0); // Load predicate row into LREG0
    TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, offset1); // Load true-value row into LREG1
    TTI_SFPSETCC(0, p_sfpu::LREG0, 0, sfpi::SFPSETCC_MOD1_LREG_EQ0); // Set LaneFlags: true where predicate==0
    TT_SFPLOAD(p_sfpu::LREG1, mod0, ADDR_MOD_7, offset2); // Load false-value row into LREG1 (only lanes where predicate==0)
    TTI_SFPENCC(0, 0, 0, sfpi::SFPENCC_MOD1_EU_R1); // Disable conditional execution, reset LaneFlags to all-true
    TT_SFPSTORE(p_sfpu::LREG1, mod0, ADDR_MOD_6, offset3); // Store result; ADDR_MOD_6 has dest.incr=2

#pragma GCC unroll 8
    for (int d = 0; d < ITERATIONS; d++)
    {
        lltt::replay(0, 6); // Replay the 6-instruction sequence for each of 8 rows per face
    }
#else
    if (dst_index_out == dst_index_in0)
    {
        // Implementation notes, see the original file for more details
        // Optimized 3-cycle path using SFPLOADMACRO when output overwrites predicate input

        load_replay_buf(
            0,
            3,
            [offset0, offset1, offset2]
            {
                TT_SFPLOADMACRO((0 << 2), mod0, ADDR_MOD_7, offset0); // Macro 0: loads predicate, schedules SFPSETCC+SFPSTORE
                TT_SFPLOADMACRO((2 << 2), mod0, ADDR_MOD_7, offset1); // Macro 2: loads true-value, schedules SFPENCC
                TT_SFPLOAD(0, mod0, ADDR_MOD_6, offset2);             // Load false-value; ADDR_MOD_6 dest.incr=2
            });

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            lltt::replay(0, 3);
        }
    }
    else
    {
        // Implementation notes, see the original file for more details
        // 4-cycle path using SFPLOADMACRO when output goes to a different register

        int offset3 = (dst_index_out * 32) << 1;

        load_replay_buf(
            0,
            4,
            [offset0, offset1, offset2, offset3]
            {
                TT_SFPLOADMACRO((1 << 2), mod0, ADDR_MOD_7, offset0); // Macro 1: loads predicate, schedules SFPSETCC
                TT_SFPLOADMACRO((2 << 2), mod0, ADDR_MOD_7, offset1); // Macro 2: loads true-value, schedules SFPENCC
                TT_SFPLOAD(0, mod0, ADDR_MOD_7, offset2);             // Load false-value
                TT_SFPSTORE(0, mod0, ADDR_MOD_6, offset3);            // Store result; ADDR_MOD_6 dest.incr=2
            });

#pragma GCC unroll 8
        for (int d = 0; d < ITERATIONS; d++)
        {
            lltt::replay(0, 4);
        }
    }
#endif
}

template <bool APPROXIMATION_MODE>
inline void _init_where_()
{
#ifndef DISABLE_SFPLOADMACRO
    // InstructionTemplate[0]: SFPSETCC with LREG_EQ0 mode — sets lane flags where predicate is zero
    TTI_SFPSETCC(0, 0, 12, 6); // imm12=12 selects template slot, mod1=6 is SFPSETCC_MOD1_LREG_EQ0

    // InstructionTemplate[1]: SFPENCC — disables conditional execution, resets LaneFlags
    TTI_SFPENCC(0, 0, 13, 0);

    // Macro 0: for where(a, b, c, a) — output overwrites first input
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 4; // =4: schedule InstructionTemplate[0] on simple unit
        constexpr std::uint32_t mad_bits    = 0;
        constexpr std::uint32_t round_bits  = 0;
        constexpr std::uint32_t store_bits  = 0x00 | 0x00 | (2 << 3) | 3; // =19: schedule SFPSTORE on store unit, delay=2

        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_LOWER, (mad_bits << 8) | simple_bits);  // Load lower 16 bits of Sequence[0]
        TTI_SFPLOADI(0, sfpi::SFPLOADI_MOD0_UPPER, (store_bits << 8) | round_bits); // Load upper 16 bits of Sequence[0]
        TTI_SFPCONFIG(0, 4 + 0, 0); // Write Sequence[0] to LoadMacroConfig (VD=4 => Sequence slot 0)
    }

    // Macro 1: for where(a, b, c, d) — output to different register
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 4; // =4: schedule InstructionTemplate[0] on simple unit
        constexpr std::uint32_t mad_bits    = 0;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 1, 1); // Write Sequence[1] lower 16 bits (VD=5)
    }

    // Macro 2: used by both paths — schedules SFPENCC (InstructionTemplate[1])
    {
        constexpr std::uint32_t simple_bits = 0x00 | 0x00 | (0 << 3) | 5; // =5: schedule InstructionTemplate[1] on simple unit
        constexpr std::uint32_t mad_bits    = 0;

        TTI_SFPCONFIG((mad_bits << 8) | simple_bits, 4 + 2, 1); // Write Sequence[2] lower 16 bits (VD=6)
    }

    // Misc config: UsesLoadMod0ForStore=1, WaitForElapsedInstructions=1 for all macros
    TTI_SFPCONFIG(0x770, 8, 1); // VD=8 => Misc configuration
#endif
}
```

### SFPU Instructions Used

| Instruction | Description |
|-------------|-------------|
| `SFPLOAD` / `TT_SFPLOAD` | Loads a row of 32 elements from a DEST register tile face into a local register (LREG). Supports LO16 (bfloat16) and INT32 data modes. |
| `SFPLOADMACRO` / `TT_SFPLOADMACRO` | Performs an SFPLOAD and simultaneously schedules additional instructions (simple, MAD, round, store) on other SFPU sub-units according to a pre-configured macro. Enables multi-issue execution achieving 3-4 cycles per row. |
| `SFPSETCC` / `TTI_SFPSETCC` | Sets per-lane condition flags (`LaneFlags`). In `SFPSETCC_MOD1_LREG_EQ0` mode, it sets the flag to true for each lane where LREG0 equals zero (i.e., where the predicate is false/zero). |
| `SFPENCC` / `TTI_SFPENCC` | Controls conditional execution mode. In `SFPENCC_MOD1_EU_R1` mode, it disables lane-flag-based predication and resets all LaneFlags to true, effectively ending the conditional block. |
| `SFPSTORE` / `TT_SFPSTORE` | Stores a local register (LREG) value back to a DEST register tile face row. Uses the same LO16/INT32 data mode as the loads. |
| `SFPLOADI` / `TTI_SFPLOADI` | Loads an immediate value into LREG0's lower or upper 16 bits. Used during init to stage macro sequence data before writing it to `LoadMacroConfig` via `SFPCONFIG`. |
| `SFPCONFIG` / `TTI_SFPCONFIG` | Writes configuration data to various SFPU control registers including `LoadMacroConfig` instruction templates, sequence slots, and misc settings. |
| `SETRWC` / `TTI_SETRWC` | Sets the read/write counter for DEST register addressing. Used in the params dispatch loop to advance the face pointer by 16 rows (8+8) between faces. |
| `STALLWAIT` / `TTI_STALLWAIT` | Stalls the math pipeline until the SFPU is ready (`STALL_SFPU`) or until SFPU completes (`WAIT_SFPU`). |
| `lltt::record` / `lltt::replay` | Software replay buffer mechanism: `record(buf, N)` records the next N instructions into buffer `buf`; `replay(buf, N)` replays them. Used to execute the same instruction sequence across all 8 rows of a 16-row face (ITERATIONS=8). |
| `load_replay_buf` | Higher-level wrapper that combines `lltt::record` with a lambda containing the instructions to record (used when SFPLOADMACRO is enabled). |

### SFPU Register Usage

| Register | Usage |
|----------|-------|
| **LREG0** | Holds the predicate row loaded from `DEST[dst_index_in0]`. Used by `SFPSETCC` to evaluate the zero-comparison condition. |
| **LREG1** | Initially loaded with the true-value row from `DEST[dst_index_in1]`. Then conditionally overwritten with the false-value row from `DEST[dst_index_in2]` — only the lanes where LREG0 was zero (predicate false) get overwritten due to `SFPSETCC`/`SFPENCC` predication. The final contents of LREG1 hold the where-selected result. (In `DISABLE_SFPLOADMACRO` mode.) |
| **LREG0 (SFPLOADMACRO path)** | When SFPLOADMACRO is enabled, all loads target LREG0. The macro-scheduled simple unit instructions (`SFPSETCC`, `SFPENCC`) operate on the previously loaded value. The conditional load of false-value into LREG0 (overwriting lanes where predicate==0) produces the result. |
| **DEST registers** | Three input tiles occupy DEST slots 0, 1, 2 (predicate, true-value, false-value). Output is written to DEST slot 0 (same as predicate in the typical `where(a, b, c, a)` calling convention). |
| **LaneFlags** | Per-lane boolean array set by `SFPSETCC`. Lanes where predicate==0 have `LaneFlags=true`, causing the subsequent false-value load to only modify those lanes. `SFPENCC` resets all flags. |
| **Replay buffer 0** | Stores the recorded instruction sequence (3, 4, or 6 instructions depending on path) for replay across all 8 rows of a face. |

### Address Mode Configuration

The WHERE operation configures two address modes:

**ADDR_MOD_7** (configured in `eltwise_ternary_sfpu_configure_addrmod<SfpuType::where>()`):
- `srca.incr = 0`, `srcb.incr = 0`, `dest.incr = 0`
- Used for SFPLOAD/SFPLOADMACRO instructions where the DEST address should not auto-increment (the replay mechanism handles row advancement implicitly via the offset parameter).

**ADDR_MOD_6** (configured in `eltwise_ternary_sfpu_configure_addrmod<SfpuType::where>()`):
- `srca.incr = 0`, `srcb.incr = 0`, `dest.incr = 2`
- Used for the final SFPSTORE (or the last SFPLOAD in the optimized path) where the DEST address needs to advance by 2 rows to move to the next row position within the face.

**Hardware generation differences:**
- **Wormhole B0**: The `_start_` function calls `math::set_addr_mod_base()` which sets the addr mod base register to 1, causing hardware to use addr mods 4..7 instead of 0..3. The SFPU kernel code references `ADDR_MOD_3` and `ADDR_MOD_2`, which with the base offset resolve to physical addr mods 7 and 6 respectively. The `_done_` function calls `math::clear_addr_mod_base()` to restore default.
- **Blackhole**: The `_start_` function does NOT call `set_addr_mod_base()`. The SFPU kernel code directly references `ADDR_MOD_7` and `ADDR_MOD_6`. The `_done_` function uses `TTI_SETC16(2, 0)` to explicitly clear the addr mod base. The net effect is the same physical address mode configuration on both architectures.

## External Knowledge Sources
### DeepWiki Queries
1. **Query**: "How is the WHERE (ternary) SFPU operation implemented? What compute kernel does it use and how does it dispatch to the SFPU?"
   **Reason**: Needed to understand the overall architecture and file locations for the WHERE operation's SFPU path.
   **Key Findings**: Identified compute kernel variants (no_bcast, col_scalar_bcast, row_bcast), the define mechanism (`TERNARY_SFPU_OP_INIT`/`TERNARY_SFPU_OP_FUNC`), and the LLK dispatch path through `llk_math_eltwise_ternary_sfpu_where`.

2. **Query**: "How is the ternary SFPU operation (where) implemented in the LLK layer?" (tt-llk repo)
   **Reason**: Needed detailed LLK call chain and SFPU kernel function signatures.
   **Key Findings**: Confirmed the params dispatch function iterates over faces, identified ADDR_MOD_7 (dest.incr=0) and ADDR_MOD_6 (dest.incr=2) configuration, and the `_calculate_where_` / `_init_where_` function pair.

3. **Query**: "What does SFPSETCC do with SFPSETCC_MOD1_LREG_EQ0? What does SFPENCC with SFPENCC_MOD1_EU_R1 do?" (tt-isa-documentation repo)
   **Reason**: Needed to understand the condition code mechanism that implements the where selection logic.
   **Key Findings**: `SFPSETCC` sets per-lane flags based on a register comparison (EQ0 sets flag where register is zero). `SFPENCC` controls whether lane flags are used for predication and can reset all flags. Together they implement conditional lane selection: set flags on predicate==0, conditionally load false-value (only on flagged lanes), then disable predication.

4. **Query**: "What is SFPLOADMACRO and how does it differ from SFPLOAD?" (tt-isa-documentation repo)
   **Reason**: The WHERE kernel heavily uses SFPLOADMACRO for performance optimization and needed to understand the macro scheduling mechanism.
   **Key Findings**: SFPLOADMACRO performs a load plus schedules up to 4 additional instructions across SFPU sub-units (simple, MAD, round, store) in parallel. Configuration uses `SFPCONFIG` to write instruction templates and sequence definitions to `LoadMacroConfig`. This enables the WHERE kernel to achieve 3-4 cycles per row instead of 6 cycles in the non-macro path.

### Confluence References
No Confluence references were needed for this analysis. The DeepWiki ISA documentation provided sufficient detail on the SFPU instructions used.

### Glean References
No Glean references were needed for this analysis.
