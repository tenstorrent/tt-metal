// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"

// Debug bus and register array dump

namespace ckernel
{

struct dbg_cfgreg
{
    constexpr static std::uint32_t THREAD_0_CFG    = 0; // Thread 0 config
    constexpr static std::uint32_t THREAD_1_CFG    = 1; // Thread 1 config
    constexpr static std::uint32_t THREAD_2_CFG    = 2; // Thread 2 config
    constexpr static std::uint32_t HW_CFG_0        = 3; // Hardware config state 0
    constexpr static std::uint32_t HW_CFG_1        = 4; // Hardware config state 1
    constexpr static std::uint32_t HW_CFG_SIZE     = 187;
    constexpr static std::uint32_t THREAD_CFG_SIZE = THD_STATE_SIZE;
};

struct dbg_array_id
{
    constexpr static std::uint32_t SRCA = 0; // SrcA last used bank
    constexpr static std::uint32_t SRCB = 1; // SrcB last used bank
    constexpr static std::uint32_t DEST = 2; // Dest acc
};

struct dbg_daisy_id
{
    constexpr static std::uint32_t INSTR_ISSUE_0 = 4;
    constexpr static std::uint32_t INSTR_ISSUE_1 = 5;
    constexpr static std::uint32_t INSTR_ISSUE_2 = 6;
};

typedef struct
{
    std::uint32_t sig_sel    : 16;
    std::uint32_t daisy_sel  : 8;
    std::uint32_t rd_sel     : 4;
    std::uint32_t reserved_0 : 1;
    std::uint32_t en         : 1;
    std::uint32_t reserved_1 : 2;
} dbg_bus_cntl_t;

typedef union
{
    std::uint32_t val;
    dbg_bus_cntl_t f;
} dbg_bus_cntl_u;

;

typedef struct
{
    std::uint32_t en       : 1;
    std::uint32_t reserved : 31;
} dbg_array_rd_en_t;

typedef union
{
    std::uint32_t val;
    dbg_array_rd_en_t f;
} dbg_array_rd_en_u;

typedef struct
{
    std::uint32_t row_addr    : 12;
    std::uint32_t row_32b_sel : 4;
    std::uint32_t array_id    : 3;
    std::uint32_t bank_id     : 1;
    std::uint32_t reserved    : 12;
} dbg_array_rd_cmd_t;

typedef union
{
    std::uint32_t val;
    dbg_array_rd_cmd_t f;
} dbg_array_rd_cmd_u;

typedef struct
{
    std::uint32_t unp      : 2;
    std::uint32_t pack     : 4;
    std::uint32_t reserved : 26;
} dbg_soft_reset_t;

typedef union
{
    std::uint32_t val;
    dbg_soft_reset_t f;
} dbg_soft_reset_u;

;

template <ThreadId thread_id>
inline void dbg_thread_halt()
{
    static_assert(
        (thread_id == ThreadId::MathThreadId) || (thread_id == ThreadId::UnpackThreadId) || (thread_id == ThreadId::PackThreadId),
        "Invalid thread id set in dbg_wait_for_thread_idle(...)");

    if constexpr (thread_id == ThreadId::UnpackThreadId)
    {
        // Wait for all instructions on the running thread to complete
        tensix_sync();
        // Notify math thread that unpack thread is idle
        mailbox_write(ThreadId::MathThreadId, 1);
        // Wait for math thread to complete debug dump
        (void)mailbox_read(ThreadId::MathThreadId);
    }
    else if constexpr (thread_id == ThreadId::MathThreadId)
    {
        // Wait for all instructions on the running thread to complete
        tensix_sync();
        // Wait for unpack thread to complete
        (void)mailbox_read(ThreadId::UnpackThreadId);
        // Wait for previous packs to finish
        while (semaphore_read(semaphore::MATH_PACK) > 0)
        {
        }
    }
}

template <ThreadId thread_id>
inline void dbg_thread_unhalt()
{
    static_assert(
        (thread_id == ThreadId::MathThreadId) || (thread_id == ThreadId::UnpackThreadId) || (thread_id == ThreadId::PackThreadId),
        "Invalid thread id set in dbg_wait_for_thread_idle(...)");

    if constexpr (thread_id == ThreadId::MathThreadId)
    {
        // Reset pack 0 (workaround)
        dbg_soft_reset_u dbg_soft_reset;
        dbg_soft_reset.val    = 0;
        dbg_soft_reset.f.pack = 1;
        reg_write(RISCV_DEBUG_REG_SOFT_RESET_0, dbg_soft_reset.val);
        wait(5);
        dbg_soft_reset.val = 0;
        reg_write(RISCV_DEBUG_REG_SOFT_RESET_0, dbg_soft_reset.val);

        // Wait for all instructions to complete
        tensix_sync();

        // Unhalt unpack thread
        mailbox_write(ThreadId::UnpackThreadId, 1);
    }
}

// SFPLOAD/SFPSTORE mode that moves a dest row through an LReg without conversion: the raw 32-bit mode
// when the SFPU sees dest as 32-bit, else the raw 16-bit mode. The implied mode converts, and its FP32
// form flushes denormal bit patterns, which small Int32 values are.
// Reads the config from the RISC, so queued config writes must have landed first.
inline InstrModLoadStore dbg_dest_row_save_mode()
{
    return (cfg_read(ALU_ACC_CTRL_SFPU_Fp32_enabled_ADDR32) & ALU_ACC_CTRL_SFPU_Fp32_enabled_MASK) ? InstrModLoadStore::INT32 : InstrModLoadStore::LO16;
}

// RISC view of the backend configuration, in 32-bit words (BackendConfiguration.md): Config[2][CFG_STATE_SIZE * 4],
// then ThreadConfig[3][THD_STATE_SIZE] of {uint16_t Value, Padding[7]}.
struct dbg_cfg_layout
{
    constexpr static std::uint32_t STATE_BANKS            = 2;
    constexpr static std::uint32_t DUAL_WRITE_BANKS       = 0; // no ConfigDualWrite on this architecture
    constexpr static std::uint32_t WORDS_PER_STATE_ENTRY  = 4; // CFG_STATE_SIZE counts 16-byte entries
    constexpr static std::uint32_t WORDS_PER_THREAD_ENTRY = 4; // {uint16_t Value, Padding[7]}
    constexpr static std::uint32_t THREAD_CFG_BASE        = (STATE_BANKS + DUAL_WRITE_BANKS) * CFG_STATE_SIZE * WORDS_PER_STATE_ENTRY;
};

// The TRISC this code executes on. SFPLOAD, MOVDBGA2D and SFPSTORE add that thread's dest offset, which
// dbg_read_dest_offset reads from the same thread's ThreadConfig bank. Every compute build defines it (0 unpack,
// 1 math, 2 pack).
#if defined(COMPILE_FOR_TRISC)
constexpr std::uint32_t DBG_EXECUTING_TRISC = COMPILE_FOR_TRISC;
static_assert(DBG_EXECUTING_TRISC <= 2, "COMPILE_FOR_TRISC must select TRISC0, TRISC1 or TRISC2");
#else
#error "ckernel_debug.h: COMPILE_FOR_TRISC must name the executing TRISC"
#endif

// Index of one of the executing TRISC's thread-CFG words in the RISC view of the backend configuration.
constexpr std::uint32_t dbg_thread_cfg_word(const std::uint32_t cfg_addr32)
{
    return dbg_cfg_layout::THREAD_CFG_BASE + (DBG_EXECUTING_TRISC * THD_STATE_SIZE + cfg_addr32) * dbg_cfg_layout::WORDS_PER_THREAD_ENTRY;
}

// The executing thread's dest offset (DEST_TARGET_REG_CFG_MATH_Offset) as its next MOVA2D, SFPLOAD or SFPSTORE would
// add it. Read from the RISC, so queued SETC16 writes must have landed first (tensix_sync).
inline std::uint32_t dbg_read_dest_offset()
{
    volatile std::uint32_t *cfg = reinterpret_cast<volatile std::uint32_t *>(TENSIX_CFG_BASE);
    return (cfg[dbg_thread_cfg_word(DEST_TARGET_REG_CFG_MATH_Offset_ADDR32)] & DEST_TARGET_REG_CFG_MATH_Offset_MASK) >> DEST_TARGET_REG_CFG_MATH_Offset_SHAMT;
}

// Physical dest row of a logical row as the debug bus addresses it (Dst.md): a 16-bit DEST maps rows one to one; a
// 32-bit DEST spreads each logical row over two physical rows, with the high halves in the row returned here and the
// low halves 8 rows further on. Reads the config from the RISC, so queued config writes must have landed first.
inline std::uint32_t dbg_dest_physical_row(std::uint32_t row)
{
    if (cfg_read(ALU_ACC_CTRL_Fp32_enabled_ADDR32) & ALU_ACC_CTRL_Fp32_enabled_MASK)
    {
        row = ((row & 0x1f8) << 1) | (row & 0x207);
    }
    return row;
}

// Dump one row of SrcA, SrcB or DEST over the debug bus into rd_data[8], from inside a dbg_thread_halt /
// dbg_thread_unhalt bracket (the math thread for every in-tree caller). With a 32-bit DEST each word holds the high
// 16 bits of two datums; the low halves live in the physical row 8 further on.
//
// The SrcA dump stages the row in dest at the executing thread's dest offset and restores what it overwrote
// before returning; the thread's dest offset is left untouched. Call it only while that offset points at a tile
// written in the current section -- after an op, before the section is handed to the packer -- because the save
// and restore footprint is the aligned group of four rows holding the row at the dest offset (and, with a 32-bit
// DEST, the rows 8 further on that hold their low halves), and those rows must have been written: rows left
// undefined by a ZEROACC must not be borrowed, as an SFPLOAD of one is undefined behaviour and the restore would
// define the row with whatever it returned.
inline void dbg_get_array_row(const std::uint32_t array_id, const std::uint32_t row_addr, std::uint32_t *rd_data)
{
    // Dest offset is added to row_addr to dump currently used half of the dest accumulator (SyncHalf dest mode)
    std::uint32_t dest_offset = 0;
    if (array_id == dbg_array_id::DEST)
    {
        dest_offset = get_dest_buffer_base();
    }

    // The same mode must save and restore the borrowed row; read it once, after queued config writes land.
    InstrModLoadStore save_mode = InstrModLoadStore::LO16;
    // Physical row the SrcA row is staged in: the row at the thread's dest offset, which SFPLOAD, MOVDBGA2D and
    // SFPSTORE all address relative to (SFPLOAD.md, the MOVA2D model). The offset is left where the caller put it,
    // so the footprint stays on the tile the caller is working on (see the contract above).
    std::uint32_t staged_row = 0;
    // SFPLOAD and SFPSTORE add the dest offset to their immediate, then take an aligned group of four rows from
    // Addr & ~3 and the even or odd columns from Addr & 2 (SFPLOAD.md, SFPSTORE.md; the sum wraps at 10 bits). An
    // offset with bit 1 set would turn immediates 0 / 2 into the odd columns of its group and the even columns of
    // the next one, leaving the even columns of the staged row unsaved. Cancel that bit in the immediates so they
    // select the even and odd columns of the group that holds the staged row.
    constexpr std::uint32_t sfpu_dest_addr_mask = 0x3ff;
    std::uint32_t even_cols_imm                 = 0;
    std::uint32_t odd_cols_imm                  = 2;
    if (array_id == dbg_array_id::SRCA)
    {
        tensix_sync();
        save_mode  = dbg_dest_row_save_mode();
        const std::uint32_t dest_offset_row = dbg_read_dest_offset();
        staged_row                          = dbg_dest_physical_row(dest_offset_row);
        even_cols_imm                       = (0u - (dest_offset_row & 2)) & sfpu_dest_addr_mask;
        odd_cols_imm                        = (even_cols_imm + 2) & sfpu_dest_addr_mask;

        // When SrcA array is selected we need to copy row from src register into dest to be able to dump data
        // Dump from SrcA array is not supported
        // Save the dest row at the thread's dest offset to SFPU registers
        // Move SrcA into that row
        // Dump that row
        // Restore it
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 0},
            .srcb = {.incr = 0, .clr = 0, .cr = 0},
            .dest = {.incr = 0, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_0);

        // Clear counters
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);
        TT_SFPLOAD(p_sfpu::LREG3, save_mode, 0 /*sfpu_addr_mode*/, even_cols_imm); // Save the staged row's group (even cols) to LREG_3
        TT_SFPLOAD(p_sfpu::LREG4, save_mode, 0 /*sfpu_addr_mode*/, odd_cols_imm);  // Save the staged row's group (odd cols)  to LREG_4

        TTI_STALLWAIT(p_stall::STALL_MATH, p_stall::SFPU1);

        // Move to the last used bank. NOTE: this hands the bank being left to the unpackers as
        // well as flipping -- see CLEARDVALID's model -- so across this bracket's two flips BOTH
        // SrcA banks end up owned by the unpackers and the caller's SrcA validity is not restored.
        // Flipping without giving the bank away needs SETRWC under CLR_DVALID_SrcA_Disable. A
        // SETDVALID pairing like the SrcB branch below is available on this architecture and is
        // the other candidate for that follow-up; it is ruled out only on Blackhole, where a raw
        // SETDVALID is unsupported.
        TTI_CLEARDVALID(0b01, 0);

        // With a 32-bit DEST the move has to write 32-bit datums at 32-bit addresses, which MOVDBGA2D does only when
        // the SrcA format is TF32 (MOVA2D model: UseDst32b = SrcAFmt == TF32; a mismatch with the DEST width is
        // undefined behaviour). Blackhole forces TF32 under fp32 accumulation, Wormhole does not, so force it here
        // the way the 32-bit datacopy moves do: through the format override, lifted once the move has completed.
        // Without it the row would land in 16-bit addressing while the save, restore and read-back use the
        // 32-bit rows. The caller may be inside a sequence that set the override itself (transpose_dest_32b), so
        // both fields are handed back exactly as they were found, not cleared.
        constexpr std::uint32_t srca_fmt_select_mask = ALU_FORMAT_SPEC_REG_SrcA_val_MASK | ALU_FORMAT_SPEC_REG_SrcA_override_MASK;
        static_assert(ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32 == ALU_FORMAT_SPEC_REG_SrcA_override_ADDR32, "SrcA_val and SrcA_override must share a word");
        const bool dest_32b                 = (cfg_read(ALU_ACC_CTRL_Fp32_enabled_ADDR32) & ALU_ACC_CTRL_Fp32_enabled_MASK) != 0;
        const std::uint32_t caller_srca_fmt = cfg_read(ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32) & srca_fmt_select_mask; // read after the entry drain
        if (dest_32b)
        {
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_override_RMW>(1);
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_RMW>(to_underlying(DataFormat::Tf32));
        }

        // Copy single row from SrcA[row_addr] to the row at the dest offset
        TT_MOVDBGA2D(0, row_addr, 0, p_mova2d::MOV_1_ROW, 0);

        // Wait for TT instructions to complete
        tensix_sync();

        if (dest_32b)
        {
            // The caller's SrcA format selection, both fields in one write; the final drain lands it before return.
            cfg_reg_rmw_tensix<ALU_FORMAT_SPEC_REG_SrcA_val_ADDR32, 0, srca_fmt_select_mask>(caller_srca_fmt);
        }
    }
    else if (array_id == dbg_array_id::SRCB)
    {
        addr_mod_t {
            .srca = {.incr = 0, .clr = 0, .cr = 0},
            .srcb = {.incr = 0, .clr = 0, .cr = 0},
            .dest = {.incr = 0, .clr = 0, .cr = 0},
        }
            .set(ADDR_MOD_0);

        // Clear counters
        TTI_SETRWC(p_setrwc::CLR_NONE, 0, 0, 0, 0, p_setrwc::SET_ABD_F);

        // Move to the last used bank
        TTI_SETDVALID(0b10);
        TTI_CLEARDVALID(0b10, 0);

        // Latch debug values
        TTI_SETDVALID(0b10);
        TTI_SHIFTXB(ADDR_MOD_0, 0, row_addr >> 1);
        TTI_CLEARDVALID(0b10, 0);

        // Wait for TT instructions to complete
        tensix_sync();

        /*
        // Get last used bank id via debug bus
        dbg_bus_cntl_u dbg_bus_cntl;
        dbg_bus_cntl.val = 0;
        dbg_bus_cntl.f.rd_sel = 6; // Read bits 111:96, 110 (write_math_id_reg)
        dbg_bus_cntl.f.sig_sel = 0x4<<1;
        dbg_bus_cntl.f.daisy_sel = dbg_daisy_id::INSTR_ISSUE_2;
        dbg_bus_cntl.f.en = 1;
        reg_write(RISCV_DEBUG_REG_DBG_BUS_CNTL_REG, dbg_bus_cntl.val);
        wait (5); // Wait for value to get stable
        srcb_bank_id = ((reg_read(RISCV_DEBUG_REG_DBG_RD_DATA)) & 0x4000) ? 0 : 1;

        // Disable debug bus
        dbg_bus_cntl.val = 0;
        reg_write(RISCV_DEBUG_REG_DBG_BUS_CNTL_REG, dbg_bus_cntl.val);
        */
    }

    // Get actual row address and array id used in hw. The SrcA row sits where MOVDBGA2D staged it.
    std::uint32_t hw_row_addr = (array_id == dbg_array_id::SRCA) ? staged_row : ((array_id == dbg_array_id::DEST) ? dest_offset + row_addr : row_addr & 0x1);

    std::uint32_t hw_array_id = (array_id == dbg_array_id::SRCA) ? dbg_array_id::DEST : array_id;

    std::uint32_t hw_bank_id = 0;

    bool sel_datums_15_8 = hw_array_id != dbg_array_id::DEST;

    dbg_array_rd_en_u dbg_array_rd_en;
    dbg_array_rd_en.val  = 0;
    dbg_array_rd_en.f.en = 0x1;
    reg_write(RISCV_DEBUG_REG_DBG_ARRAY_RD_EN, dbg_array_rd_en.val);

    dbg_array_rd_cmd_u dbg_array_rd_cmd;
    dbg_array_rd_cmd.val        = 0;
    dbg_array_rd_cmd.f.array_id = hw_array_id;
    dbg_array_rd_cmd.f.bank_id  = hw_bank_id;

    for (std::uint32_t i = 0; i < 8; i++)
    {
        dbg_array_rd_cmd.f.row_addr    = sel_datums_15_8 ? (hw_row_addr | ((i >= 4) ? (1 << 6) : 0)) : hw_row_addr;
        dbg_array_rd_cmd.f.row_32b_sel = i;
        reg_write(RISCV_DEBUG_REG_DBG_ARRAY_RD_CMD, dbg_array_rd_cmd.val);
        wait(5); // Wait for value to get stable
        rd_data[i] = reg_read(RISCV_DEBUG_REG_DBG_ARRAY_RD_DATA);
    }

    // Disable debug control
    dbg_array_rd_en.val = 0;
    reg_write(RISCV_DEBUG_REG_DBG_ARRAY_RD_EN, dbg_array_rd_en.val);

    // Restore the borrowed row
    if (array_id == dbg_array_id::SRCA)
    {
        TT_SFPSTORE(p_sfpu::LREG3, save_mode, 0 /*sfpu_addr_mode*/, even_cols_imm); // Restore the staged row's group (even cols) from LREG_3
        TT_SFPSTORE(p_sfpu::LREG4, save_mode, 0 /*sfpu_addr_mode*/, odd_cols_imm);  // Restore the staged row's group (odd cols) from LREG_4
        // Move to the current bank
        TTI_CLEARDVALID(1, 0);
        // The restore is only queued at this point; a RISC reader of dest right after the call (a debug-bus read
        // of the same row, say) would otherwise see the staged SrcA row. Drain so the restore holds on return.
        tensix_sync();
    }
}

inline std::uint32_t dbg_read_cfgreg(const std::uint32_t cfgreg_id, const std::uint32_t addr)
{
    std::uint32_t hw_base_addr = 0;

    switch (cfgreg_id)
    {
        case dbg_cfgreg::HW_CFG_1:
            hw_base_addr = dbg_cfgreg::HW_CFG_SIZE;
            break;
        case dbg_cfgreg::THREAD_0_CFG:
        case dbg_cfgreg::THREAD_1_CFG:
        case dbg_cfgreg::THREAD_2_CFG:
            hw_base_addr = 2 * dbg_cfgreg::HW_CFG_SIZE + cfgreg_id * dbg_cfgreg::THREAD_CFG_SIZE;
            break;
        default:
            break;
    }

    std::uint32_t hw_addr = hw_base_addr + (addr & 0x7ff); // hw address is 4-byte aligned
    reg_write(RISCV_DEBUG_REG_CFGREG_RD_CNTL, hw_addr);

    wait(1);

    return reg_read(RISCV_DEBUG_REG_CFGREG_RDDATA);
}

} // namespace ckernel
