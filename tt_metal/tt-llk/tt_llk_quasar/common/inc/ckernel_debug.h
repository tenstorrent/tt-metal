// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_dest.h"

// DEST-register debug helpers for Quasar.
//
// dbg_thread_halt / dbg_thread_unhalt: an inter-thread mailbox rendezvous (ported from Blackhole) that
// quiesces the unpack thread around a math-side DEST read. Pack is deferred -- only the unpack<->math
// rendezvous is implemented (dbg_thread_halt<Math> reads DEST while dbg_thread_halt<Unpack> is
// stalled). It uses the Tensix mailboxes, not the debug bus.
//
// Blackhole's dbg_thread_halt<Math> also drains pack, by spinning until semaphore::MATH_PACK reads 0.
// This port deliberately does not, and neither that spin nor an equivalent assert would be correct
// here: MATH_PACK is a counting semaphore that _llk_math_pack_sync_init_ seeds with a maximum of 1
// under DstSync::SyncFull but 2 under DstSync::SyncHalf, and _llk_math_wait_for_dest_available_ is a
// SEMWAIT on STALL_ON_MAX, so it releases math as soon as the count drops below that maximum. Under
// SyncHalf a count of 1 is therefore legal on entry here, with the previously committed tile still
// packing out of the other bank. Keeping pack off the bank being read is instead the caller's
// responsibility, discharged by reading inside the tile_regs_acquire() ... tile_regs_commit() window
// -- see dprint_tensix_dest_reg in api/debug/dprint_tensix.h.
//
// dbg_read_dest_row_*: read one DEST row through the RISC memory-mapped window at RISCV_DEST_START_ADDR.
// The caller must program the reading RISC's section via configure_dest_access() (see ckernel_dest.h)
// first.

namespace ckernel
{

// Quasar mailboxes are addressed by NEO-cluster-local role (Unpack=0, Math=1, Pack=2 == COMPILE_FOR_TRISC
// % 4); mailbox_read blocks until the peer posts. Each thread only ever names the *other* thread as the
// channel, satisfying the self-loopback assert. Unpack talks on the Math channel, Math on the Unpack
// channel.
template <ThreadId thread_id>
inline void dbg_thread_halt()
{
    static_assert(thread_id == UnpackThreadId || thread_id == MathThreadId, "dbg_thread_halt: thread_id must be Unpack or Math (pack deferred)");

    if constexpr (thread_id == UnpackThreadId)
    {
        // Wait for this thread's in-flight unpacks (issued through the MOP/replay) to finish, signal idle
        // to math, then block until math releases us in dbg_thread_unhalt.
        bstatus_u busy {};
        busy.mop    = 1;
        busy.replay = 1;
        busy.unpack = 1;
        wait_bstatus_low(busy.val);
        mailbox_write(MathThreadId, 1);
        volatile std::uint32_t ack = mailbox_read(MathThreadId);
        (void)ack;
    }
    else // MathThreadId
    {
        // Wait for this thread's FPU/SFPU writes to DEST to finish, then for unpack to report idle.
        bstatus_u busy {};
        busy.mop    = 1;
        busy.replay = 1;
        busy.fpu    = 1;
        busy._sfpu  = 1;
        wait_bstatus_low(busy.val);
        volatile std::uint32_t idle = mailbox_read(UnpackThreadId);
        (void)idle;
    }
}

template <ThreadId thread_id>
inline void dbg_thread_unhalt()
{
    static_assert(thread_id == UnpackThreadId || thread_id == MathThreadId, "dbg_thread_unhalt: thread_id must be Unpack or Math (pack deferred)");

    if constexpr (thread_id == MathThreadId)
    {
        // Release the unpack thread. Math's only Tensix work since dbg_thread_halt, the configure_dest_access
        // config writes, was waited on before the DEST read, and the read itself is RISC loads through the
        // MMIO window, so there is nothing left to wait for.
        mailbox_write(UnpackThreadId, 1);
    }
    // UnpackThreadId: nothing to do.
}

// Read one DEST row of sixteen 16-bit datums (Float16 / Float16_b / UInt16) into 8 packed uint32 words
// ([hi << 16 | lo]). logical_row is the absolute DEST row index (tile_id * NUM_ROWS_PER_TILE + row).
inline void dbg_read_dest_row_16b(std::uint32_t logical_row, std::uint32_t rd[8])
{
    volatile std::uint16_t* addr = reinterpret_cast<volatile std::uint16_t*>(RISCV_DEST_START_ADDR);
    const std::uint32_t base     = logical_row * 16;
    for (int i = 0; i < 8; ++i)
    {
        const std::uint32_t lo = addr[base + 2 * i];
        const std::uint32_t hi = addr[base + 2 * i + 1];
        rd[i]                  = lo | (hi << 16);
    }
}

// Read one DEST row of sixteen 32-bit datums (Float32 / Int32) into 16 uint32 words. With the section
// programmed for Float32 + swizzling, the window presents the reconstructed float32 values contiguously.
inline void dbg_read_dest_row_32b(std::uint32_t logical_row, std::uint32_t rd[16])
{
    volatile std::uint32_t* addr = reinterpret_cast<volatile std::uint32_t*>(RISCV_DEST_START_ADDR);
    const std::uint32_t base     = logical_row * 16;
    for (int i = 0; i < 16; ++i)
    {
        rd[i] = addr[base + i];
    }
}

// Read one DEST row of sixteen 8-bit datums (Int8) into 4 packed uint32 words, element i in byte i.
inline void dbg_read_dest_row_8b(std::uint32_t logical_row, std::uint32_t rd[4])
{
    volatile std::uint8_t* addr = reinterpret_cast<volatile std::uint8_t*>(RISCV_DEST_START_ADDR);
    const std::uint32_t base    = logical_row * 16;
    for (int i = 0; i < 4; ++i)
    {
        rd[i] =
            addr[base + 4 * i] | (addr[base + 4 * i + 1] << 8) | (addr[base + 4 * i + 2] << 16) | (static_cast<std::uint32_t>(addr[base + 4 * i + 3]) << 24);
    }
}

} // namespace ckernel
