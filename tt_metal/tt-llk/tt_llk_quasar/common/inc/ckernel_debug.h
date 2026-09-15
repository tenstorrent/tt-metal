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
// quiesces the unpack thread around a math-side DEST read, making the read safe mid-pipeline. Pack is
// deferred -- only the unpack<->math rendezvous is implemented (dbg_thread_halt<Math> reads DEST while
// dbg_thread_halt<Unpack> is stalled). It uses the Tensix mailboxes, not the debug bus.
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
        // Drain, signal idle to math, then block until math releases us in dbg_thread_unhalt.
        tensix_sync();
        mailbox_write(MathThreadId, 1);
        volatile std::uint32_t ack = mailbox_read(MathThreadId);
        (void)ack;
    }
    else // MathThreadId
    {
        // Drain, then wait for unpack to report idle before reading DEST.
        tensix_sync();
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
        // Release the unpack thread.
        tensix_sync();
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

} // namespace ckernel
