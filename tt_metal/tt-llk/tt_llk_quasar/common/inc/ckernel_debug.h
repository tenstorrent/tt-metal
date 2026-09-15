// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ckernel.h"
#include "ckernel_defs.h"
#include "ckernel_dest.h"

// Quasar HAS a Tensix debug bus: t6_debug_map.h defines its registers (DBG_BUS_CTRL,
// TENSIX_CREG_READ, DBG_ARRAY_RD_EN/CMD/DATA). What is missing is only the RISCV_DEBUG_REG_* wrapper
// macros Wormhole/Blackhole drive it through (dbg_get_array_row, dbg_read_cfgreg) -- those are
// commented out in quasar/tensix.h and not wired up here yet. So rather than program the debug-bus
// array read, DEST is read through the RISC-V memory-mapped window at RISCV_DEST_START_ADDR, after
// programming the reading RISC's RISC_DEST_ACCESS_CTRL section with configure_dest_access(fmt)
// (see ckernel_dest.h). The helpers below expose that read at row granularity for the
// dprint_tensix DEST dump; they keep only a small per-row buffer so no large stack allocation is
// needed on the TRISC.
//
// The dbg_thread_halt/dbg_thread_unhalt choreography below IS ported from Blackhole. It is not a
// debug-bus operation -- it is a pure inter-thread rendezvous over the Tensix mailboxes, and Quasar's
// mailbox register layout is identical to Blackhole's. It lets the math reader quiesce the unpack
// thread around a DEST read even mid-pipeline, where kernel-level phase separation is not available.
// PACK IS DEFERRED for now: only the unpack<->math rendezvous is implemented. Blackhole additionally
// drains the math<->pack (MATH_PACK) semaphore so pack is not reading DEST during the dump; on these
// datacopy tests tile_regs_acquire already drains MATH_PACK before the copy, so it is not needed yet.

namespace ckernel
{

// ---------------------------------------------------------------------------
// DEST-read thread rendezvous (mailbox + semaphore choreography, ported from Blackhole).
//
// Quasar mailbox specifics vs Blackhole (see ckernel.h):
//   * Mailboxes are addressed by NEO-cluster-LOCAL role 0..3 (Unpack=0, Math=1, Pack=2), which is
//     COMPILE_FOR_TRISC % 4 -- not the cluster-global processor id. The ThreadId enum values below
//     already are those local roles, so passing ThreadId::MathThreadId etc. is correct.
//   * mailbox_write/read assert against self-loopback (thread != COMPILE_FOR_TRISC % 4). This
//     choreography only ever names the *other* thread as the rendezvous channel, so it satisfies
//     the assert: unpack talks on the Math channel, math talks on the Unpack channel.
//   * The mailbox read is HW-blocking (stalls until the peer posts) -- identical to Blackhole. The
//     normal Quasar pipeline syncs on ckernel::trisc::semaphore::{UNPACK_MATH,MATH_PACK,PACK_UNPACK},
//     never on the mailboxes, so this rendezvous cannot collide with pipeline traffic.
//
// Only the unpack and math threads rendezvous. Pack is deferred (see the header note): coordinating
// pack (Blackhole's MATH_PACK drain) is not needed for the current datacopy tests because
// tile_regs_acquire already drains MATH_PACK on math before the copy that feeds the read.
// ---------------------------------------------------------------------------
template <ThreadId thread_id>
inline void dbg_thread_halt()
{
    static_assert(thread_id == UnpackThreadId || thread_id == MathThreadId, "dbg_thread_halt: thread_id must be Unpack or Math (pack deferred)");

    if constexpr (thread_id == UnpackThreadId)
    {
        // Drain this thread's own instructions, tell math we are idle, then block until math
        // releases us in dbg_thread_unhalt.
        tensix_sync();
        mailbox_write(MathThreadId, 1);
        volatile std::uint32_t ack = mailbox_read(MathThreadId);
        (void)ack;
    }
    else // MathThreadId
    {
        // Drain math, then wait for unpack to report idle before reading DEST.
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
        // Release the unpack thread. (Blackhole's unhalt also issues a pack soft-reset workaround via
        // RISCV_DEBUG_REG_SOFT_RESET_0; not applicable here -- pack is deferred and this path only
        // reads DEST, never mutating pipeline state.)
        tensix_sync();
        mailbox_write(UnpackThreadId, 1);
    }
    // UnpackThreadId: nothing to do.
}

// Read one DEST row of sixteen 16-bit datums (Float16 / Float16_b / UInt16) as 8 packed uint32
// words (each word = [hi datum << 16 | lo datum]). `logical_row` is the absolute row index within
// DEST (tile_id * NUM_ROWS_PER_TILE + row). The caller must have programmed the reading RISC's
// section via configure_dest_access(fmt) first.
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

// Read one DEST row of sixteen 32-bit datums (Float32 / Int32 / UInt32) as 16 uint32 words. With
// the section programmed for Float32 + swizzling, the memory-mapped window already presents the
// reconstructed float32 values contiguously (16 per row), matching the FPU's view.
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
