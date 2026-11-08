// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The token loops' kernel_main (mb_loop.hpp): included after the kernel's MB_LOOP_BODY. Without MB_LOOPS: kernel_main
// with MB_CHIP (the chip's args first, then the body), else nothing.
#pragma once

#include "ttnn/cpp/ttnn/operations/experimental/moe_block/common/kernels/mb_chip.hpp"

#ifdef MB_LOOPS
#include "internal/circular_buffer_init.h"
#include "internal/tt-1xx/cache.h"
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#define MB_LOOP_DM 1
#include "stream_io_map.h"
#endif

// (the loop's machinery out of line and for size: it runs a few times a call, the kernels' L1 text is tight)
#define MB_LOOP_FN __attribute__((noinline, optimize("Os")))
namespace mb_loop {
constexpr uint32_t LT = MB_LT;
#if defined(COMPILE_FOR_BRISC)
constexpr uint32_t RID = 0;
#elif defined(COMPILE_FOR_NCRISC)
constexpr uint32_t RID = 1;
#elif defined(UCK_CHLKC_UNPACK)
constexpr uint32_t RID = 2;
#elif defined(UCK_CHLKC_MATH)
constexpr uint32_t RID = 3;
#else
constexpr uint32_t RID = 4;
#endif
constexpr uint32_t GO = 80, CNT = 96;

FORCE_INLINE uint32_t t(uint32_t i) { return get_common_arg_val<uint32_t>(LT + i); }
FORCE_INLINE volatile tt_l1_ptr uint32_t* word(uint32_t off) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t(0) + off);
}
FORCE_INLINE void wait_at_least(uint32_t off, uint32_t v) {
    while (*word(off) < v) {
        invalidate_l1_cache();
    }
}

// (every RISC) loop l is over here: my NoC traffic out of L1 and acked / the Tensix pipeline idle
MB_LOOP_FN void finish() {
#ifdef MB_LOOP_DM
    noc_async_full_barrier();
    noc_async_posted_writes_flushed();
#else
    ckernel::tensix_sync();
#endif
}

#if defined(MB_LOOP_DM) && defined(MB_LOOP_LEAD)
FORCE_INLINE uint32_t me() { return get_absolute_logical_x() | (uint32_t(get_absolute_logical_y()) << 8); }
// the table's roles: past the common deltas and the per-core sections
MB_LOOP_FN uint32_t roles_at() {
    uint32_t i = 6 + 2 * t(5);
    for (uint32_t s = 0, np = t(i++); s < np; ++s) {
        i += 2 + 2 * t(i + 1);
    }
    return i;
}
// my core's role if this kernel leads it: its RISCs (bits 8..12 as the table's) and semaphores; 0: not the lead
struct Role {
    uint32_t riscs = 0, sems = 0;
};
MB_LOOP_FN Role role() {
    uint32_t i = roles_at();
    if (t(i) >> 31) {  // (every core of the kernel alike)
        return Role{t(i + 1), t(i + 2)};
    }
    const uint32_t m = me();
    for (uint32_t s = 0, nl = t(i++); s < nl; ++s, i += 3) {
        if (t(i) == m) {
            return Role{t(i + 1), t(i + 2)};
        }
    }
    return Role{};
}

// (lead) one phase of the chip's barrier: count on the coordinator, which releases every core with go = v
MB_LOOP_FN void chip_phase(uint32_t v) {
    const uint32_t c = t(3);
    noc_semaphore_inc(get_noc_addr((c >> 16) & 0xFF, c >> 24, t(0) + CNT), 1);
    noc_async_atomic_barrier();
#ifdef MB_LOOP_COORD
    const bool coord = (c & 0xFF) == get_absolute_logical_x() && ((c >> 8) & 0xFF) == get_absolute_logical_y();
    if (coord) {
        const uint32_t n = t(4);
        wait_at_least(CNT, n);
        noc_semaphore_inc(get_noc_addr(t(0) + CNT), 0u - n);  // (re-armed before anyone is released)
        noc_async_atomic_barrier();
        // (the release list past the roles)
        uint32_t i = roles_at();
        i += (t(i) >> 31) ? 3 : 1 + 3 * t(i);
        for (uint32_t s = 0, nr = t(i++); s < nr; ++s) {
            const uint32_t xy = t(i++);
            noc_inline_dw_write(get_noc_addr(xy & 0xFFFF, xy >> 16, t(0) + GO), v);
        }
        noc_async_write_barrier();
    }
#endif
    wait_at_least(GO, v);
}

#endif

#ifdef MB_LOOP_DM
// my args for loop l: the linear deltas, and on the last loop its own
MB_LOOP_FN void apply_deltas(bool last) {
    const uint32_t nc = t(5);
    for (uint32_t e = 0; e < nc; ++e) {
        const uint32_t ix = t(6 + 2 * e), d = t(7 + 2 * e);
        if (!(ix >> 31) || last) {
            *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_common_arg_addr(ix & 0x7FFFFFFF)) += d;
        }
    }
    uint32_t i = 6 + 2 * nc;
    const uint32_t m = get_absolute_logical_x() | (uint32_t(get_absolute_logical_y()) << 8);
    for (uint32_t s = 0, np = t(i++); s < np; ++s) {
        const uint32_t who = t(i), n = t(i + 1);
        if (who == m) {
            for (uint32_t e = 0; e < n; ++e) {
                const uint32_t ix = t(i + 2 + 2 * e), d = t(i + 3 + 2 * e);
                if (!(ix >> 31) || last) {
                    *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_addr(ix & 0x7FFFFFFF)) += d;
                }
            }
        }
        i += 2 + 2 * n;
    }
}
#endif

// (every RISC) my CB interfaces as the firmware sets them at a launch
MB_LOOP_FN void cb_reinit() {
#if !defined(UCK_CHLKC_MATH)
    const uint32_t li = *GET_MAILBOX_ADDRESS_DEV(launch_msg_rd_ptr);
    const auto* lm = GET_MAILBOX_ADDRESS_DEV(launch[li]);
    const uint32_t base = lm->kernel_config.kernel_config_base[PROGRAMMABLE_CORE_TYPE];
    uint32_t tt_l1_ptr* cbl = reinterpret_cast<uint32_t tt_l1_ptr*>(base + lm->kernel_config.local_cb_offset);
    const uint64_t mask = lm->kernel_config.local_cb_mask;
#ifdef MB_LOOP_DM
    setup_local_cb_read_write_interfaces<true, true, false, false>(cbl, 0, static_cast<uint32_t>(mask));
    setup_local_cb_read_write_interfaces<true, true, false, false>(cbl, 32, static_cast<uint32_t>(mask >> 32));
#elif defined(UCK_CHLKC_UNPACK)
    setup_local_cb_read_write_interfaces<true, false, false, false>(cbl, 0, static_cast<uint32_t>(mask));
    setup_local_cb_read_write_interfaces<true, false, false, false>(cbl, 32, static_cast<uint32_t>(mask >> 32));
#else
    setup_local_cb_read_write_interfaces<false, true, true, false>(cbl, 0, static_cast<uint32_t>(mask));
    setup_local_cb_read_write_interfaces<false, true, true, false>(cbl, 32, static_cast<uint32_t>(mask >> 32));
#endif
#endif
}

// between loop l and l + 1 (l + 1 < MB_LOOPS)
MB_LOOP_FN void between(uint32_t l) {
    finish();
    const uint32_t a = 2 * l + 1, b = 2 * l + 2;
    *word(16 * RID) = a;
#if defined(MB_LOOP_DM) && defined(MB_LOOP_LEAD)
    if (const Role ro = role(); ro.riscs) {  // the lead: my core's RISCs, the chip; the resets; the chip again
        for (uint32_t r = 0; r < 5; ++r) {
            if (ro.riscs >> r & 1) {
                wait_at_least(16 * r, a);
            }
        }
        chip_phase(a);
        const uint32_t sems = ro.sems;
        for (uint32_t s = 0; s < 32; ++s) {
            if (sems >> s & 1) {
                *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(s)) = 0;
            }
        }
#pragma GCC unroll 1
        for (uint32_t cb = 0; cb < NUM_CIRCULAR_BUFFERS; ++cb) {
            *get_cb_tiles_received_ptr(cb) = 0;
            *get_cb_tiles_acked_ptr(cb) = 0;
        }
        chip_phase(b);
    } else {
        wait_at_least(GO, b);
    }
#else
    wait_at_least(GO, b);
#endif
}

// after the last loop: the loop words back to 0 for the next launch (the lead, once its core's RISCs are done)
MB_LOOP_FN void done() {
    const uint32_t a = 2 * MB_LOOPS - 1;
    *word(16 * RID) = a;
#if defined(MB_LOOP_DM) && defined(MB_LOOP_LEAD)
    if (const Role ro = role(); ro.riscs) {
        for (uint32_t r = 0; r < 5; ++r) {
            if (ro.riscs >> r & 1) {
                wait_at_least(16 * r, a);
            }
        }
        for (uint32_t r = 0; r < 6; ++r) {
            *word(16 * r) = 0;
        }
    }
#endif
}

MB_LOOP_FN void enter(uint32_t l) {
    cb_reinit();
#ifdef MB_LOOP_DM
    apply_deltas(l + 1 == MB_LOOPS);
#ifdef MB_LOOP_HEADER_POOL
    PacketHeaderPool::reset();
#endif
#endif
}
}  // namespace mb_loop

void kernel_main() {
#ifdef MB_CHIP
    mb_chip::apply();
#endif
#ifdef MB_CHIP_WAIT
    mb_chip::wait();
#endif
    for (uint32_t l = 0; l < MB_LOOPS; ++l) {
        mb_loop::g_loop = l;
        if (l) {
            mb_loop::enter(l);
        }
        mb_loop_body();
        if (l + 1 < MB_LOOPS) {
            mb_loop::between(l);
        }
    }
    mb_loop::done();
}
#endif

#if (defined(MB_CHIP) || defined(MB_CHIP_WAIT)) && !defined(MB_LOOPS)
void kernel_main() {
#ifdef MB_CHIP
    mb_chip::apply();
#endif
#ifdef MB_CHIP_WAIT
    mb_chip::wait();
#endif
    mb_loop_body();
}
#endif
