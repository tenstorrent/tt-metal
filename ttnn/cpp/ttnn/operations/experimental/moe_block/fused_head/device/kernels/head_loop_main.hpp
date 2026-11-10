// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The fused head's kernel_main (head_loop.hpp): included after the kernel's HL_BODY. It runs the body once per token
// loop and, between two loops, the chip barrier below.
//
// The barrier (HL_LOOPS > 1). Every RISC of every core finishes loop l: its NoC traffic flushed (data movement) or its
// Tensix pipeline idle (compute), then it marks its done word. Each core has one lead (its NCRISC, else its BRISC):
// the lead waits for its core's RISCs, counts in on the coordinator core and waits for the go (phase A: no core still
// runs loop l, nothing of loop l is in flight). The lead then resets its core's program semaphores to 0 and counts in
// again (phase B: every core reset); after the second go every RISC runs loop l + 1.
//
// The words: persistent, zero, the same L1 address on every core (the host's "loops" words, 16 B apart): RISC r's
// done word at 16 r (BRISC 0, NCRISC 1, TRISC 2 .. 4), the go word at 80, the coordinator's count at 96. They are back
// at 0 when the kernels end, for the next launch.
//
// The table: common runtime args from HL_T (real common args, after the DRAM table's - the kernel's own are in its
// block), from the host (head_program.cpp add_head_loops):
//   [0] the words' address
//   [1] the coordinator: logical x | y << 8 | NoC x << 16 | NoC y << 24
//   [2] the program's cores (the barrier's count)
//   then (a lead kernel, HL_LEAD) the cores it leads: either 1 << 31 and one role for every core it runs on, or n and
//   n x (logical x | y << 8, role); a role = (the core's RISCs: bit r for RISC r, the core's program semaphores: bit s)
//   then (the coordinator kernel, HL_COORD) the cores it releases: n, n x (NoC x | y << 16)
#pragma once


#if HL_LOOPS > 1
#if defined(COMPILE_FOR_BRISC) || defined(COMPILE_FOR_NCRISC)
#define HL_DM 1
#endif

namespace hl {
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

inline __attribute__((always_inline)) uint32_t t(uint32_t i) { return get_common_arg_val<uint32_t>(HL_T + i); }
inline __attribute__((always_inline)) volatile tt_l1_ptr uint32_t* word(uint32_t off) {
    return reinterpret_cast<volatile tt_l1_ptr uint32_t*>(t(0) + off);
}
inline __attribute__((always_inline)) void wait_at_least(uint32_t off, uint32_t v) {
    while (*word(off) < v) {
#ifdef HL_DM
        invalidate_l1_cache();  // (the data movement RISCs' L1 data cache; the compute RISCs have none)
#endif
    }
}

// (every RISC) my loop is over here: my NoC traffic out of L1 and acked, or my Tensix pipeline idle
__attribute__((noinline)) inline void finish() {
#ifdef HL_DM
    noc_async_full_barrier();
#else
    ckernel::tensix_sync();
#endif
}

#if defined(HL_DM) && defined(HL_LEAD)
struct Role {
    uint32_t riscs = 0, sems = 0;
};
// my core's role if this kernel leads it (0: not the lead)
__attribute__((noinline)) inline Role role() {
    uint32_t i = 3;
    if (t(i) >> 31) {  // (every core of the kernel alike)
        return Role{t(i + 1), t(i + 2)};
    }
    const uint32_t me = get_absolute_logical_x() | (uint32_t(get_absolute_logical_y()) << 8);
    for (uint32_t s = 0, n = t(i++); s < n; ++s, i += 3) {
        if (t(i) == me) {
            return Role{t(i + 1), t(i + 2)};
        }
    }
    return Role{};
}
// (the release list: past the roles)
inline __attribute__((always_inline)) uint32_t releases_at() {
    const uint32_t i = 3;
    return (t(i) >> 31) ? i + 3 : i + 1 + 3 * t(i);
}
// (lead) one phase of the barrier: count in on the coordinator, which releases every core with go = v
__attribute__((noinline)) inline void chip_phase(uint32_t v) {
    const uint32_t c = t(1);
    noc_semaphore_inc(get_noc_addr((c >> 16) & 0xFF, c >> 24, t(0) + CNT), 1);
    noc_async_atomic_barrier();
#ifdef HL_COORD
    if ((c & 0xFF) == get_absolute_logical_x() && ((c >> 8) & 0xFF) == get_absolute_logical_y()) {
        const uint32_t n = t(2);
        wait_at_least(CNT, n);
        noc_semaphore_inc(get_noc_addr(t(0) + CNT), 0u - n);  // (re-armed before anyone is released)
        noc_async_atomic_barrier();
        uint32_t i = releases_at();
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

// between loop l and l + 1
__attribute__((noinline)) inline void between(uint32_t l) {
    finish();
    const uint32_t a = 2 * l + 1, b = 2 * l + 2;
    *word(16 * RID) = a;
#if defined(HL_DM) && defined(HL_LEAD)
    if (const Role ro = role(); ro.riscs) {  // the lead: my core's RISCs, the chip; my core's semaphores; the chip
        for (uint32_t r = 0; r < 5; ++r) {
            if (ro.riscs >> r & 1) {
                wait_at_least(16 * r, a);
            }
        }
        chip_phase(a);
        for (uint32_t s = 0; s < 32; ++s) {
            if (ro.sems >> s & 1) {
                noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(s)), 0);
            }
        }
        chip_phase(b);
    } else {
        wait_at_least(GO, b);
    }
#else
    wait_at_least(GO, b);
#endif
}

// after the last loop: the words back to 0 for the next launch (the lead, once its core's RISCs are done)
__attribute__((noinline)) inline void done() {
    finish();
    const uint32_t a = 2 * HL_LOOPS - 1;
    *word(16 * RID) = a;
#if defined(HL_DM) && defined(HL_LEAD)
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
}  // namespace hl
#endif

void kernel_main() {
    mb_args::load();  // (this core's args from the DRAM table; a compute kernel waits for its page)
    for (uint32_t l = 0; l < HL_LOOPS; ++l) {
        hl::g_loop = l;
        hl_body();
#if HL_LOOPS > 1
        if (l + 1 < HL_LOOPS) {
            hl::between(l);
        }
#endif
    }
#if HL_LOOPS > 1
    hl::done();
#endif
}
