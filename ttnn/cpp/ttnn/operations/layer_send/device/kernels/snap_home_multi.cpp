// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Moved verbatim from models/demos/deepseek_v41_flash_d_p/tt/snapshot_send.py _HOME_KERNEL_MULTI (SnapshotSend.multi)
// (NC.3).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"

constexpr uint32_t CODE_ZERO = 0xFFFFFFFFu;
constexpr uint32_t CODE_LOCAL = 0xFFFFFFFEu;
// block b's code from the packed word (8 bits a block: 0xFF zero, 0xFE local, else the owner's chip-linear index)
inline uint32_t snap_code(uint32_t cp, uint32_t b) {
    const uint32_t c = (cp >> (8 * b)) & 0xFFu;
    return c == 0xFFu ? CODE_ZERO : c == 0xFEu ? CODE_LOCAL : c;
}
inline int32_t snap_step(uint32_t n, uint32_t a, uint32_t b, uint32_t wrap) {
    if (wrap) {
        const uint32_t f = (b + n - a) % n;
        return f * 2 <= n ? 1 : -1;
    }
    return b > a ? 1 : -1;
}
// the parent of chip (r, c) in the broadcast tree from (r0, c0) (snapshot_send.tree)
inline void snap_parent(
    uint32_t r,
    uint32_t c,
    uint32_t r0,
    uint32_t c0,
    uint32_t R,
    uint32_t C,
    uint32_t wx,
    uint32_t wy,
    uint32_t& pr,
    uint32_t& pc) {
    if (c != c0) {
        pr = r;
        pc = (c + C + snap_step(C, c, c0, wx)) % C;
    } else {
        pr = (r + R + snap_step(R, r, r0, wy)) % R;
        pc = c;
    }
}
// chip (nr, nc) is my child in the tree of owner code o
inline bool snap_child(
    uint32_t o, uint32_t mr, uint32_t mc, uint32_t nr, uint32_t nc, uint32_t R, uint32_t C, uint32_t wx, uint32_t wy) {
    const uint32_t r0 = o / C, c0 = o % C;
    if (nr == r0 && nc == c0) {
        return false;
    }
    uint32_t pr = 0, pc = 0;
    snap_parent(nr, nc, r0, c0, R, C, wx, wy, pr, pc);
    return pr == mr && pc == mc;
}

constexpr uint32_t PAGE = get_compile_time_arg_val(0);
constexpr uint32_t NS = get_compile_time_arg_val(1);  // segments of this program
constexpr uint32_t WAIT_LIMIT = get_compile_time_arg_val(2);
constexpr uint32_t NB = 4;
constexpr uint32_t BR = 32;
constexpr auto g_args = TensorAccessorArgs<3>();
constexpr auto s_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

inline void wait_min_bounded(volatile tt_l1_ptr uint32_t* p, uint32_t v) {
    for (uint32_t i = 0; i < WAIT_LIMIT && *p < v; ++i) {
        invalidate_l1_cache();
    }
}

void kernel_main() {
    // [ring0, slab0 (io: the accessors' stand-ins), (ring, slab, page0, rr, codes) x NS | mr, mc, R, C, wx, wy,
    //  nports, (px, py, nr, nc) x nports, sem x 4 NS]
    constexpr uint32_t V = 2;
    constexpr uint32_t F = V + 5 * NS;
    const uint32_t mr = get_arg_val<uint32_t>(F), mc = get_arg_val<uint32_t>(F + 1);
    const uint32_t R = get_arg_val<uint32_t>(F + 2), C = get_arg_val<uint32_t>(F + 3);
    const uint32_t wx = get_arg_val<uint32_t>(F + 4), wy = get_arg_val<uint32_t>(F + 5);
    const uint32_t nports = get_arg_val<uint32_t>(F + 6);
    const uint32_t P0 = F + 7;
    const uint32_t S0 = P0 + 4 * nports;  // the arrival counters, one a (segment, block)
    const uint32_t me = mr * C + mc;
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t zbuf = buf + BR * PAGE;
    bool zeroed = false;
    for (uint32_t s = 0; s < NS; ++s) {
        const auto G = TensorAccessor(g_args, get_arg_val<uint32_t>(V + 5 * s), PAGE);
        const auto S = TensorAccessor(s_args, get_arg_val<uint32_t>(V + 5 * s + 1), PAGE);
        const uint32_t page0 = get_arg_val<uint32_t>(V + 5 * s + 2);
        const uint32_t rrp = get_arg_val<uint32_t>(V + 5 * s + 3), cp = get_arg_val<uint32_t>(V + 5 * s + 4);
        for (uint32_t b = 0; b < NB; ++b) {
            const uint32_t code = snap_code(cp, b);
            if (code == CODE_ZERO) {
                if (!zeroed) {
                    volatile tt_l1_ptr uint32_t* z = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(zbuf);
                    for (uint32_t i = 0; i < PAGE / 4; ++i) {
                        z[i] = 0;
                    }
                    zeroed = true;
                }
                for (uint32_t i = 0; i < BR; ++i) {
                    noc_async_write(zbuf, S.get_noc_addr(page0 + b * BR + i), PAGE);
                }
            } else if (code == CODE_LOCAL || code == me) {
                const uint32_t rr = (rrp >> (8 * b)) & 0xFFu;
                for (uint32_t i = 0; i < BR; ++i) {
                    noc_async_read(G.get_noc_addr(rr + i), buf + i * PAGE, PAGE);
                }
                noc_async_read_barrier();
                for (uint32_t i = 0; i < BR; ++i) {
                    noc_async_write(buf + i * PAGE, S.get_noc_addr(page0 + b * BR + i), PAGE);
                }
                noc_async_write_barrier();
            }
        }
    }
    // received blocks, (segment, block) order: wait for each, tell the ports that relay it (a counter a port)
    for (uint32_t s = 0; s < NS; ++s) {
        const uint32_t cp = get_arg_val<uint32_t>(V + 5 * s + 4);
        for (uint32_t b = 0; b < NB; ++b) {
            const uint32_t code = snap_code(cp, b);
            if (code == CODE_ZERO || code == CODE_LOCAL || code == me) {
                continue;
            }
            volatile tt_l1_ptr uint32_t* arrived =
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(S0 + NB * s + b));
            wait_min_bounded(arrived, 1);
            for (uint32_t k = 0; k < nports; ++k) {
                const uint32_t nr = get_arg_val<uint32_t>(P0 + 4 * k + 2), nc = get_arg_val<uint32_t>(P0 + 4 * k + 3);
                if (snap_child(code, mr, mc, nr, nc, R, C, wx, wy)) {
                    const uint32_t px = get_arg_val<uint32_t>(P0 + 4 * k);
                    const uint32_t py = get_arg_val<uint32_t>(P0 + 4 * k + 1);
                    noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(0)), 1);
                }
            }
        }
    }
    noc_async_write_barrier();
    for (uint32_t s = 0; s < NS; ++s) {
        const uint32_t cp = get_arg_val<uint32_t>(V + 5 * s + 4);
        for (uint32_t b = 0; b < NB; ++b) {
            const uint32_t code = snap_code(cp, b);
            if (!(code == CODE_ZERO || code == CODE_LOCAL || code == me)) {
                noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(S0 + NB * s + b)), 0u - 1u);
            }
        }
    }
    noc_async_atomic_barrier();
}
