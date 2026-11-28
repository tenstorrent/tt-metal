// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Moved verbatim from models/demos/deepseek_v41_flash_d_p/tt/snapshot_send.py _HOME_KERNEL_DYN (SnapshotSend._call_dyn)
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
constexpr uint32_t NB = 4;
constexpr uint32_t BR = 32;
constexpr auto g_args = TensorAccessorArgs<1>();
constexpr auto s_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

void kernel_main() {
    // [ring, slab, page0, rr (4 x 8 bits), codes (4 x 8 bits) | mr, mc, R, C, wx, wy, sem0..3, nports,
    //  (px, py, nr, nc) x nports]
    const uint32_t ring = get_arg_val<uint32_t>(0);
    const uint32_t slab = get_arg_val<uint32_t>(1);
    const uint32_t page0 = get_arg_val<uint32_t>(2);
    const uint32_t rrp = get_arg_val<uint32_t>(3), cp = get_arg_val<uint32_t>(4);
    const uint32_t mr = get_arg_val<uint32_t>(5), mc = get_arg_val<uint32_t>(6);
    const uint32_t R = get_arg_val<uint32_t>(7), C = get_arg_val<uint32_t>(8);
    const uint32_t wx = get_arg_val<uint32_t>(9), wy = get_arg_val<uint32_t>(10);
    const uint32_t nports = get_arg_val<uint32_t>(15);
    const uint32_t me = mr * C + mc;
    const auto G = TensorAccessor(g_args, ring, PAGE);
    const auto S = TensorAccessor(s_args, slab, PAGE);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    const uint32_t zbuf = buf + BR * PAGE;
    bool zeroed = false;
    uint32_t kind[NB], codes[NB];
    for (uint32_t b = 0; b < NB; ++b) {
        codes[b] = snap_code(cp, b);
    }
    for (uint32_t b = 0; b < NB; ++b) {
        const uint32_t code = codes[b];
        kind[b] = code == CODE_ZERO ? 0 : (code == CODE_LOCAL || code == me) ? 1 : 2;
        if (kind[b] == 0) {
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
        } else if (kind[b] == 1) {
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
    for (uint32_t b = 0; b < NB; ++b) {
        if (kind[b] != 2) {
            continue;
        }
        const uint32_t code = codes[b];
        volatile tt_l1_ptr uint32_t* arrived =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_arg_val<uint32_t>(11 + b));
        noc_semaphore_wait_min(arrived, 1);
        for (uint32_t k = 0; k < nports; ++k) {
            const uint32_t nr = get_arg_val<uint32_t>(18 + 4 * k), nc = get_arg_val<uint32_t>(19 + 4 * k);
            if (snap_child(code, mr, mc, nr, nc, R, C, wx, wy)) {
                const uint32_t px = get_arg_val<uint32_t>(16 + 4 * k);
                const uint32_t py = get_arg_val<uint32_t>(17 + 4 * k);
                noc_semaphore_inc(get_noc_addr(px, py, get_semaphore(b)), 1);
            }
        }
    }
    noc_async_write_barrier();
    for (uint32_t b = 0; b < NB; ++b) {
        if (kind[b] == 2) {
            noc_semaphore_inc(get_noc_addr(get_arg_val<uint32_t>(11 + b)), 0u - 1u);
        }
    }
    noc_async_atomic_barrier();
}
