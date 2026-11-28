// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Moved verbatim from models/demos/deepseek_v41_flash_d_p/tt/snapshot_send.py _PORT_KERNEL_MULTI (SnapshotSend.multi)
// (NC.3).
#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
using namespace tt::tt_fabric;

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
constexpr uint32_t K = get_compile_time_arg_val(1);
constexpr uint32_t NS = get_compile_time_arg_val(2);
constexpr uint32_t WAIT_LIMIT = get_compile_time_arg_val(3);
constexpr uint32_t NB = 4;
constexpr uint32_t BR = 32;
constexpr auto g_args = TensorAccessorArgs<4>();
constexpr auto s_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

inline void wait_min_bounded(volatile tt_l1_ptr uint32_t* p, uint32_t v) {
    for (uint32_t i = 0; i < WAIT_LIMIT && *p < v; ++i) {
        invalidate_l1_cache();
    }
}

void kernel_main() {
    // [ring0, slab0, (ring, slab, page0, rr, codes) x NS | mr, mc, R, C, wx, wy, nr, nc, hx, hy, mesh_id, chip_id,
    //  sem x 4 NS, conn..]
    constexpr uint32_t V = 2;
    size_t a = V + 5 * NS;
    const uint32_t mr = get_arg_val<uint32_t>(a++), mc = get_arg_val<uint32_t>(a++);
    const uint32_t R = get_arg_val<uint32_t>(a++), C = get_arg_val<uint32_t>(a++);
    const uint32_t wx = get_arg_val<uint32_t>(a++), wy = get_arg_val<uint32_t>(a++);
    const uint32_t nr = get_arg_val<uint32_t>(a++), nc = get_arg_val<uint32_t>(a++);
    const uint32_t hx = get_arg_val<uint32_t>(a++);
    const uint32_t hy = get_arg_val<uint32_t>(a++);
    const uint16_t mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const size_t s0 = a;
    a += NB * NS;
    const uint32_t me = mr * C + mc;
    bool busy = false;  // (a port with no block this call never touches its fabric connection)
    for (uint32_t s = 0; s < NS; ++s) {
        const uint32_t cp = get_arg_val<uint32_t>(V + 5 * s + 4);
        for (uint32_t b = 0; b < NB; ++b) {
            const uint32_t code = snap_code(cp, b);
            busy = busy || (code != CODE_ZERO && code != CODE_LOCAL && snap_child(code, mr, mc, nr, nc, R, C, wx, wy));
        }
    }
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    if (!busy) {
        return;
    }
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0))
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
#error "the snapshot send needs a 2D fabric (agent rule 4)"
#endif
    conn.open();
    uint32_t relayed = 0;  // (the home core's counter: one increment a block this port relays, in block order)
    volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0));
    for (uint32_t s = 0; s < NS; ++s) {
        const auto G = TensorAccessor(g_args, get_arg_val<uint32_t>(V + 5 * s), PAGE);
        const auto S = TensorAccessor(s_args, get_arg_val<uint32_t>(V + 5 * s + 1), PAGE);
        const uint32_t page0 = get_arg_val<uint32_t>(V + 5 * s + 2);
        const uint32_t rrp = get_arg_val<uint32_t>(V + 5 * s + 3), cp = get_arg_val<uint32_t>(V + 5 * s + 4);
        for (uint32_t b = 0; b < NB; ++b) {
            const uint32_t code = snap_code(cp, b);
            if (code == CODE_ZERO || code == CODE_LOCAL || !snap_child(code, mr, mc, nr, nc, R, C, wx, wy)) {
                continue;
            }
            const bool own = code == me;
            if (!own) {
                ++relayed;
                wait_min_bounded(flag, relayed);
            }
            const uint32_t rr = (rrp >> (8 * b)) & 0xFFu;
            const uint32_t p0 = page0 + b * BR;
            for (uint32_t i = 0; i < BR; ++i) {
                noc_async_read(own ? G.get_noc_addr(rr + i) : S.get_noc_addr(p0 + i), buf + i * PAGE, PAGE);
            }
            noc_async_read_barrier();
            const uint64_t arrival = get_noc_addr(hx, hy, get_arg_val<uint32_t>(s0 + NB * s + b));
            for (uint32_t i = 0; i < BR; i += K) {
                const uint64_t dst = S.get_noc_addr(p0 + i);
                const bool run = (K > 1) && (S.get_noc_addr(p0 + i + K - 1) == dst + (K - 1) * PAGE);
                const uint32_t n = run ? K : 1;
                for (uint32_t j = 0; j < K; j += n) {
                    const uint64_t d = run ? dst : S.get_noc_addr(p0 + i + j);
                    const uint32_t bytes = n * PAGE;
                    if (i + j + n == BR) {
                        hdr->to_noc_fused_unicast_write_atomic_inc(
                            NocUnicastAtomicIncFusedCommandHeader{d, arrival, 1, true}, bytes);
                    } else {
                        hdr->to_noc_unicast_write(NocUnicastCommandHeader{d}, bytes);
                    }
                    conn.wait_for_empty_write_slot();
                    conn.send_current_slot_non_blocking(buf + (i + j) * PAGE, bytes, reinterpret_cast<uint32_t>(hdr));
                    noc_async_writes_flushed();
                }
            }
        }
    }
    conn.close();
    if (relayed) {
        noc_semaphore_set(flag, 0);
    }
}
