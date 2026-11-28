// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// Moved verbatim from models/demos/deepseek_v41_flash_d_p/tt/snapshot_send.py _PORT_KERNEL_DYN (SnapshotSend._call_dyn)
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
constexpr uint32_t NB = 4;
constexpr uint32_t BR = 32;
constexpr auto g_args = TensorAccessorArgs<2>();
constexpr auto s_args = TensorAccessorArgs<g_args.next_compile_time_args_offset()>();

void kernel_main() {
    // [ring, slab, page0, rr (4 x 8 bits), codes (4 x 8 bits) | mr, mc, R, C, wx, wy, nr, nc, sem0..3, hx, hy,
    //  mesh_id, chip_id, conn..]
    size_t a = 0;
    const uint32_t ring = get_arg_val<uint32_t>(a++);
    const uint32_t slab = get_arg_val<uint32_t>(a++);
    const uint32_t page0 = get_arg_val<uint32_t>(a++);
    const uint32_t rrp = get_arg_val<uint32_t>(a++);
    const uint32_t cp = get_arg_val<uint32_t>(a++);
    uint32_t rr[NB], code[NB], sem[NB];
    for (uint32_t b = 0; b < NB; ++b) {
        rr[b] = (rrp >> (8 * b)) & 0xFFu;
        code[b] = snap_code(cp, b);
    }
    const uint32_t mr = get_arg_val<uint32_t>(a++), mc = get_arg_val<uint32_t>(a++);
    const uint32_t R = get_arg_val<uint32_t>(a++), C = get_arg_val<uint32_t>(a++);
    const uint32_t wx = get_arg_val<uint32_t>(a++), wy = get_arg_val<uint32_t>(a++);
    const uint32_t nr = get_arg_val<uint32_t>(a++), nc = get_arg_val<uint32_t>(a++);
    for (uint32_t b = 0; b < NB; ++b) {
        sem[b] = get_arg_val<uint32_t>(a++);
    }
    const uint32_t hx = get_arg_val<uint32_t>(a++);
    const uint32_t hy = get_arg_val<uint32_t>(a++);
    const uint16_t mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t me = mr * C + mc;
    uint32_t role[NB];  // 0 none, 1 from my ring (I own the block), 2 relay from my slab
    for (uint32_t b = 0; b < NB; ++b) {
        role[b] = 0;
        if (code[b] != CODE_ZERO && code[b] != CODE_LOCAL && snap_child(code[b], mr, mc, nr, nc, R, C, wx, wy)) {
            role[b] = code[b] == me ? 1 : 2;
        }
    }
    const auto G = TensorAccessor(g_args, ring, PAGE);
    const auto S = TensorAccessor(s_args, slab, PAGE);
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = PacketHeaderPool::allocate_header();
#if (defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0))
    (void)fabric_set_unicast_route(hdr, chip_id, mesh_id);
#else
#error "the snapshot send needs a 2D fabric (agent rule 4)"
#endif
    bool busy = false;  // (a port with no block this call never touches its fabric connection)
    for (uint32_t b = 0; b < NB; ++b) {
        busy = busy || role[b] != 0;
    }
    if (!busy) {
        return;
    }
    conn.open();
    for (uint32_t b = 0; b < NB; ++b) {
        if (role[b] == 0) {
            continue;
        }
        if (role[b] == 2) {
            volatile tt_l1_ptr uint32_t* flag = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(b));
            noc_semaphore_wait_min(flag, 1);
        }
        const uint32_t p0 = page0 + b * BR;
        for (uint32_t i = 0; i < BR; ++i) {
            noc_async_read(role[b] == 1 ? G.get_noc_addr(rr[b] + i) : S.get_noc_addr(p0 + i), buf + i * PAGE, PAGE);
        }
        noc_async_read_barrier();
        const uint64_t arrival = get_noc_addr(hx, hy, sem[b]);
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
    conn.close();
    for (uint32_t b = 0; b < NB; ++b) {
        if (role[b] == 2) {
            noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(b)), 0);
        }
    }
}
