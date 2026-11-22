// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The dispatch's port sender (RISCV_0, NoC1), one fabric connection (its link): the fence with the neighbour, then
// every row one hop into the same slot of the neighbour's output, and for each the entries the neighbour's ports need
// (continue / turn + / turn -), batched per stream. A flushed fused write + increment vouches for the rows sent before
// it on the link.
//
// Compile-time args: 0 the manifest CB (see the reader), 1 cb_rows, 2 cb_meta, 3 cb_stage, 4 row_bytes, 5 group,
// 6 MB (manifest batch), 7 ROWS_PER_PKT, then the TensorAccessorArgs of out.
// Runtime args: out, words (this port's semaphore words; the neighbours' sit at the same address), ready_x, ready_y
// (the neighbour's opposite port, NoC coordinates), dst_mesh_id, dst_chip_id, ring_offs, per output stream s
// (0: continue, 1 / 2: turn + / -) on (0 off, 1 on, 2 terminal: END(GMAX) only), x, y, ring k; then the neighbour's
// coordinates nb_r, nb_c and dir; then the fabric connection's.

#include "sparse_dispatch_2d_common.hpp"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"

namespace sa {

struct SenderArgs {
    uint32_t out_addr;
    uint32_t man_addr;  // the manifest rings' address (the same on every port core of every chip)
    uint32_t ready_addr;
    uint32_t ready_x;
    uint32_t ready_y;
    uint16_t dst_mesh_id;
    uint16_t dst_chip_id;
    uint32_t ring_offs;
    uint32_t cnt_addr[3];
    uint32_t s_on[3], s_x[3], s_y[3], s_k[3];
    uint32_t nb_r;  // the neighbour's coordinates
    uint32_t nb_c;
    uint32_t dir;
};

FORCE_INLINE SenderArgs sender_args(size_t& a, uint32_t man_addr) {
    SenderArgs s{};
    s.out_addr = get_arg_val<uint32_t>(a++);
    const uint32_t words = get_arg_val<uint32_t>(a++);
    s.man_addr = man_addr;
    s.ready_addr = words + READY_SEM * SEM_STRIDE;
    s.ready_x = get_arg_val<uint32_t>(a++);
    s.ready_y = get_arg_val<uint32_t>(a++);
    s.dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    s.dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    s.ring_offs = get_arg_val<uint32_t>(a++);
    for (uint32_t k = 0; k < 3; ++k) {
        s.cnt_addr[k] = words + k * SEM_STRIDE;
    }
    for (uint32_t i = 0; i < 3; ++i) {
        s.s_on[i] = get_arg_val<uint32_t>(a++);
        s.s_x[i] = get_arg_val<uint32_t>(a++);
        s.s_y[i] = get_arg_val<uint32_t>(a++);
        s.s_k[i] = get_arg_val<uint32_t>(a++);
    }
    s.nb_r = get_arg_val<uint32_t>(a++);
    s.nb_c = get_arg_val<uint32_t>(a++);
    s.dir = get_arg_val<uint32_t>(a++);
    return s;
}

template <typename CONN>
void send(const SenderArgs& args, CONN& conn) {
    using namespace tt::tt_fabric;
    constexpr uint32_t cb_rows = get_compile_time_arg_val(1);
    constexpr uint32_t cb_meta = get_compile_time_arg_val(2);
    constexpr uint32_t cb_stage = get_compile_time_arg_val(3);
    constexpr uint32_t row_bytes = get_compile_time_arg_val(4);
    constexpr uint32_t group = get_compile_time_arg_val(5);
    constexpr uint32_t MB = get_compile_time_arg_val(6);
    constexpr uint32_t ROWS_PER_PKT = get_compile_time_arg_val(7);
    constexpr auto out_args = TensorAccessorArgs<8>();

    const uint32_t ready_addr = args.ready_addr, man_addr = args.man_addr, ring_offs = args.ring_offs;
    const uint32_t nb_r = args.nb_r, nb_c = args.nb_c;
    const bool axis_x = args.dir < 2;
    const auto out = TensorAccessor(out_args, args.out_addr, row_bytes);

    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdrs[group];
    for (uint32_t h = 0; h < group; ++h) {
        hdrs[h] = PacketHeaderPool::allocate_header();
        (void)fabric_set_unicast_route(hdrs[h], args.dst_chip_id, args.dst_mesh_id);
    }
    volatile tt_l1_ptr PACKET_HEADER_TYPE* mhdr = PacketHeaderPool::allocate_header();
    (void)fabric_set_unicast_route(mhdr, args.dst_chip_id, args.dst_mesh_id);
    // fence: tell the neighbour's opposite port this chip has started; wait for the same from it
    hdrs[0]->to_noc_unicast_atomic_inc(
        NocUnicastAtomicIncCommandHeader{get_noc_addr(args.ready_x, args.ready_y, ready_addr), 1, true});
    conn.wait_for_empty_write_slot();
    conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdrs[0]), sizeof(PACKET_HEADER_TYPE));
    {
        volatile tt_l1_ptr uint32_t* ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(ready_addr);
        noc_semaphore_wait_min(ready, 1);
        noc_semaphore_inc(get_noc_addr(ready_addr), 0u - 1u);
        noc_async_atomic_barrier();
    }

    cb_reserve_back(cb_stage, 1);
    const uint32_t stage_l1 = get_write_ptr(cb_stage);  // 3 streams x 2 halves x MB entries of 16 bytes
    uint32_t half[3] = {0, 0, 0}, staged[3] = {0, 0, 0}, sent[3] = {0, 0, 0};
    auto send_manifest = [&](uint32_t s) {
        if (staged[s] == 0) {
            return;
        }
        const uint32_t src = stage_l1 + (s * 2 + half[s]) * MB * 16;
        const uint64_t dst =
            get_noc_addr(args.s_x[s], args.s_y[s], man_addr + (ring_off(ring_offs, args.s_k[s]) + sent[s]) * 16);
        const uint64_t cnt_noc = get_noc_addr(args.s_x[s], args.s_y[s], args.cnt_addr[args.s_k[s]]);
        mhdr->to_noc_fused_unicast_write_atomic_inc(
            NocUnicastAtomicIncFusedCommandHeader{dst, cnt_noc, staged[s], true}, staged[s] * 16);
        conn.wait_for_empty_write_slot();
        conn.send_current_slot_non_blocking(src, staged[s] * 16, reinterpret_cast<uint32_t>(mhdr));
        noc_async_writes_flushed();
        sent[s] += staged[s];
        staged[s] = 0;
        half[s] ^= 1;
    };
    auto stage = [&](uint32_t s, uint32_t w0, uint32_t w1) {
        volatile tt_l1_ptr uint32_t* p =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(stage_l1 + (s * 2 + half[s]) * MB * 16) + 4 * staged[s];
        p[0] = w0;
        p[1] = w1;
        if (++staged[s] == MB) {
            send_manifest(s);
        }
    };
    // what the neighbour does with a row this port just sent it
    auto follow = [&](uint32_t w0, uint32_t m) {
        const uint32_t slot = w0 & 0xFFFF, h = (w0 >> 16) & 0xF, g = (w0 >> 20) & 0xF, order = (w0 >> 24) & 1;
        const uint32_t nxt = (g + 1) << 20 | (order << 24) | slot;
        if (h > 1) {
            stage(0, nxt | ((h - 1) << 16), m);
        }
        if (axis_x == (order == 0)) {  // first leg: the neighbour turns into the other axis where its line needs it
            const uint32_t t = slot % S;
            const uint32_t ab = axis_x ? yspan(nb_r, rows_in_col(m, nb_c), t) : xspan(nb_c, cols_in_row(m, nb_r), t);
            if (ab & 15) {
                stage(1, nxt | ((ab & 15) << 16), m);
            }
            if (ab >> 4) {
                stage(2, nxt | ((ab >> 4) << 16), m);
            }
        }
    };

    bool done = false;
    while (!done) {
        if (!cb_pages_available_at_front(cb_meta, group)) {
            for (uint32_t s = 0; s < 3; ++s) {
                send_manifest(s);  // about to wait: nothing may sit staged
            }
        }
        cb_wait_front(cb_meta, group);
        cb_wait_front(cb_rows, group);
        invalidate_l1_cache();
        volatile tt_l1_ptr uint32_t* meta = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_read_ptr(cb_meta));
        const uint32_t rows_l1 = get_read_ptr(cb_rows);
        for (uint32_t i = 0; i < group; ++i) {
            const uint32_t w0 = meta[i * META_WORDS], m = meta[i * META_WORDS + 1];
            if (w0 & END) {
                const bool last = ((w0 >> 20) & 0xF) == GMAX;
                for (uint32_t s = 0; s < 3; ++s) {  // (s_on 2: a terminal stream - END(GMAX) only)
                    if (args.s_on[s] == 1 || (args.s_on[s] == 2 && last)) {
                        stage(s, w0, 0);
                    }
                }
                if (last) {
                    done = true;
                }
                continue;
            }
            if (w0 & SKIP) {
                continue;
            }
            volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr = hdrs[i];
            if constexpr (ROWS_PER_PKT == 2) {
                const uint32_t w0b = i + 1 < group ? meta[(i + 1) * META_WORDS] : END;
                if ((w0b & (END | SKIP)) == 0) {
                    const uint32_t mb = meta[(i + 1) * META_WORDS + 1];
                    hdr->to_noc_unicast_scatter_write(
                        NocUnicastScatterCommandHeader(
                            {out.get_noc_addr(w0 & 0xFFFF, 0, 0), out.get_noc_addr(w0b & 0xFFFF, 0, 0)},
                            {static_cast<uint16_t>(row_bytes)}),
                        2 * row_bytes);
                    conn.wait_for_empty_write_slot();
                    conn.send_current_slot_non_blocking(
                        rows_l1 + i * row_bytes, 2 * row_bytes, reinterpret_cast<uint32_t>(hdr));
                    follow(w0, m);
                    follow(w0b, mb);
                    ++i;
                    continue;
                }
            }
            hdr->to_noc_unicast_write(NocUnicastCommandHeader{out.get_noc_addr(w0 & 0xFFFF, 0, 0)}, row_bytes);
            conn.wait_for_empty_write_slot();
            conn.send_current_slot_non_blocking(rows_l1 + i * row_bytes, row_bytes, reinterpret_cast<uint32_t>(hdr));
            follow(w0, m);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_rows, group);
        cb_pop_front(cb_meta, group);
    }
    for (uint32_t s = 0; s < 3; ++s) {
        send_manifest(s);
    }
}

}  // namespace sa

void kernel_main() {
    constexpr uint32_t cb_man = get_compile_time_arg_val(0);
    size_t a = 0;
    const sa::SenderArgs args = sa::sender_args(a, get_write_ptr(cb_man));
    auto conn = tt::tt_fabric::WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
    conn.open();
    sa::send(args, conn);
    conn.close();
}
