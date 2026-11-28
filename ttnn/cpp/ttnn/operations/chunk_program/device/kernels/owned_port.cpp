// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

// tt/owned_gather.py's _PORT (after _COMMON), verbatim (NC.7).
#include "owned_common.hpp"

FORCE_INLINE uint8_t root_bits(volatile tt_l1_ptr PACKET_HEADER_TYPE* h) {
    tt_l1_ptr routing_l1_info_t* rt = reinterpret_cast<tt_l1_ptr routing_l1_info_t*>(ROUTING_TABLE_BASE);
    const uint8_t ay = h->route_buffer[rt->my_mesh_coord_y];
    const uint8_t a = ay != 0 ? ay : h->route_buffer[rt->mesh_y_size + rt->my_mesh_coord_x];
    return a & Routing2DCodec::ACTION_ETH_MASK;
}

void kernel_main() {
    // [out, me, peer chip, mesh id, link, hx, hy, started, arrived, chip id x N, conn..]
    size_t a = 0;
    const uint32_t out = get_arg_val<uint32_t>(a++);
    const uint32_t me = get_arg_val<uint32_t>(a++);
    const uint16_t peer = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint16_t mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(a++));
    const uint32_t link = get_arg_val<uint32_t>(a++);
    const uint32_t hx = get_arg_val<uint32_t>(a++);
    const uint32_t hy = get_arg_val<uint32_t>(a++);
    const uint32_t started = get_arg_val<uint32_t>(a++);
    const uint32_t arrived = get_arg_val<uint32_t>(a++);
    const uint32_t chips = a;
    a += N;
    auto conn = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(a);
#if !(defined(ROUTING_MODE) && ((ROUTING_MODE & ROUTING_MODE_2D) != 0))
#error "the owned history send needs a 2D fabric (agent rule 4)"
#endif
    volatile tt_l1_ptr PACKET_HEADER_TYPE* hdr[2 * B];
    for (uint32_t i = 0; i < 2 * B; ++i) {
        hdr[i] = PacketHeaderPool::allocate_header();
    }
    (void)fabric_set_unicast_route(hdr[0], peer, mesh_id);
    const uint8_t mine = root_bits(hdr[0]);
    auto route = [&](uint32_t d) -> bool {  // every header to chip d; true when its route starts here
        for (uint32_t i = 0; i < 2 * B; ++i) {
            (void)fabric_set_unicast_route(hdr[i], static_cast<uint16_t>(get_arg_val<uint32_t>(chips + d)), mesh_id);
        }
        return root_bits(hdr[0]) == mine;
    };
    auto bare_inc = [&](uint64_t sem) {
        hdr[0]->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{sem, 1, true});
        conn.wait_for_empty_write_slot();
        conn.send_payload_flush_blocking_from_address(reinterpret_cast<uint32_t>(hdr[0]), sizeof(PACKET_HEADER_TYPE));
        noc_async_writes_flushed();
    };
    conn.open();
    // 1. "started" to every chip of the group whose route starts here (link 0, when this chip is a destination)
    if (link == 0 && seg_of(me) != 0xFu) {
        for (uint32_t s = 0; s < N; ++s) {
            if (s == me || !in_group(s, me) || !route(s)) {
                continue;
            }
            bare_inc(get_noc_addr(hx, hy, started));
        }
    }
    // 2. released by the home core once every destination started
    wait_min(get_semaphore(0), 1);
    // 3. the runs of this link to every destination whose route starts here
    const auto O = TensorAccessor(o_args, out, PAGE);
    const uint32_t nb = banks_of(O);
    const uint32_t base = owner(me) * R;
    const uint32_t buf = (get_write_ptr(0) + 63) & ~63u;
    for (uint32_t d = 0; d < N; ++d) {
        const uint32_t k = seg_of(d);
        if (d == me || !in_group(d, me) || k == 0xFu || !route(d)) {
            continue;
        }
        const uint32_t lo = get_common_arg_val<uint32_t>(NSEG + k);
        const uint32_t hi = get_common_arg_val<uint32_t>(2 * NSEG + k);
        const auto S = TensorAccessor(l_args, get_common_arg_val<uint32_t>(k), PAGE);
        const uint64_t arrival = get_noc_addr(hx, hy, arrived);
        const uint32_t total = n_chunks(lo, hi, nb);
        if (total <= link) {  // no run of this link: still one increment
            bare_inc(arrival);
            continue;
        }
        const uint32_t mine_n = (total - 1 - link) / L + 1;  // this link's chunks: link, link + L, ...
        // two halves of B slots: the next half's DRAM reads run while this half's packets go out
        auto fill = [&](uint32_t half, uint32_t j0) {
            const uint32_t m = mine_n - j0 < B ? mine_n - j0 : B;
            uint32_t first, cnt;
            for (uint32_t i = 0; i < m; ++i) {
                chunk_at(link + (j0 + i) * L, lo, hi, nb, first, cnt);
                noc_async_read(S.get_noc_addr(first), buf + (half * B + i) * K * PAGE, cnt * PAGE);
            }
        };
        uint32_t half = 0;
        fill(0, 0);
        for (uint32_t j0 = 0; j0 < mine_n; j0 += B) {
            const uint32_t m = mine_n - j0 < B ? mine_n - j0 : B;
            noc_async_read_barrier();
            noc_async_writes_flushed();  // (the other half's packets have left L1: its slots and headers are free)
            for (uint32_t i = 0; i < m; ++i) {
                uint32_t first, cnt;
                chunk_at(link + (j0 + i) * L, lo, hi, nb, first, cnt);
                const uint64_t dst = tt::tt_fabric::linear::addrgen_detail::get_noc_address(O, base + first);
                const uint32_t bytes = cnt * PAGE;
                volatile tt_l1_ptr PACKET_HEADER_TYPE* h = hdr[half * B + i];
                if (j0 + i + 1 == mine_n) {
                    h->to_noc_fused_unicast_write_atomic_inc(
                        NocUnicastAtomicIncFusedCommandHeader{dst, arrival, 1, true}, bytes);
                } else {
                    h->to_noc_unicast_write(NocUnicastCommandHeader{dst}, bytes);
                }
                conn.wait_for_empty_write_slot();
                conn.send_current_slot_non_blocking(
                    buf + (half * B + i) * K * PAGE, bytes, reinterpret_cast<uint32_t>(h));
            }
            if (j0 + B < mine_n) {
                fill(half ^ 1, j0 + B);
            }
            half ^= 1;
        }
        noc_async_writes_flushed();
    }
    conn.close();
    noc_semaphore_set(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(get_semaphore(0)), 0);
}
