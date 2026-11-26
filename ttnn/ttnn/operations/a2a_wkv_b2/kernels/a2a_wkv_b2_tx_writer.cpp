// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// a2a_wkv_b2 fabric_tx writer (BRISC) — send_block + handshakes for one stream.
//
//   announce_ready : fabric atomic inc -> peer fabric_tx sem_ready on the destination
//   await_ready    : local wait sem_ready >= 1, reset 0 (destination has started this program)
//   send_block     : per packet, one scatter write of up to tiles_per_packet tiles straight into
//                    the destination's L1-interleaved output tiles (no receiver kernel)
//   announce_done  : fabric atomic inc with flush -> peer sem_done (after the last data packet)
//   await_done     : local wait sem_done >= 1 (the peer's data has landed here), reset 0
//
// The peer is the destination's stream that targets this chip; it sends exactly one ready and one
// done per call, so every count is 1. Resets happen right after each wait (see op_design.md
// "Synchronization contract" for why the next call always sees 0).

#include <cstdint>

#include "api/dataflow/dataflow_api.h"
#include "ttnn/cpp/ttnn/kernel_lib/ccl/ccl_helpers_dataflow.hpp"

void kernel_main() {
    constexpr uint32_t cb_tx_stage = get_compile_time_arg_val(0);
    constexpr uint32_t Mt = get_compile_time_arg_val(1);
    constexpr uint32_t Nt = get_compile_time_arg_val(2);
    constexpr uint32_t mt_block = get_compile_time_arg_val(3);
    constexpr uint32_t tiles_per_packet = get_compile_time_arg_val(4);
    constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t ring_size = get_compile_time_arg_val(6);
    constexpr uint32_t l1_alignment = get_compile_time_arg_val(7);
    constexpr auto out_args = TensorAccessorArgs<8>();

    // tiles_per_packet is clamped on the host to the scatter cap (and the fabric payload); restated
    // here only as a guard, never as a second source of the packet size.
    static_assert(tiles_per_packet >= 1 && tiles_per_packet <= NOC_SCATTER_WRITE_MAX_CHUNKS);

    constexpr uint32_t num_mt_blocks = (Mt + mt_block - 1) / mt_block;
    constexpr uint32_t tiles_per_source_head = ring_size * Mt * Nt;

    using namespace dataflow_kernel_lib::ccl;

    // Fabric connection block first (built on the host by build_ccl_fabric_rt_args).
    size_t arg_idx = 0;
    FabricStreamSender<DirectConn> sender(arg_idx, /*is_forward=*/true, l1_alignment);

    const uint32_t dst_mesh_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t dst_chip_id = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t peer_noc_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t peer_noc_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sem_ready_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t sem_done_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t out_addr = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t my_pos = get_arg_val<uint32_t>(arg_idx++);  // source index j on the destination
    const uint32_t num_heads = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t heads_base = arg_idx;  // h_local (on the destination) per head

    const auto out = TensorAccessor(out_args, out_addr, tile_bytes);

    ccl_routing_utils::line_unicast_route_info_t route{};
    route.dst_mesh_id = static_cast<uint16_t>(dst_mesh_id);
    route.dst_chip_id = static_cast<uint16_t>(dst_chip_id);

    // Fabric NoC addresses are noc0-encoded.
    const uint64_t peer_ready_noc = safe_get_noc_addr(peer_noc_x, peer_noc_y, sem_ready_addr, 0);
    const uint64_t peer_done_noc = safe_get_noc_addr(peer_noc_x, peer_noc_y, sem_done_addr, 0);
    volatile tt_l1_ptr uint32_t* sem_ready = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_ready_addr);
    volatile tt_l1_ptr uint32_t* sem_done = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(sem_done_addr);

    auto stream = sender.open(route);
    auto inc_ch = stream.arm_inc(1);  // Val|Flush armed

    // announce_ready / await_ready
    inc_ch.inc(peer_ready_noc);
    noc_semaphore_wait_min(sem_ready, 1);
    noc_semaphore_set(sem_ready, 0);

    auto unicast_ch = stream.arm_unicast_write(tile_bytes);  // single-tile tail packets
    [[maybe_unused]] auto scatter_ch =
        stream.arm_scatter_write(tile_bytes, tiles_per_packet > 1 ? tiles_per_packet : 2);

    uint64_t dst_addrs[tiles_per_packet] = {};
    for (uint32_t mb = 0; mb < num_mt_blocks; ++mb) {
        const uint32_t mt0 = mb * mt_block;
        const uint32_t mt_rows = (Mt - mt0) < mt_block ? (Mt - mt0) : mt_block;
        const uint32_t block_tiles = mt_rows * Nt;
        for (uint32_t s = 0; s < num_heads; ++s) {
            const uint32_t h_local = get_arg_val<uint32_t>(heads_base + s);
            const uint32_t tile0 = h_local * tiles_per_source_head + (my_pos * Mt + mt0) * Nt;
            for (uint32_t p = 0; p < block_tiles; p += tiles_per_packet) {
                const uint32_t chunks = (block_tiles - p) < tiles_per_packet ? (block_tiles - p) : tiles_per_packet;
                cb_wait_front(cb_tx_stage, tiles_per_packet);
                const uint32_t l1 = get_read_ptr(cb_tx_stage);
                for (uint32_t c = 0; c < chunks; ++c) {
                    dst_addrs[c] = tt::tt_fabric::linear::addrgen_detail::get_noc_address(out, tile0 + p + c, 0);
                }
                if (chunks == 1) {
                    unicast_ch.write(dst_addrs[0], l1);
                } else {
                    scatter_ch.write_scatter(dst_addrs, chunks, l1);
                }
                stream.flush();  // payload has left this L1 slot before the reader reuses it
                cb_pop_front(cb_tx_stage, tiles_per_packet);
            }
        }
    }

    // announce_done (flush: lands after every data packet of this stream)
    inc_ch.inc(peer_done_noc);
    stream.close();

    // await_done
    noc_semaphore_wait_min(sem_done, 1);
    noc_semaphore_set(sem_done, 0);
}
