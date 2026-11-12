// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Dedicated gather sender (Fabric2D Ring of up to 4): pushes this device's own K slice of in0 straight into the
// peers' gathered buffers, so the matmul's in0 cores never touch the fabric. One sender per (link, ethernet port) on
// the worker column beside the matmul grid, each with its own fabric connection: the port towards the backward
// neighbour serves that neighbour, the one towards the forward neighbour serves it, and the opposite device of a ring
// of 4 (two hops either way) is split between the two by chain parity. It reads its slice from the local input in
// DRAM and bumps a chain injector's arrival counter on the peer once that chain's piece has landed.
//
// Pieces go out in the order the receivers' K walk needs them, per M block: the neighbour halves the first remote K
// step uses, then the opposite device's blocks (step 2 of 4), then the neighbour halves of the last step. The
// receiver (dm_in0_sender, AGMM_GATHER_SENDERS) counts each source's pieces in the same order.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "matmul_dataflow_common.hpp"
#include "ttnn/operations/common/shared_with_host/runtime_length.hpp"

#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"

constexpr uint32_t K_tiles = get_compile_time_arg_val(0);
constexpr uint32_t K_tiles_per_device = get_compile_time_arg_val(1);
constexpr uint32_t K_block_tiles = get_compile_time_arg_val(2);
constexpr uint32_t M_block_tiles = get_compile_time_arg_val(3);
constexpr uint32_t in0_tile_size = get_compile_time_arg_val(4);
constexpr uint32_t num_tiles_to_write_per_packet = get_compile_time_arg_val(5);
constexpr uint32_t my_rank = get_compile_time_arg_val(6);
constexpr uint32_t in0_parallel_axis_cores = get_compile_time_arg_val(7);
constexpr uint32_t chains_per_sender = get_compile_time_arg_val(8);
constexpr uint32_t num_devices = get_compile_time_arg_val(9);
constexpr uint32_t ct_arg_count = 10;

constexpr uint32_t K_blocks_per_device = K_tiles_per_device / K_block_tiles;  // Ring: K_block_tiles | K per device
static_assert(K_blocks_per_device * K_block_tiles == K_tiles_per_device, "Ring needs K_block_tiles | K per device");
constexpr uint32_t cb_id_block = tt::CBIndex::c_0;
// A ring of 4 also has the opposite device (two hops either way), which this sender serves for one chain parity.
constexpr bool has_far = num_devices == 4;
// Peers whose senders write into this device, and so the entry announcements sent and awaited.
constexpr uint32_t num_announce = has_far ? 2 : 1;

struct Dest {
    // Two header sets: the scatter and unicast headers alternate per packet, so a packet's header is not rewritten
    // while it is in flight.
    PacketHeaders hdrs[2];
    uint32_t counter_sem;
};

void kernel_main() {
    const uint32_t gathered_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t input_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t M_tiles = get_common_arg_val<uint32_t>(2);
    // Global semaphores, common so a program-cache hit refreshes them: the barrier, the ring counters (backward,
    // forward) and the opposite device's counter.
    const uint32_t handshake_sem = get_common_arg_val<uint32_t>(3);

    size_t argidx = 0;  // size_t: FabricConnectionManager::build_from_args advances it
    const uint32_t chain_begin = get_arg_val<uint32_t>(argidx++);
    // neighbour: route, whether its first-needed half of ours is the left one
    const ccl_routing_utils::line_unicast_route_info_t nb_route = {
        .dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(argidx++)),
        .dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(argidx++))};
    const bool nb_first_left = get_arg_val<uint32_t>(argidx++) != 0;
    // opposite device (ring of 4): route, which chain parity this sender carries
    const ccl_routing_utils::line_unicast_route_info_t far_route = {
        .dst_mesh_id = static_cast<uint16_t>(get_arg_val<uint32_t>(argidx++)),
        .dst_chip_id = static_cast<uint16_t>(get_arg_val<uint32_t>(argidx++))};
    const uint32_t far_parity = get_arg_val<uint32_t>(argidx++);  // this sender's port s
    // the backward neighbour (s = 0) counts us as its +1 source (forward counter), the forward one as -1
    const uint32_t nb_counter = get_common_arg_val<uint32_t>(far_parity == 0 ? 5 : 4);
    const uint32_t far_counter = get_common_arg_val<uint32_t>(6);
    // entry-handshake targets (noc x, y): the neighbour's sender, then the opposite device's (same route index)
    uint32_t announce_x[num_announce], announce_y[num_announce];
    for (uint32_t i = 0; i < num_announce; ++i) {
        announce_x[i] = get_arg_val<uint32_t>(argidx++);
        announce_y[i] = get_arg_val<uint32_t>(argidx++);
    }
    uint32_t injector_noc_x[chains_per_sender];
    uint32_t injector_noc_y[chains_per_sender];
    for (uint32_t c = 0; c < chains_per_sender; ++c) {
        injector_noc_x[c] = get_arg_val<uint32_t>(argidx++);
        injector_noc_y[c] = get_arg_val<uint32_t>(argidx++);
    }
    auto fabric_connection = FabricConnectionManager::build_from_args(argidx);
    fabric_connection.open();
    auto* sender = &fabric_connection.get_forward_connection();  // the host passes this sender's port as forward

    constexpr auto gathered_args = TensorAccessorArgs<ct_arg_count>();
    const auto gathered = TensorAccessor(gathered_args, gathered_addr);
    constexpr auto input_args = TensorAccessorArgs<gathered_args.next_compile_time_args_offset()>();
    const auto input = TensorAccessor(input_args, input_addr);

    Noc noc_obj;
    Dest dest[2];
    for (uint32_t h = 0; h < 2; ++h) {
        dest[0].hdrs[h] =
            allocate_and_init_packet_headers(true, nb_route, gathered, num_tiles_to_write_per_packet, in0_tile_size);
        dest[1].hdrs[h] = allocate_and_init_packet_headers(
            has_far, far_route, gathered, num_tiles_to_write_per_packet, in0_tile_size);
    }
    dest[0].counter_sem = nb_counter;
    dest[1].counter_sem = far_counter;

    // Entry handshake: tell each peer's sender that writes into this device that we entered (so this device no longer
    // reads its previous call's gathered data), and wait until every peer we write into has said the same.
    for (uint32_t i = 0; i < num_announce; ++i) {
        fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            sender,
            dest[i].hdrs[0].sem_inc_hdr,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                safe_get_noc_addr(announce_x[i], announce_y[i], handshake_sem, 0), 0});
        noc_obj.async_writes_flushed();
    }
    auto* handshake_ptr = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(handshake_sem);
    noc_semaphore_wait_min(handshake_ptr, num_announce);
    noc_semaphore_inc(get_noc_addr(handshake_sem), uint32_t{0} - num_announce);
    noc_obj.async_atomic_barrier();

    CircularBuffer cb_block(cb_id_block);
    constexpr uint32_t block_bytes = M_block_tiles * K_block_tiles * in0_tile_size;
    const uint32_t block_l1[2] = {cb_block.get_write_ptr(), cb_block.get_write_ptr() + block_bytes};

    // A piece: one chain's rows of one M block, K columns [k0, k0 + kn) of this device's slice, to one destination.
    struct Piece {
        uint32_t m_tile, rows, k0, kn, dest;
        uint64_t counter;
    };
    const uint32_t M_blocks_per_core =
        ttnn::runtime_length::uniform_split(M_tiles, in0_parallel_axis_cores, chain_begin, M_block_tiles)
            .blocks_per_core;

    // Halves follow the receivers' Ring split (ring_k_left_tiles).
    auto half = [&](uint32_t b, bool left, uint32_t& k0, uint32_t& kn) {
        const uint32_t kl = ring_k_left_tiles(b & 1, K_block_tiles);
        k0 = b * K_block_tiles + (left ? 0 : kl);
        kn = left ? kl : K_block_tiles - kl;
    };
    const uint32_t phases = (num_devices == 4) ? 3 : (num_devices == 3 ? 2 : 1);
    const uint32_t pieces_per_mb_phase = K_blocks_per_device * chains_per_sender;
    const uint32_t total = M_blocks_per_core * phases * pieces_per_mb_phase;
    // Piece index -> piece; skips chains a far phase does not carry by advancing index. False past the end.
    auto piece_at = [&](uint32_t& index, Piece& p) -> bool {
        while (index < total) {
            const uint32_t mb = index / (phases * pieces_per_mb_phase);
            const uint32_t rem = index % (phases * pieces_per_mb_phase);
            const uint32_t ph = rem / pieces_per_mb_phase;
            const uint32_t b = (rem % pieces_per_mb_phase) / chains_per_sender;
            const uint32_t c = rem % chains_per_sender;
            // phases: neighbour first-needed halves, [opposite device's blocks], neighbour second-needed halves
            const bool is_far = (num_devices == 4 && ph == 1);
            if (is_far && ((chain_begin + c) & 1) != far_parity) {
                ++index;
                continue;
            }
            const auto split =
                ttnn::runtime_length::uniform_split(M_tiles, in0_parallel_axis_cores, chain_begin + c, M_block_tiles);
            p.m_tile = split.start + mb * M_block_tiles;
            const uint32_t m_end = std::min(std::min(p.m_tile + M_block_tiles, split.end), M_tiles);
            p.rows = p.m_tile < m_end ? m_end - p.m_tile : 0;
            if (is_far) {
                p.k0 = b * K_block_tiles;
                p.kn = K_block_tiles;
                p.dest = 1;
            } else {
                const bool second = (ph == phases - 1) && phases > 1;
                half(b, second ? !nb_first_left : nb_first_left, p.k0, p.kn);
                p.dest = 0;
            }
            p.counter = safe_get_noc_addr(injector_noc_x[c], injector_noc_y[c], dest[p.dest].counter_sem, 0);
            return true;
        }
        return false;
    };
    auto issue_reads = [&](const Piece& p, uint32_t l1) {
        for (uint32_t r = 0; r < p.rows; ++r) {
            for (uint32_t k = 0; k < p.kn; ++k) {
                noc_obj.async_read(
                    input,
                    CoreLocalMem<uint8_t>(l1 + (r * p.kn + k) * in0_tile_size),
                    in0_tile_size,
                    {.page_id = (p.m_tile + r) * K_tiles_per_device + p.k0 + k},
                    {});
            }
        }
    };

    uint32_t packets = 0;
    uint32_t index = 0;
    Piece cur{}, next{};
    bool have = piece_at(index, cur);
    if (have) {
        issue_reads(cur, block_l1[0]);
    }
    for (uint32_t slot = 0; have; ++slot) {
        noc_obj.async_read_barrier();    // this piece's tiles are in L1
        noc_obj.async_writes_flushed();  // the previous piece's packets have left the other buffer
        uint32_t next_index = index + 1;
        const bool have_next = piece_at(next_index, next);
        if (have_next) {
            issue_reads(next, block_l1[(slot + 1) & 1]);
        }
        const uint32_t l1 = block_l1[slot & 1];
        Dest& d = dest[cur.dest];
        const uint32_t tiles = cur.rows * cur.kn;
        for (uint32_t t = 0; t < tiles;) {
            const uint32_t n = std::min(num_tiles_to_write_per_packet, tiles - t);
            uint64_t noc_addrs[4] = {0, 0, 0, 0};
            for (uint32_t j = 0; j < n; ++j) {
                const uint32_t r = (t + j) / cur.kn, k = (t + j) % cur.kn;
                noc_addrs[j] = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                    gathered, (cur.m_tile + r) * K_tiles + my_rank * K_tiles_per_device + cur.k0 + k, 0);
            }
            if ((packets & 1) == 0 && packets > 0) {
                noc_obj.async_writes_flushed();  // both header sets used: the older one's packet must have left L1
            }
            write_tiles_packet(sender, d.hdrs[packets & 1], l1 + t * in0_tile_size, noc_addrs, n, in0_tile_size);
            ++packets;
            t += n;
        }
        // The arrival follows its piece's packets on the same route.
        noc_obj.async_writes_flushed();
        fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            sender, d.hdrs[0].sem_inc_hdr, tt::tt_fabric::NocUnicastAtomicIncCommandHeader{cur.counter, 0});
        cur = next;
        have = have_next;
        index = next_index;
    }
    noc_obj.async_writes_flushed();
    fabric_connection.close();
}
