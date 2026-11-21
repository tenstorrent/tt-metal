// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Row gather (gather_rows) data mover: a core beside the matmul grid, next to its link's Ethernet core, with its own
// fabric connection (no mux), one per (link, direction). Two kernels per core (this file, MOVER_READER or not): the
// reader (NoC0) reads this rank's own rows into a double-buffered CB, the sender (NoC1) sends them. The matmul's cores
// only compute and read; the movers carry the rows.
//
// Each rank owns M_dev_tiles rows (its band) of the whole K; the matmul's core row r owns output rows by
// runtime_length::uniform_split. Mover (link l, direction d) serves core rows l, l + num_links, ...: for each of their
// M blocks and each K super-block, in the matmul's block order, it sends this rank's rows of the block to the
// neighbour in direction d and, if the fabric routes the 2-hop antipode (ring of 4) this way, straight to the antipode
// too: the routers carry it through the middle rank, nothing is relayed. Every send waits only for the fabric. The
// block's last packet bumps that core row's injector counter at the destination: the high half from the 1-hop
// neighbour, the low half from the antipode, each half counting in its source's block order (dm_in0_sender.cpp).
// Fence first, as high_bw_all_gather's sender: tell the downstream's opposite-direction mover (which writes into this
// rank) that this rank has started, and wait for the same from downstream.
//
// K super-block: by default one K block (the finest arrival the matmul can use); num_dram_banks * run tiles makes each
// (row, DRAM bank) run of an interleaved tensor one contiguous read and one contiguous fabric write.

#include <cstdint>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "hostdev/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include "ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "ttnn/operations/common/shared_with_host/runtime_length.hpp"

using namespace tt::tt_fabric;

constexpr uint32_t K_tiles = get_compile_time_arg_val(0);
constexpr uint32_t K_block_tiles = get_compile_time_arg_val(1);
constexpr uint32_t M_block_tiles = get_compile_time_arg_val(2);
constexpr uint32_t m_ring = get_compile_time_arg_val(3);
constexpr uint32_t in0_parallel_axis_cores = get_compile_time_arg_val(4);
constexpr uint32_t tile_bytes = get_compile_time_arg_val(5);
constexpr uint32_t staging_cb_id = get_compile_time_arg_val(6);
constexpr uint32_t num_dram_banks = get_compile_time_arg_val(7);
constexpr uint32_t KS_tiles = get_compile_time_arg_val(8);
constexpr uint32_t tiles_per_packet = get_compile_time_arg_val(9);
constexpr uint32_t meta_cb_id = get_compile_time_arg_val(10);
constexpr auto gather_args = TensorAccessorArgs<11>();
constexpr auto input_args = TensorAccessorArgs<gather_args.next_compile_time_args_offset()>();
constexpr uint32_t K_super_blocks = K_tiles / KS_tiles;
constexpr uint32_t kBlocksPerSuper = KS_tiles / K_block_tiles;
// Per destination (1-hop neighbour, antipode): a RISC has 12 headers in the pool (6 * 2 * 2 / 2 on Blackhole).
constexpr uint32_t kHeaders = 6;
constexpr uint32_t kSegmentTiles = M_block_tiles * KS_tiles;

// Row gather: the gather buffer's tile pages are permuted so each (tile row i, K block) segment of K_block tiles sits
// contiguously in one DRAM bank (pages q, q + num_dram_banks, ... of an interleaved buffer are adjacent in a bank):
// the movers write a segment in a few large packets, the injectors read it in one go. A bijection on the buffer when
// the segment count is a multiple of num_dram_banks (validated on the host).
FORCE_INLINE uint32_t row_gather_page(uint32_t i, uint32_t j, uint32_t K_tiles_, uint32_t K_block_, uint32_t banks) {
    const uint32_t segment = i * (K_tiles_ / K_block_) + j / K_block_;
    return (segment / banks) * banks * K_block_ + (j % K_block_) * banks + segment % banks;
}

// A staged segment: rows [s, e) of rank `band`, K super-block k, core row `row`.
struct Segment {
    uint32_t row, s, e, k, band;
};
constexpr uint32_t kMaxSegments = 64;

// Rank `band`'s rows [s, e) of K super-block k of core row `row`, for each of this mover's core rows (all K blocks, or
// only the odd ones), in the matmul's block order (M block, K super-block, core row).
template <typename F>
FORCE_INLINE void for_each_band_segment(
    uint32_t M_tiles, uint32_t band, bool odd_only, uint32_t first_row, uint32_t row_stride, F&& f) {
    const uint32_t my_rank = band;
    const uint32_t M_dev_tiles = M_tiles / m_ring;
    const auto [t0, m_blocks, p0, s0, e0] =
        ttnn::runtime_length::uniform_split(M_tiles, in0_parallel_axis_cores, first_row, M_block_tiles);
    for (uint32_t m_block = 0; m_block < m_blocks; m_block++) {
        for (uint32_t k = odd_only ? 1 : 0; k < K_super_blocks; k += odd_only ? 2 : 1) {
            for (uint32_t row = first_row; row < in0_parallel_axis_cores; row += row_stride) {
                const auto [t, b, p, m_start, m_end] =
                    ttnn::runtime_length::uniform_split(M_tiles, in0_parallel_axis_cores, row, M_block_tiles);
                const uint32_t m_tile = m_start + m_block * M_block_tiles;
                const uint32_t s = std::max(m_tile, my_rank * M_dev_tiles);
                const uint32_t e =
                    std::min(std::min(std::min(m_tile + M_block_tiles, m_end), M_tiles), (my_rank + 1) * M_dev_tiles);
                if (s < e) {
                    f(row, s, e, k);
                }
            }
        }
    }
}

void kernel_main() {
    // Common: [0] gather buffer [1] input [2] forward counter [3] backward counter [4] fence counter [5] M_tiles
    // [6] rank [7] injector column NoC x [8..] injector NoC y per core row.
    const uint32_t gather_addr = get_common_arg_val<uint32_t>(0);
    const uint32_t input_addr = get_common_arg_val<uint32_t>(1);
    const uint32_t forward_counter = get_common_arg_val<uint32_t>(2);
    const uint32_t backward_counter = get_common_arg_val<uint32_t>(3);
    const uint32_t fence_counter = get_common_arg_val<uint32_t>(4);
    const uint32_t M_tiles = get_common_arg_val<uint32_t>(5);
    const uint32_t my_rank = get_common_arg_val<uint32_t>(6);
    const uint32_t injector_x = get_common_arg_val<uint32_t>(7);
    constexpr uint32_t kInjectorYArg = 8;
    // Per core: [0] direction (0: down to rank - 1, 1: up to rank + 1) [1] first core row [2] core row stride (the
    // links interleave core rows) [3, 4] downstream fabric node (mesh, chip) [5, 6] fence target NoC (downstream's
    // opposite mover) [7] also sends to the antipode [8, 9] antipode fabric node, then (sender) the fabric connection.
    size_t arg = 0;
    const bool down = get_arg_val<uint32_t>(arg++) == 0;
    const uint32_t first_row = get_arg_val<uint32_t>(arg++);
    const uint32_t row_stride = get_arg_val<uint32_t>(arg++);
    const uint32_t dst_mesh = get_arg_val<uint32_t>(arg++);
    const uint32_t dst_chip = get_arg_val<uint32_t>(arg++);
    const uint32_t fence_x = get_arg_val<uint32_t>(arg++);
    const uint32_t fence_y = get_arg_val<uint32_t>(arg++);
    const bool to_antipode = get_arg_val<uint32_t>(arg++) != 0;
    const uint32_t antipode_mesh = get_arg_val<uint32_t>(arg++);
    const uint32_t antipode_chip = get_arg_val<uint32_t>(arg++);
    // [10] on odd K blocks, tell the next rank's mover at the relay target [11, 12] to pass this rank's rows on (the
    // antipode's other way round, balancing the two directions) [13] pass the upstream rank's odd K blocks on.
    const bool notify_relay = get_arg_val<uint32_t>(arg++) != 0;
    const uint32_t relay_x = get_arg_val<uint32_t>(arg++);
    const uint32_t relay_y = get_arg_val<uint32_t>(arg++);
    const bool relay_upstream = get_arg_val<uint32_t>(arg++) != 0;
    const uint32_t upstream = down ? (my_rank + 1) % m_ring : (my_rank + m_ring - 1) % m_ring;
    // This core counts the upstream's relay notices on the counter of its own direction.
    const uint32_t counter = down ? forward_counter : backward_counter;

    Noc noc;
    CircularBuffer cb(staging_cb_id);
    const uint32_t M_dev_tiles = M_tiles / m_ring;

#ifdef MOVER_READER
    (void)injector_x, (void)dst_mesh, (void)dst_chip, (void)fence_x, (void)fence_y, (void)to_antipode;
    (void)antipode_mesh, (void)antipode_chip, (void)notify_relay, (void)relay_x, (void)relay_y, (void)fence_counter;
    // Own rows wait for nothing; the upstream's rows to pass on wait for their notice. A relay goes next as soon as it
    // has landed, never behind this mover's remaining own rows (a chain round the ring). Each kind keeps its order.
    Segment own[kMaxSegments], passing[kMaxSegments];
    uint32_t num_own = 0, num_passing = 0;
    for_each_band_segment(
        M_tiles, my_rank, false, first_row, row_stride, [&](uint32_t row, uint32_t s, uint32_t e, uint32_t k) {
            own[num_own++] = {row, s, e, k, my_rank};
        });
    if (relay_upstream) {
        for_each_band_segment(
            M_tiles, upstream, true, first_row, row_stride, [&](uint32_t row, uint32_t s, uint32_t e, uint32_t k) {
                passing[num_passing++] = {row, s, e, k, upstream};
            });
    }
    ASSERT(num_own <= kMaxSegments && num_passing <= kMaxSegments);
    const auto input = TensorAccessor(input_args, input_addr, tile_bytes);
    const auto gather = TensorAccessor(gather_args, gather_addr, tile_bytes);
    CircularBuffer meta(meta_cb_id);
    auto* landed = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(counter);
    uint32_t next_own = 0, next_passing = 0;
    while (next_own < num_own || next_passing < num_passing) {
        bool relay = false;
        if (next_passing < num_passing) {
            invalidate_l1_cache();
            relay = next_own == num_own || *landed > next_passing;
            if (relay) {
                noc_semaphore_wait_min(landed, next_passing + 1);
            }
        }
        const Segment seg = relay ? passing[next_passing++] : own[next_own++];
        cb.reserve_back(kSegmentTiles);
        meta.reserve_back(1);
        uint32_t ptr = cb.get_write_ptr();
        for (uint32_t i = seg.s; i < seg.e; i++) {
            if (relay) {  // landed in the gather buffer, one contiguous run (row_gather_page)
                noc_async_read(
                    gather.get_noc_addr(row_gather_page(i, seg.k * KS_tiles, K_tiles, K_block_tiles, num_dram_banks)),
                    ptr,
                    KS_tiles * tile_bytes);
                ptr += KS_tiles * tile_bytes;
            } else {  // in K order: the sender sends each row's segment as one contiguous run
                for (uint32_t j = seg.k * KS_tiles; j < (seg.k + 1) * KS_tiles; j++) {
                    noc_async_read(input.get_noc_addr((i - my_rank * M_dev_tiles) * K_tiles + j), ptr, tile_bytes);
                    ptr += tile_bytes;
                }
            }
        }
        {  // word by word: the sender (the other RISC) reads it from L1
            auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta.get_write_ptr());
            words[0] = seg.row, words[1] = seg.s, words[2] = seg.e, words[3] = seg.k, words[4] = seg.band;
        }
        noc.async_read_barrier();
        meta.push_back(1);
        cb.push_back(kSegmentTiles);
    }
    if (num_passing > 0) {  // the relay counter persists across calls: take this call's count off it
        noc_semaphore_inc(get_noc_addr(counter), 0u - num_passing);
        noc.async_atomic_barrier();
    }
#else
    (void)input_addr;
    const auto gather = TensorAccessor(gather_args, gather_addr, tile_bytes);
    auto connection = WorkerToFabricEdmSender::build_from_args<ProgrammableCoreType::TENSIX>(arg);
    // Blocks coming down (rank + 1 -> rank) count on the forward counter, up on the backward one at the destination
    // (the same address as this core's own counter).
    volatile tt_l1_ptr PACKET_HEADER_TYPE* headers[2][kHeaders];  // [1-hop neighbour, antipode]
    const ccl_routing_utils::line_unicast_route_info_t routes[2] = {
        {.dst_mesh_id = static_cast<uint16_t>(dst_mesh), .dst_chip_id = static_cast<uint16_t>(dst_chip)},
        {.dst_mesh_id = static_cast<uint16_t>(antipode_mesh), .dst_chip_id = static_cast<uint16_t>(antipode_chip)}};
    for (uint32_t d = 0; d < 2; ++d) {
        for (uint32_t i = 0; i < kHeaders; ++i) {
            headers[d][i] = PacketHeaderPool::allocate_header();
            ccl_routing_utils::fabric_set_line_unicast_route(headers[d][i], routes[d]);
        }
    }
    uint32_t next_header = 0, unflushed = 0;
    auto free_header = [&](uint32_t d) {  // a header must not change until its packet has left L1
        if (unflushed == kHeaders) {
            noc.async_writes_flushed();
            unflushed = 0;
        }
        ++unflushed;
        auto* header = headers[d][next_header];
        next_header = (next_header + 1) % kHeaders;
        return header;
    };
    auto send_header = [&](volatile tt_l1_ptr PACKET_HEADER_TYPE* header) {
        connection.wait_for_empty_write_slot();
        connection.send_payload_flush_blocking_from_address(
            reinterpret_cast<uint32_t>(header), sizeof(PACKET_HEADER_TYPE));
    };
    connection.open();

    // Fence: this rank has started (its previous call's reads of the gather buffer are done); wait for downstream's.
    {
        auto* header = free_header(0);
        header->to_noc_unicast_atomic_inc(
            NocUnicastAtomicIncCommandHeader{get_noc_addr(fence_x, fence_y, fence_counter), 1, true});
        send_header(header);
        noc_semaphore_wait_min(reinterpret_cast<volatile tt_l1_ptr uint32_t*>(fence_counter), 1);
        noc_semaphore_inc(get_noc_addr(fence_counter), 0u - 1u);
        noc.async_atomic_barrier();
    }

    uint32_t num_segments = 0;
    for_each_band_segment(M_tiles, my_rank, false, first_row, row_stride, [&](uint32_t, uint32_t, uint32_t, uint32_t) {
        num_segments++;
    });
    if (relay_upstream) {
        for_each_band_segment(
            M_tiles, upstream, true, first_row, row_stride, [&](uint32_t, uint32_t, uint32_t, uint32_t) {
                num_segments++;
            });
    }
    CircularBuffer meta(meta_cb_id);
    // Rows [s, e) of K super-block k from the staged segment to destination d (0: the neighbour, 1: the antipode),
    // then bump that core row's injector counter there by `count`.
    auto send_rows = [&](uint32_t d, uint32_t row, uint32_t s, uint32_t e, uint32_t k, uint32_t count) {
        uint32_t ptr = cb.get_read_ptr();
        for (uint32_t i = s; i < e; i++) {
            // The segment's tiles are adjacent in one bank (row_gather_page): contiguous packets.
            const uint64_t dst = tt::tt_fabric::linear::addrgen_detail::get_noc_address(
                gather, row_gather_page(i, k * K_block_tiles, K_tiles, K_block_tiles, num_dram_banks), 0);
            for (uint32_t c = 0; c < K_block_tiles; c += tiles_per_packet) {
                const uint32_t n = std::min(tiles_per_packet, K_block_tiles - c);
                auto* header = free_header(d);
                header->to_noc_unicast_write(NocUnicastCommandHeader{dst + c * tile_bytes}, n * tile_bytes);
                connection.wait_for_empty_write_slot();
                connection.send_current_slot_non_blocking(ptr, n * tile_bytes, reinterpret_cast<uint32_t>(header));
                ptr += n * tile_bytes;
            }
        }
        // The rows have landed (flush) before the injector counts them.
        auto* header = free_header(d);
        header->to_noc_unicast_atomic_inc(NocUnicastAtomicIncCommandHeader{
            get_noc_addr(injector_x, get_common_arg_val<uint32_t>(kInjectorYArg + row), counter), count, true});
        send_header(header);
    };
    for (uint32_t n_seg = 0; n_seg < num_segments; n_seg++) {
        meta.wait_front(1);
        invalidate_l1_cache();
        auto* words = reinterpret_cast<volatile tt_l1_ptr uint32_t*>(meta.get_read_ptr());
        const Segment seg{words[0], words[1], words[2], words[3], words[4]};
        meta.pop_front(1);
        cb.wait_front(kSegmentTiles);
        if (seg.band == my_rank) {
            // Own rows: to the neighbour (high half there); even K blocks straight to the antipode too (low half), odd
            // ones relayed the other way round by the next rank, which this tells.
            send_rows(0, seg.row, seg.s, seg.e, seg.k, kBlocksPerSuper << 16);
            if (to_antipode && (seg.k & 1) == 0) {
                send_rows(1, seg.row, seg.s, seg.e, seg.k, kBlocksPerSuper);
            }
            if (notify_relay && (seg.k & 1) == 1) {
                auto* header = free_header(0);
                header->to_noc_unicast_atomic_inc(
                    NocUnicastAtomicIncCommandHeader{get_noc_addr(relay_x, relay_y, counter), 1, true});
                send_header(header);
            }
        } else {  // the upstream's rows, on to their antipode (this mover's neighbour): low half there
            send_rows(0, seg.row, seg.s, seg.e, seg.k, kBlocksPerSuper);
        }
        noc.async_writes_flushed();  // the segment has left L1 before its CB slot is reused
        unflushed = 0;
        cb.pop_front(kSegmentTiles);
    }
    noc.async_writes_flushed();
    connection.close();
#endif
}
