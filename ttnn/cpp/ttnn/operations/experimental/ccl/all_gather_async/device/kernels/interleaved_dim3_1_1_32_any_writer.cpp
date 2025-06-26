// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t ring_size = get_compile_time_arg_val(0);
constexpr uint32_t my_chip_id = get_compile_time_arg_val(1);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(2);
constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(3);
constexpr BufferType buffer0_type = static_cast<BufferType>(get_compile_time_arg_val(4));
constexpr uint32_t cb0_id = get_compile_time_arg_val(5);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(6);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(7);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(8);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(9);
constexpr bool dynamic_alternate = get_compile_time_arg_val(10);
constexpr uint32_t num_max_targets = std::max(num_targets_forward_direction, num_targets_backward_direction);
constexpr uint32_t num_sync_targets_forward = dynamic_alternate ? num_max_targets : num_targets_forward_direction;
constexpr uint32_t num_sync_targets_backward = dynamic_alternate ? num_max_targets : num_targets_backward_direction;
constexpr bool last_dim = get_compile_time_arg_val(11);
constexpr uint32_t num_banks = get_compile_time_arg_val(12);
constexpr bool use_best_effort = get_compile_time_arg_val(13);

// for performance experiment by dynamic_alternate switch
inline void fabric_write_wrapper(
    uint64_t noc0_dest_noc_addr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection,
    size_t& l1_read_addr,
    size_t tensor_size) {
    write_and_advance_local_read_address_for_fabric_write(
        noc0_dest_noc_addr, pkt_hdr_forward, pkt_hdr_backward, fabric_connection, l1_read_addr, tensor_size);
    if constexpr (dynamic_alternate) {
        // In ring mode, it's more performant to balance traffic in left/right directions. For example, a ring size of 4
        // would have a chip send 2 hops in one direction and 1 hop in the other. By alternating, we which balance
        // traffic because the # bytes being sent 2 hops are now shared across 2 links/directions instead of just the
        // one that would come from not alternating
        std::swap(
            pkt_hdr_forward->routing_fields.value,
            pkt_hdr_backward->routing_fields.value);  // alternate the packet header distance for better balancing
    }
}

template <bool DRAM>
inline void fabric_send_full_contig(
    uint32_t contig_total,
    uint32_t& tile_id,
    InterleavedAddrGenFast<DRAM>& addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    uint32_t total_local = 0;
    while (total_local < contig_total) {
        cb_wait_front(cb0_id, packet_size_in_pages);
        size_t l1_read_addr = get_read_ptr(cb0_id);
        uint64_t noc0_dest_noc_addr = get_noc_addr(tile_id, addrgen, 0 /*offset*/, 0 /*noc_id*/);
        fabric_write_wrapper(
            noc0_dest_noc_addr,
            pkt_hdr_forward,
            pkt_hdr_backward,
            fabric_connection,
            l1_read_addr,
            packet_size_in_pages * tensor0_page_size);
        cb_pop_front(cb0_id, packet_size_in_pages);
        tile_id++;
        total_local++;
        if (total_local % num_banks == 0) {
            tile_id += num_banks * (packet_size_in_pages - 1);
        }
    }
}

template <bool DRAM>
inline void fabric_send_2contig_bf8(
    uint32_t contig_total,
    uint32_t& tile_id,
    InterleavedAddrGenFast<DRAM>& addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    uint32_t total_local = 0;
    while (total_local < contig_total) {
        cb_wait_front(cb0_id, 2);
        size_t l1_read_addr = get_read_ptr(cb0_id);
        uint64_t noc0_dest_noc_addr = get_noc_addr(tile_id, addrgen, 0 /*offset*/, 0 /*noc_id*/);
        fabric_write_wrapper(
            noc0_dest_noc_addr,
            pkt_hdr_forward,
            pkt_hdr_backward,
            fabric_connection,
            l1_read_addr,
            2 * tensor0_page_size);
        tile_id++;
        total_local++;
        cb_pop_front(cb0_id, 2);
        if (total_local % num_banks == 0) {
            tile_id += num_banks;
        }
    }
}

template <bool DRAM>
inline void fabric_send_non_contig(
    uint32_t num_tiles,
    uint32_t& tile_id,
    InterleavedAddrGenFast<DRAM>& addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    uint32_t total_local = 0;
    while (total_local < num_tiles) {
        uint32_t tiles_in_packet = std::min(num_tiles - total_local, packet_size_in_pages);
        cb_wait_front(cb0_id, tiles_in_packet);
        size_t l1_read_addr = get_read_ptr(cb0_id);
        for (uint32_t i = 0; i < tiles_in_packet; i++) {
            uint64_t noc0_dest_noc_addr = get_noc_addr(tile_id, addrgen, 0 /*offset*/, 0 /*noc_id*/);
            fabric_write_wrapper(
                noc0_dest_noc_addr,
                pkt_hdr_forward,
                pkt_hdr_backward,
                fabric_connection,
                l1_read_addr,
                tensor0_page_size);
            tile_id++;
        }
        cb_pop_front(cb0_id, tiles_in_packet);
        total_local += tiles_in_packet;
    }
}

template <bool DRAM>
inline void fabric_send_dim3_bf16_remain_even(
    uint32_t num_tiles,
    uint32_t tile_id,
    uint32_t tile_cols_per_chip,
    InterleavedAddrGenFast<DRAM>& addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    const uint32_t num_contig2 = (tile_cols_per_chip / (num_banks * packet_size_in_pages)) * num_banks;
    const uint32_t num_contig1 = ((tile_cols_per_chip - num_contig2 * 2) / num_banks) * num_banks;
    const uint32_t num_orphan = tile_cols_per_chip - num_contig2 * 2 - num_contig1;
    const uint32_t row = num_tiles / tile_cols_per_chip;
    for (uint32_t i = 0; i < row; i++) {
        fabric_send_full_contig(num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
        fabric_send_non_contig(num_contig1, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
        fabric_send_non_contig(num_orphan, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
        tile_id += tile_cols_per_chip * (ring_size - 1);
    }
}

template <bool DRAM>
inline void fabric_send_dim3_bf8_dram_remain048(
    uint32_t num_tiles,
    uint32_t tile_id,
    uint32_t tile_cols_per_chip,
    InterleavedAddrGenFast<DRAM>& addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    uint32_t row = num_tiles / tile_cols_per_chip;
    const uint32_t input_width = tile_cols_per_chip;
    uint32_t num_full_contig = (tile_cols_per_chip / (num_banks * packet_size_in_pages)) * num_banks;
    uint32_t num_contig2 =
        ((tile_cols_per_chip - num_full_contig * packet_size_in_pages) / (num_banks * 2)) * num_banks;
    uint32_t num_orphan = tile_cols_per_chip - num_full_contig * packet_size_in_pages - num_contig2 * 2;
    for (uint32_t i = 0; i < row; i++) {
        fabric_send_full_contig(
            num_full_contig, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
        fabric_send_2contig_bf8(num_contig2, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
        fabric_send_non_contig(num_orphan, tile_id, addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
        tile_id += input_width * (ring_size - 1);
    }
}

template <bool DRAM>
inline void fabric_send_dim2_bf8(
    uint32_t filled_bank_tiles,
    uint32_t rest_full_contig_ids,
    uint32_t& tile_id,
    uint32_t rest_tiles,
    InterleavedAddrGenFast<DRAM>& tensor0_addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    // e.g. num_banks: 4, packet_size_in_pages: 4
    //      |       chip0           |
    //    0 | id 0|    1|    2|    3|
    //      |    4|    5|    6|    7|
    //      |    8|    9|   10|   11|
    //    __|   12|   13|   14|   15|___  <- filled_bank_rows: 1, filled_bank_tiles: 1*4*4=16
    //    1 |   16|   17|   18|   19|     <- rest_tiles: 14 (16-29)
    //      |   20|   21|   22|   23|     <- rest_full_contig_ids: 2 (16, 17)
    //      |   24|   25|   26|   27|     <- rest_half_contig_ids: 2 (18, 19)
    //    __|   28|   29|            __   <- rest_orphan_tiles: 1 (26, 27)

    bool skip_num_banks = false;
    uint32_t rest_half_contig_ids, rest_orphan_tiles;
    if (num_banks * 3 < rest_tiles) {
        rest_half_contig_ids = (num_banks - rest_full_contig_ids);
        rest_orphan_tiles = rest_half_contig_ids;
        skip_num_banks = true;
    } else if (num_banks * 2 <= rest_tiles) {
        rest_half_contig_ids = num_banks;
        rest_orphan_tiles = (rest_tiles) % (num_banks * 2);
    } else if (num_banks < rest_tiles) {
        rest_half_contig_ids = (rest_tiles) % num_banks;
        rest_orphan_tiles = num_banks - rest_half_contig_ids;
    } else {
        rest_half_contig_ids = 0;
        rest_orphan_tiles = rest_tiles;
    }

    fabric_send_2contig_bf8(
        rest_half_contig_ids, tile_id, tensor0_addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
    if (skip_num_banks) {
        tile_id += 2 * num_banks - rest_half_contig_ids;
    }
    fabric_send_non_contig(
        rest_orphan_tiles, tile_id, tensor0_addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
}

template <bool DRAM>
inline void fabric_send_dim2_bf16(
    uint32_t num_tiles_per_chip,
    uint32_t& tile_id,
    uint32_t rest_tiles,
    InterleavedAddrGenFast<DRAM>& tensor0_addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    auto rest_orphan_tiles = 0;
    if (num_banks * 1 < rest_tiles) {
        rest_orphan_tiles = num_banks - (rest_tiles % (num_banks * (packet_size_in_pages - 1)));
    } else {
        rest_orphan_tiles = num_tiles_per_chip % (num_banks * packet_size_in_pages);
    }

    fabric_send_non_contig(
        rest_orphan_tiles, tile_id, tensor0_addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
}

template <bool DRAM>
inline void fabric_send_dim2(
    uint32_t num_tiles_per_chip,
    uint32_t tile_id,
    InterleavedAddrGenFast<DRAM>& tensor0_addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    // e.g. num_banks: 4, packet_size_in_pages: 2
    //      |       chip0           |___
    //    0 | id 0|    1|    2|    3|
    //    __|    4|    5|    6|    7|___
    //    1 |    8|    9|   10|   11|
    //    __|   12|   13|   14|   15|___  <- filled_bank_rows: 2, filled_bank_tiles: 2*4*2=16
    //    2 |   16|   17|   18|   19|     <- rest_tiles: 7 (16-22), rest_orphan_tiles: 1 (19)
    //    __|   20|   21|   22|           <- rest_full_contig_ids: 3 (16,17,18,20,21,22)
    //     ---------------------------------------------------
    //      |       chip1           |
    //      |   23|  24| ......

    auto filled_bank_rows = num_tiles_per_chip / (num_banks * packet_size_in_pages);
    auto rest_tiles = num_tiles_per_chip % (num_banks * packet_size_in_pages);
    auto filled_bank_tiles = filled_bank_rows * num_banks * packet_size_in_pages;
    auto rest_full_contig_ids = 0;
    auto rest_full_contig_rows = 0;
    if (num_banks * (packet_size_in_pages - 1) < rest_tiles) {
        rest_full_contig_ids = (rest_tiles) % (num_banks * (packet_size_in_pages - 1));
    }
    // send fully contig tiles. e.g. tileID: 0-15, 16-18, 20-22
    fabric_send_full_contig(
        filled_bank_rows * num_banks + rest_full_contig_ids,
        tile_id,
        tensor0_addrgen,
        pkt_hdr_forward,
        pkt_hdr_backward,
        fabric_connection);

    if constexpr (packet_size_in_pages == 2) {  // bf16
        // e.g. tileID: 19
        fabric_send_dim2_bf16(
            num_tiles_per_chip,
            tile_id,
            rest_tiles,
            tensor0_addrgen,
            pkt_hdr_forward,
            pkt_hdr_backward,
            fabric_connection);
    } else {  // bf8
        fabric_send_dim2_bf8(
            filled_bank_tiles,
            rest_full_contig_ids,
            tile_id,
            rest_tiles,
            tensor0_addrgen,
            pkt_hdr_forward,
            pkt_hdr_backward,
            fabric_connection);
    }
}

template <bool DRAM>
inline void fabric_send_generic(
    uint32_t num_tiles_per_chip,
    uint32_t tile_id,
    uint32_t tile_cols_per_chip,
    InterleavedAddrGenFast<DRAM>& tensor0_addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    for (uint32_t i = 0; i < num_tiles_per_chip; i += packet_size_in_pages) {
        uint32_t num_pages_to_read = min(num_tiles_per_chip - i, packet_size_in_pages);
        cb_wait_front(cb0_id, num_pages_to_read);
        size_t l1_read_addr = get_read_ptr(cb0_id);
        for (uint32_t j = 0; j < num_pages_to_read; j++) {
            uint64_t noc0_dest_noc_addr = get_noc_addr(tile_id, tensor0_addrgen, 0 /*offset*/, 0 /*noc_id*/);
            fabric_write_wrapper(
                noc0_dest_noc_addr,
                pkt_hdr_forward,
                pkt_hdr_backward,
                fabric_connection,
                l1_read_addr,
                tensor0_page_size);
            tile_id++;
            if constexpr (last_dim) {
                if (tile_id % tile_cols_per_chip == 0) {
                    tile_id += (tile_cols_per_chip * (ring_size - 1));
                }
            }
        }
        cb_pop_front(cb0_id, num_pages_to_read);
    }
}

/*
 * CCL Send will present various operating modes. Although there is only a single send kernel, it may (compile time)
 * dispatch implementations depending on those invocation parameters.
 */
void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////

    size_t arg_idx = 0;
    // Load the input tensor spec
    address_t tensor_address0 = get_arg_val<address_t>(arg_idx++);
    const size_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tile_id_start = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tile_id_end = get_arg_val<uint32_t>(arg_idx++);
    bool wait_output_semaphore = get_arg_val<uint32_t>(arg_idx++);
    bool reset_global_semaphore = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
    const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    uint32_t num_tiles_per_chip = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tile_cols_per_chip = get_arg_val<uint32_t>(arg_idx++);
    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_idx);

    // packet header cb
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_forward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_addr_backward = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);
    cb_reserve_back(reserved_packet_header_cb_id, 1);
    auto packet_header_buffer_seminc = get_write_ptr(reserved_packet_header_cb_id);
    cb_push_back(reserved_packet_header_cb_id, 1);

    // pre-populate packet headers
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
        reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
    pkt_hdr_forward->to_chip_multicast(
        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
    pkt_hdr_backward->to_chip_multicast(
        tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});

    // interleaved addrgen
    constexpr bool is_dram = buffer0_type == tt::tt_metal::BufferType::DRAM;
    auto tensor0_addrgen = InterleavedAddrGenFast<is_dram>{
        .bank_base_address = tensor_address0, .page_size = tensor0_page_size, .data_format = get_dataformat(cb0_id)};

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.open();
    }

    // when last_dim == true, tile_id coordinate is as follows
    //      |        chip0          |       chip1           |
    //      |  tile_cols_per_chip   |                       |
    //      | id 0|    1|    2|    3|    4|    5|    6|    7|
    // row  |    8|    9|   10|   11|   12|   13|   14|   15|
    //      |   16|   17|   18|   19|   20|   21|   22|   23|
    //      |   24|   25|   26|   27|   28|   29|   30|   31|
    //
    // else (dim == 1 or dim == 2)
    //      |                     chip0                     |
    //      | id 0|    1|    2|    3|    4|    5|    6|    7|
    //      |    8|    9|   10|   11|   12|   13|   14|   15|
    //     ---------------------------------------------------
    //      |                     chip1                     |
    //      |   16|   17|   18|   19|   20|   21|   22|   23|
    //      |   24|   25|   26|   27|   28|   29|   30|   31|
    //

    uint32_t tile_id = tile_id_start;
    if constexpr (use_best_effort) {
        if constexpr (last_dim) {
            if constexpr (packet_size_in_pages == 2) {  // bf16
                fabric_send_dim3_bf16_remain_even<is_dram>(
                    num_tiles_per_chip,
                    tile_id,
                    tile_cols_per_chip,
                    tensor0_addrgen,
                    pkt_hdr_forward,
                    pkt_hdr_backward,
                    fabric_connection);
            } else {
                fabric_send_dim3_bf8_dram_remain048<is_dram>(
                    num_tiles_per_chip,
                    tile_id,
                    tile_cols_per_chip,
                    tensor0_addrgen,
                    pkt_hdr_forward,
                    pkt_hdr_backward,
                    fabric_connection);
            }
        } else {
            fabric_send_dim2(
                num_tiles_per_chip, tile_id, tensor0_addrgen, pkt_hdr_forward, pkt_hdr_backward, fabric_connection);
        }
    } else {
        fabric_send_generic<is_dram>(
            num_tiles_per_chip,
            tile_id,
            tile_cols_per_chip,
            tensor0_addrgen,
            pkt_hdr_forward,
            pkt_hdr_backward,
            fabric_connection);
    }

    // 2. mcast output ready semaphore
    uint64_t out_ready_sem_noc_addr_in_pkt =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr, 0);
    auto* pkt_hdr = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
    pkt_hdr->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
        out_ready_sem_noc_addr_in_pkt,
        static_cast<uint16_t>(1),  // increment 1
        32});
    // Write the mcast packet (forward)
    if (fabric_connection.has_forward_connection()) {
        fabric_connection.get_forward_connection().wait_for_empty_write_slot();
        pkt_hdr->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_sync_targets_forward)});
        fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }
    // Write the mcast packet (backward)
    if (fabric_connection.has_backward_connection()) {
        pkt_hdr->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_sync_targets_backward)});
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_non_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }
    // increment locally
    uint64_t out_ready_sem_noc_addr =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);
    noc_semaphore_inc(out_ready_sem_noc_addr, 1);

    // 3. wait for mcast output ready semaphore
    if (wait_output_semaphore) {
        while (*reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr) < out_ready_sem_wait_value);
        DPRINT << "waitval done\n";
    }

    // 4. global semaphore reset
    if (reset_global_semaphore) {
        *reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr) = 0;
        DPRINT << "reset done\n";
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    noc_async_write_barrier();
}
