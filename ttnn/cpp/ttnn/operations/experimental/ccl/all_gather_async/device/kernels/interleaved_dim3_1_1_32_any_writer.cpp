// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_constants.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/common/interpreter_backends/kernel_common/noc_addr.hpp"
#include "minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;

///////////////////////////////////////////////////
// COMPILE TIME ARGS
///////////////////////////////////////////////////

constexpr uint32_t my_chip_id = get_compile_time_arg_val(0);
constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(1);
constexpr uint32_t num_packet_headers_storable = get_compile_time_arg_val(2);
constexpr BufferType buffer0_type = static_cast<BufferType>(get_compile_time_arg_val(3));
constexpr uint32_t cb0_id = get_compile_time_arg_val(4);
constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(5);
constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(6);
constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(7);
constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(8);
constexpr bool last_dim = get_compile_time_arg_val(9);
constexpr uint32_t num_banks = get_compile_time_arg_val(10);

template <bool DRAM>
inline void fabric_send_non_contig_tiles(
    uint32_t num_tiles,
    uint32_t& tile_id,
    uint32_t ring_size,
    uint32_t tile_cols_per_chip,
    uint32_t tile_id_start,
    InterleavedAddrGenFast<DRAM>& addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    uint32_t row = 0;
    DPRINT << "\t[W][" << (uint32_t)my_chip_id << "] BEGIN fabric_send_non_contig_tiles tile_id: " << tile_id
           << ", num_tiles: " << num_tiles << ", packet_size_in_pages: " << packet_size_in_pages << "\n";
    for (uint32_t i = 0; i < num_tiles; i += packet_size_in_pages) {
        uint32_t num_pages_to_read = min(num_tiles - i, packet_size_in_pages);
        DPRINT << "\t[W][" << (uint32_t)my_chip_id << "] num_pages_to_read: " << num_pages_to_read << "\n";
        cb_wait_front(cb0_id, num_pages_to_read);
        size_t l1_read_addr = get_read_ptr(cb0_id);
        for (uint32_t j = 0; j < num_pages_to_read; j++) {
            uint64_t noc0_dest_noc_addr = get_noc_addr(tile_id, addrgen, 0 /*offset*/, 0 /*noc_id*/);
            DPRINT << "\t\t\t[W][" << (uint32_t)my_chip_id << "] rest_tiles i: " << i << ", j: " << j
                   << ", tile_id: " << tile_id << ", packet_size_in_pages: " << packet_size_in_pages << "\n";
            write_and_advance_local_read_address_for_fabric_write(
                noc0_dest_noc_addr,
                pkt_hdr_forward,
                pkt_hdr_backward,
                fabric_connection,
                l1_read_addr,
                tensor0_page_size);
            tile_id++;
            if constexpr (last_dim) {
                if (tile_id % tile_cols_per_chip == 0) {
                    row++;
                    tile_id = row * (tile_cols_per_chip * ring_size) + tile_id_start;
                }
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb0_id, num_pages_to_read);
        DPRINT << "\t\t[W][" << (uint32_t)my_chip_id << "] outer DONE i: " << i << " -> " << i + num_pages_to_read
               << "\n";
    }
    DPRINT << "\t[W][" << (uint32_t)my_chip_id << "] DONE fabric_send_non_contig_tiles tile_id: " << tile_id
           << ", num_tiles: " << num_tiles << "\n";
}

inline uint32_t dim3_rel2abs_tile_id(
    uint32_t rel_tile_id, uint32_t tile_cols_per_chip, uint32_t ring_size, uint32_t ring_idx) {
    uint32_t row = rel_tile_id / tile_cols_per_chip;
    uint32_t idx = rel_tile_id % tile_cols_per_chip;
    return idx + (tile_cols_per_chip * ring_idx) + row * ring_size * tile_cols_per_chip;
}

template <bool DRAM>
inline void fabric_send_contig_tiles_dim3_bf16(
    uint32_t num_tiles,
    uint32_t ring_size,
    uint32_t tile_cols_per_chip,
    InterleavedAddrGenFast<DRAM>& addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    DPRINT << "\t[W][" << (uint32_t)my_chip_id << "] BEGIN fabric_send_contig_tiles_dim3_bf16 num_tiles: " << num_tiles
           << "\n";
    uint32_t row = 0;
    uint32_t total = 0;
    uint32_t tile_id = 0;
    uint32_t total_cols = tile_cols_per_chip * ring_size;
    uint32_t contig_cnt = 0;
    uint32_t end_abs_tile_id = dim3_rel2abs_tile_id(num_tiles - 1, tile_cols_per_chip, ring_size, my_chip_id);
    while (total < num_tiles) {
        uint32_t abs_tile_id = dim3_rel2abs_tile_id(tile_id, tile_cols_per_chip, ring_size, my_chip_id);
        size_t l1_read_addr = get_read_ptr(cb0_id);
        if ((abs_tile_id >= num_banks) &&
            ((uint32_t)my_chip_id == ((abs_tile_id - num_banks) % total_cols) / tile_cols_per_chip)) {
            if ((abs_tile_id >= 2 * num_banks) &&
                ((uint32_t)my_chip_id == ((abs_tile_id - 2 * num_banks) % total_cols) / tile_cols_per_chip)) {
                cb_wait_front(cb0_id, 2);
                DPRINT << "\t\t\t[W][" << (uint32_t)my_chip_id
                       << "] write non contig tiles abs_tile_id: " << abs_tile_id << "\n";
                for (uint32_t j = 0; j < 1; j++) {  // TODO: loop count
                    uint64_t noc0_dest_noc_addr = get_noc_addr(abs_tile_id, addrgen, 0 /*offset*/, 0 /*noc_id*/);
                    write_and_advance_local_read_address_for_fabric_write(
                        noc0_dest_noc_addr,
                        pkt_hdr_forward,
                        pkt_hdr_backward,
                        fabric_connection,
                        l1_read_addr,
                        tensor0_page_size);
                    tile_id++;
                    total++;
                    if (tile_id % tile_cols_per_chip == 0) {
                        row++;
                    }
                }
                noc_async_writes_flushed();
                cb_pop_front(cb0_id, 2);
            } else {
                DPRINT << "\t\t[W][" << (uint32_t)my_chip_id << "] SKIP!! total: " << total << ", tile_id: " << tile_id
                       << ", abs_tile_id: " << abs_tile_id << "\n";
                tile_id++;
                if (tile_id % tile_cols_per_chip == 0) {
                    row++;
                }
            }
        } else {
            DPRINT << "\t\t[W][" << (uint32_t)my_chip_id << "] total: " << total << ", tile_id: " << tile_id
                   << ", abs_tile_id: " << abs_tile_id << "\n";
            if ((abs_tile_id + num_banks) <= end_abs_tile_id &&
                (uint32_t)my_chip_id == ((abs_tile_id + num_banks) % total_cols) / tile_cols_per_chip) {
                cb_wait_front(cb0_id, packet_size_in_pages);
                DPRINT << "\t\t\t[W][" << (uint32_t)my_chip_id << "] write contig tiles: abs_tile_id: " << abs_tile_id
                       << ", contig_cnt: " << contig_cnt << "\n";
                uint64_t noc0_dest_noc_addr = get_noc_addr(abs_tile_id, addrgen, 0 /*offset*/, 0 /*noc_id*/);
                write_and_advance_local_read_address_for_fabric_write(
                    noc0_dest_noc_addr,
                    pkt_hdr_forward,
                    pkt_hdr_backward,
                    fabric_connection,
                    l1_read_addr,
                    packet_size_in_pages * tensor0_page_size);

                noc_async_writes_flushed();
                cb_pop_front(cb0_id, packet_size_in_pages);
                contig_cnt++;
                tile_id++;
                total += 2;
                if (tile_id % tile_cols_per_chip == 0) {
                    row++;
                }
            } else {
                cb_wait_front(cb0_id, 2);
                DPRINT << "\t\t\t[W][" << (uint32_t)my_chip_id
                       << "] write non contig tiles abs_tile_id: " << abs_tile_id << "\n";
                for (uint32_t j = 0; j < 1; j++) {  // TODO: loop count
                    uint64_t noc0_dest_noc_addr = get_noc_addr(abs_tile_id, addrgen, 0 /*offset*/, 0 /*noc_id*/);
                    write_and_advance_local_read_address_for_fabric_write(
                        noc0_dest_noc_addr,
                        pkt_hdr_forward,
                        pkt_hdr_backward,
                        fabric_connection,
                        l1_read_addr,
                        tensor0_page_size);
                    tile_id++;
                    total++;
                    if (tile_id % tile_cols_per_chip == 0) {
                        row++;
                    }
                }
                noc_async_writes_flushed();
                cb_pop_front(cb0_id, 2);
            }
        }
    }
    DPRINT << "\t[W][" << (uint32_t)my_chip_id << "] DONE fabric_send_contig_tiles_dim3_bf16 num_tiles: " << num_tiles
           << "\n";
}

template <bool DRAM>
inline void fabric_send_llama70_t3k_prefill(
    uint32_t num_tiles,
    uint32_t ring_size,
    uint32_t tile_cols_per_chip,
    InterleavedAddrGenFast<DRAM>& addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    static const uint32_t input_width = 2004;
    constexpr uint32_t contig_total =
        ((input_width / (num_banks * packet_size_in_pages)) * (num_banks * packet_size_in_pages));
    uint32_t tile_id = my_chip_id * input_width;  // abs
    uint32_t total = 0;

    DPRINT << "\t[W][" << (uint32_t)my_chip_id << "] BEGIN fabric_send_llama70_t3k_prefill num_tiles: " << num_tiles
           << ", contig_total: " << contig_total << ", num_banks: " << num_banks
           << ", packet_size_in_pages: " << packet_size_in_pages << "\n";

    while (total < contig_total) {  // 1792 (2004//(56*4) * 56*4) for DRAM bank
        for (uint32_t i = 0; i < num_banks; i++) {
            cb_wait_front(cb0_id, packet_size_in_pages);
            size_t l1_read_addr = get_read_ptr(cb0_id);
            DPRINT << "\t\t\t[W][" << (uint32_t)my_chip_id << "] write contig tiles: abs_tile_id: " << tile_id << "\n";
            uint64_t noc0_dest_noc_addr = get_noc_addr(tile_id, addrgen, 0 /*offset*/, 0 /*noc_id*/);
            write_and_advance_local_read_address_for_fabric_write(
                noc0_dest_noc_addr,
                pkt_hdr_forward,
                pkt_hdr_backward,
                fabric_connection,
                l1_read_addr,
                packet_size_in_pages * tensor0_page_size);

            noc_async_writes_flushed();
            cb_pop_front(cb0_id, packet_size_in_pages);
            tile_id++;
            total += packet_size_in_pages;
        }
        tile_id += num_banks * (packet_size_in_pages - 1);
    }

    while (total < num_tiles) {
        cb_wait_front(cb0_id, 1);
        size_t l1_read_addr = get_read_ptr(cb0_id);
        DPRINT << "\t\t\t[W][" << (uint32_t)my_chip_id << "] write non contig tiles: abs_tile_id: " << tile_id << "\n";
        uint64_t noc0_dest_noc_addr = get_noc_addr(tile_id, addrgen, 0 /*offset*/, 0 /*noc_id*/);
        write_and_advance_local_read_address_for_fabric_write(
            noc0_dest_noc_addr, pkt_hdr_forward, pkt_hdr_backward, fabric_connection, l1_read_addr, tensor0_page_size);
        noc_async_writes_flushed();
        cb_pop_front(cb0_id, 1);
        tile_id++;
        total++;
    }
    DPRINT << "\t[W][" << (uint32_t)my_chip_id << "] DONE fabric_send_llama70_t3k_prefill num_tiles: " << num_tiles
           << ", num_banks: " << num_banks << ", packet_size_in_pages: " << packet_size_in_pages << "\n";
}

template <bool DRAM>
inline void fabric_send_full_contig_tiles(
    uint32_t num_cols,
    uint32_t& col_id,
    InterleavedAddrGenFast<DRAM>& addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection,
    uint32_t tile_id_start) {
    DPRINT << "\t[W][" << (uint32_t)my_chip_id << "] fabric_send_full_contig_tiles num_cols: " << num_cols
           << ", col_id:" << col_id << "\n";
    while (col_id < num_cols) {
        cb_wait_front(cb0_id, packet_size_in_pages);
        size_t l1_read_addr = get_read_ptr(cb0_id);

        uint64_t noc0_dest_noc_addr = get_noc_addr(col_id + tile_id_start, addrgen, 0 /*offset*/, 0 /*noc_id*/);
        DPRINT << "\t\t[W][" << (uint32_t)my_chip_id << "] tile_id: " << col_id + tile_id_start
               << ", noc0_dest_noc_addr: " << noc0_dest_noc_addr << "\n";
        write_and_advance_local_read_address_for_fabric_write(
            noc0_dest_noc_addr,
            pkt_hdr_forward,
            pkt_hdr_backward,
            fabric_connection,
            l1_read_addr,
            packet_size_in_pages * tensor0_page_size);

        col_id++;
        if (col_id % num_banks == 0) {
            col_id += num_banks * (packet_size_in_pages - 1);
        }
        noc_async_writes_flushed();
        cb_pop_front(cb0_id, packet_size_in_pages);
    }
}

template <bool DRAM>
inline void fabric_send_dim2_bf8(
    uint32_t num_tiles_per_chip,
    uint32_t tile_id_start,
    uint32_t& total,
    uint32_t filled_bank_tiles,
    uint32_t filled_bank_rows,
    uint32_t rest_filled_bank,
    uint32_t rest_full_contig_cols,
    uint32_t ring_size,
    uint32_t tile_cols_per_chip,
    InterleavedAddrGenFast<DRAM>& tensor0_addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    auto rest_2contig_cols = 0;
    auto rest_tiles = 0;
    if (num_banks * 3 < rest_filled_bank) {
        rest_2contig_cols = (num_banks - rest_full_contig_cols);
        rest_tiles = rest_2contig_cols;
    } else if (num_banks * 2 <= rest_filled_bank) {
        rest_2contig_cols = num_banks;
        rest_tiles = (rest_filled_bank) % (num_banks * 2);
    } else if (num_banks < rest_filled_bank) {
        rest_2contig_cols = (rest_filled_bank) % num_banks;
        rest_tiles = num_banks - rest_2contig_cols;
    } else {
        rest_tiles = rest_filled_bank;
    }

    DPRINT << "\t[W][" << (uint32_t)my_chip_id << "] filled_bank_rows: " << filled_bank_rows
           << ", rest_tiles:" << rest_tiles << ", rest_2contig_cols: " << rest_2contig_cols << ", total: " << total
           << ", end: " << rest_2contig_cols + (filled_bank_tiles + rest_full_contig_cols) << "\n";

    uint32_t outer_id = 0;
    while (total < rest_2contig_cols + (filled_bank_tiles + rest_full_contig_cols)) {
        uint32_t num_2contig = min(rest_2contig_cols - outer_id, 2);
        DPRINT << "\t\t[W][" << (uint32_t)my_chip_id << "] total: " << total << ", num_2contig: " << num_2contig
               << "\n";
        // cb_wait_front(cb0_id, num_2contig * 2);
        cb_wait_front(cb0_id, 4);
        size_t l1_read_addr = get_read_ptr(cb0_id);

        uint32_t id = total + tile_id_start;
        for (uint32_t j = 0; j < num_2contig; j++) {
            uint64_t noc0_dest_noc_addr = get_noc_addr(id, tensor0_addrgen, 0 /*offset*/, 0 /*noc_id*/);
            DPRINT << "\t\t\t[W][" << (uint32_t)my_chip_id << "] tile_id: " << id << ", j: " << j << "\n";
            write_and_advance_local_read_address_for_fabric_write(
                noc0_dest_noc_addr,
                pkt_hdr_forward,
                pkt_hdr_backward,
                fabric_connection,
                l1_read_addr,
                2 * tensor0_page_size);
            id++;
        }
        outer_id += num_2contig;
        total += num_2contig;
        if (total % num_banks == 0) {
            total += num_banks + rest_full_contig_cols;
        }
        noc_async_writes_flushed();
        // cb_pop_front(cb0_id, num_2contig * 2);
        cb_pop_front(cb0_id, 4);
    }
    total += tile_id_start;
    fabric_send_non_contig_tiles<DRAM>(
        rest_tiles,
        total,
        ring_size,
        tile_cols_per_chip,
        tile_id_start,
        tensor0_addrgen,
        pkt_hdr_forward,
        pkt_hdr_backward,
        fabric_connection);
}

template <bool DRAM>
inline void fabric_send_dim2_bf16(
    uint32_t num_tiles_per_chip,
    uint32_t tile_id_start,
    uint32_t& total,
    uint32_t filled_bank_rows,
    uint32_t rest_filled_bank,
    uint32_t ring_size,
    uint32_t tile_cols_per_chip,
    InterleavedAddrGenFast<DRAM>& tensor0_addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    auto rest_tiles = 0;
    if (num_banks * 1 < rest_filled_bank) {
        rest_tiles = num_banks - (rest_filled_bank % (num_banks * (packet_size_in_pages - 1)));
    } else {
        rest_tiles = num_tiles_per_chip % (num_banks * packet_size_in_pages);
    }

    DPRINT << "[W][" << (uint32_t)my_chip_id << "] filled_bank_rows: " << filled_bank_rows
           << ", rest_tiles:" << rest_tiles << "\n";

    total += tile_id_start;
    fabric_send_non_contig_tiles<DRAM>(
        rest_tiles,
        total,
        ring_size,
        tile_cols_per_chip,
        tile_id_start,
        tensor0_addrgen,
        pkt_hdr_forward,
        pkt_hdr_backward,
        fabric_connection);
}

template <bool DRAM>
inline void fabric_send_dim2(
    uint32_t num_tiles_per_chip,
    uint32_t tile_id_start,
    uint32_t ring_size,
    uint32_t tile_cols_per_chip,
    InterleavedAddrGenFast<DRAM>& tensor0_addrgen,
    volatile PACKET_HEADER_TYPE* pkt_hdr_forward,
    volatile PACKET_HEADER_TYPE* pkt_hdr_backward,
    FabricConnectionManager& fabric_connection) {
    auto filled_bank_rows = num_tiles_per_chip / (num_banks * packet_size_in_pages);
    auto rest_filled_bank = num_tiles_per_chip % (num_banks * packet_size_in_pages);
    auto rest_full_contig_cols = 0;
    auto filled_bank_tiles = filled_bank_rows * num_banks * packet_size_in_pages;
    // uint32_t tile_id = tile_id_start;
    uint32_t total = 0;

    if (num_banks * (packet_size_in_pages - 1) < rest_filled_bank) {
        rest_full_contig_cols = (rest_filled_bank) % (num_banks * (packet_size_in_pages - 1));
    }
    DPRINT << "[W][" << (uint32_t)my_chip_id << "] filled_bank_rows:" << filled_bank_rows
           << ", rest_filled_bank: " << rest_filled_bank << ", rest_full_contig_cols: " << rest_full_contig_cols
           << "\n";
    fabric_send_full_contig_tiles<DRAM>(
        filled_bank_tiles + rest_full_contig_cols,
        total,
        tensor0_addrgen,
        pkt_hdr_forward,
        pkt_hdr_backward,
        fabric_connection,
        tile_id_start);

    // total += tile_id_start;
    if constexpr (packet_size_in_pages == 2) {  // bf16
        fabric_send_dim2_bf16(
            num_tiles_per_chip,
            tile_id_start,
            total,
            filled_bank_rows,
            rest_filled_bank,
            ring_size,
            tile_cols_per_chip,
            tensor0_addrgen,
            pkt_hdr_forward,
            pkt_hdr_backward,
            fabric_connection);
    } else {  // bf8
        fabric_send_dim2_bf8(
            num_tiles_per_chip,
            tile_id_start,
            total,
            filled_bank_tiles,
            filled_bank_rows,
            rest_filled_bank,
            rest_full_contig_cols,
            ring_size,
            tile_cols_per_chip,
            tensor0_addrgen,
            pkt_hdr_forward,
            pkt_hdr_backward,
            fabric_connection);
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
    uint32_t ring_size = get_arg_val<uint32_t>(arg_idx++);
    uint32_t tile_cols_per_chip = get_arg_val<uint32_t>(arg_idx++);
    size_t arg_for_fab = arg_idx;
    auto fabric_connection = FabricConnectionManager::build_from_args(arg_idx);

    DPRINT << "ct args: \n";
    DPRINT << "my_chip_id: " << (uint32_t)my_chip_id << "\n";
    DPRINT << "reserved_packet_header_cb_id: " << (uint32_t)reserved_packet_header_cb_id << "\n";
    DPRINT << "num_packet_headers_storable: " << (uint32_t)num_packet_headers_storable << "\n";
    DPRINT << "buffer0_type: " << (uint32_t)buffer0_type << "\n";
    DPRINT << "cb0_id: " << (uint32_t)cb0_id << "\n";
    DPRINT << "packet_size_in_pages: " << (uint32_t)packet_size_in_pages << "\n";
    DPRINT << "tensor0_page_size: " << (uint32_t)tensor0_page_size << "\n";
    DPRINT << "num_targets_forward_direction: " << (uint32_t)num_targets_forward_direction << "\n";
    DPRINT << "num_targets_backward_direction: " << (uint32_t)num_targets_backward_direction << "\n";
    DPRINT << "last_dim: " << (uint32_t)last_dim << "\n";
    DPRINT << "num_banks: " << (uint32_t)num_banks << "\n";

    DPRINT << "rt args: \n";
    DPRINT << "tensor_address0: " << (uint32_t)tensor_address0 << "\n";
    DPRINT << "tile_id_start: " << (uint32_t)tile_id_start << "\n";
    DPRINT << "tile_id_end: " << (uint32_t)tile_id_end << "\n";
    DPRINT << "wait_output_semaphore: " << (uint32_t)wait_output_semaphore << "\n";
    DPRINT << "reset_global_semaphore: " << (uint32_t)reset_global_semaphore << "\n";
    DPRINT << "out_ready_sem_bank_addr: " << (uint32_t)out_ready_sem_bank_addr << "\n";
    DPRINT << "out_ready_sem_noc0_x: " << (uint32_t)out_ready_sem_noc0_x << "\n";
    DPRINT << "out_ready_sem_noc0_y: " << (uint32_t)out_ready_sem_noc0_y << "\n";
    DPRINT << "out_ready_sem_wait_value: " << (uint32_t)out_ready_sem_wait_value << "\n";
    DPRINT << "ring_size: " << (uint32_t)ring_size << "\n";
    DPRINT << "tile_cols_per_chip: " << (uint32_t)tile_cols_per_chip << "\n";
    DPRINT << "num_tiles_per_chip: " << (uint32_t)num_tiles_per_chip << "\n";

    DPRINT << "arg_for_fab: " << (uint32_t)arg_for_fab << "\n";
    DPRINT << "fabric_connection arg 0" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 1" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 2" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 3" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";
    DPRINT << "fabric_connection arg 4" << get_arg_val<uint32_t>(arg_for_fab++) << "\n";

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
    DPRINT << "packet_header_buffer_addr_forward: " << (uint32_t)packet_header_buffer_addr_forward << "\n";
    DPRINT << "packet_header_buffer_addr_backward: " << (uint32_t)packet_header_buffer_addr_backward << "\n";
    DPRINT << "packet_header_buffer_seminc: " << (uint32_t)packet_header_buffer_seminc << "\n";

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

    // 1. mcast via fabric to remote tensor addresses
    DPRINT << "num_targets_forward_direction: " << num_targets_forward_direction << "\n";
    DPRINT << "num_targets_backward_direction: " << num_targets_backward_direction << "\n";
    DPRINT << "my_chip_id: " << my_chip_id << "\n";

    DPRINT << "tensor -> CB: " << (uint32_t)cb0_id << "\n";
    DPRINT << "packet size in pages: " << (uint32_t)packet_size_in_pages << "\n";

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

    if constexpr (last_dim) {
        if constexpr (true) {
            if constexpr (packet_size_in_pages == 2) {  // bf16
                fabric_send_contig_tiles_dim3_bf16<is_dram>(
                    num_tiles_per_chip,
                    ring_size,
                    tile_cols_per_chip,
                    tensor0_addrgen,
                    pkt_hdr_forward,
                    pkt_hdr_backward,
                    fabric_connection);
            } else {
                fabric_send_llama70_t3k_prefill<is_dram>(
                    num_tiles_per_chip,
                    ring_size,
                    tile_cols_per_chip,
                    tensor0_addrgen,
                    pkt_hdr_forward,
                    pkt_hdr_backward,
                    fabric_connection);
            }
        } else {
            fabric_send_non_contig_tiles<is_dram>(
                num_tiles_per_chip,
                tile_id_start,
                ring_size,
                tile_cols_per_chip,
                tile_id_start,
                tensor0_addrgen,
                pkt_hdr_forward,
                pkt_hdr_backward,
                fabric_connection);
        }
    } else {
        fabric_send_dim2(
            num_tiles_per_chip,
            tile_id_start,
            ring_size,
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
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
        fabric_connection.get_forward_connection().send_payload_flush_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }
    // Write the mcast packet (backward)
    if (fabric_connection.has_backward_connection()) {
        pkt_hdr->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});
        fabric_connection.get_backward_connection().wait_for_empty_write_slot();
        fabric_connection.get_backward_connection().send_payload_non_blocking_from_address(
            packet_header_buffer_seminc, sizeof(PACKET_HEADER_TYPE));
    }
    // increment locally
    uint64_t out_ready_sem_noc_addr =
        safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);
    noc_semaphore_inc(out_ready_sem_noc_addr, 1);
    DPRINT << "inc done\n";

    // 3. wait for mcast output ready semaphore
    if (wait_output_semaphore) {
        while (*reinterpret_cast<volatile uint32_t*>(out_ready_sem_bank_addr) < out_ready_sem_wait_value);
        DPRINT << "waitval done\n";
    }

    // 4. global semaphore reset
    if (reset_global_semaphore) {
        const uint64_t dest_noc_addr = get_noc_addr(my_x[0], my_y[0], out_ready_sem_bank_addr);
        noc_inline_dw_write(dest_noc_addr, 0);
        DPRINT << "reset done\n";
    }

    if (fabric_connection.is_logically_connected()) {
        fabric_connection.close();
    }

    noc_async_write_barrier();
    DPRINT << "DONE \n";
}
