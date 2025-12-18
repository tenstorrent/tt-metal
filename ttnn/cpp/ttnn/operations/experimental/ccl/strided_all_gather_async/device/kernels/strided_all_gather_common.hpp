// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "fabric/fabric_edm_packet_header.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_mux_interface.hpp"
#include "tt_metal/fabric/hw/inc/linear/addrgen_api.h"
#include <cstdint>
#include <utility>
#include "ttnn/operations/ccl/shared_with_host/sharded_tensor_addr_gen.hpp"
#include "ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "tt_metal/fabric/hw/inc/linear/api.h"

using namespace tt::tt_fabric::linear::experimental;

FORCE_INLINE uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

FORCE_INLINE uint32_t round_up(uint32_t a, uint32_t b) { return b * div_up(a, b); }

FORCE_INLINE void advance_subchunk(
    uint32_t& tensor_row, uint32_t& subchunk_start_row, uint32_t& subchunk_end_row, uint32_t subchunk_height_stride) {
    uint32_t new_subchunk_rows = tensor_row - subchunk_end_row;
    subchunk_start_row = subchunk_start_row + subchunk_height_stride;
    subchunk_end_row = subchunk_end_row + subchunk_height_stride;
    tensor_row = subchunk_start_row + (new_subchunk_rows - 1);
}

FORCE_INLINE int32_t get_chunk_tile(
    uint32_t& tensor_row,
    uint32_t& tensor_col,
    uint32_t num_workers,
    uint32_t& subchunk_start_row,
    uint32_t& subchunk_end_row,
    uint32_t subchunk_height_stride,
    uint32_t chunk_start_col,
    uint32_t chunk_end_col,
    uint32_t chunk_width,
    uint32_t tensor_Wt,
    uint32_t tensor_Ht) {
    // Compute the returned tile index
    uint32_t tile_index = -1;
    if (tensor_row < tensor_Ht) {
        tile_index = tensor_row * tensor_Wt + tensor_col;
    }

    // Update the next tensor row and col
    tensor_col = tensor_col + num_workers;
    if (tensor_col > chunk_end_col) {
        uint32_t tensor_col_chunk_space = tensor_col - chunk_start_col;
        uint32_t tensor_row_delta = tensor_col_chunk_space / chunk_width;
        tensor_row = tensor_row + tensor_row_delta;
        tensor_col = chunk_start_col + tensor_col_chunk_space - (tensor_row_delta * chunk_width);
        if (tensor_row > subchunk_end_row) {
            advance_subchunk(tensor_row, subchunk_start_row, subchunk_end_row, subchunk_height_stride);
        }
    }

    // Return the tile index
    return tile_index;
}

FORCE_INLINE uint32_t
get_sender_id(uint32_t direction, uint32_t my_chip_id, uint32_t slices_received, uint32_t ring_size) {
    int32_t sender_chip_id;
    if (direction == 1) {
        sender_chip_id = my_chip_id + slices_received + 1;
        return (sender_chip_id >= (int)ring_size) ? sender_chip_id - ring_size : sender_chip_id;
    } else {
        sender_chip_id = my_chip_id - (slices_received + 1);
        return (sender_chip_id < 0) ? ring_size + sender_chip_id : sender_chip_id;
    }
}

FORCE_INLINE uint32_t next_mm_aligned_chunk_height(
    uint32_t input_chunk_start_tile, uint32_t M_tiles_per_core, uint32_t input_tensor_Wt, uint32_t mm_block_h) {
    uint32_t input_row = input_chunk_start_tile / input_tensor_Wt;
    if ((input_row + mm_block_h) > M_tiles_per_core) {
        return M_tiles_per_core - input_row;
    } else {
        return mm_block_h;
    }
}

template <typename AddrGenType>
FORCE_INLINE uint32_t read_chunk(
    uint32_t& chunk_start_tile,
    uint32_t worker_tile_offset,
    uint32_t cb_output_id,
    uint32_t tiles_in_chunk,
    uint32_t chunk_width,
    uint32_t subchunk_height,
    uint32_t subchunk_height_stride,
    uint32_t max_tiles_per_packet,
    uint32_t ag_worker_core_id,
    uint32_t ag_worker_cores,
    AddrGenType input_tensor_addrgen,
    uint32_t input_tensor_page_size,
    AddrGenType output_tensor_addrgen,
    uint32_t input_tensor_Wt,
    uint32_t input_tensor_Ht,
    uint32_t output_tensor_Wt,
    uint32_t actual_sender_chip_id,
    bool read_output) {
    uint32_t worker_tiles_in_curr_chunk =
        (tiles_in_chunk / ag_worker_cores) + ((ag_worker_core_id < (tiles_in_chunk % ag_worker_cores)) ? 1 : 0);
    uint32_t num_tiles_per_packet = std::min(max_tiles_per_packet, worker_tiles_in_curr_chunk);
    uint32_t packets_in_curr_chunk = div_up(worker_tiles_in_curr_chunk, num_tiles_per_packet);
    uint32_t chunk_tile_iter = 0;

    // Chunk values (chunk spans all mm cores)
    // convert chunk start from linear index into row and col coord, still in input tensor space
    uint32_t chunk_start_row = chunk_start_tile / input_tensor_Wt;
    uint32_t chunk_start_col = chunk_start_tile % input_tensor_Wt;

    // Subchunk values (subchunk spans just 1 mm core)
    uint32_t subchunk_start_row = chunk_start_row;  // initialize the subchunk tracker (start and end)
    uint32_t subchunk_end_row = chunk_start_row + subchunk_height - 1;

    // Worker chunk values
    uint32_t worker_chunk_start_row_chunk_space = worker_tile_offset / chunk_width;
    uint32_t worker_chunk_start_col_chunk_space = worker_tile_offset % chunk_width;
    uint32_t worker_chunk_row = chunk_start_row + worker_chunk_start_row_chunk_space;
    uint32_t worker_chunk_col = chunk_start_col + worker_chunk_start_col_chunk_space;
    if (worker_chunk_row > subchunk_end_row) {
        advance_subchunk(worker_chunk_row, subchunk_start_row, subchunk_end_row, subchunk_height_stride);
    }
    if (read_output) {
        worker_chunk_col = worker_chunk_col + actual_sender_chip_id * input_tensor_Wt;
        chunk_start_col = chunk_start_col + actual_sender_chip_id * input_tensor_Wt;
    }
    uint32_t chunk_end_col = chunk_start_col + chunk_width - 1;
    for (uint32_t packet_idx = 0; packet_idx < packets_in_curr_chunk; packet_idx++) {
        uint32_t tiles_left_in_chunk = worker_tiles_in_curr_chunk - chunk_tile_iter;
        uint32_t tiles_to_read_in_packet = std::min(tiles_left_in_chunk, num_tiles_per_packet);

        cb_reserve_back(cb_output_id, max_tiles_per_packet);
        size_t l1_write_addr = get_write_ptr(cb_output_id);
        for (uint32_t j = 0; j < tiles_to_read_in_packet; ++j) {
            int32_t tile_id = get_chunk_tile(
                worker_chunk_row,
                worker_chunk_col,
                ag_worker_cores,
                subchunk_start_row,
                subchunk_end_row,
                subchunk_height_stride,
                chunk_start_col,
                chunk_end_col,
                chunk_width,
                read_output ? output_tensor_Wt : input_tensor_Wt,
                input_tensor_Ht);
            if (tile_id >= 0) {
                uint64_t noc_read_addr =
                    get_noc_addr(tile_id, read_output ? output_tensor_addrgen : input_tensor_addrgen);
                noc_async_read(noc_read_addr, l1_write_addr, input_tensor_page_size);

                l1_write_addr += input_tensor_page_size;
            }
            chunk_tile_iter++;
        }

        noc_async_read_barrier();
        cb_push_back(cb_output_id, max_tiles_per_packet);
    }

    uint32_t new_chunk_start_tile = chunk_start_tile + chunk_width;
    uint32_t new_chunk_row = new_chunk_start_tile / input_tensor_Wt;
    if (new_chunk_row != chunk_start_row) {
        chunk_start_tile = (chunk_start_row + subchunk_height) * input_tensor_Wt;
    } else {
        chunk_start_tile = new_chunk_start_tile;
    }
    return chunk_tile_iter;
}

template <typename AddrGenType, uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
FORCE_INLINE uint32_t write_chunk(
    uint32_t& chunk_start_tile,
    uint32_t worker_tile_offset,
    uint32_t cb_output_id,
    uint32_t tiles_in_chunk,
    uint32_t chunk_width,
    uint32_t subchunk_height,
    uint32_t subchunk_height_stride,
    uint32_t max_tiles_per_packet,
    uint32_t ag_worker_core_id,
    uint32_t ag_worker_cores,
    AddrGenType output_addrgen,
    uint32_t output_page_size,
    uint32_t input_tensor_Wt,
    uint32_t input_tensor_Ht,
    uint32_t output_tensor_Wt,
    uint32_t actual_sender_chip_id,
    tt::tt_fabric::WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& mux_connection,
    volatile PACKET_HEADER_TYPE* pkt_scatter_hdr,
    volatile PACKET_HEADER_TYPE* pkt_unicast_hdr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_sem_inc,
    uint64_t out_ready_sem_noc_addr_in_pkt,
    const bool direction,
    const uint32_t num_targets_forward_direction,
    const uint32_t num_targets_backward_direction,
    bool write_local) {
    uint32_t worker_tiles_in_curr_chunk =
        (tiles_in_chunk / ag_worker_cores) + ((ag_worker_core_id < (tiles_in_chunk % ag_worker_cores)) ? 1 : 0);
    uint32_t num_tiles_per_packet = std::min(max_tiles_per_packet, worker_tiles_in_curr_chunk);
    uint32_t packets_in_curr_chunk = div_up(worker_tiles_in_curr_chunk, num_tiles_per_packet);
    uint32_t chunk_tile_iter = 0;

    // Chunk values (chunk spans all mm cores)
    // convert chunk start from linear index into row and col coord, still in input tensor space
    uint32_t chunk_start_row = chunk_start_tile / input_tensor_Wt;
    uint32_t chunk_start_col = chunk_start_tile % input_tensor_Wt;

    // Subchunk values (subchunk spans just 1 mm core)
    uint32_t subchunk_start_row = chunk_start_row;  // initialize the subchunk tracker (start and end)
    uint32_t subchunk_end_row = chunk_start_row + subchunk_height - 1;

    // Worker chunk values
    uint32_t worker_chunk_start_row_chunk_space = worker_tile_offset / chunk_width;
    uint32_t worker_chunk_start_col_chunk_space = worker_tile_offset % chunk_width;
    uint32_t worker_chunk_row = chunk_start_row + worker_chunk_start_row_chunk_space;
    uint32_t worker_chunk_col = chunk_start_col + worker_chunk_start_col_chunk_space;
    if (worker_chunk_row > subchunk_end_row) {
        advance_subchunk(worker_chunk_row, subchunk_start_row, subchunk_end_row, subchunk_height_stride);
    }
    worker_chunk_col = worker_chunk_col + actual_sender_chip_id * input_tensor_Wt;
    chunk_start_col = chunk_start_col + actual_sender_chip_id * input_tensor_Wt;
    uint32_t chunk_end_col = chunk_start_col + chunk_width - 1;

    for (uint32_t packet_idx = 0; packet_idx < packets_in_curr_chunk; packet_idx++) {
        uint32_t tiles_left_in_chunk = worker_tiles_in_curr_chunk - chunk_tile_iter;
        uint32_t tiles_to_write_in_packet = std::min(tiles_left_in_chunk, num_tiles_per_packet);

        cb_wait_front(cb_output_id, max_tiles_per_packet);
        size_t l1_read_addr = get_read_ptr(cb_output_id);

        uint32_t padded_tiles = 0;
        int32_t tile_one_id = get_chunk_tile(
            worker_chunk_row,
            worker_chunk_col,
            ag_worker_cores,
            subchunk_start_row,
            subchunk_end_row,
            subchunk_height_stride,
            chunk_start_col,
            chunk_end_col,
            chunk_width,
            output_tensor_Wt,
            input_tensor_Ht);
        if (tile_one_id < 0) {
            padded_tiles++;
        }
        chunk_tile_iter++;
        int32_t tile_two_id = tile_one_id;
        if (tiles_to_write_in_packet == 2) {
            tile_two_id = get_chunk_tile(
                worker_chunk_row,
                worker_chunk_col,
                ag_worker_cores,
                subchunk_start_row,
                subchunk_end_row,
                subchunk_height_stride,
                chunk_start_col,
                chunk_end_col,
                chunk_width,
                output_tensor_Wt,
                input_tensor_Ht);
            if (tile_two_id < 0) {
                padded_tiles++;
            }
            chunk_tile_iter++;
        }

        tiles_to_write_in_packet = tiles_to_write_in_packet - padded_tiles;
        // Will have more cases once scatter-write supports more than 2 distinct addresses
        switch (tiles_to_write_in_packet) {
            case 2: {
                auto noc_address0 =
                    tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_addrgen, tile_one_id, 0);
                auto noc_address1 =
                    tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_addrgen, tile_two_id, 0);
                if ((direction == 1 && num_targets_backward_direction) ||
                    (direction == 0 && num_targets_forward_direction)) {
                    fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                        &mux_connection,
                        pkt_scatter_hdr,
                        l1_read_addr,
                        NocUnicastScatterCommandHeader({noc_address0, noc_address1}, {0}));
                }
                if (direction == 1 && write_local) {
                    uint64_t local_noc0_dest_noc_addr_tile_one = get_noc_addr(tile_one_id, output_addrgen);
                    uint64_t local_noc0_dest_noc_addr_tile_two = get_noc_addr(tile_two_id, output_addrgen);

                    noc_async_write(l1_read_addr, local_noc0_dest_noc_addr_tile_one, output_page_size);
                    noc_async_write(
                        l1_read_addr + output_page_size, local_noc0_dest_noc_addr_tile_two, output_page_size);
                    noc_async_write_barrier();
                }
                break;
            }
            case 1: {
                auto noc_address0 =
                    tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_addrgen, tile_one_id, 0);
                if ((direction == 1 && num_targets_backward_direction) ||
                    (direction == 0 && num_targets_forward_direction)) {
                    fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                        &mux_connection, pkt_unicast_hdr, l1_read_addr, NocUnicastCommandHeader{noc_address0});
                }
                if (direction == 1 && write_local) {
                    uint64_t local_noc0_dest_noc_addr = get_noc_addr(tile_one_id, output_addrgen);
                    noc_async_write(l1_read_addr, local_noc0_dest_noc_addr, output_page_size);
                    noc_async_write_barrier();
                }
                break;
            }
            case 0:
            default: {
                break;
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_output_id, max_tiles_per_packet);
    }
    // Write the semaphore packet
    if ((direction == 1 && num_targets_backward_direction) || (direction == 0 && num_targets_forward_direction)) {
        fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
            &mux_connection,
            pkt_hdr_sem_inc,
            tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0, 0});
    }

    uint32_t new_chunk_start_tile = chunk_start_tile + chunk_width;
    uint32_t new_chunk_row = new_chunk_start_tile / input_tensor_Wt;
    if (new_chunk_row != chunk_start_row) {
        chunk_start_tile = (chunk_start_row + subchunk_height) * input_tensor_Wt;
    } else {
        chunk_start_tile = new_chunk_start_tile;
    }
    return chunk_tile_iter;
}
