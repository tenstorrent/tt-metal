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

FORCE_INLINE uint32_t
get_next_tile_input(uint32_t local_tile_index, uint32_t input_start_tile_index, uint32_t ag_parallel_factor) {
    // Imagine the input is already permuted (this has nothing to do with our all gather, it's just ordering the output
    // of all gather such it will be ideal for matmul) We split up the work evenly amongst the all gather cores.
    // Probably the best way is just to round robin through the input amongst the various all gather cores.  Ignore
    // direction since you send the same thing forward and backward.  For now just send the whole thing in that order,
    // we can add finer grain fidelity to correspond to the syncs for matmul.  Right now just sync once when we reach
    // the end of the buffer.
    return input_start_tile_index + local_tile_index * ag_parallel_factor;
}

FORCE_INLINE uint32_t get_next_tile_output(
    uint32_t local_tile_index,
    uint32_t input_start_tile_index,
    uint32_t ag_parallel_factor,
    uint32_t input_tensor_Wt,
    uint32_t output_tensor_Wt,
    uint32_t device_index) {
    uint32_t input_tile_index = input_start_tile_index + local_tile_index * ag_parallel_factor;
    uint32_t input_row = input_tile_index / input_tensor_Wt;
    uint32_t input_col = input_tile_index % input_tensor_Wt;
    return input_row * output_tensor_Wt + device_index * input_tensor_Wt +
           input_col;  // TODO should pass device_index*input_tensor_Wt to prevent recalculating them
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

FORCE_INLINE uint32_t div_up(uint32_t a, uint32_t b) { return (a + b - 1) / b; }

template <typename AddrGenType>
FORCE_INLINE uint32_t read_chunk(
    uint32_t global_tile_index,
    uint32_t global_tile_id_start,
    uint32_t cb_output_id,
    uint32_t tiles_per_core,
    uint32_t tiles_per_chunk,
    uint32_t max_tiles_per_packet,
    uint32_t ag_worker_cores,
    AddrGenType input_tensor_addrgen,
    uint32_t input_tensor_page_size) {
    uint32_t next_tile_to_read = global_tile_index;
    uint32_t tiles_left = tiles_per_core - next_tile_to_read;
    uint32_t tiles_in_curr_chunk = std::min(tiles_left, tiles_per_chunk);
    uint32_t num_tiles_per_packet = std::min(max_tiles_per_packet, tiles_in_curr_chunk);
    uint32_t packets_in_curr_chunk = div_up(tiles_in_curr_chunk, num_tiles_per_packet);
    for (uint32_t packet_idx = 0; packet_idx < packets_in_curr_chunk; packet_idx++) {
        uint32_t chunk_tile_idx = next_tile_to_read - global_tile_index;
        uint32_t tiles_left_in_chunk = tiles_in_curr_chunk - chunk_tile_idx;
        uint32_t tiles_to_read_in_packet = std::min(tiles_left_in_chunk, num_tiles_per_packet);

        cb_reserve_back(cb_output_id, max_tiles_per_packet);
        size_t l1_write_addr = get_write_ptr(cb_output_id);
        for (uint32_t j = 0; j < tiles_to_read_in_packet; ++j) {
            uint32_t tile_id = get_next_tile_input(next_tile_to_read, global_tile_id_start, ag_worker_cores);

            uint64_t noc_read_addr = get_noc_addr(tile_id, input_tensor_addrgen);
            noc_async_read(noc_read_addr, l1_write_addr, input_tensor_page_size);

            l1_write_addr += input_tensor_page_size;
            next_tile_to_read++;
        }

        noc_async_read_barrier();
        cb_push_back(cb_output_id, max_tiles_per_packet);
    }

    return next_tile_to_read;
}

template <typename AddrGenType, uint8_t FABRIC_MUX_CHANNEL_NUM_BUFFERS = 0>
FORCE_INLINE uint32_t write_chunk(
    uint32_t global_tile_index,
    uint32_t global_tile_id_start,
    uint32_t cb_output_id,
    uint32_t tiles_per_core,
    uint32_t tiles_per_chunk,
    uint32_t max_tiles_per_packet,
    uint32_t ag_worker_cores,
    AddrGenType output_addrgen,
    uint32_t output_page_size,
    uint32_t input_tensor_Wt,
    uint32_t output_tensor_Wt,
    uint32_t my_chip_id,
    tt::tt_fabric::WorkerToFabricMuxSender<FABRIC_MUX_CHANNEL_NUM_BUFFERS>& mux_connection,
    volatile PACKET_HEADER_TYPE* pkt_scatter_hdr,
    volatile PACKET_HEADER_TYPE* pkt_unicast_hdr,
    volatile PACKET_HEADER_TYPE* pkt_hdr_sem_inc,
    uint64_t out_ready_sem_noc_addr_in_pkt,
    bool direction,
    bool write_local) {
    uint32_t next_tile_to_write = global_tile_index;
    uint32_t tiles_left = tiles_per_core - next_tile_to_write;
    uint32_t tiles_in_curr_chunk = std::min(tiles_left, tiles_per_chunk);
    uint32_t num_tiles_per_packet = std::min(max_tiles_per_packet, tiles_in_curr_chunk);
    uint32_t packets_in_curr_chunk = div_up(tiles_in_curr_chunk, num_tiles_per_packet);
    for (uint32_t packet_idx = 0; packet_idx < packets_in_curr_chunk; packet_idx++) {
        uint32_t chunk_tile_idx = next_tile_to_write - global_tile_index;
        uint32_t tiles_left_in_chunk = tiles_in_curr_chunk - chunk_tile_idx;
        uint32_t tiles_to_write_in_packet = std::min(tiles_left_in_chunk, num_tiles_per_packet);

        cb_wait_front(cb_output_id, max_tiles_per_packet);
        size_t l1_read_addr = get_read_ptr(cb_output_id);

        uint32_t tile_one_id = get_next_tile_output(
            next_tile_to_write, global_tile_id_start, ag_worker_cores, input_tensor_Wt, output_tensor_Wt, my_chip_id);
        next_tile_to_write++;
        uint32_t tile_two_id = tile_one_id;
        if (tiles_to_write_in_packet == 2) {
            tile_two_id = get_next_tile_output(
                next_tile_to_write,
                global_tile_id_start,
                ag_worker_cores,
                input_tensor_Wt,
                output_tensor_Wt,
                my_chip_id);
            next_tile_to_write++;
        }
        auto noc_address0 = tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_addrgen, tile_one_id, 0);
        auto noc_address1 = tt::tt_fabric::linear::addrgen_detail::get_noc_address(output_addrgen, tile_two_id, 0);

        // Will have more cases once scatter-write supports more than 2 distinct addresses
        switch (tiles_to_write_in_packet) {
            case 2: {
                fabric_unicast_noc_scatter_write_with_state<UnicastScatterWriteUpdateMask::DstAddrs>(
                    &mux_connection,
                    pkt_scatter_hdr,
                    l1_read_addr,
                    NocUnicastScatterCommandHeader{{noc_address0, noc_address1}, 0});
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
            case 1:
            default: {
                fabric_unicast_noc_unicast_write_with_state<UnicastWriteUpdateMask::DstAddr>(
                    &mux_connection, pkt_unicast_hdr, l1_read_addr, NocUnicastCommandHeader{noc_address0});
                if (direction == 1 && write_local) {
                    uint64_t local_noc0_dest_noc_addr = get_noc_addr(tile_one_id, output_addrgen);
                    noc_async_write(l1_read_addr, local_noc0_dest_noc_addr, output_page_size);
                    noc_async_write_barrier();
                }
                break;
            }
        }
        noc_async_writes_flushed();
        cb_pop_front(cb_output_id, max_tiles_per_packet);
    }
    // Write the semaphore packet
    fabric_unicast_noc_unicast_atomic_inc_with_state<UnicastAtomicIncUpdateMask::DstAddr>(
        &mux_connection,
        pkt_hdr_sem_inc,
        tt::tt_fabric::NocUnicastAtomicIncCommandHeader{out_ready_sem_noc_addr_in_pkt, 0, 0});
    return next_tile_to_write;
}
