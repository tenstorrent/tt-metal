// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_api.h"
#include "tt_metal/fabric/hw/inc/tt_fabric_interface.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

inline void print_full_tile(uint32_t cb_id, uint32_t tile_id = 0, bool untilize = false) {
    DPRINT << "======" << ENDL();
    for (uint16_t r = 0; r < 32; ++r) {
        DPRINT << (uint)r << " : "
               << TileSlice(
                      cb_id,
                      tile_id,
                      SliceRange{
                          .h0 = (uint8_t)r,
                          .h1 = (uint8_t)(r + 1),
                          .hs = (uint8_t)1,
                          .w0 = (uint8_t)0,
                          .w1 = (uint8_t)32,
                          .ws = (uint8_t)1},
                      true,
                      untilize)
               << ENDL();
    }
    DPRINT << "++++++" << ENDL();
}

void kernel_main() {
    constexpr uint32_t page_size = get_compile_time_arg_val(0);
    constexpr uint32_t cb_id_out = get_compile_time_arg_val(1);
    constexpr uint32_t fabric_sender_cb_index = get_compile_time_arg_val(2);
    constexpr uint32_t fabric_receiver_cb_index = get_compile_time_arg_val(3);
    constexpr uint32_t client_interface_cb_index = get_compile_time_arg_val(4);
    constexpr uint32_t start_tile = get_compile_time_arg_val(5);
    constexpr uint32_t end_tile = get_compile_time_arg_val(6);
    // constexpr uint32_t num_devices = 2;

    const uint32_t output_addr = get_arg_val<uint32_t>(0);
    const uint32_t write_output = get_arg_val<uint32_t>(1);
    uint32_t router_noc_xy = 0, offset_to_receiver = 0, my_noc_xy = 0, dst_mesh_id = 0, dst_device_id = 0;
    if (!write_output) {
        router_noc_xy = get_arg_val<uint32_t>(2);
        offset_to_receiver = get_arg_val<uint32_t>(3);
        my_noc_xy = get_arg_val<uint32_t>(4);
        dst_mesh_id = get_arg_val<uint32_t>(5);
        dst_device_id = get_arg_val<uint32_t>(6);
    }
    // ublocks size defined in tiles
    constexpr uint32_t onetile = 1;
    constexpr uint32_t tile_bytes = get_tile_size(cb_id_out);
    const DataFormat data_format = get_dataformat(cb_id_out);
    // my dst noc offset is actually the same as the noc xy my fabric receiver_cb_index but on the other device
    // uint64_t dst_noc_addr = get_noc_addr_helper(my_noc_xy, get_write_ptr(fabric_receiver_cb_index));
    uint32_t packet_size_bytes = (end_tile - start_tile) * tile_bytes + PACKET_HEADER_SIZE_BYTES;

    uint64_t output_noc_addr = get_noc_addr(get_write_ptr(cb_id_out));
    for (uint32_t tile = start_tile; tile < end_tile; ++tile) {
        if (write_output) {
            DPRINT << "This core's writer is just writing to the output buffer" << ENDL();
            print_full_tile(cb_id_out, tile, true);
        } else {
            DPRINT << "This core's writer is writing to the fabric sender buffer" << " start_tile: " << start_tile
                   << " end_tile: " << end_tile << " my_noc_xy: " << my_noc_xy << " router_noc_xy: " << router_noc_xy
                   << " offset_to_receiver: " << offset_to_receiver << " packet_size_bytes: " << packet_size_bytes
                   << ENDL();
            // write to fabric sender buffer
            uint32_t cb_out_read_ptr = get_read_ptr(cb_id_out);
            uint32_t fabric_sender_cb_read_ptr = get_write_ptr(fabric_sender_cb_index);
            tt::data_movement::common::tt_memmove<true, false, true, tile_bytes>(
                fabric_sender_cb_read_ptr + tile * tile_bytes + PACKET_HEADER_SIZE_BYTES,
                cb_out_read_ptr + tile * tile_bytes,
                tile_bytes);
        }
    }
    noc_async_read_barrier();
    if (!write_output) {
        // get_local_cb_interface(fabric_sender_cb_index).fifo_rd_ptr += PACKET_HEADER_SIZE_BYTES;
        // get_local_cb_interface(fabric_sender_cb_index).fifo_wr_ptr += PACKET_HEADER_SIZE_BYTES;
        // for (uint32_t tile = start_tile; tile < end_tile; ++tile) {
        //     print_full_tile(fabric_sender_cb_index, tile, true);
        // }
        // get_local_cb_interface(fabric_sender_cb_index).fifo_rd_ptr -= PACKET_HEADER_SIZE_BYTES;
        // get_local_cb_interface(fabric_sender_cb_index).fifo_wr_ptr -= PACKET_HEADER_SIZE_BYTES;

        uint32_t dst_addr = output_addr;
        uint64_t dst_noc_addr =
            get_noc_addr(get_read_ptr(cb_id_out));  // for now just overwrite the input buffer on the remote device to
                                                    // check if the data is correct

        uint32_t client_interface_addr = get_write_ptr(client_interface_cb_index);
        volatile tt_l1_ptr fabric_pull_client_interface_t* client_interface =
            reinterpret_cast<volatile tt_l1_ptr fabric_pull_client_interface_t*>(client_interface_addr);
        fabric_endpoint_init(client_interface, 0 /* unused */);
        DPRINT << "dest_noc_addr: " << dst_noc_addr << ENDL();
        fabric_async_write(
            client_interface,
            router_noc_xy,
            get_write_ptr(fabric_sender_cb_index),  // source address in sender’s memory
            dst_mesh_id,
            dst_device_id,
            dst_noc_addr,      // destination write address
            packet_size_bytes  // number of bytes to write to remote destination
        );

        fabric_wait_for_pull_request_flushed(client_interface);
    }
}
