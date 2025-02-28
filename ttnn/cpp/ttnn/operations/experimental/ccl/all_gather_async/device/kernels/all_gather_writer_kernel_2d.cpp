// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

// clang-format off
#include "dataflow_api.h"
#include "debug/dprint.h"
#include "fabric/hw/inc/tt_fabric.h"
#include "fabric/hw/inc/tt_fabric_interface.h"
#include "fabric/hw/inc/tt_fabric_api.h"
#include "tests/tt_metal/tt_metal/perf_microbenchmark/common/kernel_utils.hpp"

// clang-format on

using namespace tt::tt_fabric;

volatile fabric_client_interface_t* client_interface;

uint64_t xy_local_addr;

void kernel_main() {
    constexpr uint32_t client_interface_cb = get_compile_time_arg_val(0);
    constexpr uint32_t is_horizontal = get_compile_time_arg_val(1);
    constexpr uint32_t dst_is_dram = get_compile_time_arg_val(2);
    constexpr uint32_t cb_id_in0 = get_compile_time_arg_val(3);
    constexpr uint32_t num_devices = get_compile_time_arg_val(4);
    constexpr uint32_t this_device_id = get_compile_time_arg_val(5);
    constexpr uint32_t element_size = get_compile_time_arg_val(6);
    constexpr uint32_t semaphore_target_value = get_compile_time_arg_val(7);

    uint32_t rt_args_idx = 0;
    uint32_t src_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t dst_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t lower_pages = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t higher_pages = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t num_bytes = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t page_size = get_arg_val<uint32_t>(rt_args_idx++);

    uint32_t dst_mesh_id_dir0 = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t dst_device_id_dir0 = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t depth_dir0 = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t router_noc_xy_dir0 = get_arg_val<uint32_t>(rt_args_idx++);

    uint32_t dst_mesh_id_dir1 = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t dst_device_id_dir1 = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t depth_dir1 = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t router_noc_xy_dir1 = get_arg_val<uint32_t>(rt_args_idx++);

    uint32_t first_device = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t last_device = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t sephamore_src_addr = get_arg_val<uint32_t>(rt_args_idx++);
    uint32_t sephamore_noc_encoding = get_arg_val<uint32_t>(rt_args_idx++);

    uint32_t num_dirs = (!first_device && !last_device) ? 2 : 1;

    const InterleavedAddrGen<dst_is_dram> s = {.bank_base_address = dst_addr, .page_size = num_bytes};

    uint32_t packet_size_bytes = num_bytes + PACKET_HEADER_SIZE_BYTES;

    DPRINT << "dst_is_dram: " << (uint32_t)dst_is_dram << ENDL();
    DPRINT << "cb_id_in0: " << (uint32_t)cb_id_in0 << ENDL();
    DPRINT << "num_devices: " << (uint32_t)num_devices << ENDL();
    DPRINT << "this_device_id: " << (uint32_t)this_device_id << ENDL();
    DPRINT << "element_size: " << (uint32_t)element_size << ENDL();
    DPRINT << "semaphore_target_value: " << (uint32_t)semaphore_target_value << ENDL();

    DPRINT << "src_addr: " << (uint32_t)src_addr << ENDL();
    DPRINT << "dst_addr: " << (uint32_t)dst_addr << ENDL();
    DPRINT << "lower_pages: " << (uint32_t)lower_pages << ENDL();
    DPRINT << "higher_pages: " << (uint32_t)higher_pages << ENDL();
    DPRINT << "num_bytes: " << (uint32_t)num_bytes << ENDL();
    DPRINT << "page_size: " << (uint32_t)page_size << ENDL();

    DPRINT << "dst_mesh_id_dir0: " << (uint32_t)dst_mesh_id_dir0 << ENDL();
    DPRINT << "dst_device_id_dir0: " << (uint32_t)dst_device_id_dir0 << ENDL();
    DPRINT << "depth_dir0: " << (uint32_t)depth_dir0 << ENDL();
    DPRINT << "router_noc_xy_dir0: " << (uint32_t)router_noc_xy_dir0 << ENDL();

    DPRINT << "dst_mesh_id_dir1: " << (uint32_t)dst_mesh_id_dir1 << ENDL();
    DPRINT << "dst_device_id_dir1: " << (uint32_t)dst_device_id_dir1 << ENDL();
    DPRINT << "depth_dir1: " << (uint32_t)depth_dir1 << ENDL();
    DPRINT << "router_noc_xy_dir1: " << (uint32_t)router_noc_xy_dir1 << ENDL();

    DPRINT << "first_device: " << (uint32_t)first_device << ENDL();
    DPRINT << "last_device: " << (uint32_t)last_device << ENDL();
    DPRINT << "sephamore_src_addr: " << (uint32_t)sephamore_src_addr << ENDL();
    DPRINT << "sephamore_noc_encoding: " << (uint32_t)sephamore_noc_encoding << ENDL();

    DPRINT << "num_dirs: " << (uint32_t)num_dirs << ENDL();
    DPRINT << "packet_size_bytes: " << (uint32_t)packet_size_bytes << ENDL();

    uint32_t client_interface_addr = get_write_ptr(client_interface_cb);
    volatile tt_l1_ptr fabric_pull_client_interface_t* client_interface =
        reinterpret_cast<volatile tt_l1_ptr fabric_pull_client_interface_t*>(client_interface_addr);

    DPRINT << "after setting vlient interface " << ENDL();
    for (uint32_t i = 0; i < num_dirs; i++) {
        DPRINT << "before fabric endpoint init for dirextion index : " << (uint32_t)i << ENDL();
        fabric_endpoint_init(client_interface + i, 0 /* unused */);
        DPRINT << "after fabric endpoint init for dirextion index : " << (uint32_t)i << ENDL();
    }
    DPRINT << "after fabric endpoint init " << ENDL();
    uint32_t l1_read_addr = get_read_ptr(cb_id_in0);
    cb_wait_front(cb_id_in0, higher_pages * lower_pages);

    uint32_t original_addr = get_read_ptr(cb_id_in0);
    auto* ptr_orig = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(original_addr);
    for (uint32_t ii1 = 0; ii1 < 1024; ii1 = ii1 + 1) {
        if (ii1 % 16 == 0) {
            DPRINT << "CHECK HERE ";
        }
        DPRINT << "value at i1 = " << (uint32_t)ii1 << " is: " << BF16((uint16_t)ptr_orig[ii1]) << ENDL();
    }

    if constexpr (is_horizontal) {
        for (uint32_t i = 0; i < higher_pages; i++) {
            for (uint32_t k = 0; k < lower_pages; k++) {
                uint32_t src_offset = i * lower_pages * element_size + k;
                uint32_t dst_offset =
                    i * num_devices * lower_pages * element_size + k + this_device_id * lower_pages * element_size;
                uint64_t dst_noc_addr = get_noc_addr_helper(dst_offset, dst_addr);
                noc_async_write(l1_read_addr + src_offset, dst_noc_addr, num_bytes);

                if (!last_device) {
                    fabric_async_write_multicast(
                        client_interface,
                        router_noc_xy_dir0,
                        l1_read_addr + src_offset,  // source address in sender’s memory
                        dst_mesh_id_dir0,
                        dst_device_id_dir0,
                        dst_noc_addr,       // destination write address
                        packet_size_bytes,  // number of bytes to write to remote destination
                        depth_dir0,
                        0,
                        0,
                        0);
                }
            }
        }
        if (!last_device) {
            fabric_wait_for_pull_request_bytes_flushed(client_interface, PACKET_HEADER_SIZE_BYTES);
        }
        packet_header_t* packet_header = (packet_header_t*)(src_addr);

        client_interface++;

        packet_header->routing.dst_mesh_id = dst_mesh_id_dir0;
        packet_header->routing.dst_dev_id = dst_device_id_dir0;
        packet_header->packet_parameters.mcast_parameters.east = 0;
        packet_header->packet_parameters.mcast_parameters.west = depth_dir0;

        for (uint32_t i = 0; i < higher_pages; i++) {
            for (uint32_t k = 0; k < lower_pages; k++) {
                uint32_t src_offset = i * lower_pages * element_size + k;
                uint32_t dst_offset =
                    i * num_devices * lower_pages * element_size + k + this_device_id * lower_pages * element_size;
                uint64_t dst_noc_addr = get_noc_addr_helper(dst_offset, dst_addr);
                if (!first_device) {
                    fabric_async_write_multicast<AsyncWriteMode::ADD_AND_SEND_PR>(
                        client_interface,
                        router_noc_xy_dir1,
                        l1_read_addr + src_offset,  // source address in sender’s memory
                        dst_mesh_id_dir1,
                        dst_device_id_dir1,
                        dst_noc_addr,       // destination write address
                        packet_size_bytes,  // number of bytes to write to remote destination
                        0,
                        depth_dir1,
                        0,
                        0);
                }
            }
        }
    }

    else {
        for (uint32_t i = 0; i < higher_pages; i++) {
            for (uint32_t k = 0; k < lower_pages; k++) {
                uint32_t src_offset = i * lower_pages * element_size + k;
                uint32_t dst_offset =
                    i * num_devices * lower_pages * element_size + k + this_device_id * lower_pages * element_size;
                uint64_t dst_noc_addr = get_noc_addr_helper(dst_offset, dst_addr);
                noc_async_write(l1_read_addr + src_offset, dst_noc_addr, num_bytes);
                if (!last_device) {
                    fabric_async_write_multicast(
                        client_interface,
                        router_noc_xy_dir0,
                        l1_read_addr + src_offset,  // source address in sender’s memory
                        dst_mesh_id_dir0,
                        dst_device_id_dir0,
                        dst_noc_addr,       // destination write address
                        packet_size_bytes,  // number of bytes to write to remote destination
                        0,
                        0,
                        depth_dir0,
                        0);
                }
            }
        }
        if (!last_device) {
            fabric_wait_for_pull_request_bytes_flushed(client_interface, PACKET_HEADER_SIZE_BYTES);
        }
        packet_header_t* packet_header = (packet_header_t*)(src_addr);

        client_interface++;

        packet_header->routing.dst_mesh_id = dst_mesh_id_dir1;
        packet_header->routing.dst_dev_id = dst_device_id_dir1;
        packet_header->packet_parameters.mcast_parameters.north = 0;
        packet_header->packet_parameters.mcast_parameters.south = depth_dir1;

        for (uint32_t i = 0; i < higher_pages; i++) {
            for (uint32_t k = 0; k < lower_pages; k++) {
                uint32_t src_offset = i * lower_pages * element_size + k;
                uint32_t dst_offset =
                    i * num_devices * lower_pages * element_size + k + this_device_id * lower_pages * element_size;
                uint64_t dst_noc_addr = get_noc_addr_helper(dst_offset, dst_addr);
                if (!first_device) {
                    fabric_async_write_multicast<AsyncWriteMode::ADD_AND_SEND_PR>(
                        client_interface,
                        router_noc_xy_dir1,
                        l1_read_addr + src_offset,  // source address in sender’s memory
                        dst_mesh_id_dir1,
                        dst_device_id_dir1,
                        dst_noc_addr,       // destination write address
                        packet_size_bytes,  // number of bytes to write to remote destination
                        0,
                        0,
                        0,
                        depth_dir1);
                }
            }
        }
    }

    /*
    //semaphores
    //mcast output ready semaphore
    client_interface = reinterpret_cast<volatile tt_l1_ptr fabric_pull_client_interface_t*>(client_interface_addr);
    uint32_t atomic_inc = 1;
    uint32_t wrap_boundary = 31;
    uint64_t semaphore_dst_addr = get_noc_addr_helper(sephamore_noc_encoding, dst_addr);
    if (!last_device) {
        fabric_atomic_inc(
            client_interface,
            router_noc_xy_dir0,
            sephamore_src_addr,
            dst_mesh_id_dir0,
            dst_device_id_dir0,
            semaphore_dst_addr,
            atomic_inc,
            wrap_boundary);
    }
    client_interface++;
    if (!first_device) {
        fabric_atomic_inc(
            client_interface,
            router_noc_xy_dir1,
            sephamore_src_addr,
            dst_mesh_id_dir1,
            dst_device_id_dir1,
            semaphore_dst_addr,
            atomic_inc,
            wrap_boundary);
    }

    //increment locally
    //not sure about the address
    noc_semaphore_inc(sephamore_src_addr, 1);

    //wait for semaphore
    while (*reinterpret_cast<volatile uint32_t*>(sephamore_src_addr) < semaphore_target_value) {

    }

    */

    // Flush all pull requests
    client_interface = reinterpret_cast<volatile tt_l1_ptr fabric_pull_client_interface_t*>(client_interface_addr);
    for (uint32_t i = 0; i < num_dirs; i++) {
        fabric_wait_for_pull_request_flushed(client_interface);
        client_interface++;
    }
    noc_async_write_barrier();
    cb_pop_front(cb_id_in0, higher_pages * lower_pages);
}
