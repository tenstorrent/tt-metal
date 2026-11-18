// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/global_semaphore.h"
#include "api/dataflow/endpoints.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "api/buffer_types.hpp"
#include "experimental/kernel_args.h"
#include "tt_metal/fabric/hw/inc/edm_fabric/edm_fabric_worker_adapters.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_routing_utils.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/worker_sync_utils.hpp"
#include "cpp/ttnn/operations/ccl/ccl_host_types.hpp"
#include "cpp/ttnn/operations/ccl/kernel_common/sharding_addrgen.hpp"
#include "tt_metal/fabric/hw/inc/tt_fabric_status.h"
#include "tt_metal/fabric/hw/inc/packet_header_pool.h"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include <cstdint>
#include <utility>

using address_t = uint32_t;
using tt::tt_metal::BufferType;
using ttnn::ccl::Topology;

constexpr bool is_first_chip = get_arg(args::is_first_chip);
constexpr bool is_last_chip = get_arg(args::is_last_chip);
constexpr bool direction = get_arg(args::direction);

void kernel_main() {
    ///////////////////////////////////////////////////
    // ARGS
    ///////////////////////////////////////////////////
    // Load the input tensor spec
    const uint32_t stick_size = get_arg(args::stick_size);
    const uint32_t stick_start_id = get_arg(args::stick_start_id);
    const uint32_t num_sticks_to_read = get_arg(args::num_sticks_to_read);
    const uint32_t output_outer_dim_size = get_arg(args::output_outer_dim_size);
    const uint32_t outer_dims_to_forward = get_arg(args::outer_dims_to_forward);
    const uint32_t outer_dims_to_keep_start = get_arg(args::outer_dims_to_keep_start);
    const uint32_t outer_dims_to_keep_end = get_arg(args::outer_dims_to_keep_end);
    const uint32_t outer_dims_to_receive = get_arg(args::outer_dims_to_receive);
    const uint32_t outer_dims_from_forward = get_arg(args::outer_dims_from_forward);
    const uint32_t num_sticks_per_outer_dim = get_arg(args::num_sticks_per_outer_dim);
    const uint8_t out_ready_sem_noc0_x = get_arg(args::out_ready_sem_noc0_x);
    const uint8_t out_ready_sem_noc0_y = get_arg(args::out_ready_sem_noc0_y);
    experimental::GlobalSemaphore out_ready_sem(global_sem::out_ready_sem);
    bool use_barrier_sem = get_arg(args::use_barrier_sem);
    const uint8_t barrier_sem_noc0_x = get_arg(args::barrier_sem_noc0_x);
    const uint8_t barrier_sem_noc0_y = get_arg(args::barrier_sem_noc0_y);
    experimental::GlobalSemaphore barrier_sem(global_sem::barrier_sem);
    // This writer sends to at most one neighbor: the forward one in direction 1, the backward one in direction 0.
    const bool has_fabric_connection = tt::tt_fabric::has_fabric_connection(fabric::connection);
    tt::tt_fabric::WorkerToFabricEdmSender fabric_connection;
    if (has_fabric_connection) {
        fabric_connection = tt::tt_fabric::WorkerToFabricEdmSender::build_from_binding(fabric::connection);
    }

    uint32_t read_size = stick_size;
    const auto dst_accessor = TensorAccessor(tensor::dst);

    Noc noc_obj;
    DataflowBuffer dfb_output(dfb::output);

    // pre-populate packet headers
    auto pkt_hdr = PacketHeaderPool::allocate_header();
    pkt_hdr->to_chip_unicast(1);
    auto pkt_hdr_sem_inc = PacketHeaderPool::allocate_header();

    if (has_fabric_connection) {
        fabric_connection.open();
    }

    // Barrier semaphore
    if (use_barrier_sem) {
        auto pkt_hdr_barrier_sem_inc = PacketHeaderPool::allocate_header();

        if (!is_last_chip) {
            // unicast barrier semaphore
            uint64_t barrier_sem_noc_addr_in_pkt =
                safe_get_noc_addr(barrier_sem_noc0_x, barrier_sem_noc0_y, barrier_sem.address(), 0);
            pkt_hdr_barrier_sem_inc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
                barrier_sem_noc_addr_in_pkt, static_cast<uint32_t>(1)});  // increment 1

            // Write the unicast packet
            if (has_fabric_connection) {
                fabric_connection.wait_for_empty_write_slot();
                pkt_hdr_barrier_sem_inc->to_chip_unicast(1);
                fabric_connection.send_payload_flush_blocking_from_address(
                    (uint32_t)pkt_hdr_barrier_sem_inc, sizeof(PACKET_HEADER_TYPE));
            }
            noc_obj.async_writes_flushed();
        }
        if (!is_last_chip) {
            barrier_sem.wait_min(1);
        }
        barrier_sem.set(0);
    }

    if (!is_last_chip) {
        for (uint32_t outer_dim_id = 0; outer_dim_id < outer_dims_to_forward; outer_dim_id++) {
            uint32_t dst_stick_id = 0;
            if (direction) {
                dst_stick_id = outer_dim_id * num_sticks_per_outer_dim + stick_start_id;
            } else {
                dst_stick_id =
                    (output_outer_dim_size - (outer_dims_to_forward - outer_dim_id)) * num_sticks_per_outer_dim +
                    stick_start_id;
            }
            for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                dfb_output.wait_front(1);
                uint32_t l1_read_addr = dfb_output.get_read_ptr();

                uint64_t dst_noc_addr = dst_accessor.get_noc_addr(dst_stick_id, 0, 0);

                pkt_hdr->to_noc_unicast_write(tt::tt_fabric::NocUnicastCommandHeader{dst_noc_addr}, stick_size);
                fabric_connection.wait_for_empty_write_slot();
                fabric_connection.send_payload_without_header_non_blocking_from_address(l1_read_addr, stick_size);
                fabric_connection.send_payload_flush_non_blocking_from_address(
                    (uint32_t)pkt_hdr, sizeof(PACKET_HEADER_TYPE));

                noc_obj.async_writes_flushed();

                dst_stick_id++;

                noc_obj.async_write_barrier();
                dfb_output.pop_front(1);
            }
        }

        // unicast output ready semaphore
        uint64_t out_ready_sem_noc_addr_in_pkt =
            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem.address(), 0);
        pkt_hdr_sem_inc->to_noc_unicast_atomic_inc(tt::tt_fabric::NocUnicastAtomicIncCommandHeader{
            out_ready_sem_noc_addr_in_pkt, static_cast<uint32_t>(1)});  // increment 1
        // Write the unicast packet
        fabric_connection.wait_for_empty_write_slot();
        pkt_hdr_sem_inc->to_chip_unicast(1);
        fabric_connection.send_payload_flush_blocking_from_address(
            (uint32_t)pkt_hdr_sem_inc, sizeof(PACKET_HEADER_TYPE));
    } else {
        // If we need extend beyond the original input tensor, pad
        if (direction) {
            if (outer_dims_from_forward) {
                dfb_output.wait_front(1);
                uint32_t l1_read_addr = dfb_output.get_read_ptr();
                for (uint32_t outer_dim_id = 0; outer_dim_id < outer_dims_from_forward; outer_dim_id++) {
                    uint32_t dst_stick_id = (outer_dim_id + outer_dims_to_receive +
                                             (outer_dims_to_keep_end - outer_dims_to_keep_start + 1)) *
                                                num_sticks_per_outer_dim +
                                            stick_start_id;
                    for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                        noc_obj.async_write(
                            CoreLocalMem<uint8_t>(l1_read_addr),
                            dst_accessor,
                            stick_size,
                            {},
                            {.page_id = dst_stick_id});

                        dst_stick_id++;

                        noc_obj.async_write_barrier();
                    }
                }
                dfb_output.pop_front(1);
            }
        }
    }

    // Copy the entire input
    if (direction) {
        for (uint32_t outer_dim_id = 0; outer_dim_id <= (outer_dims_to_keep_end - outer_dims_to_keep_start);
             outer_dim_id++) {
            uint32_t dst_stick_id = (outer_dim_id + outer_dims_to_receive) * num_sticks_per_outer_dim + stick_start_id;
            for (uint32_t iter = 0; iter < num_sticks_to_read; ++iter) {
                dfb_output.wait_front(1);
                noc_obj.async_write(dfb_output, dst_accessor, stick_size, {}, {.page_id = dst_stick_id});

                dst_stick_id++;

                noc_obj.async_write_barrier();
                dfb_output.pop_front(1);
            }
        }
    }

    if (has_fabric_connection) {
        fabric_connection.close();
    }
}
