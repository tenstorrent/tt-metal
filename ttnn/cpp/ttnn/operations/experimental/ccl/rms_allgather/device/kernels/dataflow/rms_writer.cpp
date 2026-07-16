// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/dataflow/noc_semaphore.h"
#include "api/core_local_mem.h"
#include "hostdevcommon/common_values.hpp"
#include <tt-metalium/buffer_types.hpp>
#include "tt_metal/fabric/hw/inc/edm_fabric/fabric_connection_manager.hpp"
#include "tt_metal/fabric/hw/inc/noc_addr.h"
#include "cpp/ttnn/operations/ccl/shared_with_host/hetergeneous_data_structs.hpp"
#include "cpp/ttnn/operations/ccl/common/kernels/minimal_ccl_common.hpp"
#include "ttnn/cpp/ttnn/kernel_lib/reduce_helpers_dataflow.hpp"
#include "cpp/ttnn/kernel/dataflow/generate_bcast_scalar.hpp"
#include "reshard_writer.hpp"
#include <cstdint>
#include <utility>
#include "api/tensor/noc_traits.h"
void kernel_main() {
    constexpr bool is_all_to_all_worker = get_compile_time_arg_val(0) == 1;
    constexpr uint32_t cb_in_2 = get_compile_time_arg_val(1);
    constexpr uint32_t cb_in_4 = get_compile_time_arg_val(2);
    constexpr uint32_t cb_to_allgather_writer = get_compile_time_arg_val(3);
    // Todo add these CBs
    constexpr uint32_t reserved_packet_header_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t packet_size_in_pages = get_compile_time_arg_val(5);
    constexpr uint32_t tensor0_page_size = get_compile_time_arg_val(6);
    constexpr uint32_t num_targets_forward_direction = get_compile_time_arg_val(7);
    constexpr uint32_t num_targets_backward_direction = get_compile_time_arg_val(8);
    constexpr uint32_t num_links = get_compile_time_arg_val(9);

    constexpr bool fuse_gamma = get_compile_time_arg_val(10) == 1;
    constexpr uint32_t block_w = get_compile_time_arg_val(11);

    // Circular Buffer CTs
    constexpr uint32_t cb_out_resharded = get_compile_time_arg_val(12);
    constexpr uint32_t cb_out = get_compile_time_arg_val(13);
    constexpr uint32_t eps_cb_id = get_compile_time_arg_val(14);
    constexpr uint32_t post_cb_in_4 = get_compile_time_arg_val(15);
    constexpr uint32_t cb_gamma = get_compile_time_arg_val(16);

    // Data type CTs
    constexpr bool FLOAT32_DTYPE_GAMMA = get_compile_time_arg_val(18) == 1;

    // Reshard writer
    constexpr uint32_t worker_core_stride_w_bytes = get_compile_time_arg_val(19);
    constexpr uint32_t storage_core_stride_w_bytes = get_compile_time_arg_val(20);
    constexpr uint32_t stats_set_semaphore_id = get_compile_time_arg_val(21);
    constexpr uint32_t signaling_cb = get_compile_time_arg_val(22);
    constexpr uint32_t num_blocks = get_compile_time_arg_val(23);
    // num_mcast_dests = num_x * num_y, the cell count of the multicast bounding
    // box. For non-rectangular shard grids this differs from num_blocks (the
    // worker count). The NoC ack counter must be credited against the
    // rectangle size or noc_async_write_barrier() will wait forever.
    constexpr uint32_t num_mcast_dests = get_compile_time_arg_val(24);
    constexpr auto gamma_args = TensorAccessorArgs<25>();

    Noc noc_obj;
    DataflowBuffer cb_packet_header(reserved_packet_header_cb_id);
    DataflowBuffer cb_to_allgather_writer_obj(cb_to_allgather_writer);
    DataflowBuffer cb_signaling(signaling_cb);
    DataflowBuffer cb_gamma_obj(cb_gamma);

    Semaphore<> stats_set_sem(stats_set_semaphore_id);

    size_t arg_idx = 0;
    const uint32_t base_post_rt =
        get_arg_val<uint32_t>(arg_idx++);  // RT 0 holds how many pre RTs there are (which can vary core by core)
    const uint32_t mcast_dest_noc_start_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_dest_noc_start_y = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_dest_noc_end_x = get_arg_val<uint32_t>(arg_idx++);
    const uint32_t mcast_dest_noc_end_y = get_arg_val<uint32_t>(arg_idx++);
    dataflow_kernel_lib::calculate_and_prepare_reduce_scaler<
        cb_in_2,
        ckernel::PoolType::AVG,
        ckernel::ReduceDim::REDUCE_ROW,
        block_w * tt::constants::TILE_WIDTH>();

    if constexpr (is_all_to_all_worker) {
        const uint32_t scalar_c_bits = get_arg_val<uint32_t>(arg_idx++);
        float scalar_c_f = __builtin_bit_cast(float, scalar_c_bits);
        dataflow_kernel_lib::prepare_reduce_scaler<cb_in_4, ckernel::PoolType::AVG, ckernel::ReduceDim::REDUCE_ROW>(
            scalar_c_f);
        const uint32_t post_scalar_c_bits = get_arg_val<uint32_t>(base_post_rt + 0);
        float post_scalar_c_f = __builtin_bit_cast(float, post_scalar_c_bits);
        dataflow_kernel_lib::
            prepare_reduce_scaler<post_cb_in_4, ckernel::PoolType::AVG, ckernel::ReduceDim::REDUCE_ROW>(
                post_scalar_c_f);
    } else {
        arg_idx++;
    }
    const uint32_t gamma_addr = get_arg_val<uint32_t>(base_post_rt + 2);
    const uint32_t gamma_tile_start_id = get_arg_val<uint32_t>(base_post_rt + 3);

    // Reshard writer
#ifndef SKIP_WRITE_BACK
    const uint32_t num_segments_to_write_back = get_arg_val<uint32_t>(base_post_rt + 4);
    const uint32_t storage_core_start_offset = get_arg_val<uint32_t>(base_post_rt + 5);
    tt_l1_ptr uint32_t* segment_args = (tt_l1_ptr uint32_t*)(get_arg_addr(base_post_rt + 6));
#endif

    const uint32_t out_single_tile_size_bytes = get_tile_size(cb_out);
    const uint32_t eps = get_arg_val<uint32_t>(base_post_rt + 1);
    generate_bcast_col_scalar(CircularBuffer(eps_cb_id), eps);
    const uint32_t iteration_number = get_arg_val<uint32_t>(arg_idx++);
    const size_t out_ready_sem_bank_addr = get_arg_val<uint32_t>(arg_idx++);
    uint32_t out_ready_sem_wait_value = get_arg_val<uint32_t>(arg_idx++);
    ttnn::ccl::address_t tensor_address0 = get_arg_val<ttnn::ccl::address_t>(arg_idx++);
    // Start the all gather part
    if (iteration_number == 0) {
        // Do this only on one of the cores

        // To do add these to Program Factory on i=0 case
        uint32_t first_core_tile_start_offset = get_arg_val<uint32_t>(arg_idx++);
        uint32_t num_cores = get_arg_val<uint32_t>(arg_idx++);
        const uint8_t out_ready_sem_noc0_x = get_arg_val<uint32_t>(arg_idx++);
        const uint8_t out_ready_sem_noc0_y = get_arg_val<uint32_t>(arg_idx++);
        tt_l1_ptr uint32_t* core_noc_x = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
        arg_idx += num_cores;
        tt_l1_ptr uint32_t* core_noc_y = (tt_l1_ptr uint32_t*)(get_arg_addr(arg_idx));
        arg_idx += num_cores;
        auto fabric_connection =
            FabricConnectionManager::build_from_args<FabricConnectionManager::BUILD_AND_OPEN_CONNECTION_START_ONLY>(
                arg_idx);

        // packet header cb
        cb_packet_header.reserve_back(1);
        auto packet_header_buffer_addr_forward = cb_packet_header.get_write_ptr();
        cb_packet_header.push_back(1);
        cb_packet_header.reserve_back(1);
        auto packet_header_buffer_addr_backward = cb_packet_header.get_write_ptr();
        cb_packet_header.push_back(1);
        cb_packet_header.reserve_back(1);
        auto packet_header_buffer_seminc = cb_packet_header.get_write_ptr();
        cb_packet_header.push_back(1);
        // pre-populate packet headers
        volatile PACKET_HEADER_TYPE* pkt_hdr_forward =
            reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_forward);
        volatile PACKET_HEADER_TYPE* pkt_hdr_backward =
            reinterpret_cast<volatile PACKET_HEADER_TYPE*>(packet_header_buffer_addr_backward);
        pkt_hdr_forward->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_forward_direction)});
        pkt_hdr_backward->to_chip_multicast(
            tt::tt_fabric::MulticastRoutingCommandHeader{1, static_cast<uint8_t>(num_targets_backward_direction)});
        fabric_connection.open_finish();
        // 1. mcast via fabric to remote tensor addresses
        uint32_t tiles_read = 0;
        uint32_t shard_tile_id = first_core_tile_start_offset;
        uint32_t core_id = 0;
        auto* pkt_hdr = reinterpret_cast<PACKET_HEADER_TYPE*>(packet_header_buffer_seminc);
        uint64_t out_ready_sem_noc_addr_in_pkt =
            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr, 0);
        // Send data and the semaphore with the last tile
        uint32_t num_tiles_to_read_this_core = 1;
        cb_to_allgather_writer_obj.wait_front(num_tiles_to_read_this_core);
        size_t l1_read_addr = cb_to_allgather_writer_obj.get_read_ptr();
        uint64_t noc0_dest_noc_addr = safe_get_noc_addr(core_noc_x[core_id], core_noc_y[core_id], tensor_address0, 0);
        noc0_dest_noc_addr += shard_tile_id * tensor0_page_size;
        fused_write_atomic_and_advance_local_read_address_for_fabric_write(
            noc0_dest_noc_addr,
            pkt_hdr_forward,
            pkt_hdr_backward,
            fabric_connection,
            l1_read_addr,
            tensor0_page_size,
            out_ready_sem_noc_addr_in_pkt,
            1,  // increment 1
            false);

        fabric_connection.close_start();
        cb_to_allgather_writer_obj.pop_front(num_tiles_to_read_this_core);
        // increment locally
        uint64_t out_ready_sem_noc_addr =
            safe_get_noc_addr(out_ready_sem_noc0_x, out_ready_sem_noc0_y, out_ready_sem_bank_addr);
        volatile tt_l1_ptr uint32_t* out_ready_sem_bank_addr_ptr =
            reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr);
        if constexpr (num_links == 1) {
            // We deduct the local increment from the target semaphore value as we don't need internal synchronization
            noc_semaphore_wait(out_ready_sem_bank_addr_ptr, out_ready_sem_wait_value - 1);
        } else {
            // if multiple links then we need to also ensure the local ones have completed by having them also
            // increment the semaphore and including them in the total
            noc_semaphore_inc(out_ready_sem_noc_addr, 1);
            noc_semaphore_wait(
                reinterpret_cast<volatile tt_l1_ptr uint32_t*>(out_ready_sem_bank_addr), out_ready_sem_wait_value);
        }

        // 4. global semaphore reset
        noc_semaphore_set(out_ready_sem_bank_addr_ptr, 0);
        // Signal the other local cores that the semaphore has returned

        stats_set_sem.set(VALID);
        // num_dests counts the multicast bounding-box cells (loopback includes
        // self), not the worker count.
        stats_set_sem.set_multicast<NocOptions::MCAST_INCL_SRC>(
            noc_obj,
            mcast_dest_noc_start_x,
            mcast_dest_noc_start_y,
            mcast_dest_noc_end_x,
            mcast_dest_noc_end_y,
            num_mcast_dests);
        noc_obj.async_write_barrier();
        fabric_connection.close_finish();  // Includes a noc async write barrier
    } else {
        // Wait for the signal that the stats semaphore was written in the all gather core
        stats_set_sem.wait(VALID);
    }
    // Tell the compute kernel it is ok to proceed
    cb_signaling.reserve_back(1);
    cb_signaling.push_back(1);

    if constexpr (fuse_gamma) {
        const uint32_t gamma_tile_bytes = get_tile_size(cb_gamma);
        const auto gamma = TensorAccessor(gamma_args, gamma_addr);

        constexpr uint32_t bytes_in_faceline = FLOAT32_DTYPE_GAMMA ? 64 : 32;
        constexpr uint32_t bytes_in_two_facelines = bytes_in_faceline * 2;
        constexpr uint32_t mask_read_tile_offset_bytes = FLOAT32_DTYPE_GAMMA ? 1024 : 512;

        uint32_t l1_write_addr_gamma = cb_gamma_obj.get_write_ptr();
        cb_gamma_obj.reserve_back(block_w);
        for (uint32_t w = 0; w < block_w; w++) {
            uint32_t tile_id = gamma_tile_start_id + w;
            noc_obj.async_read(
                gamma, CoreLocalMem<uint8_t>(l1_write_addr_gamma), bytes_in_two_facelines, {.page_id = tile_id}, {});
            uint64_t gamma_noc_addr = get_noc_addr(l1_write_addr_gamma + bytes_in_faceline);
            noc_obj.async_read_barrier();  // might be faster to do two separate read instead of barrier.
            noc_async_read(gamma_noc_addr, l1_write_addr_gamma + mask_read_tile_offset_bytes, bytes_in_faceline);
            l1_write_addr_gamma += gamma_tile_bytes;
        }
        noc_obj.async_read_barrier();
        cb_gamma_obj.push_back(block_w);
    }

#ifndef SKIP_WRITE_BACK
    write_minimal_resharded_data<cb_out, cb_out_resharded, worker_core_stride_w_bytes, storage_core_stride_w_bytes>(
        noc_obj, num_segments_to_write_back, storage_core_start_offset, segment_args);
#endif
}
