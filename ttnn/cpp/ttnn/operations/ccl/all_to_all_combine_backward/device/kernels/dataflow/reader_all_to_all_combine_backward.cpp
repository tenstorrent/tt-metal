// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

// Backward kernel for all_to_all_combine.
//
// Forward all_to_all_combine:
//   Input:  [K_or_1, B_global, S, H] per device (expert contributions)
//   Output: [K, tokens_per_device, H] per device (gathered token contributions)
//
// Backward (this kernel + writer_all_to_all_combine_backward):
//   Input:  [K, tokens_per_device, H] per device (grad of forward output)
//   Output: [K_or_1, B_global, S, H] per device (grad of forward input)
//
// Reader responsibilities:
//   1. Read the mapping tensor once to build expert_to_device[] and expert_to_local_idx[].
//   2. For each global token assigned to this core, read the metadata page and K grad pages.

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/debug/dprint.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

void kernel_main() {
    // ---- Compile-time args ----
    constexpr uint32_t mapping_cb_id           = get_compile_time_arg_val(0);
    constexpr uint32_t expert_device_map_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t metadata_cb_id          = get_compile_time_arg_val(2);
    constexpr uint32_t data_cb_id              = get_compile_time_arg_val(3);
    constexpr uint32_t num_experts             = get_compile_time_arg_val(4);
    constexpr uint32_t num_devices             = get_compile_time_arg_val(5);
    constexpr uint32_t batch_size              = get_compile_time_arg_val(6);  // global batch
    constexpr uint32_t seq_size               = get_compile_time_arg_val(7);  // global seq
    constexpr uint32_t selected_experts_k     = get_compile_time_arg_val(8);
    constexpr uint32_t linearized_mesh_coord  = get_compile_time_arg_val(9);
    constexpr uint32_t grad_size_bytes        = get_compile_time_arg_val(10);
    constexpr uint32_t mapping_page_size_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t metadata_page_size_bytes = get_compile_time_arg_val(12);
    constexpr bool locally_reduced            = get_compile_time_arg_val(13);
    constexpr auto grad_args     = TensorAccessorArgs<14>();
    constexpr auto mapping_args  = TensorAccessorArgs<grad_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<mapping_args.next_compile_time_args_offset()>();

    // ---- Runtime args ----
    // [0] mapping tensor address
    // [1] metadata tensor address
    // [2] grad_output tensor address
    // [3] global token start index for this core
    // [4] global token end index for this core
    // [5] global token start index for this device (= device_in_group * tokens_per_device)
    // [6] tokens_per_device (total local tokens on this device)
    const auto mapping_tensor_addr        = get_arg_val<uint32_t>(0);
    const auto metadata_tensor_addr       = get_arg_val<uint32_t>(1);
    const auto grad_tensor_addr           = get_arg_val<uint32_t>(2);
    const auto token_global_start         = get_arg_val<uint32_t>(3);
    const auto token_global_end           = get_arg_val<uint32_t>(4);
    const auto token_global_device_start  = get_arg_val<uint32_t>(5);
    const auto tokens_per_device          = get_arg_val<uint32_t>(6);

    const auto grad_addrgen     = TensorAccessor(grad_args, grad_tensor_addr, grad_size_bytes);
    const auto mapping_addrgen  = TensorAccessor(mapping_args, mapping_tensor_addr, mapping_page_size_bytes);
    const auto metadata_addrgen = TensorAccessor(metadata_args, metadata_tensor_addr, metadata_page_size_bytes);

    // ---- Phase 1: Build expert_to_device[] and expert_to_local_idx[] ----
    // expert_device_map CB layout: [expert_to_device[0..E-1] | expert_to_local_idx[0..E-1]]
    cb_reserve_back(expert_device_map_cb_id, 1);
    auto expert_map_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(expert_device_map_cb_id));
    volatile tt_l1_ptr uint16_t* expert_to_device_ptr    = expert_map_ptr;
    volatile tt_l1_ptr uint16_t* expert_to_local_idx_ptr = expert_map_ptr + num_experts;

    // Temp buffer for one mapping page
    cb_reserve_back(mapping_cb_id, 1);
    const uint32_t mapping_buf_addr = get_write_ptr(mapping_cb_id);
    cb_push_back(mapping_cb_id, 1);

    // Track how many experts we have seen per device to compute local indices
    uint16_t local_count[num_devices] = {};

    for (uint32_t e = 0; e < num_experts; ++e) {
        const uint64_t map_noc_addr = get_noc_addr(e, mapping_addrgen);
        noc_async_read(map_noc_addr, mapping_buf_addr, mapping_page_size_bytes);
        noc_async_read_barrier();

        auto mptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mapping_buf_addr);
        for (uint32_t d = 0; d < num_devices; ++d) {
            invalidate_l1_cache();
            if (mptr[d] == 1u) {
                expert_to_device_ptr[e]    = static_cast<uint16_t>(d);
                expert_to_local_idx_ptr[e] = local_count[d]++;
                break;
            }
        }
    }
    cb_push_back(expert_device_map_cb_id, 1);
    DPRINT << "[BWD READER " << linearized_mesh_coord << "] expert map built, t=[" << token_global_start << "," << token_global_end << ") dev_start=" << token_global_device_start << "\n";

    // ---- Phase 2: Per-token metadata + K grad pages ----
    for (uint32_t t_global = token_global_start; t_global < token_global_end; ++t_global) {
        const uint32_t t_local = t_global - token_global_device_start;

        // Push metadata page for this global token
        DPRINT << "[BWD READER " << linearized_mesh_coord << "] t=" << t_global << " reserving metadata\n";
        cb_reserve_back(metadata_cb_id, 1);
        DPRINT << "[BWD READER " << linearized_mesh_coord << "] t=" << t_global << " reading metadata\n";
        const uint32_t metadata_l1_addr = get_write_ptr(metadata_cb_id);
        const uint64_t metadata_noc_addr = get_noc_addr(t_global, metadata_addrgen);
        noc_async_read(metadata_noc_addr, metadata_l1_addr, metadata_page_size_bytes);
        noc_async_read_barrier();
        DPRINT << "[BWD READER " << linearized_mesh_coord << "] t=" << t_global << " pushing metadata\n";
        cb_push_back(metadata_cb_id, 1);

        // Push K grad pages for this token
        for (uint32_t k = 0; k < selected_experts_k; ++k) {
            const uint32_t grad_page_idx = k * tokens_per_device + t_local;
            cb_reserve_back(data_cb_id, 1);
            const uint32_t grad_l1_addr = get_write_ptr(data_cb_id);
            const uint64_t grad_noc_addr = get_noc_addr(grad_page_idx, grad_addrgen);
            noc_async_read(grad_noc_addr, grad_l1_addr, grad_size_bytes);
            noc_async_read_barrier();
            cb_push_back(data_cb_id, 1);
        }
    }
}
