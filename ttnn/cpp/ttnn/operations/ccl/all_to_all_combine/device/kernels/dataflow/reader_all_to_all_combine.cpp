// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"

#include "dprint_pages.h"

namespace detail{

// (experts // devices, batch, seq, hidden_size)
template <uint32_t Batch, uint32_t Seq, bool LocallyReduced>
inline uint32_t get_data_page_idx(const uint32_t e, const uint32_t token) {
    if constexpr (LocallyReduced){
        return token;
    } else {
        return e * Batch * Seq + token;
    }
}

template <uint32_t DeviceIdx, uint32_t NumMappingPages, uint32_t MappingPageSizeBytes, typename AddrGen>
void get_device_expert_indices(
    const AddrGen& mapping_addrgen,
    const uint32_t mapping_l1_buffer_addr,
    const uint32_t mapping_page_size,
    volatile tt_l1_ptr uint16_t* output_ptr) {
    auto mapping_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mapping_l1_buffer_addr);

    for (uint32_t expert_idx = 0; expert_idx < NumMappingPages; ++expert_idx) {
        const uint64_t map_page_noc_addr = get_noc_addr(expert_idx, mapping_addrgen);
        noc_async_read(map_page_noc_addr, mapping_l1_buffer_addr,MappingPageSizeBytes);
        noc_async_read_barrier();

        if (mapping_ptr[DeviceIdx] == 1u) {
            *(output_ptr++)=expert_idx;

        }
    }
}
}// namespace detail

using ttnn::operations::ccl::common::find_if;

void kernel_main() {
    constexpr uint32_t mapping_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t local_experts_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t metadata_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t data_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t num_local_experts = get_compile_time_arg_val(4);
    constexpr uint32_t batch_size = get_compile_time_arg_val(5);
    constexpr uint32_t seq_size = get_compile_time_arg_val(6);
    constexpr uint32_t num_mapping_pages= get_compile_time_arg_val(7);
    constexpr uint32_t linearized_mesh_coord = get_compile_time_arg_val(8);
    constexpr uint32_t data_size_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t selected_experts_k = get_compile_time_arg_val(10);
    constexpr uint32_t mapping_page_size_bytes = get_compile_time_arg_val(11);
    constexpr uint32_t metadata_page_size_bytes = get_compile_time_arg_val(12);
    constexpr bool locally_reduced = get_compile_time_arg_val(13);
    constexpr auto data_args = TensorAccessorArgs<14>();
    constexpr auto mapping_args = TensorAccessorArgs<data_args.next_compile_time_args_offset()>();
    constexpr auto metadata_args = TensorAccessorArgs<mapping_args.next_compile_time_args_offset()>();

    const auto mapping_tensor_addr = get_arg_val<uint32_t>(0);
    const auto metadata_tensor_addr = get_arg_val<uint32_t>(1);
    const auto data_tensor_addr = get_arg_val<uint32_t>(2);
    const auto token_start_idx = get_arg_val<uint32_t>(3);
    const auto token_end_idx = get_arg_val<uint32_t>(4);

    const auto metadata_addrgen = TensorAccessor(metadata_args, metadata_tensor_addr, metadata_page_size_bytes);
    const auto mapping_addrgen = TensorAccessor(mapping_args, mapping_tensor_addr, mapping_page_size_bytes);
    const auto data_addrgen = TensorAccessor(data_args, data_tensor_addr, data_size_bytes);

    // this gets sent to writer
    cb_reserve_back(local_experts_cb_id,1);
    auto local_experts_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_write_ptr(local_experts_cb_id));

    // temp buffer just used here
    cb_reserve_back(mapping_cb_id,1);
    uint32_t mapping_buffer_addr = get_write_ptr(mapping_cb_id);
    cb_push_back(mapping_cb_id, 1);

    detail::get_device_expert_indices<linearized_mesh_coord, num_mapping_pages, mapping_page_size_bytes>(
        mapping_addrgen, mapping_buffer_addr, mapping_page_size_bytes, local_experts_ptr);
    cb_push_back(local_experts_cb_id,1);
    for (uint32_t token = token_start_idx; token < token_end_idx; ++token) {
        cb_reserve_back(metadata_cb_id,1);
        const uint32_t metadata_l1_addr = get_read_ptr(metadata_cb_id);
        const uint64_t metadata_noc_addr = get_noc_addr(token, metadata_addrgen);
        noc_async_read(metadata_noc_addr, metadata_l1_addr, metadata_page_size_bytes);
        noc_async_read_barrier();

        auto metadata_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(metadata_l1_addr);

        for (uint32_t e = 0; e < num_local_experts; ++e) {
            const auto & expert_idx = local_experts_ptr[e];
            if (find_if<uint16_t, selected_experts_k, false>(metadata_ptr, expert_idx)) {
                const uint32_t data_page_idx =
                    detail::get_data_page_idx<batch_size, seq_size, locally_reduced>(e, token);
                cb_reserve_back(data_cb_id, 1);

                const uint32_t data_l1_addr=get_write_ptr(data_cb_id);
                const uint64_t data_noc_addr = get_noc_addr(data_page_idx, data_addrgen);
                noc_async_read(data_noc_addr,data_l1_addr,data_size_bytes);
                noc_async_read_barrier();

                cb_push_back(data_cb_id, 1);

                if constexpr (locally_reduced){
                    break;
                }
            }
        }
        cb_push_back(metadata_cb_id, 1);
    }
}
