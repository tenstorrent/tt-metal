// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

#include "api/debug/dprint_pages.h"

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
    const Noc& noc,
    const AddrGen& mapping_addrgen,
    const uint32_t mapping_l1_buffer_addr,
    const uint32_t mapping_page_size,
    volatile tt_l1_ptr uint16_t* output_ptr) {
    auto mapping_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(mapping_l1_buffer_addr);
    CoreLocalMem<uint32_t> mapping_dst(mapping_l1_buffer_addr);

    for (uint32_t expert_idx = 0; expert_idx < NumMappingPages; ++expert_idx) {
        noc.async_read(
            mapping_addrgen,
            mapping_dst,
            MappingPageSizeBytes,
            {.page_id = expert_idx, .offset_bytes = 0},
            {.offset_bytes = 0});
        noc.async_read_barrier();

        if (mapping_ptr[DeviceIdx] == 1u) {
            *(output_ptr++) = expert_idx;
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

    const auto metadata_addrgen = TensorAccessor(metadata_args, metadata_tensor_addr);
    const auto mapping_addrgen = TensorAccessor(mapping_args, mapping_tensor_addr);
    const auto data_addrgen = TensorAccessor(data_args, data_tensor_addr);

    Noc noc;
    CircularBuffer mapping_cb(mapping_cb_id);
    CircularBuffer local_experts_cb(local_experts_cb_id);
    CircularBuffer metadata_cb(metadata_cb_id);
    CircularBuffer data_cb(data_cb_id);

    // this gets sent to writer
    local_experts_cb.reserve_back(1);
    auto local_experts_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(local_experts_cb.get_write_ptr());

    // temp buffer just used here
    mapping_cb.reserve_back(1);
    uint32_t mapping_buffer_addr = mapping_cb.get_write_ptr();
    mapping_cb.push_back(1);

    detail::get_device_expert_indices<linearized_mesh_coord, num_mapping_pages, mapping_page_size_bytes>(
        noc, mapping_addrgen, mapping_buffer_addr, mapping_page_size_bytes, local_experts_ptr);
    local_experts_cb.push_back(1);
    for (uint32_t token = token_start_idx; token < token_end_idx; ++token) {
        metadata_cb.reserve_back(1);
        const uint32_t metadata_l1_addr = metadata_cb.get_write_ptr();
        noc.async_read(
            metadata_addrgen,
            metadata_cb,
            metadata_page_size_bytes,
            {.page_id = token, .offset_bytes = 0},
            {.offset_bytes = 0});
        noc.async_read_barrier();

        auto metadata_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(metadata_l1_addr);

        for (uint32_t e = 0; e < num_local_experts; ++e) {
            const auto & expert_idx = local_experts_ptr[e];
            if (find_if<uint16_t, selected_experts_k, false>(metadata_ptr, expert_idx)) {
                const uint32_t data_page_idx =
                    detail::get_data_page_idx<batch_size, seq_size, locally_reduced>(e, token);
                data_cb.reserve_back(1);

                noc.async_read(
                    data_addrgen,
                    data_cb,
                    data_size_bytes,
                    {.page_id = data_page_idx, .offset_bytes = 0},
                    {.offset_bytes = 0});
                noc.async_read_barrier();

                data_cb.push_back(1);

                if constexpr (locally_reduced){
                    break;
                }
            }
        }
        metadata_cb.push_back(1);
    }
}
