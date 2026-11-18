// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "api/dataflow/dataflow_api.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/scratchpad.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"
#include "experimental/kernel_args.h"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"
#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"

#include "api/debug/dprint_pages.h"

namespace detail {

// (experts // devices, batch, seq, hidden_size)
template <uint32_t Batch, uint32_t Seq, bool LocallyReduced>
inline uint32_t get_data_page_idx(const uint32_t e, const uint32_t token) {
    if constexpr (LocallyReduced) {
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
}  // namespace detail

using ttnn::operations::ccl::common::find_if;

void kernel_main() {
    constexpr uint32_t num_local_experts = get_arg(args::num_local_experts);
    constexpr uint32_t batch_size = get_arg(args::batch_size);
    constexpr uint32_t seq_size = get_arg(args::seq_size);
    constexpr uint32_t num_mapping_pages = get_arg(args::num_mapping_pages);
    constexpr uint32_t linearized_mesh_coord = get_arg(args::linearized_mesh_coord);
    constexpr uint32_t data_size_bytes = get_arg(args::data_size_bytes);
    constexpr uint32_t selected_experts_k = get_arg(args::selected_experts_k);
    constexpr uint32_t mapping_page_size_bytes = get_arg(args::mapping_page_size_bytes);
    constexpr uint32_t metadata_page_size_bytes = get_arg(args::metadata_page_size_bytes);
    constexpr bool locally_reduced = get_arg(args::locally_reduced);

    const auto token_start_idx = get_arg(args::token_start_idx);
    const auto token_end_idx = get_arg(args::token_end_idx);

    const auto metadata_addrgen = TensorAccessor(tensor::metadata);
    const auto mapping_addrgen = TensorAccessor(tensor::mapping);
    const auto data_addrgen = TensorAccessor(tensor::data);

    Noc noc;
    Scratchpad<uint8_t> mapping(scratch::mapping);
    DataflowBuffer dfb_local_experts(dfb::local_experts);
    DataflowBuffer dfb_metadata(dfb::metadata);
    DataflowBuffer dfb_data(dfb::data);

    // this gets sent to writer
    dfb_local_experts.reserve_back(1);
    auto local_experts_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(dfb_local_experts.get_write_ptr());

    // temp buffer just used here
    uint32_t mapping_buffer_addr = reinterpret_cast<uint32_t>(&mapping[0]);

    detail::get_device_expert_indices<linearized_mesh_coord, num_mapping_pages, mapping_page_size_bytes>(
        noc, mapping_addrgen, mapping_buffer_addr, mapping_page_size_bytes, local_experts_ptr);
    dfb_local_experts.push_back(1);
    for (uint32_t token = token_start_idx; token < token_end_idx; ++token) {
        dfb_metadata.reserve_back(1);
        const uint32_t metadata_l1_addr = dfb_metadata.get_write_ptr();
        noc.async_read(
            metadata_addrgen,
            dfb_metadata,
            metadata_page_size_bytes,
            {.page_id = token, .offset_bytes = 0},
            {.offset_bytes = 0});
        noc.async_read_barrier();

        auto metadata_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(metadata_l1_addr);

        for (uint32_t e = 0; e < num_local_experts; ++e) {
            const auto& expert_idx = local_experts_ptr[e];
            if (find_if<uint16_t, selected_experts_k, false>(metadata_ptr, expert_idx)) {
                const uint32_t data_page_idx =
                    detail::get_data_page_idx<batch_size, seq_size, locally_reduced>(e, token);
                dfb_data.reserve_back(1);

                noc.async_read(
                    data_addrgen,
                    dfb_data,
                    data_size_bytes,
                    {.page_id = data_page_idx, .offset_bytes = 0},
                    {.offset_bytes = 0});
                noc.async_read_barrier();

                dfb_data.push_back(1);

                if constexpr (locally_reduced) {
                    break;
                }
            }
        }
        dfb_metadata.push_back(1);
    }
}
