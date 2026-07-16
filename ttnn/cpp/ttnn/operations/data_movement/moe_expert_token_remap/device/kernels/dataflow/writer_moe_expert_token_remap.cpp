// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "api/debug/dprint_pages.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    constexpr uint32_t local_experts_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t metadata_dfb_id = get_compile_time_arg_val(1);
    constexpr uint32_t data_dfb_id = get_compile_time_arg_val(2);
    constexpr uint32_t output_mapping_dfb_id = get_compile_time_arg_val(3);
    constexpr uint32_t output_reduced_dfb_id = get_compile_time_arg_val(4);
    constexpr uint32_t selected_experts_k = get_compile_time_arg_val(5);
    constexpr uint32_t num_local_experts = get_compile_time_arg_val(6);
    constexpr uint32_t output_mapping_page_size_bytes = get_compile_time_arg_val(7);  // num_local_experts * datum size
    constexpr uint32_t datum_size_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t output_reduced_page_size_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t reduction_size = get_compile_time_arg_val(10);

    constexpr auto output_mapping_args = TensorAccessorArgs<11>();
    constexpr auto output_reduced_args =
        TensorAccessorArgs<decltype(output_mapping_args)::next_compile_time_args_offset()>();

    using data_addr_t = tt::data_movement::common::ByteSizeAddressType<datum_size_bytes>::type;

    const auto output_mapping_base_addr = get_arg_val<uint32_t>(0);
    const auto start_idx = get_arg_val<uint32_t>(1);
    const auto end_idx = get_arg_val<uint32_t>(2);
    const auto output_reduced_base_addr = get_arg_val<uint32_t>(3);
    const auto reduce_start_idx = get_arg_val<uint32_t>(4);

    const auto output_mapping_addrgen = TensorAccessor(output_mapping_args, output_mapping_base_addr);
    const auto output_reduced_addrgen = TensorAccessor(output_reduced_args, output_reduced_base_addr);

    Noc noc;
    DataflowBuffer local_experts_dfb(local_experts_dfb_id);
    DataflowBuffer metadata_dfb(metadata_dfb_id);
    DataflowBuffer data_dfb(data_dfb_id);
    DataflowBuffer output_mapping_dfb(output_mapping_dfb_id);
    DataflowBuffer output_reduced_dfb(output_reduced_dfb_id);

    // scratch space for mapping
    output_mapping_dfb.reserve_back(1);
    const uint32_t output_l1_addr = output_mapping_dfb.get_write_ptr();
    output_mapping_dfb.push_back(1);

    // scratch space for reduction
    output_reduced_dfb.reserve_back(1);
    const uint32_t reduced_l1_addr = output_reduced_dfb.get_write_ptr();
    output_reduced_dfb.push_back(1);
    tt::data_movement::common::fill_with_val<uint16_t>(reduced_l1_addr, num_local_experts, 0u);
    auto reduced_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(reduced_l1_addr);

    local_experts_dfb.wait_front(1);
    auto local_experts_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(local_experts_dfb.get_read_ptr());

    for (uint32_t bs = start_idx, reduce_idx = reduce_start_idx, reduction_count = 0; bs < end_idx;
         ++bs, ++reduction_count) {
        metadata_dfb.wait_front(1);
        const uint32_t metadata_l1_addr = metadata_dfb.get_write_ptr();
        auto metadata_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(metadata_l1_addr);

        tt::data_movement::common::fill_with_val<data_addr_t>(output_l1_addr, num_local_experts, 0);

        bool found = false;
        for (uint32_t e = 0; e < num_local_experts; ++e) {
            const auto& expert_idx = local_experts_ptr[e];
            if (ttnn::operations::ccl::common::find_if<uint16_t, selected_experts_k, false>(metadata_ptr, expert_idx)) {
                if (!found) {
                    data_dfb.wait_front(1);
                    const uint32_t data_l1_addr = data_dfb.get_read_ptr();
                    found = true;
                }

                const uint32_t topk_l1_addr = data_dfb.get_read_ptr() + expert_idx * datum_size_bytes;
                const uint32_t output_l1_element_addr = output_l1_addr + e * datum_size_bytes;
                tt::data_movement::common::tt_memmove<false, false, false, datum_size_bytes>(
                    noc, output_l1_element_addr, topk_l1_addr, datum_size_bytes);

                reduced_l1_ptr[e] = 1;
            }
        }
        CoreLocalMem<uint32_t> map_src(output_l1_addr);
        noc.async_write(
            map_src,
            output_mapping_addrgen,
            output_mapping_page_size_bytes,
            {.offset_bytes = 0},
            {.page_id = bs, .offset_bytes = 0});

        if (found) {
            data_dfb.pop_front(1);
            found = false;
        }

        if (reduction_count == reduction_size - 1) {
            CoreLocalMem<uint32_t> red_src(reduced_l1_addr);
            noc.async_write(
                red_src,
                output_reduced_addrgen,
                output_reduced_page_size_bytes,
                {.offset_bytes = 0},
                {.page_id = reduce_idx++, .offset_bytes = 0});
            noc.async_write_barrier();
            tt::data_movement::common::fill_with_val<uint16_t>(reduced_l1_addr, num_local_experts, 0u);
            reduction_count = 0;
        }

        metadata_dfb.pop_front(1);
    }
    local_experts_dfb.pop_front(1);
}
