// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "dprint_pages.h"

#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

namespace detail {

template <uint32_t Size, class Enable = void>
struct DataTypeHolder {
    typedef void type;
};

template <uint32_t Size>
struct DataTypeHolder<Size, typename std::enable_if<Size == 2>::type> {
    typedef uint16_t type;
};

template <uint32_t Size>
struct DataTypeHolder<Size, typename std::enable_if<Size == 4>::type> {
    typedef uint32_t type;
};

}  // namespace detail

void kernel_main() {
    constexpr uint32_t local_experts_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t metadata_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t data_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t output_mapping_cb_id = get_compile_time_arg_val(3);
    constexpr uint32_t output_reduced_cb_id = get_compile_time_arg_val(4);
    constexpr uint32_t selected_experts_k = get_compile_time_arg_val(5);
    constexpr uint32_t num_local_experts = get_compile_time_arg_val(6);
    constexpr uint32_t output_mapping_page_size_bytes = get_compile_time_arg_val(7);  // num_local_experts * datum size
    constexpr uint32_t datum_size_bytes = get_compile_time_arg_val(8);
    constexpr uint32_t output_reduced_page_size_bytes = get_compile_time_arg_val(9);
    constexpr uint32_t reduction_size = get_compile_time_arg_val(10);

    constexpr auto output_mapping_args = TensorAccessorArgs<11>();
    constexpr auto output_reduced_args =
        TensorAccessorArgs<decltype(output_mapping_args)::next_compile_time_args_offset()>();

    using data_addr_t = detail::DataTypeHolder<datum_size_bytes>::type;

    const auto output_mapping_base_addr = get_arg_val<uint32_t>(0);
    const auto start_idx = get_arg_val<uint32_t>(1);
    const auto end_idx = get_arg_val<uint32_t>(2);
    const auto output_reduced_base_addr = get_arg_val<uint32_t>(3);
    const auto reduce_start_idx = get_arg_val<uint32_t>(4);

    const auto output_mapping_addrgen = TensorAccessor(
        output_mapping_args, output_mapping_base_addr, output_mapping_page_size_bytes);
    const auto output_reduced_addrgen = TensorAccessor(
        output_reduced_args, output_reduced_base_addr,output_reduced_page_size_bytes);

    // scratch space for mapping
    cb_reserve_back(output_mapping_cb_id, 1);
    const uint32_t output_l1_addr = get_write_ptr(output_mapping_cb_id);
    cb_push_back(output_mapping_cb_id, 1);

    // scratch space for reduction
    cb_reserve_back(output_reduced_cb_id, 1);
    const uint32_t reduced_l1_addr = get_write_ptr(output_reduced_cb_id);
    cb_push_back(output_reduced_cb_id, 1);
    tt::data_movement::common::fill_with_val<uint16_t>(reduced_l1_addr, num_local_experts, 0u);
    auto reduced_l1_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(reduced_l1_addr);

    cb_wait_front(local_experts_cb_id, 1);
    auto local_experts_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(get_read_ptr(local_experts_cb_id));

    for (uint32_t bs = start_idx, reduce_idx = reduce_start_idx, reduction_count = 0; bs < end_idx;
         ++bs, ++reduction_count) {
        cb_wait_front(metadata_cb_id, 1);
        const uint32_t metadata_l1_addr = get_write_ptr(metadata_cb_id);
        auto metadata_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(metadata_l1_addr);

        tt::data_movement::common::fill_with_val<data_addr_t>(output_l1_addr, num_local_experts, 0);

        bool found = false;
        for (uint32_t e = 0; e < num_local_experts; ++e) {
            const auto& expert_idx = local_experts_ptr[e];
            if (ttnn::operations::ccl::common::find_if<uint16_t, selected_experts_k, false>(metadata_ptr, expert_idx)) {
                if (!found) {
                    cb_wait_front(data_cb_id, 1);
                    const uint32_t data_l1_addr = get_read_ptr(data_cb_id);
                    found = true;
                }

                const uint32_t topk_l1_addr = get_read_ptr(data_cb_id) + expert_idx * datum_size_bytes;
                const uint32_t output_l1_element_addr = output_l1_addr + e * datum_size_bytes;
                tt::data_movement::common::tt_memmove<false, false, false, datum_size_bytes>(
                    output_l1_element_addr, topk_l1_addr, datum_size_bytes);

                reduced_l1_ptr[e] = 1;
            }
        }
        const uint64_t output_noc_addr = get_noc_addr(bs, output_mapping_addrgen);
        noc_async_write(output_l1_addr, output_noc_addr, output_mapping_page_size_bytes);

        if (found) {
            cb_pop_front(data_cb_id, 1);
            found = false;
        }

        if (reduction_count == reduction_size - 1) {
            const uint64_t output_reduced_noc_addr = get_noc_addr(reduce_idx++, output_reduced_addrgen);
            noc_async_write(reduced_l1_addr, output_reduced_noc_addr, output_reduced_page_size_bytes);
            noc_async_write_barrier();
            tt::data_movement::common::fill_with_val<uint16_t>(reduced_l1_addr, num_local_experts, 0u);
            reduction_count = 0;
        }

        cb_pop_front(metadata_cb_id, 1);
    }
    cb_pop_front(local_experts_cb_id, 1);
}
