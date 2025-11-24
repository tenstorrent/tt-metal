// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "dprint_pages.h"

#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

using namespace tt::data_movement::common;

void kernel_main() {
    constexpr uint32_t routing_weights_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t local_weights_idxs_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t local_weights_cb_id = get_compile_time_arg_val(2);
    constexpr uint32_t num_cluster_experts = get_compile_time_arg_val(3);
    constexpr uint32_t num_non_zero_per_device = get_compile_time_arg_val(4);
    constexpr uint32_t weight_datum_size_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t local_weights_page_size_bytes = weight_datum_size_bytes * num_cluster_experts;

    constexpr auto local_weights_args = TensorAccessorArgs<6>();
    using weight_addr_t = ByteSizeAddressType<weight_datum_size_bytes>::type;

    const auto local_weights_base_address = get_arg_val<uint32_t>(0);

    const auto local_weights_addrgen =
        TensorAccessor(local_weights_args, local_weights_base_address, local_weights_page_size_bytes);

    cb_reserve_back(local_weights_cb_id, 1);
    const uint32_t local_weights_l1_addr = get_read_ptr(local_weights_cb_id);
    cb_push_back(local_weights_cb_id, 1);

    fill_with_val<weight_addr_t>(local_weights_l1_addr, num_cluster_experts, 0u);

    cb_wait_front(routing_weights_cb_id, 1);
    cb_wait_front(local_weights_idxs_cb_id, 1);

    const uint32_t routing_weights_addr = get_read_ptr(routing_weights_cb_id);
    const uint32_t local_weights_idxs_addr = get_read_ptr(local_weights_idxs_cb_id);
    auto local_weights_idxs_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(local_weights_idxs_addr);

    for (uint32_t i = 0; i < num_non_zero_per_device; ++i) {
        const uint32_t offset = local_weights_idxs_ptr[i] * weight_datum_size_bytes;
        tt_memmove<false, false, false, weight_datum_size_bytes>(
            local_weights_l1_addr + offset, routing_weights_addr + offset, weight_datum_size_bytes);
    }
    noc_async_write_barrier();

    const uint64_t local_weights_noc_addr = get_noc_addr(0, local_weights_addrgen);
    noc_async_write(local_weights_l1_addr, local_weights_noc_addr, local_weights_page_size_bytes);
    noc_async_write_barrier();
}
