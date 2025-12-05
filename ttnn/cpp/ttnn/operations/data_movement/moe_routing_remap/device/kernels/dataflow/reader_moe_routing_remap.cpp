// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0
#include "dprint_pages.h"
#include "utils/bfloat16.h"

#include "ttnn/cpp/ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/cpp/ttnn/operations/data_movement/common/kernels/common.hpp"

void kernel_main() {
    constexpr uint32_t routing_weights_cb_id = get_compile_time_arg_val(0);
    constexpr uint32_t local_weights_idxs_cb_id = get_compile_time_arg_val(1);
    constexpr uint32_t num_cluster_experts = get_compile_time_arg_val(2);
    constexpr uint32_t num_non_zero_per_device = get_compile_time_arg_val(3);
    constexpr uint32_t weight_datum_size_bytes = get_compile_time_arg_val(4);

    constexpr uint32_t routing_weights_page_size_bytes = weight_datum_size_bytes * num_cluster_experts;

    constexpr auto routing_weights_args = TensorAccessorArgs<5>();

    using weight_addr_t = tt::data_movement::common::ByteSizeAddressType<weight_datum_size_bytes>::type;

    const auto routing_weights_base_address = get_arg_val<uint32_t>(0);
    const auto device_weights_count_offset = get_arg_val<uint32_t>(1);

    const auto routing_weights_addrgen =
        TensorAccessor(routing_weights_args, routing_weights_base_address, routing_weights_page_size_bytes);

    cb_reserve_back(routing_weights_cb_id, 1);
    const uint32_t routing_weights_l1_addr = get_read_ptr(routing_weights_cb_id);
    const uint64_t routing_weights_noc_addr = get_noc_addr(0, routing_weights_addrgen);
    noc_async_read(routing_weights_noc_addr, routing_weights_l1_addr, routing_weights_page_size_bytes);

    cb_reserve_back(local_weights_idxs_cb_id, 1);
    const uint32_t local_weights_idxs_l1_addr = get_write_ptr(local_weights_idxs_cb_id);

    auto routing_weights_ptr = reinterpret_cast<volatile tt_l1_ptr weight_addr_t*>(routing_weights_l1_addr);
    auto local_weight_idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(local_weights_idxs_l1_addr);

    noc_async_read_barrier();
    for (uint32_t i = 0, weight_offset_count = 0, device_weights_count = 0; i < num_cluster_experts; ++i) {
        const auto& val = routing_weights_ptr[i];
        if (val != 0) {
            if (weight_offset_count < device_weights_count_offset) {
                ++weight_offset_count;
            } else {
                local_weight_idx_ptr[device_weights_count++] = static_cast<uint16_t>(i);
            }
        }
        if (device_weights_count == num_non_zero_per_device) {
            break;
        }
    }
    cb_push_back(routing_weights_cb_id, 1);
    cb_push_back(local_weights_idxs_cb_id, 1);
}
