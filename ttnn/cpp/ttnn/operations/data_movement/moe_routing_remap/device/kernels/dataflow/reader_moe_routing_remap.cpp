// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "api/numeric/bfloat16.h"
#include "api/dataflow/noc.h"
#include "api/dataflow/circular_buffer.h"
#include "api/tensor/noc_traits.h"

#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

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

    const auto routing_weights_addrgen = TensorAccessor(routing_weights_args, routing_weights_base_address);

    Noc noc;
    CircularBuffer routing_weights_cb(routing_weights_cb_id);
    CircularBuffer local_weights_idxs_cb(local_weights_idxs_cb_id);

    routing_weights_cb.reserve_back(1);
    const uint32_t routing_weights_l1_addr = routing_weights_cb.get_write_ptr();
    noc.async_read(
        routing_weights_addrgen,
        routing_weights_cb,
        routing_weights_page_size_bytes,
        {.page_id = 0, .offset_bytes = 0},
        {.offset_bytes = 0});

    local_weights_idxs_cb.reserve_back(1);
    const uint32_t local_weights_idxs_l1_addr = local_weights_idxs_cb.get_write_ptr();

    auto routing_weights_ptr = reinterpret_cast<volatile tt_l1_ptr weight_addr_t*>(routing_weights_l1_addr);
    auto local_weight_idx_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(local_weights_idxs_l1_addr);

    noc.async_read_barrier();
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
    routing_weights_cb.push_back(1);
    local_weights_idxs_cb.push_back(1);
}
