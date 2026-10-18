// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "api/dataflow/noc.h"
#include "api/dataflow/dataflow_buffer.h"
#include "api/core_local_mem.h"
#include "api/tensor/noc_traits.h"

#include "ttnn/operations/ccl/common/kernels/moe_utils.hpp"
#include "ttnn/operations/data_movement/common/kernels/common.hpp"

using namespace tt::data_movement::common;

void kernel_main() {
    constexpr uint32_t routing_weights_dfb_id = get_compile_time_arg_val(0);
    constexpr uint32_t local_weights_idxs_dfb_id = get_compile_time_arg_val(1);
    constexpr uint32_t local_weights_dfb_id = get_compile_time_arg_val(2);
    constexpr uint32_t num_cluster_experts = get_compile_time_arg_val(3);
    constexpr uint32_t num_non_zero_per_device = get_compile_time_arg_val(4);
    constexpr uint32_t weight_datum_size_bytes = get_compile_time_arg_val(5);
    constexpr uint32_t local_weights_page_size_bytes = weight_datum_size_bytes * num_cluster_experts;

    constexpr auto local_weights_args = TensorAccessorArgs<6>();
    using weight_addr_t = ByteSizeAddressType<weight_datum_size_bytes>::type;

    const auto local_weights_base_address = get_arg_val<uint32_t>(0);

    const auto local_weights_addrgen = TensorAccessor(local_weights_args, local_weights_base_address);

    Noc noc;
    DataflowBuffer local_weights_dfb(local_weights_dfb_id);
    DataflowBuffer routing_weights_dfb(routing_weights_dfb_id);
    DataflowBuffer local_weights_idxs_dfb(local_weights_idxs_dfb_id);

    local_weights_dfb.reserve_back(1);
    const uint32_t local_weights_l1_addr = local_weights_dfb.get_write_ptr();
    local_weights_dfb.push_back(1);

    fill_with_val<weight_addr_t>(local_weights_l1_addr, num_cluster_experts, 0u);

    routing_weights_dfb.wait_front(1);
    local_weights_idxs_dfb.wait_front(1);

    const uint32_t routing_weights_addr = routing_weights_dfb.get_read_ptr();
    const uint32_t local_weights_idxs_addr = local_weights_idxs_dfb.get_read_ptr();
    auto local_weights_idxs_ptr = reinterpret_cast<volatile tt_l1_ptr uint16_t*>(local_weights_idxs_addr);

    for (uint32_t i = 0; i < num_non_zero_per_device; ++i) {
        const uint32_t offset = local_weights_idxs_ptr[i] * weight_datum_size_bytes;
        tt_memmove<false, false, false, weight_datum_size_bytes>(
            noc, local_weights_l1_addr + offset, routing_weights_addr + offset, weight_datum_size_bytes);
    }
    noc.async_write_barrier();

    CoreLocalMem<uint32_t> src(local_weights_l1_addr);
    noc.async_write(
        src,
        local_weights_addrgen,
        local_weights_page_size_bytes,
        {.offset_bytes = 0},
        {.page_id = 0, .offset_bytes = 0});
    noc.async_write_barrier();
}
