// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/operations/ccl/all_broadcast/device/all_broadcast_device_operation_types.hpp"
#include "ttnn/operations/ccl/all_broadcast/device/all_broadcast_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/types.hpp"
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/sub_device.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>

#include <optional>
#include <variant>
#include <vector>

namespace ttnn::operations::ccl::all_broadcast {

struct AllBroadcastDeviceOperation {
    using operation_attributes_t = all_broadcast::operation_attributes_t;
    using tensor_args_t = all_broadcast::tensor_args_t;
    using spec_return_value_t = all_broadcast::spec_return_value_t;
    using tensor_return_value_t = all_broadcast::tensor_return_value_t;
    using program_factory_t = std::variant<program::AllBroadcastProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& input_tensor,
        std::optional<uint32_t> cluster_axis,
        const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id,
        const ttnn::MemoryConfig& output_mem_config,
        uint32_t num_links,
        tt::tt_fabric::Topology topology);
};

}  // namespace ttnn::operations::ccl::all_broadcast

namespace ttnn::prim {
constexpr auto all_broadcast = ttnn::register_operation<
    "ttnn::prim::all_broadcast",
    ttnn::operations::ccl::all_broadcast::AllBroadcastDeviceOperation>();
}  // namespace ttnn::prim
