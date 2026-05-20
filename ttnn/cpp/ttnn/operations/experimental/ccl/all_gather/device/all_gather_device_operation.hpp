// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <variant>

#include "all_gather_device_operation_types.hpp"
#include "all_gather_factory.hpp"

#include "ttnn/device_operation.hpp"

namespace ttnn::experimental::prim {

struct AllGatherDeviceOperation {
    using operation_attributes_t = AllGatherParams;
    using tensor_args_t = AllGatherInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<AllGatherFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors);
};

std::tuple<AllGatherParams, AllGatherInputs> all_gather_build_operation_args(
    const Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    int32_t dim,
    uint32_t num_links,
    const std::optional<MemoryConfig>& memory_config,
    tt::tt_fabric::Topology topology,
    std::optional<uint32_t> cluster_axis);

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

Tensor all_gather_experimental(
    const Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    int32_t dim,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<uint32_t> cluster_axis);

}  // namespace ttnn::prim
