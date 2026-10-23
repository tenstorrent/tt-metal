// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <variant>

#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/device_operation.hpp"
#include "all_gather_device_operation_types.hpp"
#include "all_gather_multicast_factory.hpp"
#include "all_gather_unicast_factory.hpp"

namespace ttnn::operations::ccl {

namespace all_gather_plan {
struct AgPlan;
}

struct AllGatherDeviceOperation {
    using operation_attributes_t = AllGatherParams;
    using tensor_args_t = AllGatherInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using program_factory_t = std::variant<AllGatherMulticastFactory, AllGatherUnicastFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t& args, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensors);

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
};

tt::tt_metal::TensorSpec compute_output_specs_helper(
    const Tensor& input_tensor,
    int32_t gather_dim_from_end,
    uint32_t num_devices,
    const std::optional<MemoryConfig>& memory_config);

// Params (machine and fabric facts included, no choice yet) and inputs of one call.
std::tuple<AllGatherParams, AllGatherInputs> all_gather_build_operation_args(
    const Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    int32_t dim,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    const std::optional<CoreRangeSet>& sub_core_grid,
    const AgForce& force);

// Plans the call (all_gather_plan.hpp) and stores the choice in params; the caller takes composite when the plan says
// so. The output spec is the persistent output's when there is one.
all_gather_plan::AgPlan plan_all_gather(AllGatherParams& params, const AllGatherInputs& inputs);

}  // namespace ttnn::operations::ccl

namespace ttnn::prim {

Tensor all_gather(
    const Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& persistent_output_tensor,
    int32_t dim,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);

}  // namespace ttnn::prim
