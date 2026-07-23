// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <variant>
#include <tt-metalium/sub_device_types.hpp>
#include "ttnn/device_operation.hpp"
#include "high_bw_all_gather_device_operation_types.hpp"
#include "high_bw_all_gather_unicast_factory.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::high_bw_all_gather {

struct HighBwAllGatherDeviceOperation {
    using operation_attributes_t = HighBwAllGatherParams;
    using tensor_args_t = HighBwAllGatherInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using program_factory_t = std::variant<HighBwAllGatherUnicastFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::high_bw_all_gather

namespace ttnn::prim {

Tensor high_bw_all_gather(
    const Tensor& input_tensor,
    const ttnn::Tensor& output_tensor,
    int32_t dim,
    uint32_t cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id = std::nullopt,
    const std::optional<CoreRangeSet>& sub_core_grid = std::nullopt);

}  // namespace ttnn::prim
