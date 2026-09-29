// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "width_sharded_all_reduce_device_operation_types.hpp"
#include "width_sharded_all_reduce_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::prim {

struct WidthShardedAllReduceDeviceOperation {
    using operation_attributes_t = WidthShardedAllReduceParams;
    using tensor_args_t = WidthShardedAllReduceInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<WidthShardedAllReduceMeshWorkloadFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor width_sharded_all_reduce(
    const Tensor& input_tensor,
    std::optional<uint32_t> cluster_axis,
    const std::optional<tt::tt_metal::SubDeviceId>& subdevice_id,
    uint32_t num_links,
    tt::tt_fabric::Topology topology);

}  // namespace ttnn::prim
