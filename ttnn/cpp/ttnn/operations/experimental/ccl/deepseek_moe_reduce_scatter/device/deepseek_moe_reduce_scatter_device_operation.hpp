// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <vector>

#include "ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/deepseek_moe_reduce_scatter_device_operation_types.hpp"
#include "ttnn/operations/experimental/ccl/deepseek_moe_reduce_scatter/device/deepseek_moe_reduce_scatter_program_factory.hpp"

#include "ttnn/decorators.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/ccl/ccl_host_datastructures.hpp"

namespace ttnn::experimental::prim {

struct DeepseekMoEReduceScatterDeviceOperation {
    using operation_attributes_t = DeepseekMoEReduceScatterParams;
    using tensor_args_t = DeepseekMoEReduceScatterInputs;
    using spec_return_value_t = std::vector<ttnn::TensorSpec>;
    using tensor_return_value_t = std::vector<ttnn::Tensor>;
    using program_factory_t = std::variant<DeepseekMoEReduceScatterMeshWorkloadFactory>;
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::stl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> deepseek_moe_reduce_scatter(
    const std::vector<ttnn::Tensor>& input_tensors,
    const tt::tt_metal::MemoryConfig& output_memory_config,
    uint32_t dim,
    uint32_t num_links,
    std::optional<uint32_t> cluster_axis);

}  // namespace ttnn::prim
