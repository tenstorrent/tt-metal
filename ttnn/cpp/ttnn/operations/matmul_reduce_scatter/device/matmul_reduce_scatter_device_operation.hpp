// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include "matmul_reduce_scatter_device_operation_types.hpp"
#include "matmul_reduce_scatter_program_factory.hpp"

namespace ttnn::operations::matmul_reduce_scatter {

// Call order (ttnn/api/ttnn/device_operation.hpp): create_output_tensors, the program hash (the default key:
// every attribute, the solved plan included, and the tensor specs), then on a hit validate_on_program_cache_hit
// and the factory's override_runtime_arguments, on a miss validate_on_program_cache_miss and create_descriptor.
struct MatmulReduceScatterDeviceOperation {
    using operation_attributes_t = MatmulReduceScatterParams;
    using tensor_args_t = MatmulReduceScatterInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<MatmulReduceScatterProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // The support contract and the preallocated output, once per cached program.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // An equal key passed the miss check; only the preallocated output's aliasing can differ.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// SUPPORTED of ttnn/ttnn/operations/matmul_reduce_scatter (in its axis order): throws UnsupportedAxisValue.
// scatter_dim is canonical (negative); num_links is checked only when given.
void check_support(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    int32_t scatter_dim,
    tt::tt_fabric::Topology topology,
    std::optional<int32_t> num_links);

}  // namespace ttnn::operations::matmul_reduce_scatter

namespace ttnn::prim {

Tensor matmul_reduce_scatter(
    const Tensor& input_tensor,
    const Tensor& weight,
    const Tensor& staging,
    const std::optional<Tensor>& output,
    const ttnn::operations::matmul_reduce_scatter::MatmulReduceScatterParams& params);

}  // namespace ttnn::prim
