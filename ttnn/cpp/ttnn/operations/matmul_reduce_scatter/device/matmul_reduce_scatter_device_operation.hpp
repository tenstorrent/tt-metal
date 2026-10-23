// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include "matmul_reduce_scatter_device_operation_types.hpp"
#include "matmul_reduce_scatter_program_factory.hpp"

namespace ttnn::operations::matmul_reduce_scatter {

// The framework calls: create_output_tensors, then the cache key (the default: every attribute, the solved plan
// included, plus the tensor specs), then on a hit validate_on_program_cache_hit + override_runtime_arguments, on a
// miss validate_on_program_cache_miss + create_descriptor once per mesh coordinate.
struct MatmulReduceScatterDeviceOperation {
    using operation_attributes_t = MatmulReduceScatterParams;
    using tensor_args_t = MatmulReduceScatterInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<MatmulReduceScatterProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // The support contract and the operand checks, once per cached program.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // The key fixes every spec and the plan; only the preallocated output's buffer is checked again.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// SUPPORTED of matmul_reduce_scatter.py, in its axis order: throws UnsupportedAxisValue. scatter_dim is canonical;
// num_links is checked only when the caller gave it.
void check_support(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    int32_t scatter_dim,
    ttnn::ccl::Topology topology,
    const std::optional<int32_t>& num_links);

// The operand checks run() makes before it reads the mesh: ranks, K, DRAM-interleaved TILE operands
// (std::invalid_argument) and an interleaved output memory config (NotImplemented).
void check_operands(
    const Tensor& input_tensor, const Tensor& weight, const std::optional<tt::tt_metal::MemoryConfig>& memory_config);

}  // namespace ttnn::operations::matmul_reduce_scatter

namespace ttnn::prim {

Tensor matmul_reduce_scatter(
    const Tensor& input_tensor,
    const Tensor& weight,
    const std::optional<Tensor>& output_tensor,
    const ttnn::operations::matmul_reduce_scatter::MatmulReduceScatterParams& params);

}  // namespace ttnn::prim
