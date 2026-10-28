// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include "matmul_reduce_scatter_prefill_device_operation_types.hpp"
#include "matmul_reduce_scatter_prefill_program_factory.hpp"

namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill {

// The framework calls: create_output_tensors, then the cache key (the default: every attribute, the plan's inputs
// included, plus the tensor specs), then on a hit validate_on_program_cache_hit + override_runtime_arguments, on a
// miss validate_on_program_cache_miss + create_descriptor once per mesh coordinate.
struct MatmulReduceScatterPrefillDeviceOperation {
    using operation_attributes_t = MatmulReduceScatterPrefillParams;
    using tensor_args_t = MatmulReduceScatterPrefillInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<MatmulReduceScatterPrefillProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // The support contract and the operand checks, once per cached program.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // The key fixes every spec and the plan; only the preallocated output's buffer is checked again.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// The supported dtypes, layout, alignment, cluster_axis, scatter_dim, topology and num_links (TT_FATAL otherwise).
// scatter_dim is canonical; num_links is checked only when the caller gave it.
void check_support(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    int32_t scatter_dim,
    ttnn::ccl::Topology topology,
    const std::optional<int32_t>& num_links);

// The operand checks the entry makes before it reads the mesh: ranks, K, DRAM-interleaved TILE operands and an
// interleaved output memory config (TT_FATAL otherwise).
void check_operands(
    const Tensor& input_tensor, const Tensor& weight, const std::optional<tt::tt_metal::MemoryConfig>& memory_config);

// The group along cluster_axis: a 2D mesh, >= 2 devices on the axis and a scattered extent that splits into G tile
// blocks (TT_FATAL otherwise). scatter_dim is canonical. Returns G.
uint32_t check_group(const Tensor& input_tensor, const Tensor& weight, int32_t cluster_axis, int32_t scatter_dim);

}  // namespace ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill

namespace ttnn::prim {

Tensor matmul_reduce_scatter_prefill(
    const Tensor& input_tensor,
    const Tensor& weight,
    const std::optional<Tensor>& output_tensor,
    const Tensor& staging,
    const ttnn::operations::experimental::ccl::matmul_reduce_scatter_prefill::MatmulReduceScatterPrefillParams& params);

}  // namespace ttnn::prim
