// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include <tt-metalium/experimental/fabric/fabric_edm_types.hpp>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include "matmul_reduce_scatter_device_operation_types.hpp"
#include "matmul_reduce_scatter_program_factory.hpp"

namespace ttnn::operations::matmul_reduce_scatter {

// Fused matmul + reduce-scatter (SUM) over one mesh axis. Every attribute is structural (the solved plan
// included), so the default program key (attributes plus tensor specs) is exact; the factory's
// override_runtime_arguments rewrites the buffer, staging and mailbox addresses on a hit.
struct MatmulReduceScatterDeviceOperation {
    using operation_attributes_t = MatmulReduceScatterParams;
    using tensor_args_t = MatmulReduceScatterInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<MatmulReduceScatterProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Field comparisons only (the planner and the support contract ran in the public entry), so the hit
    // path reuses it.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// SUPPORTED of ttnn/ttnn/operations/matmul_reduce_scatter/matmul_reduce_scatter.py (validate()): throws
// UnsupportedAxisValue. Called by the public entry before anything is allocated.
void check_support(
    const Tensor& input_tensor,
    const Tensor& weight,
    int32_t cluster_axis,
    int32_t scatter_dim,
    tt::tt_fabric::Topology topology,
    const std::optional<int32_t>& num_links);

}  // namespace ttnn::operations::matmul_reduce_scatter

namespace ttnn::prim {

Tensor matmul_reduce_scatter(
    const Tensor& input_tensor,
    const Tensor& weight,
    const Tensor& staging,
    const Tensor& mailbox,
    const std::optional<Tensor>& output_tensor,
    const ttnn::operations::matmul_reduce_scatter::MatmulReduceScatterParams& params);

}  // namespace ttnn::prim
