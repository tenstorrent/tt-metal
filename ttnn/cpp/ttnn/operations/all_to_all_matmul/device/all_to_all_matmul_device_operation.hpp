// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include "all_to_all_matmul_device_operation_types.hpp"
#include "all_to_all_matmul_planner.hpp"
#include "all_to_all_matmul_program_factory.hpp"

namespace ttnn::operations::all_to_all_matmul {

// Every attribute is structural: the default key (attributes + tensor specs, scratch spec included) is used.
struct AllToAllMatmulDeviceOperation {
    using operation_attributes_t = AllToAllMatmulParams;
    using tensor_args_t = AllToAllMatmulInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<AllToAllMatmulProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Support contract, plan fit and the scratch tensor's spec, once per cached program.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // The key fixes every spec; only the scratch tensor's mesh and aliasing can differ on a hit.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// SUPPORTED of ttnn/ttnn/operations/all_to_all_matmul/all_to_all_matmul.py: throws UnsupportedAxisValue.
void check_support(const AllToAllMatmulParams& attrs, const Tensor& input, const Tensor& weight);

// What solve_plan reads from the call besides the topology.
PlanInputs plan_inputs_of(const AllToAllMatmulParams& attrs, const Tensor& input, const Tensor& weight);

}  // namespace ttnn::operations::all_to_all_matmul

namespace ttnn::prim {

Tensor all_to_all_matmul(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& scratch,
    const ttnn::operations::all_to_all_matmul::AllToAllMatmulParams& params);

}  // namespace ttnn::prim
