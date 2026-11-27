// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/device_operation.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/types.hpp"
#include "grouped_matmul_device_operation_types.hpp"
#include "grouped_matmul_program_factory.hpp"

namespace ttnn::operations::grouped_matmul {

// y[g] = x[g] @ w[g]. Every attribute is structural; the key is every attribute plus x's and w's specs
// (compute_program_hash). A hit only rewrites the addresses (the factory's override_runtime_arguments).
struct GroupedMatmulDeviceOperation {
    using operation_attributes_t = GroupedMatmulParams;
    using tensor_args_t = GroupedMatmulInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<GroupedMatmulProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // The planner's asserts (a layout exists, depth / w_inflight / x_ahead bounds, DEST fit).
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    // The key fixes everything the miss validator checks; the entry checked the per-call tensors.
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Every attribute and x's and w's specs; the caller's output_tensor is left out (see the definition).
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::grouped_matmul

namespace ttnn::prim {

Tensor grouped_matmul(
    const Tensor& x,
    const Tensor& w,
    const std::optional<Tensor>& output,
    const ttnn::operations::grouped_matmul::GroupedMatmulParams& params);

}  // namespace ttnn::prim
