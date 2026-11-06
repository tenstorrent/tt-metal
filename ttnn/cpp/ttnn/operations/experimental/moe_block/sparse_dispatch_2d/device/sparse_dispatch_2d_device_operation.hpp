// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include "sparse_dispatch_2d_device_operation_types.hpp"
#include "sparse_dispatch_2d_program_factory.hpp"
#include "ttnn/device_operation.hpp"

namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d {

struct SparseDispatch2dDeviceOperation {
    using operation_attributes_t = SparseDispatch2dParams;
    using tensor_args_t = SparseDispatch2dInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using topology_return_value_t = std::vector<tt::tt_metal::TensorTopology>;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<SparseDispatch2dProgramFactory>;

    // Structure only: plan_key, the hashed attributes and the tensor specs. No buffer address.
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static topology_return_value_t compute_output_topologies(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::moe_block::sparse_dispatch_2d

namespace ttnn::prim {

// xbuf is filled in place and returned.
Tensor moe_block_sparse_dispatch_2d(
    const Tensor& xbuf,
    const Tensor& route,
    const Tensor& sems,
    const ttnn::operations::experimental::moe_block::sparse_dispatch_2d::SparseDispatch2dParams& params,
    const std::optional<Tensor>& arena = std::nullopt);

}  // namespace ttnn::prim
