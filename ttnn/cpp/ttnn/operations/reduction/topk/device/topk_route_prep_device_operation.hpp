// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

// topk_route_prep: PRIVATE prep op for ttnn.topk's large-k Blackhole routing composite
// (topk.cpp: run_topk_large_indices_route). Fuses the route's first two stages —
// untilize (TILE -> ROW_MAJOR) and the -inf clamp — into one kernel launch:
//
//   input:  TILE bf16 interleaved [..., R, W] (DRAM or L1)
//   output: ROW_MAJOR bf16 interleaved, same logical shape, with every element
//           floored at the lowest FINITE bf16 (bit pattern 0xFF7F,
//           -3.3895313892515355e38) — see the clamp-trick contract in topk.cpp.
//
// Not registered with nanobind; called only from the routing composite via the
// ttnn::operations::reduction::topk_route_prep() entry function below.

namespace ttnn::operations::reduction::topk_route_prep {

// Everything the program needs is derived from the input tensor's shape/layout,
// so the op carries no attributes.
struct operation_attributes_t {};

struct tensor_args_t {
    Tensor input_tensor;
};

using tensor_return_value_t = Tensor;
using spec_return_value_t = tt::tt_metal::TensorSpec;

namespace program {

struct TopkRoutePrepSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle compute_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<CoreCoord> cores{};
};

struct TopkRoutePrepProgramFactory {
    using shared_variables_t = TopkRoutePrepSharedVariables;
    using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

    static cached_program_t create(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void override_runtime_arguments(
        cached_program_t& cached_program,
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);
};

}  // namespace program

struct TopkRoutePrepDeviceOperation {
    using operation_attributes_t = topk_route_prep::operation_attributes_t;
    using tensor_args_t = topk_route_prep::tensor_args_t;
    using tensor_return_value_t = topk_route_prep::tensor_return_value_t;
    using spec_return_value_t = topk_route_prep::spec_return_value_t;

    using program_factory_t = std::variant<program::TopkRoutePrepProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(const operation_attributes_t& attrs, const tensor_args_t& tensor_args);
    static void validate_on_program_cache_hit(const operation_attributes_t& attrs, const tensor_args_t& tensor_args);
    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& attrs, const tensor_args_t& tensor_args);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(const Tensor& input_tensor);
};

}  // namespace ttnn::operations::reduction::topk_route_prep

namespace ttnn::operations::reduction::topk {

// C++-only entry point (no nanobind): fused untilize + lowest-finite-bf16 clamp.
// Lives in the ::topk namespace (next to the routing composite, its only caller);
// ::reduction itself would collide with the op's topk_route_prep namespace above.
Tensor topk_route_prep(const Tensor& input_tensor);

}  // namespace ttnn::operations::reduction::topk
