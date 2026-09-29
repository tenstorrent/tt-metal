// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tuple>
#include <variant>
#include <vector>

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

// topk_route_finish: PRIVATE finish op for ttnn.topk's large-k Blackhole routing
// composite (topk.cpp: run_topk_large_indices_route). Fuses the route's whole tail —
// gather of the ORIGINAL values by index + TILE assembly of both outputs + index
// dtype emit — into one kernel launch:
//
//   inputs:  (a) the ORIGINAL TILE bf16 logits [..., R, W], interleaved (DRAM or L1)
//            (b) ROW_MAJOR UINT32 indices [..., R, k_rounded], interleaved — the
//                topk_large_indices output (k_rounded a multiple of 16)
//   outputs: values  = TILE bf16 [..., R, k_rounded] (tile padding zero-filled)
//            indices = TILE [..., R, k_rounded], dtype UINT16 when the logits'
//                      tile-padded width fits 16 bits, else UINT32 — the same
//                      boundary as the stock device op's compute_output_specs.
//
// Values are gathered element-wise from the TILE-layout source (no untilize of the
// original) and copied BIT-EXACT — -inf lanes included; see the -inf clamp-trick
// contract in topk.cpp. It replaces the unfused to_layout(ROW_MAJOR) + chunked
// ttnn::gather + 2x to_layout(TILE) + typecast tail of the routed composite.
//
// Not registered with nanobind; called only from the routing composite via the
// ttnn::operations::reduction::topk::topk_route_finish() entry function below.

namespace ttnn::operations::reduction::topk_route_finish {

// Everything the program needs is derived from the two input tensors'
// shapes/layouts, so the op carries no attributes.
struct operation_attributes_t {};

struct tensor_args_t {
    Tensor input_tensor;    // ORIGINAL TILE bf16 logits
    Tensor indices_tensor;  // ROW_MAJOR UINT32 top-k_rounded indices
};

using tensor_return_value_t = std::tuple<Tensor, Tensor>;  // {values, indices}
using spec_return_value_t = std::tuple<tt::tt_metal::TensorSpec, tt::tt_metal::TensorSpec>;

namespace program {

struct TopkRouteFinishSharedVariables {
    tt::tt_metal::KernelHandle reader_kernel_id{};
    tt::tt_metal::KernelHandle writer_kernel_id{};
    std::vector<CoreCoord> cores{};
};

struct TopkRouteFinishProgramFactory {
    using shared_variables_t = TopkRouteFinishSharedVariables;
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

struct TopkRouteFinishDeviceOperation {
    using operation_attributes_t = topk_route_finish::operation_attributes_t;
    using tensor_args_t = topk_route_finish::tensor_args_t;
    using tensor_return_value_t = topk_route_finish::tensor_return_value_t;
    using spec_return_value_t = topk_route_finish::spec_return_value_t;

    using program_factory_t = std::variant<program::TopkRouteFinishProgramFactory>;

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

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor, const Tensor& indices_tensor);
};

}  // namespace ttnn::operations::reduction::topk_route_finish

namespace ttnn::operations::reduction::topk {

// C++-only entry point (no nanobind): fused TILE-source gather + tile assembly +
// index dtype emit. Returns {values, indices}. Lives in the ::topk namespace (next
// to the routing composite, its only caller), mirroring topk_route_prep.
std::vector<Tensor> topk_route_finish(const Tensor& input_tensor, const Tensor& indices_tensor);

}  // namespace ttnn::operations::reduction::topk
