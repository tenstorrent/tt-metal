// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "routed_expert_ffn_device_operation_types.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

#include <variant>

namespace ttnn::prim::qsr {

// y = ((x @ w_gate) * (x @ w_up)) @ w_down on a single node: one reader, one compute and one writer thread.
// x (M, K), w_gate and w_up (K, H), w_down (H, K) and y (M, K) are bfloat16, TILE layout, DRAM interleaved.
struct RoutedExpertFfnDeviceOperation {
    using operation_attributes_t = RoutedExpertFfnParams;
    using tensor_args_t = RoutedExpertFfnInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct SingleNodeProgramFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<SingleNodeProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

ttnn::Tensor routed_expert_ffn(
    const ttnn::Tensor& x,
    const ttnn::Tensor& w_gate,
    const ttnn::Tensor& w_up,
    const ttnn::Tensor& w_down,
    const tt::tt_metal::MemoryConfig& output_mem_config);

}  // namespace ttnn::prim::qsr
