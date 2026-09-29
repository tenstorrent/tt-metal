// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "qkv_causal_conv1d_silu_device_operation_types.hpp"
#include "qkv_causal_conv1d_silu_program_factory.hpp"
#include "qkv_causal_conv1d_silu_tiled_program_factory.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::experimental::prim {

struct QkvCausalConv1dSiluOperation {
    using operation_attributes_t = QkvCausalConv1dSiluParams;
    using tensor_args_t = QkvCausalConv1dSiluInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    // The input layout selects the factory: ROW_MAJOR -> QkvCausalConv1dSiluProgramFactory,
    // TILE -> QkvCausalConv1dSiluTiledProgramFactory.
    using program_factory_t = std::variant<QkvCausalConv1dSiluProgramFactory, QkvCausalConv1dSiluTiledProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};

// Returns {q, k, v}, plus new_state when return_conv_state is true (TILE input only).
std::vector<Tensor> qkv_causal_conv1d_silu(
    const Tensor& input,
    const std::optional<Tensor>& history,
    const Tensor& tap0,
    const Tensor& tap1,
    const Tensor& tap2,
    const Tensor& tap3,
    uint32_t q_width,
    uint32_t k_width,
    uint32_t v_width,
    uint32_t channel_chunk_size,
    bool return_conv_state,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool fused_qk_l2_norm = false);

}  // namespace ttnn::experimental::prim
