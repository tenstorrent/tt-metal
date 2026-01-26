// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "softmax_operation_types.hpp"
#include "softmax_program_factory_general.hpp"
#include "softmax_program_factory_general_w_small.hpp"
#include "softmax_program_factory_general_w_large.hpp"
#include "softmax_program_factory_general_h_small.hpp"
#include "softmax_program_factory_general_h_large.hpp"
#include "softmax_program_factory_general_c_large.hpp"
#include "softmax_program_factory_attention_optimized.hpp"
#include "softmax_program_factory_attention_optimized_sharded.hpp"

#include "ttnn/decorators.hpp"

#include <optional>

namespace ttnn::prim {
struct SoftmaxDeviceOperation {
    using operation_attributes_t = SoftmaxParams;
    using tensor_args_t = SoftmaxInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    using program_factory_t = std::variant<
        SoftmaxProgramFactoryGeneralWSmall,
        SoftmaxProgramFactoryGeneralWLarge,
        SoftmaxProgramFactoryGeneralHSmall,
        SoftmaxProgramFactoryGeneralHLarge,
        SoftmaxProgramFactoryGeneralCLarge,
        SoftmaxShardedProgramFactoryAttentionOptimized,
        SoftmaxProgramFactoryAttentionOptimized>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::tt_metal::operation::Hash compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, const Tensor&);
};

Tensor softmax(
    const Tensor& input_tensor,
    int8_t dim = -1,
    const tt::tt_metal::MemoryConfig& output_mem_config = {},
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool numeric_stable = true);
Tensor scale_mask_softmax(
    const Tensor& input_tensor,
    std::optional<float> scale = std::nullopt,
    const std::optional<const Tensor>& mask = std::nullopt,
    const tt::tt_metal::MemoryConfig& output_mem_config = {},
    bool is_causal_mask = false,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool numeric_stable = true);
Tensor softmax_in_place(
    Tensor& input_tensor,
    int8_t dim = -1,
    SoftmaxProgramConfig program_config = {},
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool numeric_stable = true);
Tensor scale_mask_softmax_in_place(
    Tensor& input_tensor,
    std::optional<float> scale = std::nullopt,
    const std::optional<const Tensor>& mask = std::nullopt,
    SoftmaxProgramConfig program_config = {},
    bool is_causal_mask = false,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool numeric_stable = true);
Tensor scale_causal_mask_hw_dims_softmax_in_place(
    Tensor& input_tensor,
    std::optional<float> scale = std::nullopt,
    const std::optional<const Tensor>& mask = std::nullopt,
    SoftmaxProgramConfig program_config = {},
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool numeric_stable = true);

Tensor softmax(
    SoftmaxOperationType softmax_type,
    const Tensor& input_tensor,
    int8_t dim = -1,
    const std::optional<const Tensor>& mask = std::nullopt,
    std::optional<float> scale = std::nullopt,
    bool inplace = false,
    tt::tt_metal::MemoryConfig output_mem_config = {},
    SoftmaxProgramConfig program_config = {},
    bool is_causal_mask = false,
    DeviceComputeKernelConfig compute_kernel_config = {},
    bool is_scale_causal_mask_hw_dims_softmax = false,
    bool numeric_stable = true);

}  // namespace ttnn::prim
