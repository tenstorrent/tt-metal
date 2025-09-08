// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "softmax_operation_types.hpp"
#include "softmax_program_factory.hpp"

#include "ttnn/decorators.hpp"

#include <optional>

namespace ttnn::operations::normalization {
namespace softmax {
struct SoftmaxDeviceOperation {
    using operation_attributes_t = softmax::operation_attributes_t;
    using tensor_args_t = softmax::tensor_args_t;
    using spec_return_value_t = softmax::spec_return_value_t;
    using tensor_return_value_t = softmax::tensor_return_value_t;

    using program_factory_t = std::variant<
        program::SoftmaxProgramFactoryGeneralWSmall,
        program::SoftmaxProgramFactoryGeneralWLarge,
        program::SoftmaxProgramFactoryGeneralHSmall,
        program::SoftmaxProgramFactoryGeneralHLarge,
        program::SoftmaxProgramFactoryGeneralCLarge,
        program::SoftmaxShardedProgramFactoryAttentionOptimized,
        program::SoftmaxProgramFactoryAttentionOptimized>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static tt::tt_metal::operation::Hash compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, const Tensor&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
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
        bool numeric_stable = false);
};

Tensor softmax(
    QueueId queue_id,
    const Tensor& input_tensor,
    int8_t dim = -1,
    tt::tt_metal::MemoryConfig output_mem_config = {},
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool numeric_stable = false);
Tensor scale_mask_softmax(
    QueueId queue_id,
    const Tensor& input_tensor,
    std::optional<float> scale = std::nullopt,
    const std::optional<const Tensor>& mask = std::nullopt,
    tt::tt_metal::MemoryConfig output_mem_config = {},
    bool is_causal_mask = false,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool numeric_stable = false);
Tensor softmax_in_place(
    QueueId queue_id,
    Tensor& input_tensor,
    int8_t dim = -1,
    SoftmaxProgramConfig program_config = {},
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool numeric_stable = false);
Tensor scale_mask_softmax_in_place(
    QueueId queue_id,
    Tensor& input_tensor,
    std::optional<float> scale = std::nullopt,
    const std::optional<const Tensor>& mask = std::nullopt,
    SoftmaxProgramConfig program_config = {},
    bool is_causal_mask = false,
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool numeric_stable = false);
Tensor scale_causal_mask_hw_dims_softmax_in_place(
    QueueId queue_id,
    Tensor& input_tensor,
    std::optional<float> scale = std::nullopt,
    const std::optional<const Tensor>& mask = std::nullopt,
    SoftmaxProgramConfig program_config = {},
    std::optional<const DeviceComputeKernelConfig> compute_kernel_config = std::nullopt,
    bool numeric_stable = false);
}  // namespace softmax
}  // namespace ttnn::operations::normalization

namespace ttnn::prim {

constexpr auto softmax =
    ttnn::register_operation<"ttnn::prim::softmax", ttnn::operations::normalization::softmax::SoftmaxDeviceOperation>();

}  // namespace ttnn::prim
