// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "softmax_operation_types.hpp"

#include <optional>
#include <variant>
#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::prim {
struct SoftmaxDeviceOperation {
    using operation_attributes_t = SoftmaxParams;
    using tensor_args_t = SoftmaxInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    //
    // General-purpose softmax with arbitrary dimension support
    //
    struct SoftmaxProgramFactoryGeneralWSmall {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };
    struct SoftmaxProgramFactoryGeneralWLarge {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };
    struct SoftmaxProgramFactoryGeneralHSmall {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };
    struct SoftmaxProgramFactoryGeneralHLarge {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };
    struct SoftmaxProgramFactoryGeneralCLarge {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };
    //
    // Optimized for transformer attention patterns
    //
    // Sharded memory
    struct SoftmaxShardedProgramFactoryAttentionOptimized {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };
    // Interleaved memory
    struct SoftmaxProgramFactoryAttentionOptimized {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };

    using program_factory_t = std::variant<
        SoftmaxProgramFactoryGeneralWSmall,
        SoftmaxProgramFactoryGeneralWLarge,
        SoftmaxProgramFactoryGeneralHSmall,
        SoftmaxProgramFactoryGeneralHLarge,
        SoftmaxProgramFactoryGeneralCLarge,
        SoftmaxShardedProgramFactoryAttentionOptimized,
        SoftmaxProgramFactoryAttentionOptimized>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

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
