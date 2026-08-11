// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "softmax_operation_types.hpp"

#include <optional>
#include <variant>
#include <tt-metalium/program_descriptors.hpp>
#include <ttnn/metal_v2_artifacts.hpp>
#include <cstdint>
#include "ttnn/types.hpp"
#include "ttnn/operation.hpp"

namespace ttnn::prim {
struct SoftmaxDeviceOperation {
    using operation_attributes_t = SoftmaxParams;
    using tensor_args_t = SoftmaxInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    //
    // General-purpose softmax with arbitrary dimension support
    //
    // Ported to Metal 2.0 (MetalV2FactoryConcept). See device/softmax_program_factory_general_w_small.cpp.
    struct SoftmaxProgramFactoryGeneralWSmall {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };
    // Ported to Metal 2.0 (MetalV2FactoryConcept). See device/softmax_program_factory_general_w_large.cpp.
    // NOTE: the shared w_large compute kernel must be built at -O3 in the fp32 path (the factory sets the
    // compute KernelSpec opt_level to O3, matching legacy). At -O2 the LLK addrmod SETC16 asm immediate
    // fails to fold ("impossible constraint in 'asm'"); at O3 no source workaround is needed.
    struct SoftmaxProgramFactoryGeneralWLarge {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };
    // Ported to Metal 2.0 (MetalV2FactoryConcept). See device/softmax_program_factory_general_h_small.cpp.
    struct SoftmaxProgramFactoryGeneralHSmall {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };
    // Ported to Metal 2.0 (MetalV2FactoryConcept). See device/softmax_program_factory_general_h_large.cpp.
    struct SoftmaxProgramFactoryGeneralHLarge {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };
    // Ported to Metal 2.0 (MetalV2FactoryConcept). See device/softmax_program_factory_general_c_large.cpp.
    struct SoftmaxProgramFactoryGeneralCLarge {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };
    //
    // Optimized for transformer attention patterns
    //
    // Sharded memory
    // Ported to Metal 2.0 (MetalV2FactoryConcept). See device/softmax_program_factory_attention_optimized_sharded.cpp.
    struct SoftmaxShardedProgramFactoryAttentionOptimized {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
    };
    // Interleaved memory
    // Ported to Metal 2.0 (MetalV2FactoryConcept). See device/softmax_program_factory_attention_optimized.cpp.
    struct SoftmaxProgramFactoryAttentionOptimized {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
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
    std::int8_t dim = -1,
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
    std::int8_t dim = -1,
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
    std::int8_t dim = -1,
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
