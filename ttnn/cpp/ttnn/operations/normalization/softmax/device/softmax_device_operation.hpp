// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "softmax_operation_types.hpp"

#include <optional>
#include <variant>
#include <vector>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/experimental/program_descriptor_patching.hpp>
#include "ttnn/types.hpp"
#include "ttnn/distributed/types.hpp"
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

    static tt::tt_metal::operation::Hash compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, const Tensor&);

    // Opts the op into the descriptor fast-path (no create_descriptor() rebuild on a cache hit).
    // The interleaved SoftmaxProgramFactoryAttentionOptimized factory binds every per-dispatch
    // address as a Buffer* rt-arg, so it has nothing to re-apply (returns empty). The sharded
    // SoftmaxShardedProgramFactoryAttentionOptimized factory binds input/output (and sharded mask)
    // as CB `.buffer`, but bakes the mask buffer ADDRESS into reader arg index 1 (used by the
    // TensorAccessor NoC read for non-sharded masks); that address changes per dispatch and must be
    // re-applied here. All other reader args (scale, mask_start_tile_id, num_tiles_in_attn_mask) are
    // shape/attr-derived and covered by compute_program_hash.
    static std::vector<tt::tt_metal::DynamicRuntimeArg> get_dynamic_runtime_args(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&,
        const std::optional<ttnn::MeshCoordinate>& = std::nullopt);
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
