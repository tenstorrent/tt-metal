// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"

#include "reduce_op_device_operation_types.hpp"
#include "tt_stl/reflection.hpp"
#include "ttnn/types.hpp"
#include "ttnn/metal_v2_artifacts.hpp"

namespace ttnn::prim {

struct ReduceDeviceOperation {
    using operation_attributes_t = ReduceParams;
    using tensor_args_t = Tensor;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    struct ReduceSingleCoreHwProgramFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        // CustomProgramSpecFactoryConcept cache-hit hook. compute_program_hash excludes both
        // scalars, so their values are not pinned by the cache key and must be re-applied here as
        // common runtime args; everything else the kernels read is shape-derived and in the hash.
        static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    struct ReduceMultiCoreHProgramFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        // CustomProgramSpecFactoryConcept cache-hit hook. compute_program_hash excludes both
        // scalars, so their values are not pinned by the cache key and must be re-applied here as
        // common runtime args; everything else the kernels read is shape-derived and in the hash.
        static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    struct ReduceMultiCoreWProgramFactory {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        // CustomProgramSpecFactoryConcept cache-hit hook. compute_program_hash excludes both
        // scalars, so their values are not pinned by the cache key and must be re-applied here as
        // common runtime args; everything else the kernels read is shape-derived and in the hash.
        static tt::tt_metal::experimental::ProgramRunArgs override_runtime_arguments(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    using program_factory_t =
        std::variant<ReduceSingleCoreHwProgramFactory, ReduceMultiCoreHProgramFactory, ReduceMultiCoreWProgramFactory>;

    static program_factory_t select_program_factory(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static void validate_on_program_cache_miss(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    // `scaler` and `post_mul_scaler` are excluded: they reach the kernels as runtime args, so every
    // value shares one program (#54180). `scaler_mode` carries the structural half.
    static ttsl::hash::hash_t compute_program_hash(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args);
};

ttnn::Tensor reduce(
    const Tensor& input_tensor,
    tt::tt_metal::ReduceOpMath reduce_math,
    tt::tt_metal::ReduceOpDim reduce_dim,
    float scaler,
    const MemoryConfig& output_mem_config,
    const std::optional<DataType>& output_dtype,
    const ttnn::DeviceComputeKernelConfig& compute_kernel_config,
    const std::optional<CoreRangeSet>& sub_core_grids,
    bool negate = false,
    float post_mul_scaler = 1.0f,
    ScalerMode scaler_mode = ScalerMode::ScalerTile,
    bool row_major_w_dense_path = false,
    bool row_major_h_dense_path = false,
    bool use_sfpu_reduce = false,
    uint32_t num_h_slices = 1,
    tt::tt_metal::Layout output_layout = tt::tt_metal::Layout::TILE);

}  // namespace ttnn::prim
