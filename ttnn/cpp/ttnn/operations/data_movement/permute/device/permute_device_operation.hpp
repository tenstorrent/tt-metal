// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include <tt_stl/span.hpp>
#include "ttnn/operation.hpp"
#include "ttnn/metal_v2_artifacts.hpp"
#include "ttnn/distributed/types.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::data_movement {

struct PermuteDeviceOperation {
    struct operation_attributes_t {
        const ttsl::SmallVector<uint32_t> dims;
        const MemoryConfig output_mem_config;
        const float pad_value = 0.0f;
        // A row-major blocked program serves every shape of the same rank and row widths, at more host time per
        // call (see permute_blocked_generic_is_shape_free).
        const bool runtime_shapes = false;
    };
    struct tensor_args_t {
        const Tensor& input_tensor;
        std::optional<Tensor> optional_output_tensor;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;

    using tensor_return_value_t = Tensor;

    // Row-major tensor where the last dimension is not moved in the permutation.
    struct MultiCoreRowInvariant {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    // Row-major tensor where the last dimension is moved in the permutation.
    struct MultiCoreBlockedGeneric {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        // Rewrites the block ranges, counts, shape and strides on a hit at another shape, and rebinds the
        // tensors.
        static tt::tt_metal::ProgramRunArgs override_runtime_arguments(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value,
            const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate = std::nullopt);
    };

    // Tiled tensor where both tile dimensions stay in the last two positions
    // (either identity or WH swap).
    struct MultiCoreTileInvariant {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    // Tiled tensor where only one of the tile dimensions is moved out of the
    // last two positions.
    struct MultiCoreTileRowInvariant {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    // Tiled tensor where both tile dimensions are moved in the permutation.
    struct MultiCoreTiledGeneric {
        static ttnn::device_operation::ProgramArtifacts create_program_artifacts(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<
        MultiCoreRowInvariant,
        MultiCoreBlockedGeneric,
        MultiCoreTileInvariant,
        MultiCoreTileRowInvariant,
        MultiCoreTiledGeneric>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, const Tensor&);
};
// Whether a MultiCoreBlockedGeneric program can serve another shape of the same rank: with runtime_shapes, an
// interleaved row-major input and output, no preallocated output, and a non-empty input. Its kernels then take every
// dimension size at run time, and only the row widths (the page sizes) and, below one block per core, the block count
// shape the program.
bool permute_blocked_generic_is_shape_free(
    const PermuteDeviceOperation::operation_attributes_t& operation_attributes,
    const PermuteDeviceOperation::tensor_args_t& tensor_args);

// The number of blocks a MultiCoreBlockedGeneric program splits over the cores.
uint32_t permute_blocked_generic_num_blocks(
    const PermuteDeviceOperation::operation_attributes_t& operation_attributes,
    const PermuteDeviceOperation::tensor_args_t& tensor_args);

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::PermuteDeviceOperation::tensor_return_value_t permute(
    const Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    float pad_value = 0.0f,
    bool runtime_shapes = false);
}  // namespace ttnn::prim
