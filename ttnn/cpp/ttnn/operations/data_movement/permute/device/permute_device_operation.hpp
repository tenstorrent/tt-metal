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
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::data_movement {

struct PermuteDeviceOperation {
    struct operation_attributes_t {
        const ttsl::SmallVector<uint32_t> dims;
        const MemoryConfig output_mem_config;
        const float pad_value = 0.0f;
    };
    struct tensor_args_t {
        const Tensor& input_tensor;
        std::optional<Tensor> optional_output_tensor;
    };

    using spec_return_value_t = tt::tt_metal::TensorSpec;

    using tensor_return_value_t = Tensor;

    // Row-major tensor where the last dimension is not moved in the permutation.
    struct MultiCoreRowInvariant {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    // Row-major tensor where the last dimension is moved in the permutation.
    struct MultiCoreBlockedGeneric {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    // Tiled tensor where both tile dimensions stay in the last two positions
    // (either identity or WH swap).
    struct MultiCoreTileInvariant {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    // Tiled tensor where only one of the tile dimensions is moved out of the
    // last two positions.
    struct MultiCoreTileRowInvariant {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    // Tiled tensor where both tile dimensions are moved in the permutation.
    struct MultiCoreTiledGeneric {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
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

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, const Tensor&);
};
}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::PermuteDeviceOperation::tensor_return_value_t permute(
    const Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor,
    float pad_value = 0.0f);
}  // namespace ttnn::prim
