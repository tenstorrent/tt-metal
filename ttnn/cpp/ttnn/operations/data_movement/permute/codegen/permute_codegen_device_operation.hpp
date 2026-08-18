// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include <tt_stl/span.hpp>
#include "ttnn/operation.hpp"
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::operations::data_movement {

// Row-major-only codegen port of permute; see codegen/permute_codegen_supported.hpp for the
// dispatch predicate deciding whether a given call lands here vs. the native prim.
struct PermuteCodegenDeviceOperation {
    static constexpr uint32_t kMaxDims = 8;

    struct operation_attributes_t {
        const uint32_t rank;
        const std::array<uint32_t, kMaxDims> dims;
        const std::array<uint32_t, kMaxDims> input_shape;
        const std::array<uint32_t, kMaxDims> output_strides;
        const uint32_t num_rows;
        const uint32_t elem_size;
        const uint32_t num_blocks_total;
        const MemoryConfig output_mem_config;
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

    using program_factory_t = std::variant<MultiCoreRowInvariant, MultiCoreBlockedGeneric>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};
}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::PermuteCodegenDeviceOperation::tensor_return_value_t permute_codegen(
    const Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor = std::nullopt);
}  // namespace ttnn::prim
