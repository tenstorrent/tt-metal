// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
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

// ROW_MAJOR-only codegen port of permute. Branches on whether the permutation moves the last
// (contiguous) dim: dims[-1] == rank - 1 selects the row-invariant no-compute path, otherwise the
// W-changing tilize/transpose/pack_untilize path.
struct PermuteCodegenDeviceOperation {
    static constexpr uint32_t MAX_DIMS = 8;

    struct operation_attributes_t {
        const uint32_t rank = 0;
        const std::array<uint32_t, MAX_DIMS> dims{};
        const std::array<uint32_t, MAX_DIMS> input_shape{};
        const std::array<uint32_t, MAX_DIMS> output_strides{};
        const uint32_t num_rows = 0;
        const uint32_t aligned_stick_bytes = 0;
        const uint32_t elem_size = 0;
        const uint32_t num_blocks_total = 0;
        const MemoryConfig output_mem_config;
    };
    struct tensor_args_t {
        const Tensor& input_tensor;
        std::optional<Tensor> optional_output_tensor;
    };

    using spec_return_value_t = ttnn::TensorSpec;

    using tensor_return_value_t = Tensor;

    // Row-major, row-invariant: stick reader + inverse-permutation writer, no compute.
    struct RowInvariant {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    // Row-major, W-changing: blocked reader -> tilize/transpose_tile/pack_untilize -> writer.
    struct BlockedGeneric {
        static tt::tt_metal::ProgramDescriptor create_descriptor(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    using program_factory_t = std::variant<RowInvariant, BlockedGeneric>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, const Tensor&);
};
}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::PermuteCodegenDeviceOperation::tensor_return_value_t permute_codegen(
    const Tensor& input_tensor,
    const ttsl::SmallVector<uint32_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor);
}  // namespace ttnn::prim
