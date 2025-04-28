// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"
#include <tt_stl/span.hpp>

namespace ttnn::operations::data_movement {

struct PermuteDeviceOperation {
    struct operation_attributes_t {
        const SmallVector<uint32_t> dims;
        const MemoryConfig output_mem_config;
        const std::optional<float> pad_value;
    };
    struct tensor_args_t {
        const Tensor& input_tensor;
        std::optional<Tensor> optional_output_tensor;
    };

    using spec_return_value_t = ttnn::TensorSpec;

    using tensor_return_value_t = Tensor;

    // Implementation for a row major tensor where the row dimension is not moved in the permutation
    struct MultiCoreRowInvariant {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle unary_reader_kernel_id;
            tt::tt_metal::KernelHandle unary_writer_kernel_id;
            tt::tt_metal::CoreRangeSet core_range;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    // Implementation for a row major tensor where the row dimension is moved in the permutation
    struct MultiCoreBlockedGeneric {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle unary_reader_kernel_id;
            tt::tt_metal::KernelHandle unary_writer_kernel_id;
            tt::tt_metal::KernelHandle compute_kernel_id;
            CoreRangeSet core_range;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    // Implementation for when the tile is not broken apart (either dims = {..., rank - 2, rank - 1} or {..., rank - 1,
    // rank - 2})
    struct MultiCoreTileInvariant {
        // Shared variables are the variables that are shared between the create and override_runtime_arguments methods
        struct shared_variables_t {
            tt::tt_metal::KernelHandle unary_reader_kernel_id;
            tt::tt_metal::KernelHandle unary_writer_kernel_id;
            tt::tt_metal::KernelHandle compute_kernel_id;
            CoreRangeSet core_range;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    // Implemention for when only one of the height dimension (rank - 2) and the width dimension is swapped with another
    // dimension (dims = {..., rank - 2,
    // ..., i, rank - 1})
    struct MultiCoreTileRowInvariant {
        // Shared variables are the variables that are shared between the create and override_runtime_arguments methods
        struct shared_variables_t {
            tt::tt_metal::KernelHandle unary_reader_kernel_id;
            tt::tt_metal::KernelHandle unary_writer_kernel_id;
            tt::tt_metal::KernelHandle compute_kernel_id;
            CoreRangeSet core_range;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);
    };

    // Implementation for when both the height and width dimension is swapped around in the permutation
    struct MultiCoreTiledGeneric {
        // Shared variables are the variables that are shared between the create and override_runtime_arguments methods
        struct shared_variables_t {
            tt::tt_metal::KernelHandle unary_reader_kernel_id;
            tt::tt_metal::KernelHandle unary_writer_kernel_id;
            tt::tt_metal::KernelHandle compute_kernel_id;
            CoreRangeSet core_range;
        };
        using cached_program_t = ttnn::device_operation::CachedProgram<shared_variables_t>;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value);

        static void override_runtime_arguments(
            cached_program_t& cached_program,
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

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it creates a program.
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Empty as there doesn't seem to be any complicated hashing requirement
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Compute the output shapes based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // API call to map user arguments to operation attributes and tensor args.
    // This is the only method that is called by the user
    // The user will be able to call the operation using `tensor_return_value_t output =
    // ttnn::prim::example(input_tensor)` after the op is registered Keep in mind that the the overload with `queue_id`
    // argument will be added automatically for primitive operations So, the user can also call this operation using
    // `tensor_return_value_t output = ttnn::prim::example(queue_id, input_tensor)`
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const SmallVector<uint32_t>& dims,
        const std::optional<MemoryConfig>& memory_config,
        std::optional<Tensor> optional_output_tensor,
        const std::optional<float>& pad_value = std::nullopt);
};
}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
// Register the operation with the ttnn::register_operation API to make it available to the user as ttnn::prim::example
constexpr auto permute =
    ttnn::register_operation<"ttnn::prim::permute", ttnn::operations::data_movement::PermuteDeviceOperation>();
}  // namespace ttnn::prim
