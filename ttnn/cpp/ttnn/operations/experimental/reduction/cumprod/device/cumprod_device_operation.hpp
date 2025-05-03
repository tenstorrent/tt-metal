// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/core.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::reduction {

using tt::tt_metal::MemoryConfig;
using tt::tt_metal::Tensor;
using tt::tt_metal::TensorLayout;
using tt::tt_metal::TensorSpec;

struct CumprodDeviceOperation {
    struct operation_attributes_t {
        const int32_t dim;
        const tt::tt_metal::MemoryConfig output_memory_config;
    };

    struct tensor_args_t {
        const Tensor& input_tensor;
        std::optional<Tensor> optional_out{std::nullopt};
    };

    using spec_return_value_t = ttnn::TensorSpec;

    using tensor_return_value_t = Tensor;
    struct CumprodProgramFactory {
        struct shared_variables_t {
            tt::tt_metal::KernelHandle unary_reader_kernel_id;
            tt::tt_metal::KernelHandle unary_writer_kernel_id;
            std::size_t num_cores;
            std::size_t num_cores_y;
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

    using program_factory_t = std::variant<CumprodProgramFactory>;

    using invocation_result_t = std::tuple<operation_attributes_t, tensor_args_t>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it creates a program. Usually will have more checks
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it reuses a program. Usually will have less checks
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Compute the output specs based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static invocation_result_t invoke(
        const Tensor& input_tensor,
        const int32_t dim,
        std::optional<Tensor> optional_out,
        const MemoryConfig& memory_confi,
        const QueueId& queue_id = DefaultQueueId);
};

}  // namespace ttnn::operations::experimental::reduction

// Register the operation with the ttnn::register_operation API to make it available to the user as ttnn::prim::cumprod
namespace ttnn::prim {
constexpr auto cumprod = ttnn::
    register_operation<"ttnn::prim::cumprod", ttnn::operations::experimental::reduction::CumprodDeviceOperation>();
}  // namespace ttnn::prim
