// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
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

namespace ttnn::operations::ax_plus_b {

struct AX_plus_B_DeviceOperation {
    // Define the operation attributes. This is it to store all variables needed by operations that aren't tensors
    struct operation_attributes_t {
        bool attribute;
        int some_other_attribute;
    };

    struct tensor_args_t {
        const Tensor& tensor_a;
        const Tensor& tensor_x;
        const Tensor& tensor_b;
        Tensor& tensor_y;
    };

    using spec_return_value_t = ttnn::TensorSpec;

    using tensor_return_value_t = Tensor;

    // Shared variables are the variables that are shared between the create and override_runtime_arguments methods
    struct shared_variables_t {
        tt::tt_metal::KernelHandle reader_kernel_id;
        tt::tt_metal::KernelHandle writer_kernel_id;
        std::size_t num_cores;
        std::size_t num_cores_y;
    };

    struct SingleCore {
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

    struct MultiCore {
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

    using program_factory_t = std::variant<SingleCore, MultiCore>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it creates a program. Usually will have more checks
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it reuses a program. Usually will have less checks
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Compute the output specs based on the operation attributes and tensor args
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // API call to map user arguments to operation attributes and tensor args.
    // This is the only method that is called by the user
    // The user will be able to call the operation using `tensor_return_value_t output =
    // ttnn::prim::example(input_tensor)` after the op is registered
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& tensor_a, const Tensor& tensor_x, const Tensor& tensor_b, Tensor& tensor_y);

    // Optional methods

    // In case the operation need a custom hash function, the following method can be implemented
    /* static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t&, const tensor_args_t&);
    */

    // In case the operation needs a custom create_op_performance_model, this method can be implemented
    /*
    static tt::tt_metal::tt::tt_metal::operation::OpPerformanceModel create_op_performance_model(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&);
    */
};

}  // namespace ttnn::operations::ax_plus_b

// Register the operation with the ttnn::register_operation API to make it available to the user as ttnn::prim::example
namespace ttnn::prim {
constexpr auto example =
    ttnn::register_operation<"ttnn::prim::ax_plus_b", ttnn::operations::ax_plus_b::AX_plus_B_DeviceOperation>();
}  // namespace ttnn::prim
