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

namespace ttnn::operations::examples {

struct ExampleMultipleReturnDeviceOperation {
    // Define the operation attributes. This is it to store all variables needed by operations that aren't tensors
    struct operation_attributes_t {
        bool attribute = true;
        int some_other_attribute = 42;
        uint32_t return_output1 = true;
        uint32_t return_output2 = true;
    };

    // Define the tensor arguments. This is it to store all tensors passed in and/or out of the operation
    // Tensor arguments don't need to be just input tensors, they can be output tensors, input/output tensors, optional
    // tensors, etc.
    struct tensor_args_t {
        // This example will use a tensor that can only be used as an input
        const Tensor& input_tensor;

        // However, the following examples show what else can be done with tensor_args_t

        // An example of the tensor that can be used for input/output or just for pre-allocated output
        // Tensor& io_tensor;

        // An example of an optional tensor
        // std::optional<Tensor> optional_output_tensor;

        // An example of a vector of tensors
        // std::vector<Tensor> vector_of_tensors;

        // An example of a tuple of tensors
        // std::tuple<Tensor, ...> tuple_of_tensors;

        // An example of a vector of optional tensors
        // std::vector<std::optional<Tensor>> vector_of_optional_tensors;

        // An example of a tuple of tensors
        // std::tuple<std::vector<std::optional<Tensor>>, std::optional<Tensor>> some_crazy_tuple_of_tensors;
    };

    // Define the return types for the shape(s) of the operation
    // Can be a single ttnn::Shape, std::optional<ttnn::Shape>, std::vector<ttnn::Shape>, std::tuple<ttnn::Shape> etc.
    using shape_return_value_t = std::tuple<std::optional<ttnn::Shape>, std::optional<ttnn::Shape>>;

    // Define the return types for the tensor(s) of the operation
    // Can be a single Tensor, std::optional<Tensor, ...>, std::vector<Tensor>, std::tuple<Tensor, ...> etc.
    using tensor_return_value_t = std::vector<std::optional<Tensor>>;

    // Note shape_return_value_t and tensor_return_value_t should follow the same pattern
    // i.e. if shape_return_value_t is a std::vector<std::optional<ttnn::Shape>> then tensor_return_value_t should be
    // std::vector<std::optional<Tensor>>

    struct SingleCore {
        // Shared variables are the variables that are shared between the create and override_runtime_arguments methods
        struct shared_variables_t {
            KernelHandle unary_reader_kernel_id;
            KernelHandle unary_writer_kernel_id;
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

    using program_factory_t = std::variant<SingleCore>;

    // Mandatory methods

    // Select the program factory based on the operation attributes and tensor args
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it creates a program. Usually will have more checks
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    // Validate the operation when it reuses a program. Usually will have less checks
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    // Compute the output shapes based on the operation attributes and tensor args
    static shape_return_value_t compute_output_shapes(const operation_attributes_t&, const tensor_args_t&);

    // Create the output tensors based on the operation attributes and tensor args
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    // API call to map user arguments to operation attributes and tensor args.
    // This is the only method that is called by the user
    // The user will be able to call the operation using `tensor_return_value_t output = ttnn::prim::example(input_tensor)` after the op is registered
    // Keep in mind that the the overload with `queue_id` argument will be added automatically for primitive operations
    // So, the user can also call this operation using `tensor_return_value_t output = ttnn::prim::example(queue_id, input_tensor)`
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(const Tensor& input_tensor, bool return_output1, bool return_output2);

    // Optional methods

    // In case the operation need a custom hash function, the following method can be implemented
    /* static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t&, const tensor_args_t&);
    */

    // In case the operation needs a custom create_op_performance_model, this method can be implemented
    /*
    static operation::OpPerformanceModel create_op_performance_model(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&);
    */
};

}  // namespace ttnn::operations::examples

// Register the operation with the ttnn::register_operation API to make it available to the user as ttnn::prim::example
namespace ttnn::prim {
constexpr auto example_multiple_return = ttnn::register_operation<
    "ttnn::prim::example_multiple_return",
    ttnn::operations::examples::ExampleMultipleReturnDeviceOperation>();
}  // namespace ttnn::prim
