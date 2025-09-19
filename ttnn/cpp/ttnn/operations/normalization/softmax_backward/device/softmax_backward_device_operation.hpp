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

namespace ttnn::operations::normalization {
namespace softmax_backward {

struct SoftmaxBackwardDeviceOperation {
    // Define the operation attributes. This is it to store all variables needed by operations that aren't tensors
    struct operation_attributes_t {
        const uint32_t dim;
    };

    // Define the tensor arguments. This is it to store all tensors passed in and/or out of the operation
    // Tensor arguments don't need to be just input tensors, they can be output tensors, input/output tensors, optional
    // tensors, etc.
    struct tensor_args_t {
        const Tensor& softmax_output;
        const Tensor& upstream_grad;
    };

    // Define the return types for the spec(s) of the operation
    // Can be a single ttnn::TensorSpec, std::optional<ttnn::TensorSpec>, std::vector<ttnn::TensorSpec>,
    // std::tuple<ttnn::TensorSpec> etc.
    using spec_return_value_t = ttnn::TensorSpec;

    // Define the return types for the tensor(s) of the operation
    // Can be a single Tensor, std::optional<Tensor, ...>, std::vector<Tensor>, std::tuple<Tensor, ...> etc.
    using tensor_return_value_t = Tensor;

    // Note spec_return_value_t and tensor_return_value_t should follow the same pattern
    // i.e. if spec_return_value_t is a std::vector<std::optional<ttnn::TensorSpec>> then tensor_return_value_t should
    // be std::vector<std::optional<Tensor>>
    struct SingleCore {
        // Shared variables are the variables that are shared between the create and override_runtime_arguments methods
        struct shared_variables_t {
            tt::tt_metal::KernelHandle unary_reader_kernel_id;
            tt::tt_metal::KernelHandle unary_writer_kernel_id;
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
    struct MultiCore {
        // Shared variables are the variables that are shared between the create and override_runtime_arguments methods
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
    // ttnn::prim::example(input_tensor)` after the op is registered Keep in mind that the the overload with `queue_id`
    // argument will be added automatically for primitive operations So, the user can also call this operation using
    // `tensor_return_value_t output = ttnn::prim::example(queue_id, input_tensor)`
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor, const Tensor& grad, uint32_t dim);

    // Optional methods

    // In case the operation need a custom hash function, the following method can be implemented
    /* static tt::stl::hash::hash_t compute_program_hash(
        const operation_attributes_t&, const tensor_args_t&);
    */

    // In case the operation needs a custom create_op_performance_model, this method can be implemented
    /*
    static tt::tt_metal::tt::tt_metal::operation::OpPerformanceMode±l create_op_performance_model(
        const operation_attributes_t&,
        const tensor_args_t&,
        tensor_return_value_t&);
    */
};

Tensor softmax_backward(
    const Tensor& y,     // softmax output
    const Tensor& grad,  // upstream grad dL/dy
    uint32_t dim         // reduction dimension (same as fwd)
);

}  // namespace softmax_backward
}  // namespace ttnn::operations::normalization

namespace ttnn::prim {

constexpr auto softmax_backward = ttnn::register_operation<
    "ttnn::prim::softmax_backward",
    ttnn::operations::normalization::softmax_backward::SoftmaxBackwardDeviceOperation>();

}  // namespace ttnn::prim
