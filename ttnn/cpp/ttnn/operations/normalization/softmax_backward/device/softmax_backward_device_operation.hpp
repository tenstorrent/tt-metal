// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "softmax_backward_operation_types.hpp"
#include "softmax_backward_program_factory.hpp"

#include "ttnn/decorators.hpp"

#include <tuple>
#include <variant>

namespace ttnn::operations::normalization {
namespace softmax_backward {

struct SoftmaxBackwardDeviceOperation {
    using operation_attributes_t = softmax_backward::operation_attributes_t;
    using tensor_args_t = softmax_backward::tensor_args_t;
    using spec_return_value_t = softmax_backward::spec_return_value_t;
    using tensor_return_value_t = softmax_backward::tensor_return_value_t;

    struct MultiCore {
        using shared_variables_t = SoftmaxBackwardProgramFactory::shared_variables_t;
        using cached_program_t = SoftmaxBackwardProgramFactory::cached_program_t;

        static cached_program_t create(
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            return SoftmaxBackwardProgramFactory::create(operation_attributes, tensor_args, tensor_return_value);
        }

        static void override_runtime_arguments(
            cached_program_t& cached_program,
            const operation_attributes_t& operation_attributes,
            const tensor_args_t& tensor_args,
            tensor_return_value_t& tensor_return_value) {
            SoftmaxBackwardProgramFactory::override_runtime_arguments(
                cached_program, operation_attributes, tensor_args, tensor_return_value);
        }
    };

    using program_factory_t = std::variant<MultiCore>;

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
    // ttnn::prim::softmax_backward(softmax_output, upstream_grad, dim)` after the op is registered.
    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const ttnn::Tensor& softmax_output, const ttnn::Tensor& upstream_grad, uint32_t dim);

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

ttnn::Tensor softmax_backward(
    const ttnn::Tensor& softmax_output,  // softmax output
    const ttnn::Tensor& upstream_grad,   // upstream grad dL/dy
    uint32_t dim                         // reduction dimension (same as fwd)
);

}  // namespace softmax_backward
}  // namespace ttnn::operations::normalization

namespace ttnn::prim {

constexpr auto softmax_backward = ttnn::register_operation<
    "ttnn::prim::softmax_backward",
    ttnn::operations::normalization::softmax_backward::SoftmaxBackwardDeviceOperation>();

}  // namespace ttnn::prim
