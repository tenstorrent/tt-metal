// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "softmax_backward_operation_types.hpp"
#include "softmax_backward_non_streaming_program_factory.hpp"
#include "softmax_backward_streaming_program_factory.hpp"

#include <variant>

namespace ttnn::operations::normalization {
namespace softmax_backward {

struct SoftmaxBackwardDeviceOperation {
    using operation_attributes_t = softmax_backward::operation_attributes_t;
    using tensor_args_t = softmax_backward::tensor_args_t;
    using spec_return_value_t = softmax_backward::spec_return_value_t;
    using tensor_return_value_t = softmax_backward::tensor_return_value_t;

    using program_factory_t = std::variant<SoftmaxBackwardNonStreamingFactory, SoftmaxBackwardStreamingFactory>;

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

    static tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> create_op_performance_model(
        const operation_attributes_t&, const tensor_args_t&, tensor_return_value_t&);
};

ttnn::Tensor softmax_backward(
    const ttnn::Tensor& softmax_output,  // softmax output
    const ttnn::Tensor& upstream_grad,   // upstream grad dL/dy
    uint32_t dim                         // reduction dimension (same as fwd)
);

}  // namespace softmax_backward
}  // namespace ttnn::operations::normalization

namespace ttnn::prim {

ttnn::operations::normalization::softmax_backward::SoftmaxBackwardDeviceOperation::tensor_return_value_t
softmax_backward(
    const ttnn::Tensor& softmax_output,  // softmax output
    const ttnn::Tensor& upstream_grad,   // upstream grad dL/dy
    uint32_t dim                         // reduction dimension (same as fwd)
);

}  // namespace ttnn::prim
