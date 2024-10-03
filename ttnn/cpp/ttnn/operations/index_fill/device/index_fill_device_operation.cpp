// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include "index_fill_device_operation.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/deprecated/tt_dnn/op_library/moreh_helper_functions.hpp"
namespace ttnn::operations::index_fill {
IndexFillOperation::program_factory_t IndexFillOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // For now we litteraly don't care and return a single factory. Whatever
    return MultiCore{};
}
void IndexFillOperation::validate(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    // Do nothing xD
}
void IndexFillOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
}
void IndexFillOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate(operation_attributes, tensor_args);
}
IndexFillOperation::shape_return_value_t IndexFillOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return tensor_args.input.get_shape();
}
IndexFillOperation::tensor_return_value_t IndexFillOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto output_shape = compute_output_shapes(operation_attributes, tensor_args);
    const auto& input = tensor_args.input;
    return create_device_tensor(
        output_shape,
        input.tensor_attributes->dtype,
        input.tensor_attributes->layout,
        input.device(),
        operation_attributes.memory_config);
}
std::tuple<IndexFillOperation::operation_attributes_t, IndexFillOperation::tensor_args_t>
IndexFillOperation::invoke(
        const Tensor &input,
        const Tensor &batch_ids,
        const int64_t dim,
        const uint32_t value,
        const std::optional<MemoryConfig> &memory_config) {
    return {
        operation_attributes_t{dim, value, memory_config.value_or(input.memory_config())},
        tensor_args_t{input, batch_ids}
    };
}
}
