// SPDX-filecopyrighttext: © 2026 tenstorrent inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-logger/tt-logger.hpp>

#include "ttnn/tensor/types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "flip_device_operation.hpp"

namespace ttnn::operations::data_movement {

FlipDeviceOperation::program_factory_t FlipDeviceOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
    return MultiCoreRowMajor{};
}

void FlipDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    const auto& dims = operation_attributes.dims;
    const auto input_rank = input_tensor.logical_shape().rank();

    TT_FATAL(input_rank <= 5, "Flip operation supports tensors with rank up to 5, got rank {}", input_rank);
    TT_FATAL(is_device_tensor(input_tensor), "Input tensor must be on device");

    // Validate flip dimensions are within tensor rank
    for (auto dim : dims) {
        auto normalized_dim = input_tensor.logical_shape().get_normalized_index(dim);
        TT_FATAL(
            normalized_dim < input_rank, "Flip dimension {} is out of bounds for tensor with rank {}", dim, input_rank);
    }

    // Validate no duplicate dimensions
    std::set<int64_t> unique_dims;
    for (auto dim : dims) {
        auto normalized_dim = input_tensor.logical_shape().get_normalized_index(dim);
        TT_FATAL(unique_dims.insert(normalized_dim).second, "Duplicate dimension {} in flip dimensions", dim);
    }

    TT_FATAL(tensor_args.input_tensor.is_sharded() == false, "Flip operation does not support sharded input tensor");
}

FlipDeviceOperation::spec_return_value_t FlipDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    return input_tensor.tensor_spec();
}

tt::tt_metal::operation::OpPerformanceModelGeneral<FlipDeviceOperation::tensor_return_value_t>
FlipDeviceOperation::create_op_performance_model(const tensor_args_t& inputs, const Tensor& output) {
    const auto& input_tensor = inputs.input_tensor;
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output, false, 0, true);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output}, ideal_dev_clock_cycles);
    return result;
}

FlipDeviceOperation::tensor_return_value_t FlipDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input_tensor;
    // Create output tensor with same shape as input
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), input_tensor.device());
}

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::FlipDeviceOperation::tensor_return_value_t flip(
    const Tensor& input_tensor,
    const SmallVector<uint32_t>& dims,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<Tensor> optional_output_tensor) {
    using OperationType = ttnn::operations::data_movement::FlipDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .dims = dims, .output_mem_config = memory_config.value_or(input_tensor.memory_config())},
        OperationType::tensor_args_t{
            .input_tensor = input_tensor, .optional_output_tensor = std::move(optional_output_tensor)});
}
}  // namespace ttnn::prim
