// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/sampling/device/sampling_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"

#include <optional>

#include "ttnn/operations/reduction/sampling/device/sampling_device_operation_types.hpp"
#include "ttnn/operations/reduction/sampling/device/sampling_program_factory.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::reduction::sampling {

SamplingDeviceOperation::program_factory_t SamplingDeviceOperation::select_program_factory(
    const operation_attributes_t&, const tensor_args_t&) {
    return program::SamplingProgramFactory{};
}

void SamplingDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void SamplingDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_values_tensor = tensor_args.input_values;
    const auto& input_indices_tensor = tensor_args.input_indices;
    const auto& k = tensor_args.k;
    const auto& p = tensor_args.p;
    const auto& temp = tensor_args.temp;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;

    TT_FATAL(
        input_values_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only INTERLEAVED memory layout is supported for inputs!");

    TT_FATAL(input_values_tensor.dtype() == DataType::BFLOAT16, "Only BFLOAT16 is supported for inputs!");
    TT_FATAL(input_values_tensor.layout() == Layout::TILE, "Only TILE_LAYOUT is supported for inputs!");

    TT_FATAL(
        input_indices_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only INTERLEAVED memory layout is supported for inputs!");

    TT_FATAL(
        input_indices_tensor.dtype() == DataType::UINT32 || input_indices_tensor.dtype() == DataType::INT32,
        "Only UINT32 & INT32 dtypes are supported for input indices!");

    TT_FATAL(input_indices_tensor.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR is supported for input indices!");

    TT_FATAL(
        input_indices_tensor.logical_shape() == input_values_tensor.logical_shape(),
        "Input values and indices must have the same shape!");
    auto input_shape = input_values_tensor.logical_shape();
    TT_FATAL(input_shape[0] * input_shape[1] * input_shape[2] == 32, "Input must have 32 users!");
    TT_FATAL(input_shape[3] % 32 == 0, "Input inner dim ({}) must be divisible by 32, pad if needed!", input_shape[3]);

    if (args.sub_core_grids.has_value()) {
        TT_FATAL(
            args.sub_core_grids.value().num_cores() == input_shape[0] * input_shape[1] * input_shape[2],
            "Subcore grid expects num_users cores, but found {}!",
            args.sub_core_grids.value().num_cores());
    }
    if (preallocated_output_tensor.has_value()) {
        TT_FATAL(
            preallocated_output_tensor.value().dtype() == DataType::UINT32 ||
                preallocated_output_tensor.value().dtype() == DataType::INT32,
            "Only UINT32 & INT32 dtypes are supported for outputs!");

        TT_FATAL(
            preallocated_output_tensor.value().memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Only INTERLEAVED memory layout is supported for outputs!");
    }

    // Check size, layout and dtype of k, p, temp
    TT_FATAL(k.dtype() == DataType::UINT32, "Only UINT32 dtypes are supported for k!");
    TT_FATAL(p.dtype() == DataType::BFLOAT16, "Only BFLOAT16 dtypes are supported for p!");
    TT_FATAL(temp.dtype() == DataType::BFLOAT16, "Only BFLOAT16 dtypes are supported for temp!");
    TT_FATAL(k.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for k!");
    TT_FATAL(p.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for p!");
    TT_FATAL(temp.layout() == Layout::ROW_MAJOR, "Only ROW_MAJOR layout is supported for temp!");
    TT_FATAL(k.logical_shape() == Shape({32}), "k must have shape [32]!");
    TT_FATAL(p.logical_shape() == Shape({32}), "p must have shape [32]!");
    TT_FATAL(temp.logical_shape() == Shape({32}), "temp must have shape [32]!");
}

TensorSpec SamplingDeviceOperation::compute_output_specs(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto& input_values_tensor = tensor_args.input_values;
    auto input_shape = input_values_tensor.logical_shape();
    ttnn::Shape output_shape({1, 1, 1, input_shape[2]});

    return TensorSpec(
        output_shape,
        TensorLayout(DataType::UINT32, PageConfig(Layout::ROW_MAJOR), input_values_tensor.memory_config()));
}

Tensor SamplingDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }

    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_values.device());
}

}  // namespace ttnn::operations::reduction::sampling

namespace ttnn::prim {
ttnn::Tensor sampling(
    const Tensor& input_values_tensor,
    const Tensor& input_indices_tensor,
    const Tensor& k,
    const Tensor& p,
    const Tensor& temp,
    const std::optional<uint32_t>& seed,
    const std::optional<tt::tt_metal::CoreRangeSet>& sub_core_grids,
    const std::optional<Tensor>& preallocated_output_tensor) {
    using OperationType = ttnn::operations::reduction::sampling::SamplingDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{.seed = seed, .sub_core_grids = sub_core_grids},
        OperationType::tensor_args_t{
            .input_values = input_values_tensor,
            .input_indices = input_indices_tensor,
            .k = k,
            .p = p,
            .temp = temp,
            .preallocated_output = preallocated_output_tensor});
}
}  // namespace ttnn::prim
