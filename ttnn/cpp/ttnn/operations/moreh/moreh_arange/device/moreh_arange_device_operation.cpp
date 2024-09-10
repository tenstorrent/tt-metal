// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_arange_device_operation.hpp"
#include "tt_dnn/op_library/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::moreh::moreh_arange {
void MorehArangeOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto step = operation_attributes.step;
    auto start = operation_attributes.start;
    auto end = operation_attributes.end;
    TT_FATAL(step != 0, "step must be nonzero.");
    TT_FATAL(
        ((step > 0) && (end >= start)) || ((step < 0) && (end <= start)),
        "Upper bound and larger bound inconsistent with step sign.");

    const auto& output_dtype = operation_attributes.output_dtype;
    TT_FATAL(output_dtype != DataType::BFLOAT8_B, "moreh arange not support bfloat8_b dtype.");
    TT_FATAL(output_dtype != DataType::UINT32, "moreh arange not support uint32 dtype.");

    const auto& output_tensor = tensor_args.output_tensor;
    if (!output_tensor.has_value())
        return;
    TT_FATAL(output_tensor->buffer() != nullptr, "Must have 1 output tensor.");
    TT_FATAL(
        output_dtype == output_tensor->get_dtype(),
        "If output_tensor is provided as input, its dtype should match the output_dtype parameter.");
    TT_FATAL(output_tensor->memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED);

    auto output_layout = output_tensor->get_layout();
    if (operation_attributes.untilize_out) {
        TT_FATAL(output_layout == Layout::ROW_MAJOR);
    } else {
        TT_FATAL(output_layout == Layout::TILE);
    }
}

MorehArangeOperation::program_factory_t MorehArangeOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ProgramFactory{};
}

void MorehArangeOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

void MorehArangeOperation::validate_on_program_cache_hit(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
};

MorehArangeOperation::shape_return_value_t MorehArangeOperation::compute_output_shapes(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    uint32_t num_elems = static_cast<uint32_t>(
        ceil((operation_attributes.end - operation_attributes.start) / operation_attributes.step));

    if (operation_attributes.untilize_out)
        return ttnn::Shape(tt::tt_metal::Shape({num_elems}));

    std::vector<uint32_t> output_size_vec = {
        tt::constants::TILE_HEIGHT, tt::round_up(num_elems, tt::constants::TILE_WIDTH)};

    auto dimensions_pads = std::vector<Padding::PadDimension>();
    dimensions_pads.push_back(Padding::PadDimension{.front = 0, .back = 31});
    dimensions_pads.push_back(
        Padding::PadDimension{.front = 0, .back = tt::round_up(num_elems, tt::constants::TILE_WIDTH) - num_elems});
    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);

    return ttnn::Shape{tt::tt_metal::Shape(output_size_vec, padding)};
};

MorehArangeOperation::tensor_return_value_t MorehArangeOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output_tensor = tensor_args.output_tensor;
    if (output_tensor.has_value())
        return {output_tensor.value()};
    return create_device_tensor(
        compute_output_shapes(operation_attributes, tensor_args),
        operation_attributes.output_dtype,
        operation_attributes.untilize_out ? Layout::ROW_MAJOR : Layout::TILE,
        tensor_args.any.device(),
        operation_attributes.output_memory_config);
}

std::tuple<MorehArangeOperation::operation_attributes_t, MorehArangeOperation::tensor_args_t>
MorehArangeOperation::invoke(
    float start,
    float end,
    float step,
    const Tensor& any,
    const std::optional<Tensor>& output_tensor,
    bool untilize_out,
    const std::optional<DataType>& output_dtype,
    const std::optional<MemoryConfig>& output_memory_config) {
    return {
        operation_attributes_t{
            start,
            end,
            step,
            untilize_out,
            output_dtype.value_or(any.get_dtype()),
            output_memory_config.value_or(any.memory_config()),
        },
        tensor_args_t{
            any,
            output_tensor,
        },
    };
}
}  // namespace ttnn::operations::moreh::moreh_arange
