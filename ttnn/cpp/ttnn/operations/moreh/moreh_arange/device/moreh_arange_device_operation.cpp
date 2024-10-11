// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_arange_device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
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

    const auto& dtype = operation_attributes.dtype;
    TT_FATAL(dtype != DataType::BFLOAT8_B, "moreh arange not support bfloat8_b dtype.");
    TT_FATAL(dtype != DataType::UINT32, "moreh arange not support uint32 dtype.");

    const auto& output = tensor_args.output;
    if (!output.has_value())
        return;
    TT_FATAL(output->buffer() != nullptr, "Must have 1 output tensor.");
    TT_FATAL(
        dtype == output->get_dtype(), "If output is provided as input, its dtype should match the dtype parameter.");
    TT_FATAL(output->memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Error");

    auto output_layout = output->get_layout();
    if (operation_attributes.untilize_out) {
        TT_FATAL(
            output_layout == Layout::ROW_MAJOR, "Error: output_layout must be Layout::ROW_MAJOR when untilize_out");
    } else {
        TT_FATAL(output_layout == Layout::TILE, "Error: output_layout must be Layout::TILE when !untilize_out");
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
        return ttnn::Shape(tt::tt_metal::LegacyShape({num_elems}));

    std::vector<uint32_t> output_size = {
        tt::constants::TILE_HEIGHT, tt::round_up(num_elems, tt::constants::TILE_WIDTH)};

    auto dimensions_pads = std::vector<Padding::PadDimension>();
    dimensions_pads.push_back(Padding::PadDimension{.front = 0, .back = 31});
    dimensions_pads.push_back(
        Padding::PadDimension{.front = 0, .back = tt::round_up(num_elems, tt::constants::TILE_WIDTH) - num_elems});
    const auto padding = Padding(dimensions_pads, Padding::PadValue::Any);

    return ttnn::Shape{tt::tt_metal::LegacyShape(output_size, padding)};
};

MorehArangeOperation::tensor_return_value_t MorehArangeOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& output = tensor_args.output;
    if (output.has_value())
        return output.value();
    return create_device_tensor(
        compute_output_shapes(operation_attributes, tensor_args),
        operation_attributes.dtype,
        operation_attributes.untilize_out ? Layout::ROW_MAJOR : Layout::TILE,
        tensor_args.any.device(),
        operation_attributes.memory_config);
}

std::tuple<MorehArangeOperation::operation_attributes_t, MorehArangeOperation::tensor_args_t>
MorehArangeOperation::invoke(
    float start,
    float end,
    float step,
    const Tensor& any,
    const std::optional<Tensor>& output,
    bool untilize_out,
    const std::optional<DataType>& dtype,
    const std::optional<MemoryConfig>& memory_config) {
    return {
        operation_attributes_t{
            start,
            end,
            step,
            untilize_out,
            dtype.value_or(any.get_dtype()),
            memory_config.value_or(any.memory_config()),
        },
        tensor_args_t{
            any,
            output,
        },
    };
}
}  // namespace ttnn::operations::moreh::moreh_arange
