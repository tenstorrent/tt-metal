// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_arange_device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"

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
    if (!output.has_value()) {
        return;
    }
    TT_FATAL(output->buffer() != nullptr, "Must have 1 output tensor.");
    TT_FATAL(dtype == output->dtype(), "If output is provided as input, its dtype should match the dtype parameter.");
    TT_FATAL(
        output->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Output memory config layout must be INTERLEAVED but got {}",
        output->memory_config().memory_layout());

    auto output_layout = output->layout();
    if (operation_attributes.untilize_out) {
        TT_FATAL(
            output_layout == Layout::ROW_MAJOR, "Error: output_layout must be Layout::ROW_MAJOR when untilize_out");
    } else {
        TT_FATAL(output_layout == Layout::TILE, "Error: output_layout must be Layout::TILE when !untilize_out");
    }
}

MorehArangeOperation::program_factory_t MorehArangeOperation::select_program_factory(
    const operation_attributes_t& /*operation_attributes*/, const tensor_args_t& /*tensor_args*/) {
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

MorehArangeOperation::spec_return_value_t MorehArangeOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output->tensor_spec();
    }

    uint32_t num_elems = static_cast<uint32_t>(
        std::ceil((operation_attributes.end - operation_attributes.start) / operation_attributes.step));

    if (operation_attributes.untilize_out) {
        return TensorSpec(
            Shape({num_elems}),
            TensorLayout(
                operation_attributes.dtype, PageConfig(Layout::ROW_MAJOR), operation_attributes.memory_config));
    }

    return TensorSpec(
        Shape({1, num_elems}),
        TensorLayout(operation_attributes.dtype, PageConfig(Layout::TILE), operation_attributes.memory_config));
};

MorehArangeOperation::tensor_return_value_t MorehArangeOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.output.has_value()) {
        return tensor_args.output.value();
    }
    return create_device_tensor(
        compute_output_specs(operation_attributes, tensor_args), operation_attributes.mesh_device);
}

}  // namespace ttnn::operations::moreh::moreh_arange

namespace ttnn::prim {

ttnn::operations::moreh::moreh_arange::MorehArangeOperation::tensor_return_value_t moreh_arange(
    float start,
    float end,
    float step,
    ttnn::MeshDevice* mesh_device,
    const std::optional<Tensor>& output,
    bool untilize_out,
    const DataType& dtype,
    const MemoryConfig& memory_config) {
    using OperationType = ttnn::operations::moreh::moreh_arange::MorehArangeOperation;

    auto operation_attributes =
        OperationType::operation_attributes_t{start, end, step, untilize_out, mesh_device, dtype, memory_config};
    auto tensor_args = OperationType::tensor_args_t{output};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
