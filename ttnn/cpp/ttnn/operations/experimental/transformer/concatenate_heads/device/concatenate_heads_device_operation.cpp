// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concatenate_heads_device_operation.hpp"

#include "concatenate_heads_program_factory.hpp"

namespace ttnn::operations::experimental::transformer {

void ConcatenateHeadsDeviceOperation::validate_with_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto batch_size = input_tensor.padded_shape()[0];
    // TODO: See issue #1744
    TT_FATAL(batch_size >= 7 && batch_size <= 9, "Input batch size must be between 7 to 9 for bert large TM ops!");

    TT_FATAL(input_tensor.storage_type() == tt::tt_metal::StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");

    TT_FATAL((input_tensor.padded_shape() == ttnn::Shape({batch_size, 16, 384, 64})), "Unsupported input shape");

    TT_FATAL(output_tensors.size() == 1, "Must have 1 output tensors");
    const auto& optional_output_tensor = output_tensors.at(0);
    if (optional_output_tensor.has_value()) {
        TT_FATAL(
            optional_output_tensor.value().dtype() == input_tensor.dtype(),
            "Output dtype must be same as input dtype!");

        TT_FATAL(
            optional_output_tensor.value().padded_shape() == ttnn::Shape({batch_size, 1, 384, 1024}),
            "Output shape must be (batch_size, 1, 384, 1024)!");
    }
}

std::vector<ttnn::TensorSpec> ConcatenateHeadsDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0)->tensor_spec()};
    }
    const auto& input_tensor = input_tensors.at(0);
    const auto batch_size = input_tensor.padded_shape()[0];
    ttnn::Shape output_shape({batch_size, 1, 384, 1024});
    return {TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout(
            input_tensor.dtype(), tt::tt_metal::PageConfig(tt::tt_metal::Layout::TILE), output_mem_config))};
}

std::vector<Tensor> ConcatenateHeadsDeviceOperation::create_output_tensors(
    const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>>& output_tensors) const {
    if (output_tensors.at(0).has_value()) {
        return {output_tensors.at(0).value()};
    }

    return {create_device_tensor(compute_output_specs(input_tensors, output_tensors)[0], input_tensors.at(0).device())};
}

tt::tt_metal::operation::ProgramWithCallbacks ConcatenateHeadsDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    auto device_compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();
    TT_FATAL(
        (this->compute_with_storage_grid_size.x <= device_compute_with_storage_grid_size.x &&
         this->compute_with_storage_grid_size.y <= device_compute_with_storage_grid_size.y),
        "Unsupported grid shape");

    return detail::concatenate_heads_multi_core(input_tensor, output_tensor, this->compute_with_storage_grid_size);
}

}  // namespace ttnn::operations::experimental::transformer
