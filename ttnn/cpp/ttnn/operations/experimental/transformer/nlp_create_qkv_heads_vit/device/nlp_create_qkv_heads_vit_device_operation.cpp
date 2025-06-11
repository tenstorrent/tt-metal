// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_vit_device_operation.hpp"

#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::transformer {

// Hard-coded for Vit
void NlpCreateHeadsVitDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.padded_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor.dtype() == tt::tt_metal::DataType::FLOAT32 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor.dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format");
    TT_FATAL(input_tensor.layout() == Layout::TILE, "Error");

    TT_FATAL(input_shape[2] % tt::constants::TILE_HEIGHT == 0, "Error");
    TT_FATAL((input_shape == ttnn::Shape({input_shape[0], 1, input_shape[2], 2304})), "Unsupported input shape");
    TT_FATAL(this->output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
}

std::vector<ttnn::TensorSpec> NlpCreateHeadsVitDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    if (this->output_mem_config.is_sharded()) {
        TT_ASSERT(false);
        return {};
    }
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.padded_shape();
    TensorSpec spec(
        Shape({input_shape[0], 12, input_shape[2], 64}),
        tt::tt_metal::TensorLayout(input_tensor.dtype(), tt::tt_metal::PageConfig(Layout::TILE), output_mem_config));
    return {spec, spec, spec};
}

tt::tt_metal::operation::ProgramWithCallbacks NlpCreateHeadsVitDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    return multi_core_nlp_create_qkv_heads_vit(input_tensor, output_tensors, compute_with_storage_grid_size);
}
}  // namespace ttnn::operations::experimental::transformer
