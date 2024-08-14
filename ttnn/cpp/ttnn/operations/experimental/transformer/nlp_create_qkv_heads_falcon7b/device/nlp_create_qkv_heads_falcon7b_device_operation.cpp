// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_falcon7b_device_operation.hpp"

#include "ttnn/deprecated/tt_dnn/op_library/work_split.hpp"

namespace ttnn::operations::experimental::transformer {

// Hard-coded for Falcon7B
void NlpCreateHeadsFalcon7BDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.get_dtype() == tt::tt_metal::DataType::FLOAT32 || input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT16 || input_tensor.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B, "Unsupported data format");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE);

    TT_FATAL(input_shape[2] % TILE_HEIGHT == 0);
    TT_FATAL((input_shape == tt::tt_metal::Shape({input_shape[0], 1, input_shape[2], 4672})), "Unsupported input shape");
    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED);
}

std::vector<tt::tt_metal::Shape> NlpCreateHeadsFalcon7BDeviceOperation::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    std::vector<tt::tt_metal::Shape> output_shape_vec;
    const auto& input_tensor = input_tensors.at(0);
    const auto input_shape = input_tensor.get_legacy_shape();
    output_shape_vec = {(tt::tt_metal::Shape) {input_shape[0], 71, input_shape[2], 64}, (tt::tt_metal::Shape) {input_shape[0], 1, input_shape[2], 64}, (tt::tt_metal::Shape) {input_shape[0], 1, input_shape[2], 64}};
    return output_shape_vec;
}

std::vector<Tensor> NlpCreateHeadsFalcon7BDeviceOperation::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (this->output_mem_config.is_sharded()) {
        TT_ASSERT(false);
        return {};
    } else {
        return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), Layout::TILE, this->output_mem_config);
    }
}

operation::ProgramWithCallbacks NlpCreateHeadsFalcon7BDeviceOperation::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);

    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    return multi_core_nlp_create_qkv_heads_falcon7b(input_tensor, output_tensors, compute_with_storage_grid_size);
}
}  // namespace ttnn::operations::experimental::transformer
