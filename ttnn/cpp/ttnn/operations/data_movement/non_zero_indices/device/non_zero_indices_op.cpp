// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "non_zero_indices_op.hpp"

namespace ttnn {

namespace operations::data_movement {

void NonZeroIndices::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto input_tensor_a_shape = input_tensor_a.get_legacy_shape();
    TT_FATAL(input_tensor_a_shape[0] == 1 and
            input_tensor_a_shape[1] == 1 and
            input_tensor_a_shape[2] == 1
        , "Input should be 1D");
    TT_FATAL(input_tensor_a.get_layout() == Layout::ROW_MAJOR, "Currently only supporting row major layout");
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to Non-zero need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands to Non-zero need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Non-zero does not currently support sharding");
}

std::vector<tt::tt_metal::LegacyShape> NonZeroIndices::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    tt::tt_metal::LegacyShape num_non_zero_shape({1,1,1,8});
    return {num_non_zero_shape, input_tensor.get_legacy_shape()};
}

std::vector<Tensor> NonZeroIndices::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    return operation::generic_create_output_tensors(*this, input_tensors, DataType::UINT32, Layout::ROW_MAJOR, this->output_mem_config);
}

operation::ProgramWithCallbacks NonZeroIndices::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& out_num_indices = output_tensors.at(0);
    const auto& out_indices = output_tensors.at(1);

    return non_zero_indices_single_core(input_tensor, out_num_indices, out_indices);
}


}  // namespace operations::data_movement

}  // namespace ttnn
