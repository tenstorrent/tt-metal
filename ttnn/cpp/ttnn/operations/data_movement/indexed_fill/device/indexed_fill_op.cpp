// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_op.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

void IndexedFill::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(1);
    const auto& input_tensor_b = input_tensors.at(2);
    const auto& batch_ids = input_tensors.at(0);
    auto input_tensor_a_shape = input_tensor_a.padded_shape();
    auto input_tensor_b_shape = input_tensor_b.padded_shape();
    TT_FATAL(input_tensor_a.layout() == Layout::ROW_MAJOR, "Currently only supporting row major layout");
    TT_FATAL(input_tensor_b.layout() == input_tensor_a.layout(), "Inputs must be same layout");
    TT_FATAL(
        input_tensor_a_shape[1] == input_tensor_b_shape[1] && input_tensor_a_shape[2] == input_tensor_b_shape[2] &&
            input_tensor_a_shape[3] == input_tensor_b_shape[3],
        "Dims except batch dim must be the same on inputs");
    TT_FATAL(
        input_tensor_b_shape[0] == batch_ids.padded_shape()[-1], "Second input and batch ids must be same outer dim");
    TT_FATAL(batch_ids.layout() == Layout::ROW_MAJOR, "Batch IDs must be ROW MAJOR");
    TT_FATAL(this->dim == 0, "Currently only supporting batch dimension");
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to Index Fill need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to Index Fill need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED, "Error");
    TT_FATAL(
        input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Index Fill does not currently support sharding");
}

std::vector<ttnn::TensorSpec> IndexedFill::compute_output_specs(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(1);
    return {TensorSpec(
        input_tensor.logical_shape(),
        TensorLayout(input_tensor.dtype(), PageConfig(input_tensor.layout()), output_mem_config))};
}

operation::ProgramWithCallbacks IndexedFill::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& batch_ids = input_tensors.at(0);
    const auto& input_tensor_a = input_tensors.at(1);
    const auto& input_tensor_b = input_tensors.at(2);
    const auto& output_tensor = output_tensors.at(0);

    return indexed_fill_multi_core(batch_ids, input_tensor_a, input_tensor_b, output_tensor);
}

}  // namespace ttnn::operations::data_movement
