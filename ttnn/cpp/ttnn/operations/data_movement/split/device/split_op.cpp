// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "split_op.hpp"

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

#include "split_program_factory.hpp"
using namespace tt::constants;


namespace ttnn::operations::data_movement {

void SplitDeviceOperation::validate(const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    tt::tt_metal::Buffer *in0_buffer = input_tensor.buffer();

    TT_FATAL(this->dim == 3 || this->dim == 2, "Split is possible along dim 2 or 3 only");
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to TM need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to TM need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Split does not currently support sharding");
    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Split does not currently support sharding");


    TT_FATAL(input_tensor.get_legacy_shape()[0] == 1, "shape[0] must be 1 (batch 1 only)");
    TT_FATAL(input_tensor.get_legacy_shape()[this->dim] % this->num_splits == 0, "Dim being split must be evenly divisible by number of splits");
    TT_FATAL(this->dim <= input_tensor.get_legacy_shape().rank() && this->dim >= 0, "Dim being split must be from 0 to rank - 1");
    TT_FATAL(input_tensor.get_legacy_shape().rank() == 4, "Tensor needs to be rank 4");
    TT_FATAL(input_tensor.get_layout() == Layout::TILE, "Tensor needs to be in TILE Layout");

}


std::vector<tt::tt_metal::LegacyShape> SplitDeviceOperation::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor = input_tensors.at(0);
    auto input_shape_array = input_tensor.get_legacy_shape().to_array_4D();
    auto output_shape_array = input_shape_array;
    output_shape_array[this->dim] /= this->num_splits;
    tt::tt_metal::LegacyShape output_shape(output_shape_array);
    std::vector<tt::tt_metal::LegacyShape> output_shape_vector(this->num_splits, output_shape);
    return output_shape_vector;
}


std::vector<Tensor> SplitDeviceOperation::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor.get_dtype(), input_tensor.get_layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks SplitDeviceOperation::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return detail::split_last_dim_two_chunks_tiled(input_tensor, output_tensors, this->output_mem_config);
}


}
