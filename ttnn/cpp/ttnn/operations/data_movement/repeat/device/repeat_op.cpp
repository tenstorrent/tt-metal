// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_op.hpp"
#include "repeat_program_factory.hpp"
#include "ttnn/tensor/tensor_utils.hpp"

using namespace tt::constants;

namespace ttnn::operations::data_movement {

void RepeatDeviceOperation::validate(const std::vector<Tensor> &input_tensors) const {
    const auto &input_tensor = input_tensors[0];
    tt::tt_metal::LegacyShape input_shape = input_tensor.get_legacy_shape();
    TT_FATAL(this->repeat_dim < input_shape.rank(), "Repeat dim specified is larger than input tensor rank.");
    if (input_tensor.get_layout() == Layout::ROW_MAJOR && this->repeat_dim == input_shape.rank() - 1) {
        TT_FATAL(
            (input_shape[this->repeat_dim] * input_tensor.element_size()) % input_tensor.buffer()->alignment() == 0,
            "The last dim of tensor being repeated must be 32 byte aligned for DRAM Tensor and 16 byte aligned for L1 "
            "tensor");
    }
    TT_FATAL(this->num_repeats > 0, "Number of repeats should be greater than 0");
    TT_FATAL(input_tensor.buffer(), "Operand to repeat needs to be allocated in a buffer on device.");
    TT_FATAL(input_tensor.device(), "Operand to repeat needs to be on device.");
    TT_FATAL(input_tensor.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
             "Input to repeat must be interleaved.");
    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED,
             "Output of repeat must be interleaved.");
}

std::vector<ttnn::SimpleShape> RepeatDeviceOperation::compute_output_shapes(
    const std::vector<Tensor> &input_tensors) const {
    ttnn::SimpleShape shape_out = input_tensors[0].get_logical_shape();
    shape_out[this->repeat_dim] *= this->num_repeats;
    return {shape_out};
}

std::vector<Tensor> RepeatDeviceOperation::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const Tensor &ref_in_tensor = input_tensors[0];

    return operation::generic_create_output_tensors(
        *this, input_tensors, ref_in_tensor.get_dtype(), ref_in_tensor.get_layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks RepeatDeviceOperation::create_program(const std::vector<Tensor> &input_tensors,
                                                                      std::vector<Tensor> &output_tensors) const {
    return detail::repeat_multi_core(input_tensors[0], this->repeat_dim, this->num_repeats, output_tensors[0]);
}

}  // namespace ttnn::operations::data_movement
