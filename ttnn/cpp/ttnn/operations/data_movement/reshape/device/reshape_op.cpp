// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_op.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"

#include "ttnn/tensor/tensor_utils.hpp"
#include "reshape_program_factory.hpp"
using namespace tt::constants;


namespace ttnn::operations::data_movement {

void ReshapeDeviceOperation::validate(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr , "Operands to reshape need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.get_dtype() == DataType::BFLOAT16);

    TT_FATAL(input_tensor_a.get_layout() == Layout::TILE || input_tensor_a.get_layout() == Layout::ROW_MAJOR, "Only tile and row major reshape supported!");

    auto output_shape = infer_dims_for_reshape(this->N, this->C, this->H, this->W, input_tensor_a.volume());
    TT_FATAL(input_tensor_a.volume() == output_shape[0] * output_shape[1] * output_shape[2] * output_shape[3], "New shape volume must match old shape volume");

    TT_FATAL(input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED, "Reshape does not currently support sharding");
    TT_FATAL(this->output_mem_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Reshape does not currently support sharding");

    if (input_tensor_a.get_layout() == Layout::TILE) {
        TT_FATAL(input_tensor_a.volume() % TILE_HW == 0);
        TT_FATAL(output_shape[2] % TILE_HEIGHT == 0 && output_shape[3] % TILE_WIDTH == 0 && "Expected a multiple of 32 for H, W (or -1 evaluating to such) for reshape!");
    } else if (input_tensor_a.get_layout() == Layout::ROW_MAJOR) {
        uint32_t ROW_MAJOR_WIDTH = 8;
        TT_FATAL(input_tensor_a.get_legacy_shape()[3] % ROW_MAJOR_WIDTH == 0 && output_shape[3] % ROW_MAJOR_WIDTH == 0, "Operand/target width must be a multiple of 8");
        uint32_t num_old_sticks = input_tensor_a.get_legacy_shape()[0] * input_tensor_a.get_legacy_shape()[1] * input_tensor_a.get_legacy_shape()[2];
        uint32_t num_new_sticks = output_shape[0] * output_shape[1] * output_shape[2];
    } else {
        TT_FATAL(false, "Unsupported layout for reshape");
    }
}


std::vector<tt::tt_metal::Shape> ReshapeDeviceOperation::compute_output_shapes(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return {tt::tt_metal::infer_dims_for_reshape(this->N, this->C, this->H, this->W, input_tensor_a.volume())};
}


std::vector<Tensor> ReshapeDeviceOperation::create_output_tensors(const std::vector<Tensor> &input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return operation::generic_create_output_tensors(*this, input_tensors, input_tensor_a.get_dtype(), input_tensor_a.get_layout(), this->output_mem_config);
}

operation::ProgramWithCallbacks ReshapeDeviceOperation::create_program(const std::vector<Tensor>& input_tensors, std::vector<Tensor> &output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    if (input_tensor_a.get_layout() == Layout::ROW_MAJOR) {
        return {detail::reshape_rm_multi_core(input_tensor_a, output_tensor, this->N, this->C, this->H, this->W)};
    } else if (input_tensor_a.get_layout() == Layout::TILE) {
        return {detail::reshape_tile_single_core(input_tensor_a, output_tensor, this->N, this->C, this->H, this->W)};
    } else {
        TT_ASSERT(false, "Unsupported layout for reshape");
        return {};
    }
}


}
