// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshard_op.hpp"

#include "third_party/magic_enum/magic_enum.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/host_api.hpp"
#include "reshard_program_factory.hpp"
#include "tt_metal/common/work_split.hpp"

using namespace tt::constants;


namespace ttnn::operations::data_movement {

void ReshardDeviceOperation::validate_with_output_tensors(const std::vector<Tensor>& input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Operands to shard need to be on device!");
    TT_FATAL(input_tensor.buffer() != nullptr, "Operands to shard need to be allocated in buffers on device!");
    TT_FATAL(input_tensor.is_sharded(), "input must be sharded");
    bool has_output_tensor = output_tensors.size() == 1 && output_tensors[0].has_value();
    if (has_output_tensor) {
        const auto& output_tensor = output_tensors[0].value();
        TT_FATAL(input_tensor.get_shape() == output_tensor.get_shape());
        TT_FATAL(input_tensor.get_dtype() == output_tensor.get_dtype());
        TT_FATAL(input_tensor.get_layout() == output_tensor.get_layout());
    }
    const auto& out_mem_config = has_output_tensor ? output_tensors[0].value().memory_config() : this->output_mem_config;
    TT_FATAL(out_mem_config.is_sharded(), "output must be sharded");
    TT_FATAL(out_mem_config.buffer_type == BufferType::L1);
    if(input_tensor.get_layout() == Layout::ROW_MAJOR) {
        bool same_row_size = input_tensor.memory_config().shard_spec.value().shape[1] == out_mem_config.shard_spec.value().shape[1];
        TT_FATAL(same_row_size, "row major must have shard_spec[1] be the same on both input and output");
    }
}

std::vector<tt::tt_metal::Shape> ReshardDeviceOperation::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {input_tensor.get_legacy_shape()};
}

operation::ProgramWithCallbacks ReshardDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {

    const auto& input_tensor = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    //each tensor has its respective shard_spec within its memory_config
    return detail::reshard_multi_core(input_tensor, output_tensor);
}

std::vector<Tensor> ReshardDeviceOperation::create_output_tensors(const std::vector<Tensor> &input_tensors, const std::vector<std::optional<Tensor>> &output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    if (output_tensors.size() == 1 && output_tensors[0].has_value()) {
        return {output_tensors[0].value()};
    } else {
        auto mem_config = this->output_mem_config;

        return {create_device_tensor(
            this->compute_output_shapes(input_tensors).at(0),
            input_tensor.get_dtype(),
            input_tensor.get_layout(),
            input_tensor.device(),
            mem_config
            )};
    }
}


}  // namespace ttnn::operations::data_movement
