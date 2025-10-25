// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reshape_op.hpp"
#include <tt-metalium/constants.hpp>

#include "ttnn/tensor/tensor_utils.hpp"
#include "reshape_program_factory.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

std::vector<ttnn::Tensor> ReshapeDeviceOperation::invoke(std::vector<Tensor> input_tensors) {
    // // No-op (Will do a tensor copy)
    // if (((input_tensor.layout() == Layout::TILE or input_tensor.layout() == Layout::ROW_MAJOR) &&
    //      padded_output_shape[3] == input_tensor.padded_shape()[3])) {
    //     // Don't need to do a check here to see the H and W both divisible by 32
    //     // since handled within the tensor reshape method
    //     return ttnn::experimental::view(input_tensor, logical_output_shape, padded_output_shape);
    // }
    // if (input_tensor.padded_shape() == padded_output_shape) {
    //     return ttnn::operations::experimental::auto_format::AutoFormat::move_tensor_to_mem_config(
    //         input_tensor, output_mem_config);
    // }
    // uint32_t ROW_MAJOR_WIDTH = 8;
    // if (input_tensor.layout() == Layout::ROW_MAJOR &&
    //     (input_tensor.padded_shape()[3] % ROW_MAJOR_WIDTH != 0 || padded_output_shape[3] % ROW_MAJOR_WIDTH != 0) &&
    //     ((padded_output_shape.volume() / padded_output_shape[-1]) % TILE_HEIGHT != 0 ||
    //      padded_output_shape[-1] % TILE_WIDTH != 0 || input_tensor.padded_shape()[-1] % TILE_WIDTH != 0 ||
    //      (input_tensor.physical_volume() / input_tensor.padded_shape()[-1]) % TILE_HEIGHT != 0)) {
    //     TT_FATAL(input_tensor.dtype() == DataType::BFLOAT16, "Error");

    //     return detail::manual_insertion(
    //         (tt::tt_metal::Tensor)input_tensor,
    //         logical_output_shape,
    //         padded_output_shape,
    //         input_tensor.device(),
    //         output_mem_config);
    // }
    return tt::tt_metal::operation::run(*this, input_tensors);
}

void ReshapeDeviceOperation::set_output_tensors(std::vector<Tensor> output_tensors) {
    output_tensors_ = output_tensors;
}

std::vector<Tensor> ReshapeDeviceOperation::get_output_tensors() const { return output_tensors_; }

std::vector<Tensor> ReshapeDeviceOperation::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    if (output_tensors_.size() > 0) {
        return output_tensors_;
    }

    auto output_specs = compute_output_specs(input_tensors);

    // Create output tensors for each spec
    std::vector<ttnn::Tensor> output_tensors;
    for (const auto& spec : output_specs) {
        // I am not entirely sure if this is true in any device...
        auto output_tensor = tt::tt_metal::create_device_tensor(spec, input_tensors[0].device());
        output_tensor = tt::tt_metal::set_tensor_id(output_tensor);
        output_tensors.push_back(output_tensor);
    }

    return output_tensors;
}

void ReshapeDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to reshape need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to reshape need to be allocated in buffers on device!");
    TT_FATAL(input_tensor_a.dtype() == DataType::BFLOAT16 or input_tensor_a.dtype() == DataType::FLOAT32, "Error");

    // Maybe call validate again?
}

void ReshapeDeviceOperation::validate(const std::vector<ttnn::experimental::jit::LazyTensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL(
        input_tensor_a.get_tensor_spec().data_type() == DataType::BFLOAT16 ||
            input_tensor_a.get_tensor_spec().data_type() == DataType::FLOAT32,
        "Error BFloat16 or Float32 only supported!");

    TT_FATAL(
        input_tensor_a.get_tensor_spec().layout() == Layout::TILE ||
            input_tensor_a.get_tensor_spec().layout() == Layout::ROW_MAJOR,
        "Only tile and row major reshape supported!");

    TT_FATAL(
        input_tensor_a.get_tensor_spec().memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only interleaved memory layout is supported for inputs. Use ttnn::reshape for reshaping sharded inputs");

    if (input_tensor_a.get_tensor_spec().tensor_layout().get_layout() == Layout::TILE) {
        TT_FATAL(input_tensor_a.get_tensor_spec().padded_shape().volume() % TILE_HW == 0, "Error");
    } else if (input_tensor_a.get_tensor_spec().tensor_layout().get_layout() == Layout::ROW_MAJOR) {
        uint32_t ROW_MAJOR_WIDTH = 8;
        TT_FATAL(
            input_tensor_a.get_tensor_spec().padded_shape()[3] % ROW_MAJOR_WIDTH == 0,
            "Operand/target width must be a multiple of 8");
        TT_FATAL(padded_output_shape[3] % ROW_MAJOR_WIDTH == 0, "Operand/target width must be a multiple of 8");
    } else {
        TT_THROW("Unsupported layout for reshape");
    }
}

std::vector<ttnn::TensorSpec> ReshapeDeviceOperation::compute_output_specs(
    const std::vector<ttnn::experimental::jit::LazyTensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    return {input_tensor_a.get_tensor_spec()};
}

std::vector<ttnn::TensorSpec> ReshapeDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    return {TensorSpec(
        logical_output_shape,
        TensorLayout::fromPaddedShape(
            input_tensor.dtype(),
            input_tensor.tensor_spec().page_config(),
            output_mem_config,
            logical_output_shape,
            padded_output_shape))};
}

tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>>
ReshapeDeviceOperation::create_op_performance_model(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& optional_input_tensors,
    std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor = input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output_tensor);
    tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> result(
        input_tensors, output_tensors, ideal_dev_clock_cycles);
    return result;
}

operation::ProgramWithCallbacks ReshapeDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    if (input_tensor_a.layout() == Layout::ROW_MAJOR) {
        return {detail::reshape_rm_multi_core(input_tensor_a, output_tensor)};
    } else if (input_tensor_a.layout() == Layout::TILE) {
        return {detail::reshape_tile_single_core(input_tensor_a, output_tensor)};
    } else {
        TT_ASSERT(false, "Unsupported layout for reshape");
        return {};
    }
}

}  // namespace ttnn::operations::data_movement
