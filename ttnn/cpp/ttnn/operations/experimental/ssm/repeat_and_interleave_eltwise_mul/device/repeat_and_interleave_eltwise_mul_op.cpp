// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "repeat_and_interleave_eltwise_mul_op.hpp"

#include "repeat_and_interleave_eltwise_mul_program_factory.hpp"
#include "tt_metal/common/constants.hpp"

namespace ttnn::operations::experimental::ssm {

void RepeatAndInterleaveEltwiseMul::validate(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    TT_FATAL(input_tensors.size() == 2);
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    TT_FATAL(
        (input_tensor_a.get_layout() == Layout::TILE && input_tensor_b.get_layout() == Layout::TILE),
        "Inputs to ssm_eltwise_mul must be tilized");

    // TODO: Uplift to support BFLOAT8_B and mixed precision
    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE and input_tensor_b.storage_type() == StorageType::DEVICE,
        "Operands to ssm_eltwise_mul need to be on device!");
    TT_FATAL(
        input_tensor_a.buffer() != nullptr and input_tensor_b.buffer() != nullptr,
        "Operands to ssm_eltwise_mul need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor_a.device() == input_tensor_b.device(),
        "Operands to ssm_eltwise_mul need to be on the same device!");

    TT_FATAL(
        input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Unsupported memory layout for input a!");
    TT_FATAL(
        input_tensor_b.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Unsupported memory layout for input b!");
    TT_FATAL(
        input_tensor_a.get_dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor_a.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format for input a!");
    TT_FATAL(
        input_tensor_b.get_dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor_b.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format for input b!");

    TT_FATAL(
        this->memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Unsupported memory layout for output!");
    TT_FATAL(
        this->dtype == tt::tt_metal::DataType::BFLOAT16 || this->dtype == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format for output!");

    const auto ashape = input_tensor_a.get_legacy_shape();
    const auto bshape = input_tensor_b.get_legacy_shape();
    TT_FATAL((ashape[0] == 1 and ashape[1] == 1), "Batch not supported for input a!");
    TT_FATAL((bshape[0] == 1 and bshape[1] == 1), "Batch not supported for input b!");
    TT_FATAL((ashape[2] % TILE_HEIGHT == 0), "Num of users must be multiple of 32 for input a!");
    TT_FATAL((bshape[2] % TILE_HEIGHT == 0), "Num of users must be multiple of 32 for input b!");
    TT_FATAL((ashape[2] == bshape[2]), "Num of users must match in both of the input!");
    TT_FATAL((ashape[3] != bshape[3]), "Use eltwise mul for same size inputs!");
    TT_FATAL(
        (ashape[3] == TILE_WIDTH || ashape[3] == TILE_WIDTH * HIDDEN_SIZE), "Input a width must be 32 or 32*5120!");
    TT_FATAL(
        (bshape[3] == HIDDEN_SIZE || bshape[3] == TILE_WIDTH * HIDDEN_SIZE), "Input b width must be 32 or 32*5120!");
}

std::vector<tt::tt_metal::LegacyShape> RepeatAndInterleaveEltwiseMul::compute_output_shapes(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    const auto shape_a = input_tensor_a.get_legacy_shape();
    const auto shape_b = input_tensor_b.get_legacy_shape();
    return {{shape_a[0], shape_a[1], shape_a[2], tt::constants::TILE_WIDTH * HIDDEN_SIZE}};
}

std::vector<Tensor> RepeatAndInterleaveEltwiseMul::create_output_tensors(
    const std::vector<Tensor>& input_tensors) const {
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->dtype, Layout::TILE, this->memory_config);
}

operation::ProgramWithCallbacks RepeatAndInterleaveEltwiseMul::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);
    const auto hidden_size = HIDDEN_SIZE;

    auto device_compute_with_storage_grid_size = input_tensor_a.device()->compute_with_storage_grid_size();

    return detail::multi_core_ssm_eltwise_mul(
        input_tensor_a,
        input_tensor_b,
        output_tensor,
        hidden_size,
        this->math_fidelity,
        device_compute_with_storage_grid_size);
}

}  // namespace ttnn::operations::experimental::ssm
