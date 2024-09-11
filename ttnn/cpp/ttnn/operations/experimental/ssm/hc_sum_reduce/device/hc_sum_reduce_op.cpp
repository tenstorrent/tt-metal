// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "hc_sum_reduce_op.hpp"

#include "hc_sum_reduce_program_factory.hpp"

namespace ttnn::operations::experimental::ssm {

void HCSumReduce::validate(const std::vector<Tensor>& input_tensors) const {
    using namespace tt::constants;
    TT_FATAL(input_tensors.size() == 1, "Error");
    const auto& input_tensor_a = input_tensors.at(0);
    TT_FATAL((input_tensor_a.get_layout() == Layout::TILE), "Inputs to ssm_1d_sum_reduce must be tilized");

    // TODO: Uplift to support mixed precision
    TT_FATAL(
        input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to ssm_1d_sum_reduce need to be on device!");
    TT_FATAL(
        input_tensor_a.buffer() != nullptr, "Operands to ssm_1d_sum_reduce need to be allocated in buffers on device!");

    TT_FATAL(
        input_tensor_a.memory_config().memory_layout == TensorMemoryLayout::INTERLEAVED,
        "Unsupported memory layout for input a!");
    TT_FATAL(
        input_tensor_a.get_dtype() == tt::tt_metal::DataType::BFLOAT16 ||
            input_tensor_a.get_dtype() == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format for input a!");

    TT_FATAL(
        this->memory_config.memory_layout == TensorMemoryLayout::INTERLEAVED, "Unsupported memory layout for output!");
    TT_FATAL(
        this->dtype == tt::tt_metal::DataType::BFLOAT16 || this->dtype == tt::tt_metal::DataType::BFLOAT8_B,
        "Unsupported data format for output!");

    constexpr uint32_t latent = 32;
    const auto ashape = input_tensor_a.get_legacy_shape();
    TT_FATAL((ashape[0] == 1 and ashape[1] == 1), "Dim 1 and 2 are expected to be 1 in input a!");
    TT_FATAL((ashape[2] % TILE_HEIGHT == 0), "Batch size must be divisible by 32 for input a!");
    TT_FATAL((ashape[3] % TILE_WIDTH == 0), "Final dim must be a multiple of 32!");
    TT_FATAL(((ashape[3] / TILE_WIDTH) % latent == 0), "Final dim/TILE_SIZE must be a multiple of latent size!");
}

std::vector<tt::tt_metal::Shape> HCSumReduce::compute_output_shapes(const std::vector<Tensor>& input_tensors) const {
    constexpr uint32_t latent = 32;
    const auto& input_tensor_a = input_tensors.at(0);
    const auto shape_a = input_tensor_a.get_legacy_shape();
    return {{shape_a[0], shape_a[1], shape_a[2], shape_a[3] / latent}};
}

std::vector<Tensor> HCSumReduce::create_output_tensors(const std::vector<Tensor>& input_tensors) const {
    return operation::generic_create_output_tensors(
        *this, input_tensors, this->dtype, Layout::TILE, this->memory_config);
}

operation::ProgramWithCallbacks HCSumReduce::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    auto& output_tensor = output_tensors.at(0);
    auto device_compute_with_storage_grid_size = input_tensor_a.device()->compute_with_storage_grid_size();
    return detail::multi_core_ssm_1d_sum_reduce(
        input_tensor_a, output_tensor, math_fidelity, device_compute_with_storage_grid_size);
}

}  // namespace ttnn::operations::experimental::ssm
