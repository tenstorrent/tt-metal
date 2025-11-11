// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "matmul_1d_device_operation.hpp"
#include "matmul_1d_program_factory.hpp"
#include "ttnn/operations/matmul/device/matmul_op.hpp"
#include <tt-metalium/constants.hpp>

using namespace ttnn::operations::matmul;

namespace ttnn::operations::experimental::deepseek_b1::matmul_1d {

using namespace tt::constants;

void Matmul1DDeviceOperation::validate(const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Input A must be on device");
    TT_FATAL(input_tensor_b.storage_type() == StorageType::DEVICE, "Input B must be on device");
    TT_FATAL(input_tensor_a.layout() == Layout::TILE, "Input A must be tilized");
    TT_FATAL(input_tensor_b.layout() == Layout::TILE, "Input B must be tilized");

    const auto& ashape = input_tensor_a.padded_shape();
    const auto& bshape = input_tensor_b.padded_shape();

    auto in0_tile = input_tensor_a.tensor_spec().tile();
    auto in1_tile = input_tensor_b.tensor_spec().tile();
    auto in0_tile_shape = in0_tile.get_tile_shape();
    auto in1_tile_shape = in1_tile.get_tile_shape();

    // Shape compatibility checks
    TT_FATAL(ashape[-1] == bshape[-2], "Dimension K (A.shape[-1] and B.shape[-2]) must match for A and B in matmul");

    // Tile divisibility checks for input A
    TT_FATAL(
        ashape[-2] % in0_tile_shape[0] == 0,
        "A.shape[-2] ({}) must be divisible by tile shape[0] ({})",
        ashape[-2],
        in0_tile_shape[0]);
    TT_FATAL(
        ashape[-1] % in0_tile_shape[1] == 0,
        "A.shape[-1] ({}) must be divisible by tile shape[1] ({})",
        ashape[-1],
        in0_tile_shape[1]);

    // Tile divisibility checks for input B
    TT_FATAL(
        bshape[-2] % in1_tile_shape[0] == 0,
        "B.shape[-2] ({}) must be divisible by tile shape[0] ({})",
        bshape[-2],
        in1_tile_shape[0]);
    TT_FATAL(
        bshape[-1] % in1_tile_shape[1] == 0,
        "B.shape[-1] ({}) must be divisible by tile shape[1] ({})",
        bshape[-1],
        in1_tile_shape[1]);

    // Buffer size checks
    tt::DataFormat in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_a.dtype());
    tt::DataFormat in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_b.dtype());
    uint32_t in0_single_tile_size = in0_tile.get_tile_size(in0_data_format);
    uint32_t in1_single_tile_size = in1_tile.get_tile_size(in1_data_format);
    tt::tt_metal::Buffer* in0_buffer = input_tensor_a.buffer();
    tt::tt_metal::Buffer* in1_buffer = input_tensor_b.buffer();

    TT_FATAL(
        in0_buffer->size() % in0_single_tile_size == 0,
        "Input A buffer size ({}) must be divisible by single tile size ({})",
        in0_buffer->size(),
        in0_single_tile_size);
    TT_FATAL(
        in1_buffer->size() % in1_single_tile_size == 0,
        "Input B buffer size ({}) must be divisible by single tile size ({})",
        in1_buffer->size(),
        in1_single_tile_size);
}

std::vector<ttnn::TensorSpec> Matmul1DDeviceOperation::compute_output_specs(
    const std::vector<Tensor>& input_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);

    // Use the matmul helper function to compute output shape
    const auto output_shape = compute_matmul_output_shape(input_tensor_a, input_tensor_b);

    auto dtype = output_dtype.value_or(input_tensor_a.dtype());
    auto mem_config = output_mem_config.value_or(input_tensor_a.memory_config());
    auto tile = input_tensor_a.tensor_spec().tile();

    return {TensorSpec(
        output_shape, tt::tt_metal::TensorLayout(dtype, tt::tt_metal::PageConfig(Layout::TILE, tile), mem_config))};
}

tt::tt_metal::operation::ProgramWithCallbacks Matmul1DDeviceOperation::create_program(
    const std::vector<Tensor>& input_tensors, std::vector<Tensor>& output_tensors) const {
    const auto& input_tensor_a = input_tensors.at(0);
    const auto& input_tensor_b = input_tensors.at(1);
    auto& output_tensor = output_tensors.at(0);

    // Call our local program factory with simplified signature
    return deepseek_b1_matmul_multi_core_reuse_mcast_1d_optimized(
        input_tensor_a,
        input_tensor_b,
        output_tensor,
        program_config.compute_with_storage_grid_size,
        compute_kernel_config.value_or(DeviceComputeKernelConfig{}));
}

}  // namespace ttnn::operations::experimental::deepseek_b1::matmul_1d
