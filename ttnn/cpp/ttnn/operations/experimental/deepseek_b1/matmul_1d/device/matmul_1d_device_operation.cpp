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

    auto tile = input_tensor_a.tensor_spec().tile();
    TT_FATAL(ashape[-1] == bshape[-2], "Input A width must match Input B height");
    TT_FATAL(ashape[-1] % tile.get_tile_shape()[1] == 0, "Input A width must be divisible by tile width");
    TT_FATAL(ashape[-2] % tile.get_tile_shape()[0] == 0, "Input A height must be divisible by tile height");
    TT_FATAL(bshape[-1] % TILE_WIDTH == 0, "Input B width must be divisible by tile width");
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

    // Convert our config to the matmul program config
    MatmulMultiCoreReuseMultiCast1DProgramConfig matmul_config{
        .compute_with_storage_grid_size = program_config.compute_with_storage_grid_size,
        .in0_block_w = program_config.in0_block_w,
        .out_subblock_h = program_config.out_subblock_h,
        .out_subblock_w = program_config.out_subblock_w,
        .out_block_h = program_config.per_core_M,
        .out_block_w = program_config.per_core_N,
        .per_core_M = program_config.per_core_M,
        .per_core_N = program_config.per_core_N,
        .fuse_batch = program_config.fuse_batch,
        .fused_activation = std::nullopt,
        .mcast_in0 = program_config.mcast_in0,
        .gather_in0 = false,
        .hop_cores = CoreRangeSet{},
        .num_global_cb_receivers = 0,
        .untilize_out = false,
    };

    // Call our local program factory (copied from matmul)
    return deepseek_b1_matmul_multi_core_reuse_mcast_1d_optimized(
        input_tensor_a,
        input_tensor_b,
        /*bias=*/std::nullopt,
        output_tensor,
        /*bcast_batch=*/false,
        matmul_config.compute_with_storage_grid_size,
        compute_kernel_config.value_or(DeviceComputeKernelConfig{}),
        matmul_config.in0_block_w,
        matmul_config.out_subblock_h,
        matmul_config.out_subblock_w,
        matmul_config.out_block_h,
        matmul_config.out_block_w,
        matmul_config.per_core_M,
        matmul_config.per_core_N,
        matmul_config.fuse_batch,
        matmul_config.fused_activation,
        matmul_config.mcast_in0,
        matmul_config.gather_in0,
        matmul_config.hop_cores,
        matmul_config.untilize_out,
        /*fused_op_signaler=*/std::nullopt,
        /*global_cb=*/std::nullopt,
        matmul_config.num_global_cb_receivers,
        /*sub_device_id=*/std::nullopt);
}

}  // namespace ttnn::operations::experimental::deepseek_b1::matmul_1d
