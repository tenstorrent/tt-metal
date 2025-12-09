// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/tmp/utilities/matmul_utilities.hpp"

#include "tt-metalium/allocator.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::operations::matmul::utilities {

uint32_t get_estimated_size_of_cbs(
    uint32_t per_core_M,
    uint32_t per_core_N,
    uint32_t in0_block_w,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    uint32_t interm_single_tile_size,
    uint32_t bias_single_tile_size) {
    // Circular Buffer sizes:
    // src0   CB: per_core_M * in0_block_w * 2 (for double buffer)
    // src1   CB: per_core_N * in0_block_w * 2 (for double buffer)
    // interm CB: per_core_M * per_core_N * interm_single_tile_size
    // out    CB: per_core_M * per_core_N
    // bias   CB: per_core_M * in0_block_w
    // Ignore optional intermediate CB because not needed when need to create a
    // program config.
    tt::DataFormat in0_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_a.dtype());
    tt::DataFormat in1_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_b.dtype());
    uint32_t in0_single_tile_size = tt::tile_size(in0_data_format);  // use as estimate for output as well
    uint32_t in1_single_tile_size = tt::tile_size(in1_data_format);
    uint32_t output_single_tile_size = in0_single_tile_size;
    auto* in0_buffer = input_tensor_a.buffer();
    auto in0_tile = input_tensor_a.tensor_spec().tile();
    uint32_t in2_block_tiles = 0;
    uint32_t in0_shard_width_in_tiles = 0;
    if (input_tensor_a.is_sharded()) {
        in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_tile_shape()[1];
    }
    in2_block_tiles = per_core_M * in0_shard_width_in_tiles;

    // Calculate individual buffer sizes in bytes - use constant for buffering depth
    uint32_t in0_size = per_core_M * in0_block_w * MCAST_INPUT_BUFFERING_DEPTH * in0_single_tile_size;
    uint32_t in1_size = per_core_N * in0_block_w * MCAST_INPUT_BUFFERING_DEPTH * in1_single_tile_size;
    uint32_t out_size = per_core_M * per_core_N * output_single_tile_size;
    uint32_t in2_size = in2_block_tiles * in0_single_tile_size;
    uint32_t interm_size = per_core_M * per_core_N * interm_single_tile_size;
    uint32_t bias_size = in0_block_w * bias_single_tile_size;
    return in0_size + in1_size + out_size + interm_size + bias_size + in2_size;
}

uint32_t estimate_interm_tile_size(
    const std::optional<const ttnn::DeviceComputeKernelConfig>& compute_kernel_config,
    const tt::tt_metal::DataType output_dtype) {
    if (get_fp32_dest_acc_en(compute_kernel_config)) {
        return tt::tile_size(tt::DataFormat::Float32);
    }
    uint32_t result = tt::tile_size(tt::DataFormat::Float16_b);  // packer l1 acc
    tt::DataFormat output_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_dtype);
    uint32_t output_tile_size = tt::tile_size(output_data_format);
    result = std::max(output_tile_size, result);
    return result;
}

uint32_t get_max_l1_space(const tt::tt_metal::Tensor& input_tensor_a) {
    auto* device = input_tensor_a.device();
    auto lowest_address = device->lowest_occupied_compute_l1_address();
    uint32_t max_l1_space = lowest_address.has_value() ? lowest_address.value() : device->l1_size_per_core();
    max_l1_space = max_l1_space - device->allocator()->get_base_allocator_addr(tt::tt_metal::HalMemType::L1);
    return max_l1_space;
}

bool is_input_batched(const ttnn::Shape& shape) {
    if (shape.rank() < 2) [[unlikely]] {
        return false;
    }

    auto is_batched = false;
    for (auto i = 0; i < shape.rank() - 2; ++i) {
        if (shape[i] > 1) {
            is_batched = true;
            break;
        }
    }
    return is_batched;
}

std::optional<ttnn::operations::unary::UnaryWithParam> get_fused_activation(
    const std::optional<const Activation>& activation) {
    if (!activation.has_value()) {
        return std::nullopt;
    }
    const auto& act = activation.value();
    if (std::holds_alternative<std::string>(act)) {
        return ttnn::operations::unary::utils::string_to_unary_with_param(std::get<std::string>(act));
    }
    return std::get<ttnn::operations::unary::UnaryWithParam>(act);
}

}  // namespace ttnn::operations::matmul::utilities
