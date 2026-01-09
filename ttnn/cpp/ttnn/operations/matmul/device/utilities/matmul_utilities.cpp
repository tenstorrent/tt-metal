// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"

#include "tt-metalium/allocator.hpp"
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::operations::matmul::utilities {

uint32_t get_estimated_size_of_cbs(
    uint32_t per_core_M,
    uint32_t per_core_N,
    uint32_t in0_block_w,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const bool transpose_a,
    const bool transpose_b,
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
    uint32_t in2_block_tiles = 0;
    uint32_t in0_shard_width_in_tiles = 0;
    if (input_tensor_a.is_sharded()) {
        auto* in0_buffer = input_tensor_a.buffer();
        const auto in0_tile = utilities::get_matmul_tile(input_tensor_a, transpose_a);
        in0_shard_width_in_tiles = in0_buffer->shard_spec().shape()[1] / in0_tile.get_width();
        if (transpose_a) {
            // An intermediate CB (c_10) of same size is needed to hold the transposed data
            in0_shard_width_in_tiles *= 2;
        }
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

ttnn::Shape compute_matmul_output_shape(
    const Tensor& input_tensor_a, const Tensor& input_tensor_b, bool transpose_a, bool transpose_b) {
    const auto& input_shape_a = get_matmul_tensor_logical_shape(input_tensor_a, transpose_a);
    const auto& input_shape_b = get_matmul_tensor_logical_shape(input_tensor_b, transpose_b);

    const auto a_rank = input_shape_a.rank();
    const auto b_rank = input_shape_b.rank();

    // Rank difference will be used to align batch dimensions
    const int32_t out_rank = std::max<int32_t>(a_rank, b_rank) - (a_rank == 1 || b_rank == 1);
    const int32_t rank_difference = std::max<int32_t>(0, out_rank - a_rank);

    // Initialize output shape based on the tensor with higher rank
    ttnn::Shape output_shape = (b_rank > a_rank) ? input_shape_b : input_shape_a;

    // Handle batch dimensions for the case where b_rank > a_rank
    for (auto index = 0; index < rank_difference; ++index) {
        TT_FATAL(input_shape_b[index] == 1, "When in1 rank greater than in0 rank front dimensions need to be 1");
        output_shape[index] = input_shape_b[index];
    }

    // Copy dimensions from input_shape_a except the last one
    for (auto index = 0; index < a_rank - 1; ++index) {
        output_shape[rank_difference + index] = input_shape_a[index];
    }

    // The last dimension comes from input_tensor_b
    output_shape[-1] = input_shape_b[-1];

    // Handle the vector matmul case: if a_rank == 1, remove the second-to-last dimension
    if (a_rank == 1 && output_shape.rank() > 1) [[unlikely]] {
        ttnn::SmallVector<uint32_t> new_shape(output_shape.rank() - 1);
        // Copy all elements except the second-to-last dimension
        size_t dst_idx = 0;
        for (size_t src_idx = 0; src_idx < output_shape.rank(); ++src_idx) {
            if (src_idx != output_shape.rank() - 2) {
                new_shape[dst_idx++] = output_shape[src_idx];
            }
        }
        output_shape = ttnn::Shape(new_shape);
    }

    // Handle the case where b_rank == 1, remove the last dimension
    if (b_rank == 1) [[unlikely]] {
        ttnn::SmallVector<uint32_t> new_shape(output_shape.rank() - 1);
        for (auto index = 0; index < output_shape.rank() - 1; ++index) {
            new_shape[index] = output_shape[index];
        }
        output_shape = ttnn::Shape(new_shape);
    }

    return output_shape;
}

tt::tt_metal::Tile get_output_tile(
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const tt::tt_metal::Tile& in0_tile,
    const tt::tt_metal::Tile& in1_tile,
    const std::optional<const tt::tt_metal::Tile>& output_tile,
    const std::optional<const tt::tt_metal::Tile>& optional_output_tensor_tile) {
    using namespace tt;
    if (output_tile.has_value() or optional_output_tensor_tile.has_value()) {
        TT_FATAL(
            !(optional_output_tensor_tile.has_value() && output_tile.has_value()),
            "Matmul cannot have both an output_tile and an optional_output_tensor. Configure the tile type of the "
            "output tensor instead if both are required.");
        const auto& override_output_tile =
            output_tile.has_value() ? output_tile.value() : optional_output_tensor_tile.value();
        const auto& out_tile_shape = override_output_tile.get_tile_shape();

        const uint32_t in0_tile_h = in0_tile.get_height();
        const uint32_t in1_tile_w = in1_tile.get_width();

        TT_FATAL(out_tile_shape[1] > 0, "the override output tile width needs to be greater than zero");
        TT_FATAL(
            out_tile_shape[1] % in1_tile_w == 0,
            "the override output tile width ({}) must be a multiple of in1 tile width ({})",
            out_tile_shape[1],
            in1_tile_w);
        TT_FATAL(out_tile_shape[0] > 0, "the override output tile height needs to be greater than zero");
        TT_FATAL(
            out_tile_shape[0] == in0_tile_h,
            "the override output tile height ({}) must equal to the in0 tile height ({})",
            out_tile_shape[0],
            in0_tile_h);
        if (out_tile_shape[1] != in1_tile_w) {
            TT_FATAL(
                out_tile_shape[0] <= constants::FACE_HEIGHT,
                "the override output tile height ({}) must equal or less to face height ({})",
                out_tile_shape[0],
                constants::FACE_HEIGHT);
        }
        if (!output_mem_config.is_sharded()) {
            TT_FATAL(
                out_tile_shape[1] == in1_tile_w,
                "the override output tile width ({}) must equal the in0 tile width ({})",
                out_tile_shape[1],
                in1_tile_w);
        }

        return override_output_tile;
    }
    return tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()});
}

tt::tt_metal::Tile get_matmul_tile(const Tensor& input_tensor, bool transpose) {
    auto curr_tile = input_tensor.tensor_spec().tile();
    if (!transpose) {
        return curr_tile;
    }

    // If the tile is already transposed and we are asked to transpose it again,
    // the result should be the original orientation (double-transpose cancels out).
    // Therefore, we negate the transpose flag.
    const auto transpose_was_set = curr_tile.get_transpose_of_faces();
    TT_FATAL(
        (!transpose_was_set) || curr_tile.get_transpose_within_face(),
        "The tile spec must have both transpose_within_face {} and transpose_of_faces {} set or neither set",
        curr_tile.get_transpose_within_face(),
        curr_tile.get_transpose_of_faces());
    return tt::tt_metal::Tile({curr_tile.get_width(), curr_tile.get_height()}, !transpose_was_set);
}

}  // namespace ttnn::operations::matmul::utilities
