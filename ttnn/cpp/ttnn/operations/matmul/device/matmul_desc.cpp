// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/matmul/device/matmul_desc.hpp"

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/matmul/device/utilities/matmul_utilities.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::matmul {

namespace {

constexpr uint32_t TILE_DIM = 32;

// A memory config (and the tensor's shard spec, when it has one), with shard sizes in the tensor's tiles
Placement placement_of(
    const tt::tt_metal::MemoryConfig& memory_config,
    const std::optional<tt::tt_metal::ShardSpec>& shard_spec,
    uint32_t tile_h,
    uint32_t tile_w) {
    using tt::tt_metal::TensorMemoryLayout;
    Placement placement;
    placement.in_l1 = memory_config.buffer_type() == tt::tt_metal::BufferType::L1;
    switch (memory_config.memory_layout()) {
        case TensorMemoryLayout::INTERLEAVED: return placement;
        case TensorMemoryLayout::HEIGHT_SHARDED: placement.layout = MemoryLayout::HeightSharded; break;
        case TensorMemoryLayout::WIDTH_SHARDED: placement.layout = MemoryLayout::WidthSharded; break;
        case TensorMemoryLayout::BLOCK_SHARDED: placement.layout = MemoryLayout::BlockSharded; break;
        default: placement.layout = MemoryLayout::NdSharded; break;
    }
    // An ND shard spec without a 2D equivalent
    if (memory_config.nd_shard_spec().has_value() && !shard_spec.has_value()) {
        placement.layout = MemoryLayout::NdSharded;
        return placement;
    }
    if (shard_spec.has_value()) {
        const auto& spec = shard_spec.value();
        placement.has_shard_spec = true;
        placement.shard_whole_tiles = spec.shape[0] % tile_h == 0 && spec.shape[1] % tile_w == 0;
        placement.shard_grid = spec.grid.bounding_box();
        placement.shard_cores = spec.grid.num_cores();
        placement.shard_h = spec.shape[0] / tile_h;
        placement.shard_w = spec.shape[1] / tile_w;
        placement.col_major = spec.orientation == tt::tt_metal::ShardOrientation::COL_MAJOR;
    }
    return placement;
}

}  // namespace

std::optional<MatmulDesc> describe_matmul(
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const bool transpose_a,
    const bool transpose_b,
    const uint32_t bias_single_tile_size,
    const ttnn::prim::MatmulParams& attributes,
    std::string& why) {
    // Tiles: A's are in0_tile_h x 32, B's 32 x in1_tile_w
    const auto in0_tile = utilities::get_matmul_tile(input_tensor_a, transpose_a);
    const auto in1_tile = utilities::get_matmul_tile(input_tensor_b, transpose_b);
    const auto out_tile =
        attributes.output_tile.value_or(tt::tt_metal::Tile({in0_tile.get_height(), in1_tile.get_width()}));
    if (in0_tile.get_width() != TILE_DIM || in1_tile.get_height() != TILE_DIM) {
        why = "tile K side not 32";
        return std::nullopt;
    }
    const auto a_shape = utilities::get_matmul_tensor_padded_shape(input_tensor_a, transpose_a);
    const auto b_shape = utilities::get_matmul_tensor_padded_shape(input_tensor_b, transpose_b);
    if (a_shape.rank() < 2 || b_shape.rank() < 2) {
        why = "rank below 2";
        return std::nullopt;
    }
    if (!attributes.compute_kernel_config.has_value()) {
        why = "no compute kernel config";
        return std::nullopt;
    }

    MatmulDesc p;
    p.batch_a = a_shape.volume() / (a_shape[-2] * a_shape[-1]);
    p.batch_b = b_shape.volume() / (b_shape[-2] * b_shape[-1]);
    p.rank_a = a_shape.rank();
    p.rank_b = b_shape.rank();
    p.in0_tile_h = in0_tile.get_height();
    p.in1_tile_w = in1_tile.get_width();
    p.out_tile_h = out_tile.get_height();
    p.out_tile_w = out_tile.get_width();
    p.Mt = a_shape[-2] / p.in0_tile_h;
    p.Kt = a_shape[-1] / TILE_DIM;
    p.Nt = b_shape[-1] / p.in1_tile_w;

    const auto& output_mem_config = attributes.output_mem_config;
    p.a = placement_of(input_tensor_a.memory_config(), input_tensor_a.shard_spec(), in0_tile.get_height(), TILE_DIM);
    p.b = placement_of(input_tensor_b.memory_config(), input_tensor_b.shard_spec(), TILE_DIM, in1_tile.get_width());
    p.out =
        placement_of(output_mem_config, output_mem_config.shard_spec(), in0_tile.get_height(), in1_tile.get_width());
    if (p.a.has_shard_spec && p.b.has_shard_spec) {
        const auto& sa = input_tensor_a.shard_spec().value();
        const auto& sb = input_tensor_b.shard_spec().value();
        p.b_shard_matches_a = p.a.layout == p.b.layout && sa.grid == sb.grid && sa.orientation == sb.orientation;
    }
    p.global_cb = attributes.global_cb.has_value();

    const auto arch = input_tensor_a.device()->arch();
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, attributes.compute_kernel_config.value());
    p.in0_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_a.dtype());
    p.in1_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor_b.dtype());
    p.out_format =
        tt::tt_metal::datatype_to_dataformat_converter(attributes.output_dtype.value_or(input_tensor_a.dtype()));
    p.bias_tile_bytes = bias_single_tile_size;
    p.transpose_a = transpose_a;
    p.in0_tile_transposed = in0_tile.get_transpose_of_faces() && in0_tile.get_transpose_within_face();
    p.untilize_out = attributes.untilize_out;
    p.math_fidelity = math_fidelity;
    p.fp32_dest_acc_en = fp32_dest_acc_en;
    p.packer_l1_acc = packer_l1_acc;
    p.dst_full_sync_en = dst_full_sync_en;
    // Fused only if the kernels support it; otherwise matmul applies it as a separate op
    if (attributes.user_fused_activation.has_value() &&
        utilities::is_fusable_activation(attributes.user_fused_activation->op_type)) {
        p.activation = attributes.user_fused_activation;
    }
    return p;
}

}  // namespace ttnn::operations::matmul
