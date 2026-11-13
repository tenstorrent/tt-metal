// SPDX-FileCopyrightText: 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "pirotski_matmul_device_operation.hpp"

#include <string_view>
#include <tuple>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/matmul/device/config/matmul_program_config.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace ttnn::experimental::nomic_embed_text_v2_moe {

namespace {

using tt::tt_metal::BufferType;
using tt::tt_metal::DataType;
using tt::tt_metal::Layout;
using tt::tt_metal::ShardOrientation;
using tt::tt_metal::TensorMemoryLayout;

constexpr uint32_t kTileSize = tt::constants::TILE_HEIGHT;

void check_tile_operand(const Tensor& tensor, std::string_view name) {
    TT_FATAL(
        tensor.storage_type() == StorageType::DEVICE && tensor.is_allocated(),
        "pirotski_matmul: {} must be an allocated device tensor",
        name);
    TT_FATAL(
        tensor.dtype() == DataType::BFLOAT16 || tensor.dtype() == DataType::BFLOAT8_B,
        "pirotski_matmul: {} must be BFLOAT16 or BFLOAT8_B, got {}",
        name,
        tensor.dtype());
    TT_FATAL(tensor.layout() == Layout::TILE, "pirotski_matmul: {} must be TILE layout, got {}", name, tensor.layout());
    const auto tile = tensor.tensor_spec().tile();
    TT_FATAL(
        tile.get_height() == kTileSize && tile.get_width() == kTileSize && !tile.get_transpose_of_faces() &&
            !tile.get_transpose_within_face(),
        "pirotski_matmul: {} must use plain 32 x 32 tiles",
        name);
    TT_FATAL(
        tensor.logical_shape().rank() >= 2 && tensor.padded_shape()[-1] % kTileSize == 0,
        "pirotski_matmul: {} must be at least 2-D with whole tiles across, got {}",
        name,
        tensor.logical_shape());
}

// A block-sharded, column-major L1 layout whose shard (x, y) sits on core (x, y) of a grid starting at (0, 0).
bool is_block_grid(const tt::tt_metal::MemoryConfig& memory_config, uint32_t columns, uint32_t rows) {
    if (memory_config.memory_layout() != TensorMemoryLayout::BLOCK_SHARDED ||
        memory_config.buffer_type() != BufferType::L1 || !memory_config.shard_spec().has_value()) {
        return false;
    }
    const auto& spec = *memory_config.shard_spec();
    const auto box = spec.grid.bounding_box();
    return spec.orientation == ShardOrientation::COL_MAJOR && spec.grid.ranges().size() == 1 &&
           box.start_coord == CoreCoord(0, 0) && box.end_coord == CoreCoord(columns - 1, rows - 1);
}

}  // namespace

PirotskiMatmulLayout PirotskiMatmulDeviceOperation::compute_layout(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    const auto& weight = tensor_args.weight;
    const auto& config = attributes.config;
    check_tile_operand(input, "input_tensor");
    check_tile_operand(weight, "weight");
    TT_FATAL(weight.device() == input.device(), "pirotski_matmul: weight must be on the device of input_tensor");
    TT_FATAL(
        weight.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "pirotski_matmul: weight must be interleaved, got {}",
        weight.memory_config().memory_layout());
    const auto& weight_shape = weight.padded_shape();
    TT_FATAL(
        weight_shape.volume() == weight_shape[-2] * weight_shape[-1],
        "pirotski_matmul: weight must be (K, N) with any leading dims 1, got {}",
        weight.logical_shape());

    // Upper dims fold into M, each sequence at its tile-padded length, as minimal_matmul folds them.
    const uint32_t k = input.padded_shape()[-1];
    const uint32_t m = input.physical_volume() / k;
    const uint32_t n = weight_shape[-1];
    const uint32_t logical_n = weight.logical_shape()[-1];
    TT_FATAL(
        weight_shape[-2] == k, "pirotski_matmul: input_tensor has K = {} but weight has K = {}", k, weight_shape[-2]);
    TT_FATAL(
        m > n, "pirotski_matmul: M = {} must exceed N = {}; minimal_matmul puts M on the grid's rows otherwise", m, n);

    if (tensor_args.bias.has_value()) {
        const auto& bias = *tensor_args.bias;
        TT_FATAL(bias.device() == input.device(), "pirotski_matmul: bias must be on the device of input_tensor");
        TT_FATAL(
            bias.dtype() == DataType::BFLOAT16 && bias.layout() == Layout::TILE &&
                bias.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "pirotski_matmul: bias must be BFLOAT16, TILE and interleaved, got {} {}",
            bias.dtype(),
            bias.layout());
        TT_FATAL(
            bias.logical_shape()[-1] == logical_n && bias.logical_shape().volume() == logical_n,
            "pirotski_matmul: bias must be (1, N) with N = {}, got {}",
            logical_n,
            bias.logical_shape());
    }
    TT_FATAL(
        attributes.dtype == DataType::BFLOAT16 || attributes.dtype == DataType::BFLOAT8_B,
        "pirotski_matmul: dtype must be BFLOAT16 or BFLOAT8_B, got {}",
        attributes.dtype);

    const auto grid = config.compute_with_storage_grid_size;
    const auto device_grid = input.device()->compute_with_storage_grid_size();
    TT_FATAL(
        grid.x >= 1 && grid.y >= 1 && grid.x <= device_grid.x && grid.y <= device_grid.y,
        "pirotski_matmul: the config's grid {} must fit the device's {}",
        grid,
        device_grid);
    const uint32_t columns = grid.x;
    const uint32_t rows = grid.y;
    const uint32_t m_tiles = m / kTileSize;
    const uint32_t k_tiles = k / kTileSize;
    const uint32_t n_tiles = n / kTileSize;
    TT_FATAL(
        n_tiles % rows == 0, "pirotski_matmul: N's {} tiles must split evenly over the grid's {} rows", n_tiles, rows);
    const uint32_t m_tiles_per_core = tt::round_up(m_tiles, columns) / columns;
    const uint32_t n_tiles_per_core = n_tiles / rows;
    TT_FATAL(
        config.N_block_size >= 1 && n_tiles_per_core % config.N_block_size == 0,
        "pirotski_matmul: N_block_size {} must divide a core's N share of {} tiles",
        config.N_block_size,
        n_tiles_per_core);
    const uint32_t n_blocks_per_core = n_tiles_per_core / config.N_block_size;
    TT_FATAL(
        config.K_block_size >= 1 && k_tiles % config.K_block_size == 0,
        "pirotski_matmul: K_block_size {} must divide K's {} tiles",
        config.K_block_size,
        k_tiles);
    TT_FATAL(config.M_block_size >= 1, "pirotski_matmul: M_block_size must be at least 1");
    TT_FATAL(
        config.subblock_h >= 1 && config.subblock_w >= 1 && config.M_block_size % config.subblock_h == 0 &&
            config.N_block_size % config.subblock_w == 0,
        "pirotski_matmul: subblock {} x {} must divide the {} x {} output block",
        config.subblock_h,
        config.subblock_w,
        config.M_block_size,
        config.N_block_size);
    // The compute kernel runs with a half-sync destination whatever dst_full_sync_en says (create_descriptor).
    auto half_sync = attributes.compute_kernel_config;
    half_sync.dst_full_sync_en = false;
    const uint32_t dest_tiles = get_dest_reg_count(half_sync);
    TT_FATAL(
        config.subblock_h * config.subblock_w <= dest_tiles,
        "pirotski_matmul: a {} x {} subblock exceeds the destination's {} tiles",
        config.subblock_h,
        config.subblock_w,
        dest_tiles);

    PirotskiMatmulLayout layout{
        .m_tiles = m_tiles,
        .k_tiles = k_tiles,
        .n_tiles = n_tiles,
        .columns = columns,
        .rows = rows,
        .m_tiles_per_core = m_tiles_per_core,
        .n_tiles_per_core = n_tiles_per_core,
        .n_blocks_per_core = n_blocks_per_core,
        .k_blocks = k_tiles / config.K_block_size,
        .main_m_tiles = m_tiles_per_core,
    };
    if (attributes.strip_m_tiles > 0) {
        layout.strip_m_tiles = attributes.strip_m_tiles;
        layout.strip_rows = device_grid.y - rows;
        TT_FATAL(
            layout.strip_m_tiles < m_tiles_per_core,
            "pirotski_matmul: strip_m_tiles {} must be below a core's M share of {} tiles",
            layout.strip_m_tiles,
            m_tiles_per_core);
        TT_FATAL(
            layout.strip_rows > 0 && rows % layout.strip_rows == 0,
            "pirotski_matmul: a strip needs device grid rows below the config grid that divide its {} rows; the "
            "device has {}",
            rows,
            device_grid.y);
        TT_FATAL(
            n_blocks_per_core == 1 && !attributes.resident_weights,
            "pirotski_matmul: a strip takes one N block a core and streamed weights");
        layout.main_m_tiles = m_tiles_per_core - layout.strip_m_tiles;
        layout.strip_lanes = rows / layout.strip_rows;
        layout.strip_n_tiles = layout.strip_lanes * n_tiles_per_core;
        // A strip core writes a packed output's rows into strip_lanes shards, whose coordinates its kernel holds in
        // fixed arrays (in1_stream.cpp).
        constexpr uint32_t kMaxStripLanes = 16;
        TT_FATAL(
            layout.strip_lanes <= kMaxStripLanes,
            "pirotski_matmul: a strip row covers {} grid rows, at most {}",
            layout.strip_lanes,
            kMaxStripLanes);
        std::tie(layout.strip_subblock_h, layout.strip_subblock_w) =
            ttnn::operations::matmul::bmm_op_utils::get_matmul_subblock_params(
                layout.strip_m_tiles, layout.strip_n_tiles, false, false, half_sync.fp32_dest_acc_en);
    }
    TT_FATAL(
        attributes.strip_m_tiles > 0 || !attributes.strip_read_ahead,
        "pirotski_matmul: strip_read_ahead needs a strip (strip_m_tiles > 0)");
    layout.m_blocks_per_core = tt::div_up(layout.main_m_tiles, config.M_block_size);
    TT_FATAL(
        !attributes.read_ahead ||
            (layout.m_blocks_per_core == 1 && n_blocks_per_core == 1 && !attributes.resident_weights),
        "pirotski_matmul: read_ahead needs one M block and one N block a core and streamed weights");

    // The split layer norm's layout: shard (x, s) holds column x's rows and K tiles [s * width, (s + 1) * width).
    const auto& input_memory = input.memory_config();
    if (input_memory.shard_spec().has_value()) {
        const auto& spec = *input_memory.shard_spec();
        const uint32_t shard_height = spec.shape[0] / kTileSize;
        const uint32_t shard_width = spec.shape[1] / kTileSize;
        // The strip's rows sit in the shards too, at an offset its reader does not take.
        if (shard_width > 0 && k_tiles % shard_width == 0 && shard_height == m_tiles_per_core &&
            layout.strip_m_tiles == 0 && is_block_grid(input_memory, columns, k_tiles / shard_width)) {
            layout.input_shard_width = shard_width;
        }
    }
    // The output's shard (x, y) is column x's M share for core (x, y)'s N share, row-major: core (x, y) packs its
    // rows in the order compute pushes them, and a partial last M block would wrap onto the shard's first rows; the
    // strip writes the share's last rows.
    const auto& output_memory = attributes.memory_config;
    if (output_memory.shard_spec().has_value() && is_block_grid(output_memory, columns, rows)) {
        const auto& spec = *output_memory.shard_spec();
        layout.direct_pack = spec.shape[0] == m_tiles_per_core * kTileSize &&
                             spec.shape[1] == n_tiles_per_core * kTileSize &&
                             layout.main_m_tiles % config.M_block_size == 0 && n_blocks_per_core == 1;
    }
    return layout;
}

void PirotskiMatmulDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    compute_layout(attributes, tensor_args);
}

PirotskiMatmulDeviceOperation::spec_return_value_t PirotskiMatmulDeviceOperation::compute_output_specs(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    using tt::tt_metal::PageConfig;
    using tt::tt_metal::TensorLayout;

    auto shape = tensor_args.input.logical_shape();
    shape[-1] = tensor_args.weight.logical_shape()[-1];
    return tt::tt_metal::TensorSpec(
        shape, TensorLayout(attributes.dtype, PageConfig(Layout::TILE), attributes.memory_config));
}

PirotskiMatmulDeviceOperation::tensor_return_value_t PirotskiMatmulDeviceOperation::create_output_tensors(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(attributes, tensor_args), tensor_args.input.device());
}

}  // namespace ttnn::experimental::nomic_embed_text_v2_moe

namespace ttnn::prim {

Tensor pirotski_matmul(
    const Tensor& input,
    const Tensor& weight,
    const std::optional<Tensor>& bias,
    const ttnn::experimental::prim::MinimalMatmulConfig& config,
    tt::tt_metal::DataType dtype,
    const tt::tt_metal::MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool resident_weights,
    uint32_t strip_m_tiles,
    bool strip_read_ahead,
    bool read_ahead) {
    using Operation = ttnn::experimental::nomic_embed_text_v2_moe::PirotskiMatmulDeviceOperation;
    return ttnn::device_operation::launch<Operation>(
        Operation::operation_attributes_t{
            .config = config,
            .dtype = dtype,
            .memory_config = memory_config,
            .compute_kernel_config = compute_kernel_config,
            .resident_weights = resident_weights,
            .strip_m_tiles = strip_m_tiles,
            .strip_read_ahead = strip_read_ahead,
            .read_ahead = read_ahead},
        Operation::tensor_args_t{.input = input, .weight = weight, .bias = bias});
}

}  // namespace ttnn::prim
