// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "groupnorm_device_operation.hpp"
#include <tt-metalium/constants.hpp>

#include "ttnn/tensor/tensor_ops.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/core/program_cache_l1.hpp"
#include "ttnn/operations/normalization/groupnorm/groupnorm_grid_utils.hpp"
#include "ttnn/operations/normalization/groupnorm/device/groupnorm_program_utils.hpp"
#include "ttnn/operations/normalization/shard_spec_validation.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

GroupNormDeviceOperation::program_factory_t GroupNormDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    if (input.is_sharded()) {
        return GroupNormDeviceOperation::GroupNormShardedProgramFactory{};
    }

    // For non-sharded: determine if we need mcast or no-mcast based on batch vs virtual rows
    const auto& program_config = std::get<GroupNormMultiCoreProgramConfig>(args.program_config);
    CoreCoord grid_size = program_config.compute_with_storage_grid_size;
    uint32_t batch = input.padded_shape()[0];
    uint32_t W = input.padded_shape()[3];
    uint32_t num_virtual_cols =
        ttnn::operations::normalization::compute_num_virtual_cols(grid_size.x, args.num_groups, W);
    TT_FATAL(
        num_virtual_cols > 0,
        "group_norm: No valid num_virtual_cols for grid_x={}, num_groups={}, W={}. "
        "Channels must be aligned to tile width and divisible by num_groups.",
        grid_size.x,
        args.num_groups,
        W);

    uint32_t num_actual_rows = grid_size.y;
    uint32_t num_virtual_rows = (grid_size.x / num_virtual_cols) * num_actual_rows;

    if (batch >= num_virtual_rows) {
        return GroupNormDeviceOperation::GroupNormNoMcastProgramFactory{};
    }
    return GroupNormDeviceOperation::GroupNormMcastProgramFactory{};
}

GroupNormInterleavedPlan GroupNormDeviceOperation::select_interleaved_plan(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;
    if (input.is_sharded() || !args.use_welford) {
        return {};
    }

    const auto& config = std::get<GroupNormMultiCoreProgramConfig>(args.program_config);
    const auto& shape = input.padded_shape();
    const std::uint32_t tile_height = input.tensor_spec().tile().get_height();
    const std::uint32_t tile_width = input.tensor_spec().tile().get_width();
    const std::uint32_t num_batches = shape[0];
    const std::uint32_t height = shape[1] * shape[2] * num_batches;
    const std::uint32_t width = shape[3];
    const std::uint32_t num_groups = args.num_groups;
    const auto grid = config.compute_with_storage_grid_size;
    if (tile_height == 0 || tile_width == 0 || height == 0 || width == 0 || num_groups == 0 || grid.x == 0 ||
        grid.y == 0) {
        return {};
    }
    const std::uint32_t height_tiles = height / tile_height;

    std::uint32_t num_virtual_cols = std::min<std::uint32_t>(grid.x, num_groups);
    while (num_virtual_cols > 0 &&
           ((width / num_virtual_cols) % tile_width != 0 || num_groups % num_virtual_cols != 0)) {
        --num_virtual_cols;
    }
    if (num_virtual_cols == 0) {
        return {};
    }
    const std::uint32_t num_virtual_rows = (grid.x / num_virtual_cols) * grid.y;
    if (height_tiles < num_virtual_rows || height_tiles % num_virtual_rows != 0) {
        return {};
    }
    std::uint32_t per_core_height_tiles = height_tiles / num_virtual_rows;
    std::uint32_t per_core_height = per_core_height_tiles * tile_height;
    const std::uint32_t per_core_width = width / num_virtual_cols;
    const std::uint32_t per_core_width_tiles = (per_core_width + tile_width - 1) / tile_width;
    const std::uint32_t channels_per_group = width / num_groups;
    const std::uint32_t num_row_shards = height / per_core_height;
    std::uint32_t batches_group_1 = num_batches > num_row_shards ? num_batches / num_row_shards : 1;
    std::uint32_t batches_group_2 = batches_group_1;
    const std::uint32_t num_col_shards = width / per_core_width;
    const std::uint32_t groups_per_core = num_groups > num_col_shards ? num_groups / num_col_shards : 1;
    const std::uint32_t block_width_tiles = find_max_tile_span(per_core_width, channels_per_group).first;
    std::uint32_t block_height_group_1 = per_core_height_tiles / batches_group_1;
    std::uint32_t block_height_group_2 = 0;

    bool equal_batches_per_core = true;
    if (num_batches >= num_row_shards) {
        equal_batches_per_core = num_batches % num_row_shards == 0;
    }
    if (!equal_batches_per_core) {
        batches_group_2 = num_batches / num_row_shards;
        batches_group_1 = batches_group_2 + 1;
        const std::uint32_t per_batch_tiles = height_tiles / num_batches;
        block_height_group_1 = per_batch_tiles;
        block_height_group_2 = per_batch_tiles;
    }

    std::uint32_t num_out_blocks = config.num_out_blocks;
    if (num_out_blocks == static_cast<std::uint32_t>(-1)) {
        num_out_blocks =
            groupnorm_heuristic_num_out_blocks(shape[1] * shape[2] * shape[3], num_virtual_cols * num_virtual_rows);
    }
    if (num_out_blocks == 0 || num_out_blocks > block_height_group_1 ||
        (block_height_group_2 > 0 && num_out_blocks > block_height_group_2)) {
        return {};
    }

    const auto input_format = datatype_to_dataformat_converter(input.dtype());
    const auto output_format = datatype_to_dataformat_converter(config.out_data_format);
    const auto intermediate_format = datatype_to_dataformat_converter(config.im_data_format);
    auto gamma_beta_format = tt::DataFormat::Float16_b;
    if (tensor_args.gamma.has_value()) {
        gamma_beta_format = datatype_to_dataformat_converter(tensor_args.gamma->dtype());
    }
    if (tensor_args.beta.has_value()) {
        gamma_beta_format = datatype_to_dataformat_converter(tensor_args.beta->dtype());
    }
    const auto mask_format = tensor_args.input_mask.has_value()
                                 ? datatype_to_dataformat_converter(tensor_args.input_mask->dtype())
                                 : tt::DataFormat::Float16_b;
    const std::uint32_t input_tile_size = tt::tile_size(input_format);
    const std::uint32_t output_tile_size = tt::tile_size(output_format);
    const std::uint32_t intermediate_tile_size = tt::tile_size(intermediate_format);
    const std::uint32_t gamma_beta_tile_size = tt::tile_size(gamma_beta_format);
    const std::uint32_t mask_tile_size = tt::tile_size(mask_format);
    const std::uint32_t epsilon_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    const bool reader_repack_output = per_core_width % tile_width != 0;
    const bool untilize_output = config.output_layout == Layout::ROW_MAJOR;
    const std::uint32_t input_mask_size = block_width_tiles * groups_per_core * mask_tile_size;
    const std::uint32_t repack_size = per_core_width_tiles * input_tile_size * 2;
    const std::uint32_t gamma_size = per_core_width_tiles * gamma_beta_tile_size;
    const std::uint32_t beta_size = per_core_width_tiles * gamma_beta_tile_size;
    const std::uint32_t partial_stats_size = 2 * intermediate_tile_size;
    const std::uint32_t global_stats_size = partial_stats_size * groups_per_core;
    const std::uint32_t normalisation_stats_size = intermediate_tile_size * groups_per_core;

    const auto fits_group = [&](std::uint32_t block_height) {
        if (block_height == 0) {
            return false;
        }
        const std::uint32_t block_tiles = block_height / num_out_blocks * block_width_tiles;
        const std::uint32_t input_staging_size = block_tiles * input_tile_size;
        const GroupNormInterleavedCbFootprint footprint{
            .output = block_tiles * output_tile_size,
            .input_staging = input_staging_size,
            .untilize_output = untilize_output ? input_staging_size : 0,
            .scaler = intermediate_tile_size,
            .epsilon = epsilon_tile_size,
            .column_scaler = intermediate_tile_size,
            .gamma = tensor_args.gamma.has_value() ? gamma_size : 0,
            .beta = tensor_args.beta.has_value() ? beta_size : 0,
            .input_mask = input_mask_size,
            .repack = reader_repack_output ? repack_size : 0,
            .x = intermediate_tile_size,
            .xmm = 2 * intermediate_tile_size,
            .xmm2 = block_tiles * intermediate_tile_size,
            .xmm3 = block_tiles * intermediate_tile_size,
            .partial_stats = partial_stats_size,
            .global_stats = global_stats_size,
            .normalisation_stats = normalisation_stats_size,
        };
        const std::uint64_t replay_size =
            static_cast<std::uint64_t>(block_height) * per_core_width_tiles * input_tile_size;
        return footprint.total_with_input(replay_size) <
               ttnn::operations::core::usable_program_l1_capacity(input.device());
    };

    return {
        .replay_group_1 = fits_group(block_height_group_1),
        .replay_group_2 = fits_group(block_height_group_2),
    };
}

ttsl::hash::hash_t GroupNormDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    if (tensor_args.input.is_sharded()) {
        return ttsl::hash::hash_objects_with_default_seed(
            ttsl::hash::type_hash<GroupNormDeviceOperation>, operation_attributes, tensor_args);
    }
    const auto plan = select_interleaved_plan(operation_attributes, tensor_args);
    return ttsl::hash::hash_objects_with_default_seed(
        ttsl::hash::type_hash<GroupNormDeviceOperation>,
        operation_attributes,
        tensor_args,
        plan.replay_group_1,
        plan.replay_group_2);
}

void GroupNormDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& a = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;
    const auto& input_mask = tensor_args.input_mask;
    const auto& negative_mask = tensor_args.negative_mask;
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();

    TT_FATAL(
        a.dtype() == DataType::BFLOAT16 || a.dtype() == DataType::FLOAT32,
        "Input tensor must be BFLOAT16 or FLOAT32, got: {}",
        a.dtype());
    TT_FATAL(a.storage_type() == StorageType::DEVICE, "Operands to groupnorm need to be on device!");
    TT_FATAL(a.buffer() != nullptr, "Operands to groupnorm need to be allocated in buffers on device!");
    TT_FATAL(
        !(args.use_welford && a.device()->arch() == tt::ARCH::QUASAR),
        "group_norm with use_welford=True is not supported on Quasar; the two-pass SFPU implementation currently "
        "supports Wormhole and Blackhole only.");
    if (a.layout() == Layout::TILE) {
        TT_FATAL(
            tile_height == tt::constants::TILE_HEIGHT && tile_width == tt::constants::TILE_WIDTH,
            "GroupNorm TILE input requires tile shape {}x{}, got: {}x{}",
            tt::constants::TILE_HEIGHT,
            tt::constants::TILE_WIDTH,
            tile_height,
            tile_width);
    }
    TT_FATAL(a.padded_shape()[3] % args.num_groups == 0, "channel must be divisible by num_groups!");
    TT_FATAL(a.padded_shape()[1] == 1, "input tensor shape[1] must be 1!");
    TT_FATAL(
        (a.padded_shape()[1] * a.padded_shape()[2]) % tile_height == 0,
        "H*W ({}*{}) must be a multiple of the tile height ({})",
        a.padded_shape()[1],
        a.padded_shape()[2],
        tile_height);

    // ROW_MAJOR (interleaved) input/output is only supported on the tile-reduction group_norm path.
    if (args.use_welford && !a.is_sharded()) {
        const Layout output_layout =
            std::visit([](const auto& config) -> Layout { return config.output_layout; }, args.program_config);
        TT_FATAL(
            a.layout() == Layout::TILE && output_layout == Layout::TILE,
            "group_norm: ROW_MAJOR interleaved input/output is not supported on the SFPU two-pass path yet. "
            "Use TILE layout for both input and output, or use the tile-reduction path "
            "(use_welford=false).");
    }

    if (a.is_sharded()) {
        const auto& shard_spec = a.shard_spec().value();
        const auto bbox = shard_spec.grid.bounding_box();
        const auto bbox_grid = ttnn::operations::normalization::core_grid_from_shard_bounding_box(bbox);
        const uint32_t bbox_num_cores = bbox_grid.x * bbox_grid.y;
        TT_FATAL(
            shard_spec.grid.num_cores() == bbox_num_cores,
            "Sharded groupnorm does not support non-rectangular core grids. "
            "The shard spec grid has {} cores but its bounding box spans {} cores ({} x {}).",
            shard_spec.grid.num_cores(),
            bbox_num_cores,
            bbox_grid.x,
            bbox_grid.y);

        const auto program_grid =
            std::visit([](const auto& config) { return config.compute_with_storage_grid_size; }, args.program_config);
        ttnn::operations::normalization::detail::validate_sharded_input(
            a, program_grid, /*require_shard_width_tile_aligned=*/false);
    }
    if (gamma.has_value()) {
        if (gamma.value().layout() == Layout::TILE) {
            TT_FATAL(
                a.padded_shape()[3] == gamma.value().padded_shape()[3],
                "{} != {}",
                a.padded_shape()[3],
                gamma.value().padded_shape()[3]);
            TT_FATAL(a.device() == gamma.value().device(), "Input and gamma tensors must be on same device");
            TT_FATAL(
                gamma.value().buffer() != nullptr, "Operands to groupnorm need to be allocated in buffers on device!");
            TT_FATAL(
                gamma.value().padded_shape()[2] == tile_height,
                "Gamma tensor height must equal tile height ({}), got: {}",
                tile_height,
                gamma.value().padded_shape()[2]);
        } else {
            TT_FATAL(
                gamma.value().layout() == Layout::ROW_MAJOR,
                "Gamma tensor must have ROW_MAJOR layout, got: {}",
                gamma.value().layout());
            TT_FATAL(
                (gamma.value().padded_shape()[3] == tile_width),
                "Gamma tensor inner dimension must equal tile width ({}), got: {}",
                tile_width,
                gamma.value().padded_shape()[3]);
            TT_FATAL(a.device() == gamma.value().device(), "Input and gamma tensors must be on same device");
            TT_FATAL(
                gamma.value().buffer() != nullptr, "Operands to groupnorm need to be allocated in buffers on device!");
            TT_FATAL(
                gamma.value().dtype() == DataType::BFLOAT16 || gamma.value().dtype() == DataType::FLOAT32,
                "Gamma tensor must be BFLOAT16 or FLOAT32, got: {}",
                gamma.value().dtype());
        }
        if (beta.has_value()) {
            TT_FATAL(
                gamma.value().layout() == beta.value().layout(),
                "Gamma and beta must have the same layout, got gamma: {} vs beta: {}",
                gamma.value().layout(),
                beta.value().layout());
            TT_FATAL(
                gamma.value().dtype() == beta.value().dtype(),
                "Gamma and beta must have the same dtype (the program factories use a single gamma/beta "
                "CB format for both), got gamma: {} vs beta: {}",
                gamma.value().dtype(),
                beta.value().dtype());
        }
    }

    if (beta.has_value()) {
        if (beta.value().layout() == Layout::TILE) {
            TT_FATAL(
                a.padded_shape()[3] == beta.value().padded_shape()[3],
                "Input and beta inner dimensions must match, got input: {} vs beta: {}",
                a.padded_shape()[3],
                beta.value().padded_shape()[3]);
            TT_FATAL(a.device() == beta.value().device(), "Input and beta tensors must be on same device");
            TT_FATAL(
                beta.value().buffer() != nullptr, "Operands to groupnorm need to be allocated in buffers on device!");
            TT_FATAL(
                beta.value().padded_shape()[2] == tile_height,
                "Beta tensor height must equal tile height ({}), got: {}",
                tile_height,
                beta.value().padded_shape()[2]);
        } else {
            TT_FATAL(
                beta.value().layout() == Layout::ROW_MAJOR,
                "Beta tensor must have ROW_MAJOR layout, got: {}",
                beta.value().layout());
            TT_FATAL(
                beta.value().padded_shape()[3] == tile_width,
                "Beta tensor inner dimension must equal tile width ({}), got: {}",
                tile_width,
                beta.value().padded_shape()[3]);
            TT_FATAL(a.device() == beta.value().device(), "Input and beta tensors must be on same device");
            TT_FATAL(
                beta.value().buffer() != nullptr, "Operands to groupnorm need to be allocated in buffers on device!");
            TT_FATAL(
                beta.value().dtype() == DataType::BFLOAT16 || beta.value().dtype() == DataType::FLOAT32,
                "Beta tensor must be BFLOAT16 or FLOAT32, got: {}",
                beta.value().dtype());
        }
    }

    if (input_mask.has_value()) {
        TT_FATAL(
            input_mask.value().layout() == Layout::TILE,
            "Input mask must have TILE layout, got: {}",
            input_mask.value().layout());
        TT_FATAL(
            input_mask.value().storage_type() == StorageType::DEVICE,
            "Input mask must be on device, got storage type: {}",
            input_mask.value().storage_type());
        TT_FATAL(input_mask.value().buffer() != nullptr, "Input mask must be allocated in buffers on device!");
        TT_FATAL(a.device() == input_mask.value().device(), "Input and input mask tensors must be on same device");
        // For non-tile-aligned H*W on the tile-reduction path the mask carries a second, row-masked copy
        // of every group; that is the only reason dim1 may be 2 * num_groups.
        const bool row_mask_doubled = !args.use_welford && (a.logical_shape()[2] != a.padded_shape()[2]);
        const uint32_t expected_mask_groups = args.num_groups * (row_mask_doubled ? 2 : 1);
        TT_FATAL(
            input_mask.value().padded_shape()[1] == expected_mask_groups,
            "Input mask dim1 must be {} ({}num_groups={}), got: {}",
            expected_mask_groups,
            row_mask_doubled ? "2 x " : "",
            args.num_groups,
            input_mask.value().padded_shape()[1]);
        TT_FATAL(
            input_mask.value().padded_shape()[2] == tile_height,
            "Input mask height must equal tile height ({}), got: {}",
            tile_height,
            input_mask.value().padded_shape()[2]);
        TT_FATAL(
            input_mask.value().padded_shape()[3] % tile_width == 0,
            "Input mask inner dimension must be divisible by tile width ({}), got: {}",
            tile_width,
            input_mask.value().padded_shape()[3]);
    }

    // Negative mask tensor is used to reduce the number of CB's used in the sharded version of the kernel by
    // overlapping the CB's used for tilized input and output. (The kernel is in fact row major variant, but is
    // internally tilizing RM into tilized inputs) Valid only if sharded program is used, and input and output tensors
    // are in row major layout.
    if (negative_mask.has_value()) {
        TT_FATAL(
            negative_mask.value().layout() == Layout::TILE,
            "Negative mask must have TILE layout, got: {}",
            negative_mask.value().layout());
        TT_FATAL(
            negative_mask.value().storage_type() == StorageType::DEVICE,
            "Negative mask must be on device, got storage type: {}",
            negative_mask.value().storage_type());
        TT_FATAL(
            negative_mask.value().buffer() != nullptr, "Negative mask must be allocated in buffers on device!");
        TT_FATAL(
            a.device() == negative_mask.value().device(), "Input and negative mask tensors must be on same device");
        TT_FATAL(
            negative_mask.value().padded_shape()[1] == args.num_groups,
            "Negative mask padded shape[1] must be equal to num_groups, but is {} and num_groups is {}",
            negative_mask.value().padded_shape()[1],
            args.num_groups);
        TT_FATAL(
            negative_mask.value().padded_shape()[2] == tile_height,
            "Negative mask padded shape[2] must equal tile height, but is {} and tile_height is {}",
            negative_mask.value().padded_shape()[2],
            tile_height);
        TT_FATAL(
            negative_mask.value().padded_shape()[3] % tile_width == 0,
            "Negative mask padded shape[3] must be divisible by tile_width, but is {} and tile_width is {}",
            negative_mask.value().padded_shape()[3],
            tile_width);
        TT_FATAL(a.is_sharded(), "Negative mask support is only available for sharded input tensors.");
        // The Welford compute kernels have no negative-mask path.
        TT_FATAL(!args.use_welford, "Negative mask is not supported with use_welford=True.");
        TT_FATAL(
            a.layout() == Layout::ROW_MAJOR,
            "If using negative mask, input tensor must be in ROW_MAJOR layout, but layout is {}",
            a.layout());
        Layout output_layout =
            std::visit([](const auto& config) -> Layout { return config.output_layout; }, args.program_config);
        TT_FATAL(
            output_layout == Layout::ROW_MAJOR,
            "If using negative mask, output tensor must be in ROW_MAJOR layout, but layout is {}",
            output_layout);
    }

    // synthesize_negative_mask makes the sharded writer kernel build the per-group
    // selector directly in L1 instead of reading a tensor.
    // This attribute is determined by ttnn::group_norm itself (see needs_negative_mask_overlap)
    if (args.synthesize_negative_mask) {
        TT_FATAL(
            a.is_sharded(),
            "group_norm: synthesize_negative_mask is only supported for sharded inputs (the interleaved factories "
            "have no negative-mask code path).");
        // The Welford kernels have no negative-mask path at all.
        TT_FATAL(!args.use_welford, "group_norm: synthesize_negative_mask is not supported with use_welford=True.");
        // Mirrors the layout requirements enforced above for a caller-supplied negative_mask tensor.
        TT_FATAL(
            a.layout() == Layout::ROW_MAJOR,
            "group_norm: synthesize_negative_mask requires a ROW_MAJOR input tensor, but layout is {}",
            a.layout());
        Layout output_layout =
            std::visit([](const auto& config) -> Layout { return config.output_layout; }, args.program_config);
        TT_FATAL(
            output_layout == Layout::ROW_MAJOR,
            "group_norm: synthesize_negative_mask requires a ROW_MAJOR output layout, but layout is {}",
            output_layout);
    }

    // For non-sharded DRAM tensors, validate that the grid produces uniform
    // multicast groups.  Non-uniform groups cause a deadlock because the sender
    // kernel waits for an exact semaphore count equal to (group_size - 1).
    if (!a.is_sharded()) {
        if (const auto* mc_config = std::get_if<GroupNormMultiCoreProgramConfig>(&args.program_config)) {
            CoreCoord grid_size = mc_config->compute_with_storage_grid_size;
            uint32_t W = a.padded_shape()[3];
            uint32_t num_batches = a.padded_shape()[0];
            uint32_t nvc = ttnn::operations::normalization::compute_num_virtual_cols(grid_size.x, args.num_groups, W);
            if (nvc > 0) {
                uint32_t num_virtual_rows = (grid_size.x / nvc) * grid_size.y;
                TT_FATAL(
                    num_virtual_rows < num_batches || num_virtual_rows % num_batches == 0,
                    "group_norm: The core grid (x={}, y={}) produces num_virtual_rows={} which is not "
                    "divisible by num_batches={}. This creates non-uniform multicast groups and will "
                    "deadlock. Use determine_expected_group_norm_dram_grid_size() with num_batches to select a valid "
                    "grid.",
                    grid_size.x,
                    grid_size.y,
                    num_virtual_rows,
                    num_batches);
            }
        }
    }
}

tt::tt_metal::TensorSpec GroupNormDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;

    return std::visit(
        [&](const auto& program_config) -> spec_return_value_t {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if (program_config.inplace) {
                if constexpr (std::is_same_v<ProgramConfigType, GroupNormShardedMultiCoreProgramConfig>) {
                    return input_tensor.tensor_spec();
                } else {
                    TT_THROW("inplace groupnorm not supported for unsharded tensors");
                }
            }

            auto mem_config = args.output_mem_config;
            return tt::tt_metal::TensorSpec(
                input_tensor.logical_shape(),
                TensorLayout::fromPaddedShape(
                    program_config.out_data_format,
                    PageConfig(program_config.output_layout),
                    mem_config,
                    input_tensor.logical_shape(),
                    input_tensor.padded_shape()));
        },
        args.program_config);
}

Tensor GroupNormDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;

    return std::visit(
        [&](const auto& program_config) -> tensor_return_value_t {
            using ProgramConfigType = std::decay_t<decltype(program_config)>;
            if (program_config.inplace) {
                if constexpr (std::is_same_v<ProgramConfigType, GroupNormShardedMultiCoreProgramConfig>) {
                    return input_tensor;
                } else {
                    TT_THROW("inplace groupnorm not supported for unsharded tensors");
                }
            }
            return create_device_tensor(compute_output_specs(args, tensor_args), input_tensor.device());
        },
        args.program_config);
}

Tensor group_norm(
    const Tensor& input,
    float eps,
    uint32_t num_groups,
    const MemoryConfig& output_mem_config,
    const GroupNormProgramConfig& program_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool use_welford,
    std::optional<Tensor> gamma,
    std::optional<Tensor> beta,
    std::optional<Tensor> input_mask,
    std::optional<Tensor> negative_mask,
    bool synthesize_negative_mask) {
    if (negative_mask.has_value()) {
        TT_FATAL(
            negative_mask.value().storage_type() == StorageType::DEVICE,
            "Negative mask must be on device, got storage type: {}",
            negative_mask.value().storage_type());
        TT_FATAL(
            negative_mask.value().buffer() != nullptr, "Negative mask must be allocated in buffers on device!");
        TT_FATAL(input.device() == negative_mask.value().device(), "Input and negative mask tensors must be on same device");
    }
    TT_FATAL(
        !(synthesize_negative_mask && negative_mask.has_value()),
        "synthesize_negative_mask=True is mutually exclusive with a caller-supplied negative_mask tensor.");
    using OperationType = GroupNormDeviceOperation;
    auto operation_attributes = OperationType::operation_attributes_t{
        .eps = eps,
        .num_groups = num_groups,
        .output_mem_config = output_mem_config,
        .program_config = program_config,
        .compute_kernel_config = compute_kernel_config,
        .use_welford = use_welford,
        .synthesize_negative_mask = synthesize_negative_mask,
    };
    auto tensor_args = OperationType::tensor_args_t{
        .input = input,
        .gamma = std::move(gamma),
        .beta = std::move(beta),
        .input_mask = std::move(input_mask),
        .negative_mask = std::move(negative_mask)};

    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}

}  // namespace ttnn::prim
