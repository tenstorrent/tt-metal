// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "fold_device_op.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::data_movement {

using tt::tt_metal::ShardSpec;

namespace {

// Synthesise an output shard spec when the user requested sharded output without a spec.
// Mirrors generate_transpose_shard_spec: uses the full compute grid with div_up/round_up so we
// always return a valid ShardSpec — any residual mis-fit surfaces from TensorSpec's own validation.
ShardSpec generate_fold_shard_spec(
    const Tensor& input_tensor, const ttnn::Shape& folded_shape, tt::tt_metal::TensorMemoryLayout requested_layout) {
    const auto grid = input_tensor.device()->compute_with_storage_grid_size();
    const CoreRangeSet all_cores(tt::tt_metal::CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
    const uint32_t num_cores = all_cores.num_cores();
    const uint32_t total_pixels = folded_shape[0] * folded_shape[1] * folded_shape[2];
    const uint32_t channels = folded_shape[3];
    const auto orientation = input_tensor.memory_config().shard_spec().has_value()
                                 ? input_tensor.memory_config().shard_spec().value().orientation
                                 : tt::tt_metal::ShardOrientation::ROW_MAJOR;

    switch (requested_layout) {
        case tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED: {
            const uint32_t shard_h = tt::div_up(total_pixels, num_cores);
            return ShardSpec(all_cores, {shard_h, channels}, orientation);
        }
        case tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED: {
            // noc_async_write_sharded splits per-pixel along W; round up to TILE_WIDTH for NOC alignment.
            const uint32_t shard_w = tt::round_up(tt::div_up(channels, num_cores), tt::constants::TILE_WIDTH);
            return ShardSpec(all_cores, {total_pixels, shard_w}, orientation);
        }
        case tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED: {
            const uint32_t shard_h = tt::div_up(total_pixels, static_cast<uint32_t>(grid.y));
            const uint32_t shard_w =
                tt::round_up(tt::div_up(channels, static_cast<uint32_t>(grid.x)), tt::constants::TILE_WIDTH);
            return ShardSpec(all_cores, {shard_h, shard_w}, orientation);
        }
        default:
            TT_THROW(
                "Fold: unsupported memory layout for shard-spec synthesis: {}", static_cast<int>(requested_layout));
    }
}

}  // namespace

bool override_compatible_with_fast_path(const std::optional<MemoryConfig>& override_mc) {
    // Fast path emits HEIGHT_SHARDED + L1 + ROW_MAJOR with a shape derived from the input; any
    // explicit shard_spec forces Path C so the requested spec is honoured byte-for-byte.
    if (!override_mc.has_value()) {
        return true;
    }
    return override_mc->is_l1() && override_mc->memory_layout() == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED &&
           !override_mc->shard_spec().has_value();
}

Fold::program_factory_t Fold::select_program_factory(
    const operation_attributes_t& op_attr, const tensor_args_t& /*tensors*/) {
    // `MultiCore` is RM-only zero-NOC; everything else uses TensorAccessor-based MultiCoreDRAMFold.
    if (op_attr.is_height_sharded_rm_fast_path) {
        return MultiCore{};
    }
    return MultiCoreDRAMFold{};
}

void validate_fold(const Fold::tensor_args_t& tensors, const Fold::operation_attributes_t& op_attr) {
    const Tensor& input_tensor = tensors.input_tensor;

    // Use logical_shape() to stay consistent with compute_output_specs (padded_shape can hide
    // a stride-non-divisible logical H/W behind TILE padding and produce wrong folded dims).
    const auto& input_shape = input_tensor.logical_shape();
    const uint32_t sh_sw = op_attr.stride_h * op_attr.stride_w;

    TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Fold: Expect input tensor to be stored on device.");
    TT_FATAL(input_tensor.buffer() != nullptr, "Fold: Expect input tensor to be allocated on a device buffer.");

    // Mathematical fold constraints — apply to every path.
    TT_FATAL(input_shape[1] % op_attr.stride_h == 0, "Fold: Input height must be divisible by stride_h.");
    TT_FATAL(input_shape[2] % op_attr.stride_w == 0, "Fold: Input width must be divisible by stride_w.");

    // Fast path requires the stride_h x stride_w neighbourhood to live on one core (RM L1 sticks).
    // Layout/sharding/buffer-type invariants are already encoded by the flag's construction.
    if (op_attr.is_height_sharded_rm_fast_path) {
        auto shard_shape = input_tensor.shard_spec().value().shape;
        TT_FATAL(
            shard_shape[0] % (input_shape[2] * op_attr.stride_h) == 0,
            "Fold (RM fast path): shard height must be divisible by input_width * stride_h.");
        return;
    }

    // Sharded-RM divide-safety for `out_shard_spec.shape[0] /= sh*sw` in compute_output_specs
    // (also inline-guarded there because validate runs after create_output_tensors).
    if (input_tensor.is_sharded() && input_tensor.layout() == Layout::ROW_MAJOR) {
        const auto& shard_shape = input_tensor.shard_spec().value().shape;
        TT_FATAL(
            shard_shape[0] % sh_sw == 0,
            "Fold (sharded RM): shard height ({}) must be divisible by stride_h*stride_w ({}).",
            shard_shape[0],
            sh_sw);
    }

    // TILE + sharded (no explicit override spec): composite's post-prim reshape does /= sh*sw on
    // the preserved shard_h. Reject inputs whose shard_h cannot fit whole folded pixel-rows.
    const bool has_override_shard_spec =
        op_attr.output_memory_config.has_value() && op_attr.output_memory_config->shard_spec().has_value();
    if (input_tensor.is_sharded() && input_tensor.layout() == Layout::TILE && !has_override_shard_spec) {
        const auto& shard_shape = input_tensor.shard_spec().value().shape;
        TT_FATAL(
            shard_shape[0] % sh_sw == 0 && shard_shape[0] >= sh_sw,
            "Fold (TILE + sharded): input shard_h ({}) must be ≥ stride_h*stride_w={} and evenly "
            "divisible by it.",
            shard_shape[0],
            sh_sw);
    }
}

void Fold::validate_on_program_cache_miss(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    validate_fold(tensors, op_attr);
}

void Fold::validate_on_program_cache_hit(const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    validate_fold(tensors, op_attr);
}

Fold::spec_return_value_t Fold::compute_output_specs(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    auto input_tensor = tensors.input_tensor;
    const ttnn::Shape& input_shape = input_tensor.logical_shape();
    auto input_dtype = input_tensor.dtype();

    tt::tt_metal::DataType output_dtype =
        (input_dtype == tt::tt_metal::DataType::FLOAT32 ||
         input_dtype == tt::tt_metal::DataType::UINT16)
            ? input_dtype
            : tt::tt_metal::DataType::BFLOAT16;

    // Folded 4D: (N, H/sh, W/sw, C*sh*sw).
    const ttnn::Shape folded_4d_shape(
        {input_shape[0],
         input_shape[1] / op_attr.stride_h,
         input_shape[2] / op_attr.stride_w,
         input_shape[3] * op_attr.stride_h * op_attr.stride_w});

    // Legacy collapsed: (1, 1, N*H/sh*W/sw, C*sh*sw).
    const ttnn::Shape collapsed_shape(
        {1,
         1,
         input_shape[0] * input_shape[1] * input_shape[2] / (op_attr.stride_h * op_attr.stride_w),
         input_shape[3] * op_attr.stride_h * op_attr.stride_w});

    if (op_attr.is_height_sharded_rm_fast_path) {
        // Fast path emits collapsed HEIGHT_SHARDED L1 RM; rescale input spec by sh*sw. Any override
        // carrying an explicit shard_spec has been routed to Path C by override_compatible_with_fast_path.
        auto shard_spec = input_tensor.shard_spec().value();
        shard_spec.shape[0] /= op_attr.stride_h * op_attr.stride_w;
        shard_spec.shape[1] *= op_attr.stride_h * op_attr.stride_w;
        auto mem_config = MemoryConfig(
            input_tensor.memory_config().memory_layout(), input_tensor.memory_config().buffer_type(), shard_spec);
        return {TensorSpec(
            collapsed_shape,
            tt::tt_metal::TensorLayout(
                output_dtype, tt::tt_metal::PageConfig(tt::tt_metal::Layout::ROW_MAJOR), mem_config))};
    }

    const bool input_is_tile = input_tensor.layout() == Layout::TILE;
    const ttnn::Shape preserved_4d_shape({input_shape[0], input_shape[1], input_shape[2], input_shape[3]});
    const auto& input_mc = input_tensor.memory_config();
    MemoryConfig output_mc = input_mc;

    if (op_attr.output_memory_config.has_value()) {
        // User-requested output config (mirrors transpose/slice/repeat): honour it.
        const auto& req = op_attr.output_memory_config.value();
        if (req.is_sharded()) {
            // Synthesiser always returns a spec; residual misfits surface from TensorSpec below.
            ShardSpec spec =
                req.shard_spec().value_or(generate_fold_shard_spec(input_tensor, folded_4d_shape, req.memory_layout()));
            if (input_is_tile) {
                // Device op writes preserved_4d; composite reshape rescales by sh*sw — invert here
                // so the final output's spec matches what the user asked for.
                const uint32_t sh_sw = op_attr.stride_h * op_attr.stride_w;
                // Inverse-rescale must divide evenly; otherwise the user's shard width is silently truncated.
                TT_FATAL(
                    spec.shape[1] % sh_sw == 0,
                    "Fold (TILE + sharded override): shard width {} not divisible by stride_h*stride_w={}; "
                    "requested spec=({}, {}).",
                    spec.shape[1],
                    sh_sw,
                    spec.shape[0],
                    spec.shape[1]);
                spec.shape[0] *= sh_sw;
                spec.shape[1] /= sh_sw;
            }
            output_mc = MemoryConfig(req.memory_layout(), req.buffer_type(), spec);
        } else {
            output_mc = req;
        }
    } else if (input_tensor.is_sharded()) {
        // Default universal-IO contract: sharded in → sharded out, same layout/grid/orientation.
        auto out_shard_spec = input_tensor.shard_spec().value();
        if (!input_is_tile) {
            const uint32_t sh_sw = op_attr.stride_h * op_attr.stride_w;
            // Guard integer-divide: shard_h < sh*sw would truncate to 0 and SIGFPE inside TensorSpec.
            TT_FATAL(
                out_shard_spec.shape[0] % sh_sw == 0 && out_shard_spec.shape[0] >= sh_sw,
                "Fold (sharded RM default): input shard_h ({}) must be ≥ stride_h*stride_w={} and evenly "
                "divisible by it. Reshape input or pass override_memory_config with an aligned shard_spec.",
                out_shard_spec.shape[0],
                sh_sw);
            out_shard_spec.shape[0] /= sh_sw;
            out_shard_spec.shape[1] *= sh_sw;
        }
        output_mc = MemoryConfig(input_mc.memory_layout(), input_mc.buffer_type(), out_shard_spec);
    }

    // Output logical shape: TILE → preserved (composite reshape finalises); sharded RM or
    // user-override or DRAM RM → folded_4d; L1 RM default → legacy collapsed (1,1,X,Y).
    ttnn::Shape output_logical_shape;
    if (input_is_tile) {
        output_logical_shape = preserved_4d_shape;
    } else if (input_tensor.is_sharded() || op_attr.output_memory_config.has_value() || input_mc.is_dram()) {
        output_logical_shape = folded_4d_shape;
    } else {
        output_logical_shape = collapsed_shape;
    }
    return {TensorSpec(
        output_logical_shape,
        tt::tt_metal::TensorLayout(output_dtype, tt::tt_metal::PageConfig(Layout::ROW_MAJOR), output_mc))};
}

Fold::tensor_return_value_t Fold::create_output_tensors(
    const operation_attributes_t& op_attr, const tensor_args_t& tensors) {
    return create_device_tensor(compute_output_specs(op_attr, tensors), tensors.input_tensor.device());
}

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::Fold::tensor_return_value_t fold(
    const ttnn::Tensor& input_tensor,
    uint32_t stride_h,
    uint32_t stride_w,
    bool is_height_sharded_rm_fast_path,
    const std::optional<tt::tt_metal::MemoryConfig>& output_memory_config) {
    using OperationType = ttnn::operations::data_movement::Fold;
    auto operation_attributes = OperationType::operation_attributes_t{
        .stride_h = stride_h,
        .stride_w = stride_w,
        .is_height_sharded_rm_fast_path = is_height_sharded_rm_fast_path,
        .output_memory_config = output_memory_config};
    auto tensor_args = OperationType::tensor_args_t{.input_tensor = input_tensor};
    return ttnn::device_operation::launch<OperationType>(operation_attributes, tensor_args);
}
}  // namespace ttnn::prim
