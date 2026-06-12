// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <ttnn/operations/pool/rotate/device/rotate_device_operation.hpp>
#include <ttnn/operations/pool/device/kernels/fixed_point_arithmetic.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>
#include <ttnn/tensor/types.hpp>
#include <ttnn/tensor/tensor_spec.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::rotate {

RotateDeviceOperation::program_factory_t RotateDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& /* tensor_args */) {
    if (operation_attributes.interpolation_mode == "bilinear") {
        return BilinearProgramFactory{};
    }
    return NearestProgramFactory{};
}

void RotateDeviceOperation::validate_inputs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    // Tensor rank
    TT_FATAL(
        input.logical_shape().rank() == 4,
        "Input tensor must be 4D (N, H, W, C), got rank {}",
        input.logical_shape().rank());

    // Layout
    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "Input tensor must be in ROW_MAJOR layout");

    // Dtype
    TT_FATAL(
        input.dtype() == DataType::BFLOAT16 || input.dtype() == DataType::FLOAT32,
        "Input tensor dtype must be bfloat16 or float32, got {}",
        input.dtype());

    // On device
    TT_FATAL(input.storage_type() == StorageType::DEVICE, "Input tensor must be on device");

    // Buffer allocated
    TT_FATAL(input.buffer() != nullptr, "Input tensor must be allocated in buffers on device");

    // Expand parameter
    TT_FATAL(
        !operation_attributes.expand,
        "expand=True is not supported. Only same-size rotation (expand=False) is implemented");

    // Interpolation mode validation
    TT_FATAL(
        operation_attributes.interpolation_mode == "nearest" || operation_attributes.interpolation_mode == "bilinear",
        "Only 'nearest' and 'bilinear' interpolation_mode are supported, got '{}'",
        operation_attributes.interpolation_mode);

    // Memory layout validation - only height sharding is supported
    if (input.is_sharded()) {
        auto mem_layout = input.memory_config().memory_layout();
        TT_FATAL(
            mem_layout == tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED,
            "Only height sharding is supported for rotate operation. Got memory layout {}",
            static_cast<int>(mem_layout));
    }

    // Wide reduction validation for bilinear mode
    if (operation_attributes.interpolation_mode == "bilinear") {
        constexpr uint32_t MAX_TILES_PER_REDUCTION = 8;
        const uint32_t input_channels = input.padded_shape()[-1];
        const uint32_t in_ntiles_c =
            static_cast<uint32_t>(std::ceil(static_cast<float>(input_channels) / tt::constants::TILE_WIDTH));
        TT_FATAL(
            in_ntiles_c <= MAX_TILES_PER_REDUCTION,
            "Wide reduction (in_ntiles_c > MAX_TILES_PER_REDUCTION) is not supported for bilinear rotate. "
            "in_ntiles_c={} exceeds MAX_TILES_PER_REDUCTION={}. Reduce channel count to <= {}.",
            in_ntiles_c,
            MAX_TILES_PER_REDUCTION,
            MAX_TILES_PER_REDUCTION * tt::constants::TILE_WIDTH);
    }
}

void RotateDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    validate_inputs(operation_attributes, tensor_args);
}

RotateDeviceOperation::spec_return_value_t RotateDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    ttnn::Shape output_shape = input.logical_shape();
    ttnn::Shape output_padded = input.padded_shape();

    if (operation_attributes.memory_config.is_sharded()) {
        if (operation_attributes.memory_config.shard_spec().has_value()) {
            auto shard_spec = operation_attributes.memory_config.shard_spec().value();
            MemoryConfig mem_config = operation_attributes.memory_config.with_shard_spec(shard_spec);
            return TensorSpec(
                output_shape,
                tt::tt_metal::TensorLayout(input.dtype(), tt::tt_metal::PageConfig(Layout::ROW_MAJOR), mem_config));
        }
        if (operation_attributes.memory_config.nd_shard_spec().has_value()) {
            return TensorSpec(
                output_shape,
                tt::tt_metal::TensorLayout(
                    input.dtype(), tt::tt_metal::PageConfig(Layout::ROW_MAJOR), operation_attributes.memory_config));
        }
    }

    return TensorSpec(
        output_shape,
        tt::tt_metal::TensorLayout::fromPaddedShape(
            input.dtype(),
            tt::tt_metal::PageConfig(Layout::ROW_MAJOR),
            operation_attributes.memory_config,
            output_shape,
            output_padded));
}

RotateDeviceOperation::tensor_return_value_t RotateDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(operation_attributes, tensor_args), tensor_args.input.device());
}

ttsl::hash::hash_t RotateDeviceOperation::compute_program_hash(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    return ttsl::hash::hash_objects_with_default_seed(
        operation_attributes.memory_config,
        operation_attributes.interpolation_mode,
        tensor_args.input.logical_shape(),
        tensor_args.input.dtype());
}

std::vector<tt::tt_metal::DynamicRuntimeArg> RotateDeviceOperation::get_dynamic_runtime_args(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /* mesh_dispatch_coordinate */) {
    using tt::tt_metal::CoreCoord;

    const auto& input_tensor = tensor_args.input;

    // --- Recompute cos/sin/center EXACTLY as both program factories do --------------------------
    const float angle_rad = operation_attributes.angle * M_PI / 180.0f;
    const float cos_angle = std::cos(angle_rad);
    const float sin_angle = std::sin(angle_rad);

    const auto& input_shape = input_tensor.padded_shape();
    const uint32_t input_batch = input_shape[0];
    const uint32_t input_height = input_shape[1];
    const uint32_t input_width = input_shape[2];

    float center_x = 0.0f;
    float center_y = 0.0f;
    if (operation_attributes.center.has_value()) {
        center_x = std::get<0>(operation_attributes.center.value()) - 0.5f;
        center_y = std::get<1>(operation_attributes.center.value()) - 0.5f;
    } else {
        center_x = (static_cast<float>(input_width) - 1.0f) / 2.0f;
        center_y = (static_cast<float>(input_height) - 1.0f) / 2.0f;
    }

    const bool is_bilinear = operation_attributes.interpolation_mode == "bilinear";

    // Fill value bits: bilinear keeps the float32 bit pattern for fp32 inputs and bf16 bits for bf16
    // inputs; nearest always emits the bf16 bit pattern (matches the respective factory).
    uint32_t fill_value_bits = 0;
    {
        const bfloat16 bf16_value(operation_attributes.fill);
        const uint16_t fill_bf16 = std::bit_cast<uint16_t>(bf16_value);
        if (is_bilinear) {
            const bool is_bfloat16 = input_tensor.dtype() == DataType::BFLOAT16;
            fill_value_bits =
                is_bfloat16 ? static_cast<uint32_t>(fill_bf16) : std::bit_cast<uint32_t>(operation_attributes.fill);
        } else {
            fill_value_bits = static_cast<uint32_t>(fill_bf16);
        }
    }

    const uint32_t cos_angle_q16 = static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(cos_angle));
    const uint32_t sin_angle_q16 = static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(sin_angle));
    const uint32_t center_x_q16 = static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(center_x));
    const uint32_t center_y_q16 = static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(center_y));

    // --- Re-derive the work-core list EXACTLY as the selected factory does ----------------------
    // The reader is pushed first into desc.kernels in both factories => reader kernel index == 0.
    // Its angle-derived args occupy reader runtime args [3]=cos, [4]=sin, [5]=center_x, [6]=center_y,
    // [7]=fill (args [0..2] are buffer*/num_sticks/start_stick_id and are NOT re-applied here).
    constexpr uint32_t kReaderKernelIdx = 0;
    constexpr uint32_t kCosArgIdx = 3;
    constexpr uint32_t kSinArgIdx = 4;
    constexpr uint32_t kCenterXArgIdx = 5;
    constexpr uint32_t kCenterYArgIdx = 6;
    constexpr uint32_t kFillArgIdx = 7;

    const bool is_input_sharded = input_tensor.is_sharded();
    const bool is_nd_sharded = input_tensor.memory_config().nd_shard_spec().has_value();

    std::vector<CoreCoord> logical_cores;
    uint32_t num_cores = 0;

    auto* device = input_tensor.device();
    const uint32_t total_output_sticks = input_batch * input_height * input_width;

    if (is_bilinear) {
        // Mirror BilinearProgramFactory::create_descriptor's work-core derivation.
        // Note: bilinear has no ND-sharded path; it keys off input.shard_spec() then output.shard_spec().
        if (is_input_sharded) {
            const auto input_shard_spec = input_tensor.shard_spec().value();
            num_cores = input_shard_spec.num_cores();
            logical_cores = tt::tt_metal::corerange_to_cores(
                input_shard_spec.grid,
                num_cores,
                input_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
        } else if (tensor_return_value.is_sharded()) {
            // Output-sharded, input interleaved: factory derives cores from the OUTPUT shard spec.
            const auto output_shard_spec = tensor_return_value.shard_spec().value();
            const uint32_t output_nsticks_per_core = output_shard_spec.shape[0];
            num_cores = (total_output_sticks + output_nsticks_per_core - 1) / output_nsticks_per_core;
            logical_cores = tt::tt_metal::corerange_to_cores(
                output_shard_spec.grid,
                num_cores,
                output_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
        } else {
            const auto compute_grid_size = device->compute_with_storage_grid_size();
            auto [num_cores_used, all_cores_range, cg1, cg2, ns1, ns2] =
                tt::tt_metal::split_work_to_cores(compute_grid_size, total_output_sticks);
            num_cores = num_cores_used;
            logical_cores = tt::tt_metal::corerange_to_cores(all_cores_range, num_cores, true);
        }
    } else {
        // Mirror NearestProgramFactory::create_descriptor's work-core derivation.
        if (is_input_sharded && !is_nd_sharded) {
            const auto input_shard_spec = input_tensor.shard_spec().value();
            num_cores = input_shard_spec.num_cores();
            logical_cores = tt::tt_metal::corerange_to_cores(
                input_shard_spec.grid,
                num_cores,
                input_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
        } else if (is_nd_sharded) {
            const auto& nd_shard_spec = input_tensor.memory_config().nd_shard_spec().value();
            num_cores = nd_shard_spec.grid.num_cores();
            logical_cores = tt::tt_metal::corerange_to_cores(
                nd_shard_spec.grid, num_cores, nd_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR);
        } else {
            const auto compute_grid_size = device->compute_with_storage_grid_size();
            auto [num_cores_used, all_cores_range, cg1, cg2, ns1, ns2] =
                tt::tt_metal::split_work_to_cores(compute_grid_size, total_output_sticks);
            num_cores = num_cores_used;
            logical_cores = tt::tt_metal::corerange_to_cores(all_cores_range, num_cores, true);
        }
    }

    std::vector<tt::tt_metal::DynamicRuntimeArg> dynamic_args;
    dynamic_args.reserve(static_cast<size_t>(num_cores) * 5);
    for (uint32_t i = 0; i < num_cores; ++i) {
        const CoreCoord& core = logical_cores[i];
        dynamic_args.push_back({kReaderKernelIdx, core, kCosArgIdx, cos_angle_q16});
        dynamic_args.push_back({kReaderKernelIdx, core, kSinArgIdx, sin_angle_q16});
        dynamic_args.push_back({kReaderKernelIdx, core, kCenterXArgIdx, center_x_q16});
        dynamic_args.push_back({kReaderKernelIdx, core, kCenterYArgIdx, center_y_q16});
        dynamic_args.push_back({kReaderKernelIdx, core, kFillArgIdx, fill_value_bits});
    }
    return dynamic_args;
}

}  // namespace ttnn::operations::rotate

namespace ttnn::prim {

ttnn::Tensor rotate(
    const Tensor& input,
    float angle,
    const std::optional<std::tuple<float, float>>& center,
    float fill,
    bool expand,
    const std::string& interpolation_mode,
    const std::optional<MemoryConfig>& memory_config) {
    using Op = ttnn::operations::rotate::RotateDeviceOperation;
    return ttnn::device_operation::launch<Op>(
        Op::operation_attributes_t{
            angle, center, fill, expand, interpolation_mode, memory_config.value_or(input.memory_config())},
        Op::tensor_args_t{input});
}

}  // namespace ttnn::prim
