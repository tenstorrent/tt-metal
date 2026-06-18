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
            MemoryConfig mem_config = MemoryConfig(
                operation_attributes.memory_config.memory_layout(),
                operation_attributes.memory_config.buffer_type(),
                shard_spec);
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

RotatePerCoreArgs compute_rotate_per_core_args(
    const RotateDeviceOperation::operation_attributes_t& operation_attributes,
    const Tensor& input,
    const Tensor& output,
    bool is_bilinear) {
    using tt::tt_metal::CoreCoord;

    RotatePerCoreArgs result;

    // --- Angle/center/fill scalars (identical in both factories) --------------------------------
    const float angle_rad = operation_attributes.angle * M_PI / 180.0f;
    const float cos_angle = std::cos(angle_rad);
    const float sin_angle = std::sin(angle_rad);

    const auto& input_shape = input.padded_shape();
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

    // Fill value bits: bilinear keeps the float32 bit pattern for fp32 inputs and bf16 bits for bf16
    // inputs; nearest always emits the bf16 bit pattern.
    {
        const bfloat16 bf16_value(operation_attributes.fill);
        const uint16_t fill_bf16 = std::bit_cast<uint16_t>(bf16_value);
        if (is_bilinear) {
            const bool is_bfloat16 = input.dtype() == DataType::BFLOAT16;
            result.fill_value_bits =
                is_bfloat16 ? static_cast<uint32_t>(fill_bf16) : std::bit_cast<uint32_t>(operation_attributes.fill);
        } else {
            result.fill_value_bits = static_cast<uint32_t>(fill_bf16);
        }
    }

    result.cos_angle_q16 = static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(cos_angle));
    result.sin_angle_q16 = static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(sin_angle));
    result.center_x_q16 = static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(center_x));
    result.center_y_q16 = static_cast<uint32_t>(fixed_point_arithmetic::float_to_fixed(center_y));

    // --- Work-core layout (matches the selected factory's derivation) ---------------------------
    const bool is_input_sharded = input.is_sharded();
    const bool is_nd_sharded = input.memory_config().nd_shard_spec().has_value();
    const uint32_t total_output_sticks = input_batch * input_height * input_width;
    auto* device = input.device();

    std::vector<CoreCoord>& logical_cores = result.cores;

    if (is_input_sharded && (is_bilinear || !is_nd_sharded)) {
        // Input-sharded path: shared by both factories (bilinear has no ND-sharded path, so it always
        // takes this branch when input is sharded; nearest takes it only when not ND-sharded).
        const auto input_shard_spec = input.shard_spec().value();
        const uint32_t num_cores = input_shard_spec.num_cores();
        const bool row_major = input_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR;
        logical_cores = tt::tt_metal::corerange_to_cores(input_shard_spec.grid, num_cores, row_major);

        const uint32_t input_nsticks_per_core = input_shard_spec.shape[0];
        const bool is_block_sharded =
            input.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::BLOCK_SHARDED;
        const bool is_width_sharded =
            input.memory_config().memory_layout() == tt::tt_metal::TensorMemoryLayout::WIDTH_SHARDED;
        const uint32_t num_cores_x = input_shard_spec.grid.bounding_box().grid_size().x;

        result.num_sticks.resize(num_cores, input_nsticks_per_core);
        result.start_stick_id.resize(num_cores);
        for (uint32_t i = 0; i < num_cores; ++i) {
            if (is_width_sharded) {
                result.start_stick_id[i] = 0;
            } else if (is_block_sharded) {
                result.start_stick_id[i] = (i / num_cores_x) * input_nsticks_per_core;
            } else {
                result.start_stick_id[i] = i * input_nsticks_per_core;
            }
        }
    } else if (!is_bilinear && is_nd_sharded) {
        // Nearest ND-sharded path.
        const auto& nd_shard_spec = input.memory_config().nd_shard_spec().value();
        const uint32_t num_cores = nd_shard_spec.grid.num_cores();
        const bool row_major = nd_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR;
        logical_cores = tt::tt_metal::corerange_to_cores(nd_shard_spec.grid, num_cores, row_major);

        const auto& shard_shape = nd_shard_spec.shard_shape;
        const uint32_t input_nsticks_per_core = shard_shape[-3] * shard_shape[-2];
        result.num_sticks.resize(num_cores, input_nsticks_per_core);
        result.start_stick_id.resize(num_cores);
        for (uint32_t i = 0; i < num_cores; ++i) {
            result.start_stick_id[i] = i * input_nsticks_per_core;
        }
    } else if (is_bilinear && output.is_sharded()) {
        // Bilinear output-sharded (input interleaved): cores from the OUTPUT shard spec.
        const auto output_shard_spec = output.shard_spec().value();
        const uint32_t output_nsticks_per_core = output_shard_spec.shape[0];
        const uint32_t num_cores = (total_output_sticks + output_nsticks_per_core - 1) / output_nsticks_per_core;
        const bool row_major = output_shard_spec.orientation == tt::tt_metal::ShardOrientation::ROW_MAJOR;
        logical_cores = tt::tt_metal::corerange_to_cores(output_shard_spec.grid, num_cores, row_major);

        result.num_sticks.resize(num_cores, output_nsticks_per_core);
        result.start_stick_id.resize(num_cores);
        uint32_t sticks_processed = 0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            result.start_stick_id[i] = sticks_processed;
            sticks_processed += output_nsticks_per_core;
        }
    } else {
        // Interleaved path: shared by both factories.
        const auto compute_grid_size = device->compute_with_storage_grid_size();
        auto [num_cores, all_cores_range, core_group_1, core_group_2, num_sticks_1, num_sticks_2] =
            tt::tt_metal::split_work_to_cores(compute_grid_size, total_output_sticks);
        logical_cores = tt::tt_metal::corerange_to_cores(all_cores_range, num_cores, true);

        result.num_sticks.resize(num_cores);
        result.start_stick_id.resize(num_cores);
        uint32_t sticks_processed = 0;
        for (uint32_t i = 0; i < num_cores; ++i) {
            const uint32_t num_sticks = core_group_1.contains(logical_cores[i]) ? num_sticks_1 : num_sticks_2;
            result.num_sticks[i] = num_sticks;
            result.start_stick_id[i] = sticks_processed;
            sticks_processed += num_sticks;
        }
    }

    return result;
}

std::vector<tt::tt_metal::DynamicRuntimeArg> RotateDeviceOperation::get_dynamic_runtime_args(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /* mesh_dispatch_coordinate */) {
    const bool is_bilinear = operation_attributes.interpolation_mode == "bilinear";

    // Same single-source-of-truth derivation the selected program factory uses on a cache miss.
    const RotatePerCoreArgs per_core =
        compute_rotate_per_core_args(operation_attributes, tensor_args.input, tensor_return_value, is_bilinear);

    // Re-apply the hash-excluded angle/center/fill reader args for every work core. The input/output
    // base addresses (reader/writer arg [0]) ride on patchable Buffer* bindings and are not touched here.
    std::vector<tt::tt_metal::DynamicRuntimeArg> dynamic_args;
    dynamic_args.reserve(per_core.cores.size() * 5);
    for (const auto& core : per_core.cores) {
        dynamic_args.push_back(
            {RotatePerCoreArgs::kReaderKernelIdx, core, RotatePerCoreArgs::kCosArgIdx, per_core.cos_angle_q16});
        dynamic_args.push_back(
            {RotatePerCoreArgs::kReaderKernelIdx, core, RotatePerCoreArgs::kSinArgIdx, per_core.sin_angle_q16});
        dynamic_args.push_back(
            {RotatePerCoreArgs::kReaderKernelIdx, core, RotatePerCoreArgs::kCenterXArgIdx, per_core.center_x_q16});
        dynamic_args.push_back(
            {RotatePerCoreArgs::kReaderKernelIdx, core, RotatePerCoreArgs::kCenterYArgIdx, per_core.center_y_q16});
        dynamic_args.push_back(
            {RotatePerCoreArgs::kReaderKernelIdx, core, RotatePerCoreArgs::kFillArgIdx, per_core.fill_value_bits});
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
