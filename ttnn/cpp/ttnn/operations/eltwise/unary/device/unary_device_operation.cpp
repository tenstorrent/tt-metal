// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "unary_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "tools/profiler/op_profiler.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::unary {

namespace {
void validate_supported_arch_dtype(
    tt::ARCH arch, DataType input_datatype, DataType output_datatype, UnaryOpType op_type) {
    switch (op_type) {
        case UnaryOpType::BITWISE_XOR:
        case UnaryOpType::BITWISE_NOT:
        case UnaryOpType::BITWISE_AND:
        case UnaryOpType::BITWISE_OR:
            TT_FATAL(
                input_datatype == DataType::INT32,
                "Unsupported input data type '{}' for UnaryOpType '{}' (Bitwise operation).",
                static_cast<int>(input_datatype),
                static_cast<int>(op_type));
            TT_FATAL(
                output_datatype == DataType::INT32,
                "Unsupported output data type '{}' for UnaryOpType '{}' (Bitwise operation).",
                static_cast<int>(output_datatype),
                static_cast<int>(op_type));
            break;
        case UnaryOpType::FMOD:
            TT_FATAL(
                (input_datatype == DataType::BFLOAT16 || input_datatype == DataType::FLOAT32),
                "Unsupported input data type '{}' for UnaryOpType '{}' (FMOD operation).",
                static_cast<int>(input_datatype),
                static_cast<int>(op_type));
            TT_FATAL(
                (output_datatype == DataType::BFLOAT16 || output_datatype == DataType::FLOAT32),
                "Unsupported output data type '{}' for UnaryOpType '{}' (FMOD operation).",
                static_cast<int>(output_datatype),
                static_cast<int>(op_type));
            break;
        default: return;
    }
}
}  // namespace

UnaryDeviceOperation::program_factory_t UnaryDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.input.is_sharded()) {
        return program::UnaryShardedProgramFactory{};
    } else {
        return program::UnaryProgramFactory{};
    }
}

void UnaryDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void UnaryDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& preallocated_output_tensor = tensor_args.preallocated_output;

    auto out_memory_config = args.output_memory_config;
    auto output_datatype = args.output_dtype;
    if (preallocated_output_tensor.has_value()) {
        out_memory_config = preallocated_output_tensor->memory_config();
        output_datatype = preallocated_output_tensor->dtype();
    }

    auto arch = input_tensor.device()->arch();
    auto input_datatype = input_tensor.dtype();
    for (const auto& unary_op : args.op_chain) {
        validate_supported_arch_dtype(arch, input_datatype, output_datatype, unary_op.type());
    }

    TT_FATAL(
        input_tensor.storage_type() == StorageType::DEVICE,
        "Unary operation requires input to be on Device. Input storage type: {}",
        static_cast<int>(input_tensor.storage_type()));

    TT_FATAL(
        input_tensor.buffer() != nullptr,
        "Operands to eltwise unary need to be allocated in buffers on the device. Buffer is null.");

    // Allow sharded output from non-sharded input if shard_spec can be created automatically
    // Only skip layout check when output is sharded but missing shard_spec (will be auto-created)
    bool allow_sharded_output = out_memory_config.is_sharded() &&
                                !out_memory_config.shard_spec().has_value();

    if (!allow_sharded_output) {
        TT_FATAL(
            input_tensor.memory_config().memory_layout() == out_memory_config.memory_layout(),
            "Unary operation requires Input and Output memory layout to match. Input layout: {}, Output layout: {}",
            static_cast<int>(input_tensor.memory_config().memory_layout()),
            static_cast<int>(out_memory_config.memory_layout()));
    }

    if (!input_tensor.is_sharded()) {
        TT_FATAL(
            input_tensor.layout() == Layout::TILE,
            "Unary operation requires tensor to be in Tile layout when working with non-sharded input tensor. Input "
            "tensor layout: {}",
            static_cast<int>(input_tensor.layout()));

        TT_FATAL(
            input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Unary operation requires Interleaved memory layout when working with non-sharded input tensor. Input "
            "memory layout: `{}`",
            static_cast<int>(input_tensor.memory_config().memory_layout()));
    }

    if (preallocated_output_tensor.has_value()) {
        const auto computed_output_shape = compute_output_specs(args, tensor_args).logical_shape();
        const auto preallocated_output_shape = preallocated_output_tensor.value().logical_shape();
        TT_FATAL(
            preallocated_output_shape == computed_output_shape,
            "When preallocted output tensor is used, Unary operation requires its shape to match the computed "
            "shape. Computed shape: {}, Shape in preallocated output tensor: {}",
            computed_output_shape,
            preallocated_output_shape);

        if(!input_tensor.is_sharded()){
            TT_FATAL(
                (preallocated_output_tensor.value().layout() == Layout::TILE),
                "Unary operation requires output tensor to be in Tile layout when working with non-sharded tensor.");
        }
    }
}

spec_return_value_t UnaryDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    auto output_layout = Layout::TILE;
    auto output_memory_config = args.output_memory_config;

    // If output memory config is sharded but doesn't have a shard_spec,
    // automatically create one from the input tensor
    if (output_memory_config.is_sharded() && !output_memory_config.shard_spec().has_value()) {
        if (tensor_args.input.is_sharded() && tensor_args.input.shard_spec().has_value()) {
            // Use input tensor's shard_spec if available
            output_memory_config = output_memory_config.with_shard_spec(tensor_args.input.shard_spec().value());
        } else {
            // Create a default shard_spec based on input shape and memory layout
            const auto& input = tensor_args.input;
            const auto& input_shape = input.padded_shape();
            const auto& device = input.device();
            const auto memory_layout = output_memory_config.memory_layout();

            // Get device compute grid
            CoreCoord grid_size = device->compute_with_storage_grid_size();

            // Calculate shard shape based on memory layout
            // Shard dimensions must satisfy:
            // 1. Tile alignment: multiples of tile size (32x32)
            // 2. L1 alignment: (shard_width * datum_size) % L1_alignment == 0
            constexpr uint32_t TILE_WIDTH = 32;
            constexpr uint32_t TILE_HEIGHT = 32;

            // Get L1 alignment requirement and datum size for alignment calculation
            uint32_t l1_alignment = hal::get_l1_alignment();
            DataType input_dtype = input.dtype();
            tt::DataFormat input_df = tt::tt_metal::datatype_to_dataformat_converter(input_dtype);
            uint32_t datum_size_bytes = datum_size(input_df);
            // Calculate minimum shard width in elements to satisfy L1 alignment
            // L1 alignment is in bytes, so we need: shard_width_elements * datum_size_bytes >= l1_alignment
            // and shard_width_elements * datum_size_bytes % l1_alignment == 0
            uint32_t min_shard_width_for_l1 = tt::div_up(l1_alignment, datum_size_bytes);
            // Round up to tile boundary for L1-aligned shard width
            uint32_t l1_aligned_tile_multiple = tt::round_up(min_shard_width_for_l1, TILE_WIDTH);

            std::array<uint32_t, 2> shard_shape;
            uint32_t num_cores = 0;
            uint32_t total_height = input.physical_volume() / input_shape[-1];
            uint32_t total_width = input_shape[-1];

            switch (memory_layout) {
                case TensorMemoryLayout::WIDTH_SHARDED: {
                    // For width sharding, try to maximize parallelism while ensuring alignment
                    TT_FATAL(
                        total_width >= l1_aligned_tile_multiple,
                        "Invalid configuration for WIDTH_SHARDED: total_width ({}) must be >= l1_aligned_tile_multiple ({})",
                        total_width,
                        l1_aligned_tile_multiple);

                    uint32_t max_grid_cores = static_cast<uint32_t>(grid_size.x * grid_size.y);
                    // Start with trying to use all available cores
                    uint32_t target_cores = std::min(max_grid_cores, total_width / l1_aligned_tile_multiple);
                    target_cores = std::max(1u, target_cores); // At least use 1 core

                    // Calculate shard width that maximizes core usage while satisfying alignment
                    uint32_t shard_width = tt::round_up(total_width / target_cores, l1_aligned_tile_multiple);
                    shard_width = std::max(shard_width, l1_aligned_tile_multiple); // Ensure minimum
                    shard_width = std::min(shard_width, total_width); // Don't exceed total width
                    shard_shape = {total_height, shard_width};
                    num_cores = tt::div_up(total_width, shard_shape[1]);
                    num_cores = std::min(num_cores, max_grid_cores); // Limit to available grid
                    break;
                }
                case TensorMemoryLayout::HEIGHT_SHARDED: {
                    // For height sharding, divide height across available cores
                    TT_FATAL(
                        total_height >= TILE_HEIGHT,
                        "Invalid configuration for HEIGHT_SHARDED: total_height ({}) must be >= TILE_HEIGHT ({})",
                        total_height,
                        TILE_HEIGHT);

                    uint32_t max_grid_cores = static_cast<uint32_t>(grid_size.x * grid_size.y);
                    uint32_t target_cores = std::min(max_grid_cores, total_height / TILE_HEIGHT);
                    target_cores = std::max(1u, target_cores);

                    uint32_t shard_height = tt::round_up(total_height / target_cores, TILE_HEIGHT);
                    shard_height = std::max(shard_height, TILE_HEIGHT);
                    shard_height = std::min(shard_height, total_height); // Don't exceed total height
                    shard_shape = {shard_height, total_width};
                    num_cores = tt::div_up(total_height, shard_shape[0]);
                    num_cores = std::min(num_cores, max_grid_cores); // Limit to available grid
                    break;
                }
                case TensorMemoryLayout::BLOCK_SHARDED: {
                    // For block sharding, divide both dimensions across grid
                    // For COL_MAJOR orientation: num_shards_along_width <= grid.y, num_shards_along_height <= grid.x
                    uint32_t grid_y = static_cast<uint32_t>(grid_size.y);  // rows
                    uint32_t grid_x = static_cast<uint32_t>(grid_size.x);   // columns

                    // We need: num_shards_height <= grid_x and num_shards_width <= grid_y
                    // For COL_MAJOR: shard_height >= ceil(total_height / grid_x) and shard_width >= ceil(total_width / grid_y)

                    // Calculate shard height: ensure num_shards_height <= grid_x
                    // Minimum shard_height needed: ceil(total_height / grid_x), rounded to tile boundary
                    uint32_t min_shard_height = tt::div_up(total_height, grid_x);
                    uint32_t shard_height = tt::round_up(min_shard_height, TILE_HEIGHT);
                    shard_height = std::max(shard_height, TILE_HEIGHT);
                    shard_height = std::min(shard_height, total_height);  // Don't exceed total height
                    uint32_t num_shards_height = tt::div_up(total_height, shard_height);
                    // If rounding caused violation, increase shard_height by one tile to guarantee satisfaction
                    if (num_shards_height > grid_x) {
                        shard_height += TILE_HEIGHT;
                        shard_height = std::min(shard_height, total_height);
                        num_shards_height = tt::div_up(total_height, shard_height);
                    }

                    // Calculate shard width: ensure num_shards_width <= grid_y
                    // Minimum shard_width needed: ceil(total_width / grid_y), rounded to alignment boundary
                    uint32_t min_shard_width = tt::div_up(total_width, grid_y);
                    uint32_t shard_width = tt::round_up(min_shard_width, l1_aligned_tile_multiple);
                    shard_width = std::max(shard_width, l1_aligned_tile_multiple);
                    shard_width = std::min(shard_width, total_width);  // Don't exceed total width
                    uint32_t num_shards_width = tt::div_up(total_width, shard_width);
                    // If rounding caused violation, increase shard_width by one alignment unit to guarantee satisfaction
                    if (num_shards_width > grid_y) {
                        shard_width += l1_aligned_tile_multiple;
                        shard_width = std::min(shard_width, total_width);
                        num_shards_width = tt::div_up(total_width, shard_width);
                    }

                    // Verify constraints are satisfied
                    TT_FATAL(num_shards_height <= grid_x, "Number of shards along height {} must not exceed grid columns {}", num_shards_height, grid_x);
                    TT_FATAL(num_shards_width <= grid_y, "Number of shards along width {} must not exceed grid rows {} for COL_MAJOR orientation", num_shards_width, grid_y);

                    shard_shape = {shard_height, shard_width};
                    num_cores = num_shards_height * num_shards_width;
                    // Limit to available grid (already guaranteed by constraints, but ensure consistency)
                    uint32_t max_grid_cores = static_cast<uint32_t>(grid_size.x * grid_size.y);
                    num_cores = std::min(num_cores, max_grid_cores);
                    break;
                }
                default:
                    TT_FATAL(false, "Unsupported sharding scheme for automatic shard_spec creation");
            }

            // Create core range set
            CoreRangeSet grid_set;
            if (memory_layout == TensorMemoryLayout::BLOCK_SHARDED) {
                // For BLOCK_SHARDED, create a rectangular grid directly
                // For COL_MAJOR orientation: num_shards_height maps to grid.x (columns), num_shards_width maps to grid.y (rows)
                // CoreRange uses (x, y) coordinates where x=columns, y=rows
                uint32_t num_shards_h = tt::div_up(total_height, shard_shape[0]);  // num_shards_height
                uint32_t num_shards_w = tt::div_up(total_width, shard_shape[1]);    // num_shards_width
                // TT_FATAL assertions above guarantee num_shards_h <= grid_size.x and num_shards_w <= grid_size.y
                // Create a single rectangular CoreRange: (x=num_shards_height, y=num_shards_width)
                CoreRange block_range({0, 0}, {num_shards_h - 1, num_shards_w - 1});
                grid_set = CoreRangeSet({block_range});
            } else {
                bool row_wise = (memory_layout == TensorMemoryLayout::WIDTH_SHARDED);
                grid_set = tt::tt_metal::num_cores_to_corerangeset(num_cores, grid_size, row_wise);
            }

            // Determine shard orientation
            ShardOrientation shard_orientation = (memory_layout == TensorMemoryLayout::WIDTH_SHARDED)
                ? ShardOrientation::ROW_MAJOR
                : ShardOrientation::COL_MAJOR;

            // Create shard spec
            ShardSpec shard_spec(grid_set, shard_shape, shard_orientation);
            output_memory_config = output_memory_config.with_shard_spec(shard_spec);
        }
    }

    if (output_memory_config.is_sharded()) {
        output_layout = tensor_args.input.layout();
    }

    const auto output_shape = tensor_args.input.logical_shape();
    return TensorSpec(output_shape, TensorLayout(args.output_dtype, output_layout, output_memory_config));
}

tensor_return_value_t UnaryDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return *tensor_args.preallocated_output;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t UnaryDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_shape = input_tensor.padded_shape();

    auto program_factory = select_program_factory(args, tensor_args);
    operation::Hash hash = operation::hash_operation<UnaryDeviceOperation>(
        args,
        program_factory.index(),
        input_tensor.dtype(),
        input_tensor.memory_config(),
        input_shape.volume());

    return hash;
}

bool UnaryDeviceOperation::skip_launch(
    const operation_attributes_t& attributes,
    const tensor_args_t& tensor_args,
    const tensor_return_value_t& tensor_return_value) {
    return tensor_return_value.logical_shape().volume() == 0;
}

std::tuple<UnaryDeviceOperation::operation_attributes_t, UnaryDeviceOperation::tensor_args_t>
UnaryDeviceOperation::invoke(
    const Tensor& input,
    const std::vector<EltwiseUnaryWithParam>& op_chain,
    DataType output_dtype,
    const MemoryConfig& output_memory_config,
    bool fp32_dest_acc_en,
    bool preserve_fp32_precision,
    bool bfp8_pack_precise,
    const std::optional<Tensor>& preallocated_output) {
    return {
        operation_attributes_t{
            .op_chain = op_chain,
            .output_dtype = output_dtype,
            .output_memory_config = output_memory_config,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .preserve_fp32_precision = preserve_fp32_precision,
            .bfp8_pack_precise = bfp8_pack_precise,
        },
        tensor_args_t{.input = input, .preallocated_output = preallocated_output}};
}

}  // namespace ttnn::operations::unary
