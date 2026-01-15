// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/slice/device/slice_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm_sharded.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_rm_stride.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_program_factory_tile_tensor_args.hpp"

#include <tt-metalium/constants.hpp>

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement {

inline __attribute__((always_inline)) uint32_t get_upper_dims_compressed(const ttnn::Shape& shape) {
    return std::accumulate(shape.cbegin(), shape.cend() - 2, 1, std::multiplies<uint32_t>{});
}

inline __attribute__((always_inline)) uint32_t
get_upper_start_offset(const ttnn::Shape& shape, Layout layout, const ttnn::Shape& slice_start) {
    // offset for every dim except last 2
    uint32_t start_offset = 0;

    uint32_t num_pages = shape.volume();
    if (layout == Layout::TILE) {
        num_pages /= tt::constants::TILE_HW;
    } else {
        uint32_t page_width = shape[-1];
        num_pages /= page_width;
    }

    for (uint32_t dim_outer = 0; dim_outer < shape.rank() - 2; dim_outer++) {
        uint32_t compressed_dims = 1;
        for (uint32_t dim_inner = 0; dim_inner <= dim_outer; dim_inner++) {
            compressed_dims *= shape[dim_inner];
        }
        start_offset += (num_pages / compressed_dims) * slice_start[dim_outer];
    }
    return start_offset;
}

inline __attribute__((always_inline)) uint32_t
get_upper_start_offset(const Tensor& tensor, const ttnn::Shape& slice_start) {
    return get_upper_start_offset(tensor.padded_shape(), tensor.layout(), slice_start);
}

// Returns the start offset for a tiled tensor, given the input tensor and the slice start shape.
// If round_up is true, and the slice_start is not aligned to a tile boundary, it will round up to the next tile.
uint32_t get_tiled_start_offset(const ttnn::Shape& input_shape, const ttnn::Shape& slice_start, bool round_up) {
    using namespace tt::constants;
    uint32_t num_input_pages = input_shape.volume() / (TILE_HW);
    uint32_t upper_dims_compressed = get_upper_dims_compressed(input_shape);
    uint32_t num_pages_width = num_input_pages / (upper_dims_compressed * tt::div_up(input_shape[-2], TILE_HEIGHT));

    // offset for every dim except last 2
    uint32_t start_offset = get_upper_start_offset(input_shape, Layout::TILE, slice_start);

    if (round_up) {
        start_offset +=
            tt::div_up(slice_start[-2], TILE_HEIGHT) * num_pages_width + tt::div_up(slice_start[-1], TILE_WIDTH);
    } else {
        start_offset += slice_start[-2] / TILE_HEIGHT * num_pages_width + slice_start[-1] / TILE_WIDTH;
    }
    return start_offset;
}

uint32_t get_tiled_start_offset(const Tensor& input_tensor, const ttnn::Shape& slice_start, bool round_up) {
    const auto& shape = input_tensor.padded_shape();
    return get_tiled_start_offset(shape, slice_start, round_up);
}

uint32_t get_rm_start_offset(const Tensor& tensor, const ttnn::Shape& slice_start) {
    uint32_t start_offset = 0;

    if (tensor.padded_shape().rank() >= 2) {
        start_offset = get_upper_start_offset(tensor, slice_start);
        start_offset += slice_start[-2];
    }

    return start_offset;
}

}  // namespace ttnn::operations::data_movement

namespace ttnn::operations::data_movement::slice {

void SliceDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::constants;
    const bool has_step = std::any_of(args.step.cbegin(), args.step.cend(), [](uint32_t s) { return s != 1; });
    TT_FATAL(tensor_args.input.storage_type() == StorageType::DEVICE, "Operands to unpad need to be on device!");
    TT_FATAL(tensor_args.input.buffer() != nullptr, "Operands to unpad need to be allocated in buffers on device!");
    TT_FATAL(
        tensor_args.input.layout() == Layout::TILE || tensor_args.input.layout() == Layout::ROW_MAJOR,
        "Input tensor layout must be TILE or ROW_MAJOR but got {}",
        tensor_args.input.layout());
    TT_FATAL(
        tensor_args.input.padded_shape().rank() == args.slice_start.rank() &&
            args.slice_start.rank() == args.slice_end.rank(),
        "Input tensor rank ({}), slice start rank ({}), and slice end rank ({}) must all be equal",
        tensor_args.input.padded_shape().rank(),
        args.slice_start.rank(),
        args.slice_end.rank());
    for (uint32_t i = 0; i < tensor_args.input.padded_shape().rank(); i++) {
        TT_FATAL(
            args.slice_start[i] < tensor_args.input.padded_shape()[i],
            "Slice start[{}] ({}) must be less than input tensor shape[{}] ({})",
            i,
            args.slice_start[i],
            i,
            tensor_args.input.padded_shape()[i]);
        TT_FATAL(
            args.slice_end[i] <= tensor_args.input.padded_shape()[i],
            "Ends {} must be less than or equal to the shape of the tensor {}",
            args.slice_end[i],
            tensor_args.input.padded_shape()[i]);
        // Check if start shape is <= end shape
        TT_FATAL(
            args.slice_start[i] <= args.slice_end[i],
            "Slice start[{}] ({}) must be <= slice end[{}] ({})",
            i,
            args.slice_start[i],
            i,
            args.slice_end[i]);
    }
    if (tensor_args.preallocated_output.has_value()) {
        const auto output_shape_required = compute_output_specs(args, tensor_args).logical_shape();
        const auto& out_tensor = tensor_args.preallocated_output.value();
        TT_FATAL(
            out_tensor.padded_shape() == output_shape_required,
            "The preallocated output tensor needs a shape of {}, however it is {}",
            output_shape_required,
            out_tensor.padded_shape());
    }
    auto output_tensor_shape = compute_output_specs(args, tensor_args).logical_shape();
    if (has_step) {  // if all ones modify before passing in to function
        TT_FATAL(
            tensor_args.input.layout() == Layout::ROW_MAJOR, "Strided slice is only supported for row major layout");
        TT_FATAL(!tensor_args.input.is_sharded(), "Strided slice is not supported for sharded tensor");
        TT_FATAL(
            args.step.size() == args.slice_end.rank(),
            "Number of steps {} must match number of ends/starts {}",
            args.step.size(),
            args.slice_end.rank());
    }
    if (tensor_args.input.layout() == Layout::TILE) {
        TT_FATAL(
            tensor_args.input.physical_volume() % TILE_HW == 0,
            "Input tensor physical volume ({}) must be divisible by TILE_HW ({})",
            tensor_args.input.physical_volume(),
            TILE_HW);
        TT_FATAL(
            (output_tensor_shape[-2] % TILE_HEIGHT == 0) && (args.slice_start[-2] % TILE_HEIGHT == 0),
            "Can only slice tilized tensor with height begin index aligned to tiles");
        TT_FATAL(
            (output_tensor_shape[-1] % TILE_WIDTH == 0) && (args.slice_start[-1] % TILE_WIDTH == 0),
            "Can only slice tilized tensor with width begin index aligned to tiles");
    } else if (tensor_args.input.layout() == Layout::ROW_MAJOR) {
        if (has_step) {
            for (uint32_t i = 0; i < tensor_args.input.padded_shape().rank(); i++) {
                TT_FATAL(args.step[i] > 0, "Step({}) = {} should be positive", i, args.step[i]);
            }
        }
    }
}

void SliceDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

SliceDeviceOperation::spec_return_value_t SliceDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    SmallVector<uint32_t> out_shape(input_tensor.logical_shape().rank());

    if (args.use_tensor_args) {
        TT_FATAL(
            args.slice_dim.has_value() && args.num_devices.has_value(),
            "slice_dim and num_devices must be provided for tensor args path");

        uint32_t slice_dimension = args.slice_dim.value();
        uint32_t num_parts = args.num_devices.value();

        for (uint32_t i = 0; i < out_shape.size(); i++) {
            out_shape[i] = input_tensor.logical_shape()[i];
        }
        TT_FATAL(
            slice_dimension < out_shape.size(),
            "slice_dim ({}) must be less than tensor rank ({})",
            slice_dimension,
            out_shape.size());

        uint32_t original_size = out_shape[slice_dimension];
        uint32_t slice_size = original_size / num_parts;

        TT_FATAL(
            original_size % num_parts == 0,
            "Input dimension {} (size={}) must be evenly divisible by num_devices ({})",
            slice_dimension,
            original_size,
            num_parts);

        out_shape[slice_dimension] = slice_size;
    } else {
        // Regular path using slice_start, slice_end, step
        auto output_dim_i = [&args](size_t i) {
            return (args.slice_end[i] - args.slice_start[i] + args.step[i] - 1) / args.step[i];
        };
        for (uint32_t i = 0; i < out_shape.size(); i++) {
            out_shape[i] = output_dim_i(i);
        }
    }
    ttnn::Shape output_tensor_shape(std::move(out_shape));
    return ttnn::TensorSpec(
        output_tensor_shape,
        tt::tt_metal::TensorLayout(input_tensor.dtype(), PageConfig(input_tensor.layout()), args.output_mem_config));
}

Tensor SliceDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output.value();
    }

    const auto& input = tensor_args.input;
    const auto output_spec = compute_output_specs(args, tensor_args);

    return create_device_tensor(output_spec, input.device());
}

SliceDeviceOperation::program_factory_t SliceDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    if (args.use_tensor_args) {
        return program::SliceTileTensorArgsProgramFactory{};
    }

    // Check if we have step != 1
    bool has_step = std::any_of(args.step.cbegin(), args.step.cend(), [](uint32_t s) { return s != 1; });

    if (input.layout() == Layout::ROW_MAJOR) {
        if (input.is_sharded()) {
            return program::SliceRmShardedProgramFactory{};
        }
        if (has_step) {
            return program::SliceRmStrideProgramFactory{};
        }
        return program::SliceRmProgramFactory{};
    }
    // Layout::TILE
    return program::SliceTileProgramFactory{};
}

tt::tt_metal::operation::OpPerformanceModelGeneral<SliceDeviceOperation::tensor_return_value_t>
SliceDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args, const Tensor& output) {
    const auto& input_tensor = tensor_args.input;
    const auto& output_tensor = output;
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output_tensor, true);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output}, ideal_dev_clock_cycles);
    return result;
}

}  // namespace ttnn::operations::data_movement::slice

namespace ttnn::prim {
ttnn::operations::data_movement::slice::SliceDeviceOperation::tensor_return_value_t slice(
    const Tensor& input,
    const ttnn::Shape& slice_start,
    const ttnn::Shape& slice_end,
    const ttnn::Shape& step,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    bool use_tensor_args,
    std::optional<Tensor> start_tensor,
    std::optional<Tensor> end_tensor,
    const std::optional<uint32_t>& slice_dim,
    const std::optional<uint32_t>& num_devices,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const std::optional<Tensor>& preallocated_output) {
    using OperationType = ttnn::operations::data_movement::slice::SliceDeviceOperation;
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            slice_start, slice_end, step, output_mem_config, use_tensor_args, slice_dim, num_devices, sub_core_grids},
        OperationType::tensor_args_t{input, std::move(start_tensor), std::move(end_tensor), preallocated_output});
}
}  // namespace ttnn::prim
