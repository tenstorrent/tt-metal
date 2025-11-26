// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "argmax_device_operation.hpp"

#include "ttnn/tensor/tensor.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::reduction::argmax {

namespace {

// Helper to generate output shape for the reduction operation
ttnn::SmallVector<uint32_t> get_output_shape(const Tensor& input_tensor, const std::optional<int>& dim, bool keepdim) {
    auto input_shape = input_tensor.logical_shape();
    int rank = input_shape.size();
    ttnn::SmallVector<uint32_t> output_shape;

    // If no reduction dims are specified, we reduce all dimensions
    auto all_dim_reduce = !dim.has_value();
    auto red_dim = dim.value_or(0);
    TT_FATAL(
        (rank == 0) || ((red_dim >= -rank) && (red_dim < rank)),
        "Invalid reduction dimension {} for input tensor with rank {}",
        red_dim,
        rank);

    // Adjust negative reduction dimension to positive
    red_dim = red_dim < 0 ? red_dim + rank : red_dim;

    // Generate output shape
    for (int d = 0; d < rank; ++d) {
        bool is_reduction_dim = all_dim_reduce || (d == red_dim);

        if (is_reduction_dim) {
            TT_FATAL(input_shape[d] != 0, "Expected reduction dim {} to have non-zero size", d);
            if (keepdim) {
                output_shape.push_back(1);
            }
        } else {
            output_shape.push_back(input_shape[d]);
        }
    }

    return output_shape;
}

}  // namespace

ArgMaxDeviceOperation::program_factory_t ArgMaxDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input = tensor_args.input;

    if (input.layout() == Layout::TILE) {
        return program::ArgMaxSingleCoreTileFactory{};
    } else if (args.use_multicore) {
        return program::ArgMaxMultiCoreRowMajorFactory{};
    } else {
        return program::ArgMaxSingleCoreRowMajorFactory{};
    }
}

void ArgMaxDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void ArgMaxDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto input_layout = input_tensor.layout();
    const auto& preallocated_output = tensor_args.preallocated_output;

    TT_FATAL(
        input_tensor.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only INTERLEAVED memory layout is supported for inputs!");

    if (input_layout == Layout::ROW_MAJOR) {
        TT_FATAL(
            input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::FLOAT32 ||
                input_tensor.dtype() == DataType::INT32 || input_tensor.dtype() == DataType::UINT32 ||
                input_tensor.dtype() == DataType::UINT16,
            "Only BFLOAT16, FLOAT32, INT32, UINT32, and UINT16 are supported for inputs with ROW_MAJOR layout!");
    } else {
        TT_FATAL(
            input_tensor.dtype() == DataType::BFLOAT16 || input_tensor.dtype() == DataType::FLOAT32,
            "Only BFLOAT16, FLOAT32 are supported for inputs with TILE layout!");

        const auto& input_shape = input_tensor.padded_shape();
        auto rank = input_shape.size();
        TT_FATAL(rank > 1, "Invalid rank for input tensor with TILE layout");
        TT_FATAL(input_shape[rank - 1] % tt::constants::TILE_WIDTH == 0, "Invalid input tensor shape");
        TT_FATAL(input_shape[rank - 2] % tt::constants::TILE_HEIGHT == 0, "Invalid input tensor shape");
    }

    TT_FATAL(args.output_dtype == DataType::UINT32, "Only UINT32 is supported for outputs!");

    TT_FATAL(
        args.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "Only INTERLEAVED memory layout is supported for outputs!");

    if (preallocated_output.has_value()) {
        TT_FATAL(preallocated_output->dtype() == DataType::UINT32, "Only UINT32 is supported for outputs!");
        TT_FATAL(
            preallocated_output->memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Only INTERLEAVED memory layout is supported for outputs!");
        TT_FATAL(preallocated_output->layout() == Layout::ROW_MAJOR, "Output tensor must have ROW_MAJOR layout!");
    }

    if (args.dim.has_value()) {
        const uint32_t input_rank = input_tensor.padded_shape().rank();
        const int dim_val = args.dim.value();
        const uint32_t normalized_dim = dim_val < 0 ? dim_val + input_rank : dim_val;

        TT_FATAL(normalized_dim == (input_rank - 1), "Only argmax on last dim is supported!");
    } else {
        TT_FATAL(input_layout != Layout::TILE, "For inputs with TILE layout, dim parameter must be specified!");
    }

    if (args.use_multicore) {
        if (args.sub_core_grids.has_value()) {
            TT_FATAL(
                args.sub_core_grids->ranges().size() <= 2,
                "Multicore argmax only supports up to 2 core grid ranges, but got {} ranges",
                args.sub_core_grids->ranges().size());
        }
        TT_FATAL(
            input_tensor.layout() == Layout::ROW_MAJOR, "Multicore argmax only supports ROW_MAJOR layout for inputs!");
    }
}

spec_return_value_t ArgMaxDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return tensor_args.preallocated_output->tensor_spec();
    }

    const auto& input_tensor = tensor_args.input;
    auto output_shape = get_output_shape(input_tensor, args.dim, args.keepdim);
    return TensorSpec(
        ttnn::Shape(output_shape),
        TensorLayout(args.output_dtype, PageConfig(Layout::ROW_MAJOR), args.output_mem_config));
}

tensor_return_value_t ArgMaxDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_output.has_value()) {
        return *tensor_args.preallocated_output;
    }
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input.device());
}

tt::stl::hash::hash_t ArgMaxDeviceOperation::compute_program_hash(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& input_shape = input_tensor.padded_shape();
    auto program_factory = select_program_factory(args, tensor_args);

    return operation::hash_operation<ArgMaxDeviceOperation>(
        args,
        program_factory.index(),
        input_tensor.dtype(),
        input_tensor.layout(),
        input_tensor.memory_config(),
        input_shape.volume());
}

std::tuple<ArgMaxDeviceOperation::operation_attributes_t, ArgMaxDeviceOperation::tensor_args_t>
ArgMaxDeviceOperation::invoke(
    const Tensor& input_tensor,
    std::optional<int> dim,
    bool keepdim,
    const std::optional<CoreRangeSet>& sub_core_grids,
    bool use_multicore,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& optional_output_tensor) {
    auto output_memory_config = memory_config.value_or(input_tensor.memory_config());

    return {
        operation_attributes_t{
            .output_dtype = DataType::UINT32,
            .dim = dim,
            .keepdim = keepdim,
            .sub_core_grids = sub_core_grids,
            .use_multicore = use_multicore,
            .output_mem_config = output_memory_config,
        },
        tensor_args_t{
            .input = input_tensor,
            .preallocated_output = optional_output_tensor,
        }};
}

}  // namespace ttnn::operations::reduction::argmax
