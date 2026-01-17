// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/topk/device/topk_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

#include <optional>
#include <tuple>

#include <tt_stl/assert.hpp>
#include "tt-metalium/allocator.hpp"
#include "ttnn/operations/reduction/topk/device/topk_device_operation_types.hpp"
#include "ttnn/operations/reduction/topk/device/topk_single_core_program_factory.hpp"
#include "ttnn/operations/reduction/topk/device/topk_multi_core_program_factory.hpp"
#include "ttnn/operations/reduction/topk/device/topk_constants.hpp"
#include "ttnn/operations/reduction/topk/device/topk_utils.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::reduction::topk {

TopKDeviceOperation::program_factory_t TopKDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;

    ttnn::Shape input_shape = input_tensor.padded_shape();
    bool uint16_output = (input_shape[args.dim] < 65536);

    bool multicore_supported = (input_tensor.padded_shape()[args.dim] >= topk::constants::multi_core_min_width);

    // for now multicore does not support uint32 output, so if uint16 is not
    // supported, we default to single core
    // multicore implementation only supports k <= 64
    multicore_supported &= uint16_output;
    multicore_supported &= (args.k <= 64);

    // don't bother with longer check if already false
    if (multicore_supported) {
        auto* device = input_tensor.device();
        tt::DataFormat value_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
        tt::DataFormat index_cb_data_format = tt::DataFormat::UInt16;

        uint32_t value_tile_size = tile_size(value_cb_data_format);
        uint32_t index_tile_size = tile_size(index_cb_data_format);

        const auto core_range = args.sub_core_grids.ranges().at(0);
        multicore_supported &= topk::utils::verify_multi_core_cost(
            input_shape[args.dim],
            topk::constants::min_dim_per_core,
            input_shape[args.dim] / 2,
            args.k,
            core_range,
            device->l1_size_per_core(),
            value_tile_size,
            index_tile_size);
    }

    if (multicore_supported) {
        return program::TopKMultiCoreProgramFactory{};
    }

    return program::TopKSingleCoreProgramFactory{};
}

void TopKDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void TopKDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& indices_tensor = tensor_args.indices;
    const auto& preallocated_outputs = tensor_args.preallocated_outputs;

    auto input_shape = input_tensor.padded_shape();
    TT_FATAL(input_shape.rank() == 4, "Input shape must be 4D, got {}", input_shape.rank());

    TT_FATAL(
        input_shape[-1] >= topk::constants::min_dim_per_core,
        "Input shape inner dim {} must be >= {}, pad with +/-infinity if necessary",
        input_shape[-1],
        topk::constants::min_dim_per_core);
    TT_FATAL(
        (input_shape[0] * input_shape[1] * input_shape[2]) % 32 == 0,
        "Input height (combined input_shape[0-3]) {} must be a multiple of 32",
        input_shape[0] * input_shape[1] * input_shape[2]);

    TT_FATAL(args.output_memory_config.is_sharded() == false, "Sharded implementation not supported yet");
    TT_FATAL(input_tensor.layout() == Layout::TILE, "The input must be in tiled format");

    const auto input_tensor_dtype = input_tensor.dtype();
    TT_FATAL(
        input_tensor_dtype == DataType::BFLOAT16 || input_tensor_dtype == DataType::BFLOAT8_B,
        "Input tensor must be BFLOAT16, or BFLOAT8_B, got: {}",
        input_tensor_dtype);
    if (indices_tensor.has_value()) {
        const auto indices_tensor_dtype = indices_tensor->dtype();
        TT_FATAL(
            indices_tensor_dtype == DataType::UINT16 || indices_tensor_dtype == DataType::UINT32,
            "Optional input tensor must be UINT16, or UINT32, got: {}",
            indices_tensor_dtype);
    }
    if (preallocated_outputs.has_value()) {
        const auto output_tensor0_dtype = std::get<0>(preallocated_outputs.value()).dtype();
        const auto output_tensor1_dtype = std::get<1>(preallocated_outputs.value()).dtype();
        TT_FATAL(
            output_tensor0_dtype == DataType::BFLOAT16 || output_tensor0_dtype == DataType::BFLOAT8_B,
            "Preallocated output tensor must be BFLOAT16 or BFLOAT8_B got: {}",
            output_tensor0_dtype);
        TT_FATAL(
            output_tensor1_dtype == DataType::UINT16 || output_tensor1_dtype == DataType::UINT32,
            "Preallocated indices tensor must be UINT16 or UINT32 got: {}",
            output_tensor1_dtype);
        TT_FATAL(
            output_tensor0_dtype == input_tensor_dtype,
            "Preallocated output tensor dtype must match input tensor dtype. Got output: {}, input: {}",
            output_tensor0_dtype,
            input_tensor_dtype);
    }

    bool can_run = false;
    bool uint16_output = (input_shape[args.dim] <= std::numeric_limits<uint16_t>::max());

    if (input_shape[args.dim] >= topk::constants::multi_core_min_width) {  // multicore implementation
        auto* device = input_tensor.device();
        tt::DataFormat value_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
        tt::DataFormat index_cb_data_format = tt::DataFormat::UInt16;

        uint32_t value_tile_size = tile_size(value_cb_data_format);
        uint32_t index_tile_size = tile_size(index_cb_data_format);

        const auto core_range = args.sub_core_grids.ranges().at(0);

        can_run = topk::utils::verify_multi_core_cost(
            input_shape[args.dim],
            topk::constants::min_dim_per_core,
            input_shape[args.dim] / 2,
            args.k,
            core_range,
            device->allocator()->get_statistics(tt::tt_metal::BufferType::L1).largest_free_block_bytes,
            value_tile_size,
            index_tile_size);

        TT_FATAL(
            args.sub_core_grids.ranges().size() == 1,
            "Only one core range is supported right now, got {}",
            args.sub_core_grids.ranges().size());

        if (!can_run) {  // can we default to new topk implementation on single core
            can_run = topk::utils::verify_single_core_cost(input_tensor, args.k, uint16_output);
        }
    } else {
        can_run = topk::utils::verify_single_core_cost(input_tensor, args.k, uint16_output);
    }
    TT_FATAL(can_run, "Not enough cores or cache size available to run topk operation");
}

spec_return_value_t TopKDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& preallocated_outputs = tensor_args.preallocated_outputs;

    if (preallocated_outputs.has_value()) {
        return {
            std::get<0>(preallocated_outputs.value()).tensor_spec(),
            std::get<1>(preallocated_outputs.value()).tensor_spec()};
    }

    auto output_shape = input_tensor.logical_shape();
    output_shape[-1] = args.k;
    ttnn::Shape input_shape = input_tensor.padded_shape();
    bool uint16_output = (input_shape[args.dim] < 65536);

    auto values_spec = TensorSpec(
        output_shape, TensorLayout(input_tensor.dtype(), PageConfig(Layout::TILE), args.output_memory_config));
    DataType index_dtype = uint16_output ? DataType::UINT16 : DataType::UINT32;
    auto index_spec =
        TensorSpec(output_shape, TensorLayout(index_dtype, PageConfig(Layout::TILE), args.output_memory_config));

    return {values_spec, index_spec};
}

tensor_return_value_t TopKDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_outputs.has_value()) {
        return tensor_args.preallocated_outputs.value();
    }
    auto output_specs = compute_output_specs(args, tensor_args);
    return {
        create_device_tensor(std::get<0>(output_specs), tensor_args.input.device()),
        create_device_tensor(std::get<1>(output_specs), tensor_args.input.device()),
    };
}

}  // namespace ttnn::operations::reduction::topk

namespace ttnn::prim {
ttnn::operations::reduction::topk::TopKDeviceOperation::tensor_return_value_t topk(
    const Tensor& input_tensor,
    uint32_t k,
    int8_t dim,
    bool largest,
    bool sorted,
    const tt::tt_metal::MemoryConfig& memory_config,
    const tt::tt_metal::CoreRangeSet& sub_core_grids,
    const std::optional<Tensor>& indices_tensor,
    const std::optional<std::tuple<Tensor, Tensor>>& preallocated_output_tensors) {
    using OperationType = ttnn::operations::reduction::topk::TopKDeviceOperation;

    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .k = k,
            .dim = dim,
            .largest = largest,
            .sorted = sorted,
            .output_memory_config = memory_config,
            .sub_core_grids = sub_core_grids},
        OperationType::tensor_args_t{
            .input = input_tensor, .indices = indices_tensor, .preallocated_outputs = preallocated_output_tensors});
}
}  // namespace ttnn::prim
