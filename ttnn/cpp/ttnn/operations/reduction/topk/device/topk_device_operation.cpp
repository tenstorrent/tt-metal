// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/topk/device/topk_device_operation.hpp"

#include "ttnn/operations/reduction/topk/device/topk_device_operation_types.hpp"
#include "ttnn/operations/reduction/topk/device/topk_single_core_program_factory.hpp"
#include "ttnn/operations/reduction/topk/device/topk_multi_core_program_factory.hpp"
#include "ttnn/operations/reduction/topk/device/topk_constants.hpp"
#include "ttnn/operations/reduction/topk/device/topk_utils.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt_stl/assert.hpp>
#include "tt-metalium/allocator.hpp"
#include "ttnn/operations/math.hpp"

#include <optional>
#include <tuple>

using namespace tt::tt_metal;

namespace ttnn::prim {
/**
 * @brief Selects the optimal program factory (single-core vs multi-core) for TopK execution
 *
 * This function analyzes the input tensor, operation parameters, and hardware constraints
 * to determine the most efficient execution strategy. It implements a hierarchical decision
 * tree that prioritizes multi-core execution when beneficial and feasible.
 *
 * MULTICORE EXECUTION REQUIREMENTS:
 * All of the following conditions must be met for multi-core execution:
 *
 * 1. DIMENSION SIZE: Input dimension >= multi_core_min_width
 *    - Ensures sufficient work to justify parallel execution overhead
 *    - Dimension size must be a power of 2 for bitonic sort
 *
 * 2. OUTPUT DATA TYPE: Must support 16-bit indices (dimension size < 65536)
 *    - Multi-core implementation currently only supports UInt16 indices
 *    - Dimensions >= 65536 force single-core execution with UInt32 indices
 *
 * 3. K VALUE LIMIT: K <= 64
 *    - Multi-core algorithm has optimized paths for small K values
 *    - Larger K values may not benefit from parallel execution
 *
 * 4. MEMORY AND CORE CONSTRAINTS: Pass verify_multi_core_cost() checks
 *    - Work must be divisible across available cores without remainder
 *    - Memory costs (gather + local per core) must fit within L1 cache limits
 *    - Contiguous rectangular core arrangement must be possible
 *    - Split size must meet minimum dimension per core requirements
 *    - Must be genuinely multi-core beneficial (require > 1 core)
 *
 * If any condition fails, falls back to single-core execution.
 *
 * @param args Operation attributes (K, dimension, memory config, core grids)
 * @param tensor_args Input and output tensor specifications
 * @return Program factory for either multi-core or single-core execution
 */
TopKDeviceOperation::program_factory_t TopKDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;

    const ttnn::Shape input_shape = input_tensor.padded_shape();

    // Check requirement #1: Minimum dimension size for multi-core efficiency
    bool multicore_supported = (input_tensor.padded_shape()[args.dim] >= ttnn::prim::constants::multi_core_min_width);

    // Apply requirement #2: Multi-core implementation constraint
    // Multi-core only supports UInt16 indices (dimension size must fit in 16 bits)
    multicore_supported &= (input_shape[args.dim] < std::numeric_limits<uint16_t>::max());
    // Dimension size must be a power of two for bitonic sort
    multicore_supported &= is_power_of_two(input_shape[args.dim]);

    // Apply requirement #3: K value limitation for multi-core optimization
    multicore_supported &= (args.k <= 64);

    // Check requirement #4: Memory and core availability constraints
    // Only perform expensive verification if basic requirements are met
    if (multicore_supported) {
        auto* device = input_tensor.device();

        // Determine data formats for memory cost calculation
        const tt::DataFormat value_cb_data_format =
            tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
        const tt::DataFormat index_cb_data_format = tt::DataFormat::UInt16;  // Multi-core always uses UInt16

        // Calculate tile sizes for memory cost analysis
        const uint32_t value_tile_size = tile_size(value_cb_data_format);
        const uint32_t index_tile_size = tile_size(index_cb_data_format);

        const auto core_range = args.sub_core_grids.ranges().at(0);

        // Perform comprehensive multi-core feasibility analysis
        // This checks: memory constraints, core availability, work divisibility,
        // and ensures optimal core grid arrangement is possible
        multicore_supported &= verify_multi_core_cost(
            input_shape[args.dim],                    // Total width to process
            ttnn::prim::constants::min_dim_per_core,  // Minimum split size
            input_shape[args.dim] / 2,                // Maximum split size
            args.k,                                   // Number of top elements
            core_range,                               // Available core grid
            device->l1_size_per_core(),               // L1 memory per core
            value_tile_size,                          // Value tile memory size
            index_tile_size);                         // Index tile memory size
    }

    // Select program factory based on feasibility analysis
    if (multicore_supported) {
        return TopKMultiCoreProgramFactory{};
    }

    return TopKSingleCoreProgramFactory{};
}

void TopKDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& indices_tensor = tensor_args.indices;
    const auto& preallocated_outputs = tensor_args.preallocated_outputs;

    // Tensor shape validation
    const auto input_shape = input_tensor.padded_shape();
    TT_FATAL(input_shape.rank() == 4, "Input shape must be 4D, got {}", input_shape.rank());
    TT_FATAL(
        input_shape[-1] >= ttnn::prim::constants::min_dim_per_core,
        "Input shape inner dim {} must be >= {}, pad with +/-infinity if necessary",
        input_shape[-1],
        ttnn::prim::constants::min_dim_per_core);
    TT_FATAL(
        (input_shape[0] * input_shape[1] * input_shape[2]) % 32 == 0,
        "Input height (combined input_shape[0-3]) {} must be a multiple of 32",
        input_shape[0] * input_shape[1] * input_shape[2]);

    // Memory configuration validation
    TT_FATAL(args.output_memory_config.is_sharded() == false, "Sharded implementation not supported yet");

    // Tensor layout validation
    TT_FATAL(input_tensor.layout() == Layout::TILE, "The input must be in tiled format");

    // Data type validation
    const auto input_tensor_dtype = input_tensor.dtype();
    TT_FATAL(
        input_tensor_dtype == DataType::BFLOAT16 || input_tensor_dtype == DataType::BFLOAT8_B,
        "Input tensor must be BFLOAT16, or BFLOAT8_B, got: {}",
        input_tensor_dtype);

    // Optional indices tensor validation (for pre-allocated indices)
    if (indices_tensor.has_value()) {
        const auto indices_tensor_dtype = indices_tensor->dtype();
        TT_FATAL(
            indices_tensor_dtype == DataType::UINT16 || indices_tensor_dtype == DataType::UINT32,
            "Optional input tensor must be UINT16, or UINT32, got: {}",
            indices_tensor_dtype);
    }

    // Preallocated output tensor validation
    if (preallocated_outputs.has_value()) {
        const auto output_tensor0_dtype = std::get<0>(preallocated_outputs.value()).dtype();  // Values tensor
        const auto output_tensor1_dtype = std::get<1>(preallocated_outputs.value()).dtype();  // Indices tensor
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

    // Execution feasibility validation
    // Verify that the operation can be executed with available hardware resources
    bool can_run = false;
    bool uint16_output = (input_shape[args.dim] <= std::numeric_limits<uint16_t>::max());

    // Try multi-core execution first if dimension is large enough
    if (input_shape[args.dim] >= ttnn::prim::constants::multi_core_min_width) {
        auto* device = input_tensor.device();

        // Set up data formats for memory cost calculations
        tt::DataFormat value_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
        tt::DataFormat index_cb_data_format = tt::DataFormat::UInt16;

        uint32_t value_tile_size = tile_size(value_cb_data_format);
        uint32_t index_tile_size = tile_size(index_cb_data_format);

        // Validate core range configuration
        TT_FATAL(
            args.sub_core_grids.ranges().size() == 1,
            "Only one core range is supported right now, got {}",
            args.sub_core_grids.ranges().size());

        const auto core_range = args.sub_core_grids.ranges().at(0);

        // Check if multi-core execution is feasible with current memory and core constraints
        can_run = verify_multi_core_cost(
            input_shape[args.dim],                    // Dimension size
            ttnn::prim::constants::min_dim_per_core,  // Min split size
            input_shape[args.dim] / 2,                // Max split size
            args.k,                                   // Top-K value
            core_range,                               // Available cores
            device->allocator()->get_statistics(tt::tt_metal::BufferType::L1).largest_free_block_bytes,  // L1 memory
            value_tile_size,   // Value tile size
            index_tile_size);  // Index tile size

        // Fallback to single-core if multi-core is not feasible
        if (!can_run) {
            can_run = ttnn::prim::verify_single_core_cost(input_tensor, args.k, uint16_output);
        }
    } else {
        // Dimension too small for multi-core, check single-core feasibility
        can_run = ttnn::prim::verify_single_core_cost(input_tensor, args.k, uint16_output);
    }

    // Final check: ensure the operation can be executed with available resources
    TT_FATAL(can_run, "Not enough cores or cache size available to run TopK operation");
}

TopKDeviceOperation::spec_return_value_t TopKDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor = tensor_args.input;
    const auto& preallocated_outputs = tensor_args.preallocated_outputs;

    // Use preallocated tensor specifications if provided
    if (preallocated_outputs.has_value()) {
        return {
            std::get<0>(preallocated_outputs.value()).tensor_spec(),   // Values tensor spec
            std::get<1>(preallocated_outputs.value()).tensor_spec()};  // Indices tensor spec
    }

    // Compute output specifications dynamically
    auto output_shape = input_tensor.logical_shape();
    output_shape[-1] = args.k;  // Set last dimension to K (number of top elements)

    ttnn::Shape input_shape = input_tensor.padded_shape();
    // Choose index data type based on dimension size (16-bit vs 32-bit indices)
    const bool uint16_output = (input_shape[args.dim] <= std::numeric_limits<uint16_t>::max());  // 65535

    // Create values tensor specification (same data type as input)
    const auto values_spec = TensorSpec(
        output_shape, TensorLayout(input_tensor.dtype(), PageConfig(Layout::TILE), args.output_memory_config));

    // Create indices tensor specification (integer type based on dimension size)
    const DataType index_dtype = uint16_output ? DataType::UINT16 : DataType::UINT32;
    const auto index_spec =
        TensorSpec(output_shape, TensorLayout(index_dtype, PageConfig(Layout::TILE), args.output_memory_config));

    return {values_spec, index_spec};
}

TopKDeviceOperation::tensor_return_value_t TopKDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    // Return preallocated tensors if provided
    if (tensor_args.preallocated_outputs.has_value()) {
        return tensor_args.preallocated_outputs.value();
    }

    // Create new tensors based on computed specifications
    const auto output_specs = compute_output_specs(args, tensor_args);

    return {
        create_device_tensor(std::get<0>(output_specs), tensor_args.input.device()),  // Values tensor
        create_device_tensor(std::get<1>(output_specs), tensor_args.input.device()),  // Indices tensor
    };
}

std::tuple<ttnn::Tensor, ttnn::Tensor> topk(
    const Tensor& input_tensor,
    uint32_t k,
    int8_t dim,
    bool largest,
    bool sorted,
    const tt::tt_metal::MemoryConfig& memory_config,
    const tt::tt_metal::CoreRangeSet& sub_core_grids,
    const std::optional<Tensor>& indices_tensor,
    const std::optional<std::tuple<Tensor, Tensor>>& preallocated_output_tensors) {
    return ttnn::device_operation::launch<TopKDeviceOperation>(
        TopkParams{
            .k = k,
            .dim = dim,
            .largest = largest,
            .sorted = sorted,
            .output_memory_config = memory_config,
            .sub_core_grids = sub_core_grids},
        TopkInputs{
            .input = input_tensor, .indices = indices_tensor, .preallocated_outputs = preallocated_output_tensors});
}
}  // namespace ttnn::prim
