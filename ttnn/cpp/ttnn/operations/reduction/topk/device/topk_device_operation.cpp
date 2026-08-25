// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/topk/device/topk_device_operation.hpp"

#include "ttnn/operations/reduction/topk/device/topk_device_operation_types.hpp"
#include "ttnn/operations/reduction/topk/device/topk_constants.hpp"
#include "ttnn/operations/reduction/topk/device/topk_utils.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt_stl/assert.hpp>
#include "tt-metalium/allocator.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/reduction/reduce_op_validation.hpp"

#include <optional>
#include <tuple>

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {
// The index dtype the op runs at: index CB, generated iota and output tensor all follow it.
// A preallocated indices output pins it; a 32-bit indices_tensor widens it. A UINT16
// indices_tensor never narrows it -- validate rejects one narrower than the input requires.
DataType resolve_index_dtype(
    const TopKDeviceOperation::operation_attributes_t& args, const TopKDeviceOperation::tensor_args_t& tensor_args) {
    if (tensor_args.preallocated_outputs.has_value()) {
        return std::get<1>(tensor_args.preallocated_outputs.value()).dtype();
    }
    if (tensor_args.indices.has_value()) {
        const auto indices_dtype = tensor_args.indices->dtype();
        if (indices_dtype == DataType::UINT32 || indices_dtype == DataType::INT32) {
            return indices_dtype;
        }
    }
    return is_uint32_index_required(tensor_args.input, args.dim) ? DataType::UINT32 : DataType::UINT16;
}

// Maps the resolved index dtype onto the circular-buffer data format used by the sort datapath. 16-bit indices
// use UInt16; 32-bit indices (UINT32/INT32) both use UInt32, since INT32 shares the same 4-byte layout for the
// non-negative positions TopK produces and the sort LLKs only handle UInt32 among the 32-bit formats.
tt::DataFormat index_cb_data_format_for(
    const TopKDeviceOperation::operation_attributes_t& args, const TopKDeviceOperation::tensor_args_t& tensor_args) {
    return (resolve_index_dtype(args, tensor_args) == DataType::UINT16) ? tt::DataFormat::UInt16
                                                                        : tt::DataFormat::UInt32;
}
}  // namespace

/**
 * @brief Selects the optimal program factory (single-core vs multi-core) for TopK execution
 *
 * This function analyzes the input tensor, operation parameters, and hardware constraints
 * to determine the most efficient execution strategy. It implements a hierarchical decision
 * tree that prioritizes multi-core execution when beneficial and feasible.
 *
 * MULTICORE EXECUTION REQUIREMENTS:
 * Requirements #1-#3 (shape/K structure) are implemented by the shared
 * ttnn::prim::topk_multicore_structurally_eligible() helper (topk_utils.cpp),
 * with thresholds named in topk_constants.hpp. The same helper gates
 * validate_on_program_cache_miss and the composite router in topk.cpp, so the
 * three sites cannot drift. (The router evaluates it with the low-tile-row
 * relaxation deliberately disabled: the composite measured faster than the
 * stock multi-core bitonic on that cell — see
 * should_route_to_topk_large_indices in topk.cpp.)
 *
 * 1. DIMENSION SIZE: Input dimension >= multi_core_min_width, OR the input has
 *    at most multi_core_low_ht_max_tile_rows tile rows and the dimension is
 *    >= multi_core_low_ht_min_width (the single-core factory parallelizes
 *    across tile rows, so low-tile-row shapes column-split instead)
 *    - Ensures sufficient work to justify parallel execution overhead
 *
 * 2. DIMENSION SIZE: Reduced dimension must be < 65535 and a power of two
 *    - Required by the multi-core bitonic sort network (16-bit element indices)
 *    - Both 16-bit (UInt16) and 32-bit (UInt32/INT32) index outputs are supported
 *
 * 3. K VALUE LIMIT: K <= multi_core_max_k (64)
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

    // Requirements #1-#3 (shape/K structure) via the shared eligibility helper — the
    // single source of truth also used by validate_on_program_cache_miss and the
    // composite router in topk.cpp. Use the tensor's actual tile height (the program
    // factories size their work the same way via tensor_spec().tile()) so custom tile
    // shapes count tile rows correctly. Validation guarantees width >= min_dim_per_core
    // and a tiled layout, so both divisors are non-zero.
    const uint32_t reduced_width = input_shape[args.dim];
    const uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    const uint32_t num_tile_rows = input_shape.volume() / reduced_width / tile_height;
    bool multicore_supported = topk_multicore_structurally_eligible(reduced_width, num_tile_rows, args.k);

    // The multi-core path takes the first (and only supported) core range below; a
    // malformed grid (zero or multiple ranges) falls back to single-core here and is
    // reported loudly by validate_on_program_cache_miss. Guarding before ranges().at(0)
    // matters because program-hash computation can reach this before validation runs.
    multicore_supported &= (args.sub_core_grids.ranges().size() == 1);

    // Check requirement #4: Memory and core availability constraints
    // Only perform expensive verification if basic requirements are met
    if (multicore_supported) {
        auto* device = input_tensor.device();

        // Determine data formats for memory cost calculation
        const tt::DataFormat value_cb_data_format =
            tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
        // The index CB uses UInt16 for 16-bit indices and UInt32 for 32-bit indices (UINT32/INT32 share the
        // same 4-byte layout). Multi-core supports both widths; use the width that will actually be allocated
        // so the memory-cost analysis reflects the real footprint.
        const tt::DataFormat index_cb_data_format = index_cb_data_format_for(args, tensor_args);

        // Calculate tile sizes for memory cost analysis
        const uint32_t value_tile_size = tile_size(value_cb_data_format);
        const uint32_t index_tile_size = tile_size(index_cb_data_format);

        const auto core_range = args.sub_core_grids.ranges().at(0);

        // Perform comprehensive multi-core feasibility analysis
        // This checks: memory constraints, core availability, work divisibility,
        // and ensures optimal core grid arrangement is possible
        multicore_supported &= verify_multi_core_cost(
            reduced_width,                            // Total width to process
            ttnn::prim::constants::min_dim_per_core,  // Minimum split size
            reduced_width / 2,                        // Maximum split size
            args.k,                                   // Number of top elements
            core_range,                               // Available core grid
            device->l1_size_per_core(),               // L1 memory per core
            value_tile_size,                          // Value tile memory size
            index_tile_size,                          // Index tile memory size
            input_tensor.tensor_spec().tile().get_width());
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

    TT_FATAL(args.k != 0, "K must be non-zero");

    // The stable bitonic network is only implemented in the WH/BH LLKs; the Quasar LLK
    // static_asserts STABLE_SORT == false. Reject it here so the caller gets an actionable error
    // instead of a kernel JIT failure.
    if (args.stable) {
        const auto arch = input_tensor.device()->arch();
        TT_FATAL(
            arch == tt::ARCH::WORMHOLE_B0 || arch == tt::ARCH::BLACKHOLE,
            "TopK stable=true is not supported on {}: the bitonic top-k LLK only implements the stable "
            "network on Wormhole and Blackhole",
            arch);
    }

    {
        const int8_t logical_rank = static_cast<int8_t>(input_tensor.logical_shape().rank());
        const int8_t last_dim = logical_rank - 1;
        TT_FATAL(
            args.dim == -1 || args.dim == last_dim,
            "TopK device operation expects reduction on the last dimension (dim=-1 or dim={} for logical rank "
            "{}), got {})",
            last_dim,
            logical_rank,
            args.dim);
    }

    // Memory configuration validation
    TT_FATAL(args.output_memory_config.is_sharded() == false, "Sharded implementation not supported yet");

    // Tensor layout validation
    TT_FATAL(input_tensor.layout() == Layout::TILE, "The input must be in tiled format");

    // Data type validation
    const auto input_tensor_dtype = input_tensor.dtype();
    TT_FATAL(
        input_tensor_dtype == DataType::BFLOAT16 || input_tensor_dtype == DataType::BFLOAT8_B ||
            input_tensor_dtype == DataType::FLOAT32,
        "Input tensor must be BFLOAT16, BFLOAT8_B, or FLOAT32, got: {}",
        input_tensor_dtype);

    // Optional indices tensor validation (for pre-allocated indices)
    if (indices_tensor.has_value()) {
        TT_FATAL(
            indices_tensor->layout() == Layout::TILE,
            "Optional indices tensor must be in tiled format, got: {}",
            indices_tensor->layout());
        const auto indices_tensor_dtype = indices_tensor->dtype();
        TT_FATAL(
            indices_tensor_dtype == DataType::UINT16 || indices_tensor_dtype == DataType::UINT32 ||
                indices_tensor_dtype == DataType::INT32,
            "Optional indices tensor must be UINT16, UINT32, or INT32, got: {}",
            indices_tensor_dtype);
        // fp32 input forces UINT32 index CBs (see compute_output_specs); UINT16 indices would be wrong.
        TT_FATAL(
            !(input_tensor_dtype == DataType::FLOAT32 && indices_tensor_dtype == DataType::UINT16),
            "Optional indices tensor must be UINT32 when input tensor is FLOAT32, got UINT16");
        // The reader reads this tensor into the index CB one entry at a time, so the widths must match.
        // UINT16 is the only 16-bit index dtype; UINT32 and INT32 are both 32-bit and interchangeable.
        const DataType resolved_index_dtype = resolve_index_dtype(args, tensor_args);
        TT_FATAL(
            (indices_tensor_dtype == DataType::UINT16) == (resolved_index_dtype == DataType::UINT16),
            "Optional indices tensor must be the same width as the output indices dtype {}, got: {}",
            resolved_index_dtype,
            indices_tensor_dtype);
        // The reader kernels page a caller-supplied indices tensor with the input's page index
        // (page_id = i * Wt + j), so the two must describe the same tile grid: same padded width
        // and same total number of tiles. A narrower indices tensor is read past the end of its
        // buffer and returns indices that do not belong to the returned values. The composite
        // topk() front-end checks the logical shapes too, but ttnn::prim::topk is a public entry
        // point of its own. Ranks may legitimately differ (the front-end normalizes only the input
        // to 4D), so compare the tile grid rather than the full shape.
        TT_FATAL(
            indices_tensor->padded_shape()[-1] == input_tensor.padded_shape()[-1] &&
                indices_tensor->physical_volume() == input_tensor.physical_volume(),
            "Optional indices tensor must span the same tile grid as the input tensor, got shape: {}, expected: {}",
            indices_tensor->padded_shape(),
            input_tensor.padded_shape());
    }

    // Preallocated output tensor validation
    if (preallocated_outputs.has_value()) {
        const auto& output_tensor0 = std::get<0>(preallocated_outputs.value());  // Values tensor
        const auto& output_tensor1 = std::get<1>(preallocated_outputs.value());  // Indices tensor
        TT_FATAL(
            output_tensor0.layout() == Layout::TILE,
            "Preallocated output tensor must be in tiled format, got: {}",
            output_tensor0.layout());
        TT_FATAL(
            output_tensor1.layout() == Layout::TILE,
            "Preallocated indices tensor must be in tiled format, got: {}",
            output_tensor1.layout());
        const auto output_tensor0_dtype = output_tensor0.dtype();
        const auto output_tensor1_dtype = output_tensor1.dtype();
        TT_FATAL(
            output_tensor0_dtype == DataType::BFLOAT16 || output_tensor0_dtype == DataType::BFLOAT8_B ||
                output_tensor0_dtype == DataType::FLOAT32,
            "Preallocated output tensor must be BFLOAT16, BFLOAT8_B, or FLOAT32 got: {}",
            output_tensor0_dtype);
        TT_FATAL(
            output_tensor1_dtype == DataType::UINT16 || output_tensor1_dtype == DataType::UINT32 ||
                output_tensor1_dtype == DataType::INT32,
            "Preallocated indices tensor must be UINT16, UINT32, or INT32 got: {}",
            output_tensor1_dtype);
        // The preallocated indices tensor sets the index width for the whole op. A 16-bit one
        // on an input that needs 32 bits wraps past 65535 and returns the wrong columns.
        const bool indices_too_narrow =
            output_tensor1_dtype == DataType::UINT16 && is_uint32_index_required(input_tensor, args.dim);
        TT_FATAL(
            !indices_too_narrow,
            "Preallocated indices tensor must be 32-bit (UINT32 or INT32) for this input, got: {}",
            output_tensor1_dtype);
        TT_FATAL(
            output_tensor0_dtype == input_tensor_dtype,
            "Preallocated output tensor dtype must match input tensor dtype. Got output: {}, input: {}",
            output_tensor0_dtype,
            input_tensor_dtype);
    }

    ReduceOpDeviceGridValidationOptions topk_grid_opts;
    topk_grid_opts.sub_grid_contained_in_device_grid = &args.sub_core_grids;
    topk_grid_opts.sub_grid_label = "sub_core_grids";
    validate_reduce_op_tensor(input_tensor, "TopK", "input", &topk_grid_opts);
    if (indices_tensor.has_value()) {
        validate_reduce_op_tensor(indices_tensor.value(), "TopK", "indices");
    }
    if (preallocated_outputs.has_value()) {
        validate_reduce_op_tensor(std::get<0>(preallocated_outputs.value()), "TopK", "preallocated_values");
        validate_reduce_op_tensor(std::get<1>(preallocated_outputs.value()), "TopK", "preallocated_indices");
    }
    // Execution feasibility validation
    // Verify that the operation can be executed with available hardware resources
    bool can_run = false;
    // 16-bit indices are used only when the resolved index dtype is UINT16 (auto for small dims, or a
    // preallocated UINT16 output); UINT32/INT32 outputs use the 32-bit path with a correspondingly larger tile.
    bool uint16_output = (resolve_index_dtype(args, tensor_args) == DataType::UINT16);

    // Try multi-core execution first when the shape/K structure is multi-core
    // eligible — the same shared predicate select_program_factory uses, so the
    // allocator-aware L1 check and the core-range validation below cover exactly
    // the shapes that will actually run the multi-core factory.
    const uint32_t reduced_width = input_shape[args.dim];
    const uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    const uint32_t num_tile_rows = input_shape.volume() / reduced_width / tile_height;
    if (topk_multicore_structurally_eligible(reduced_width, num_tile_rows, args.k)) {
        auto* device = input_tensor.device();

        // Set up data formats for memory cost calculations
        tt::DataFormat value_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
        tt::DataFormat index_cb_data_format = index_cb_data_format_for(args, tensor_args);

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
            reduced_width,                            // Dimension size
            ttnn::prim::constants::min_dim_per_core,  // Min split size
            reduced_width / 2,                        // Max split size
            args.k,                                   // Top-K value
            core_range,                               // Available cores
            device->allocator()->get_statistics(tt::tt_metal::BufferType::L1).largest_free_block_bytes,  // L1 memory
            value_tile_size,  // Value tile size
            index_tile_size,  // Index tile size
            input_tensor.tensor_spec().tile().get_width());

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

    // Create values tensor specification (same data type as input)
    const auto values_spec = tt::tt_metal::TensorSpec(
        output_shape, TensorLayout(input_tensor.dtype(), PageConfig(Layout::TILE), args.output_memory_config));

    const DataType index_dtype = resolve_index_dtype(args, tensor_args);
    const auto index_spec = tt::tt_metal::TensorSpec(
        output_shape, TensorLayout(index_dtype, PageConfig(Layout::TILE), args.output_memory_config));

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
    bool stable,
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
            .stable = stable,
            .output_memory_config = memory_config,
            .sub_core_grids = sub_core_grids},
        TopkInputs{
            .input = input_tensor, .indices = indices_tensor, .preallocated_outputs = preallocated_output_tensors});
}
}  // namespace ttnn::prim
