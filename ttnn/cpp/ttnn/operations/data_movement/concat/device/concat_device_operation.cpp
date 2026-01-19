// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "concat_device_operation.hpp"
#include "ttnn/device_operation.hpp"
#include "concat_program_factory.hpp"

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/data_movement/clone/clone.hpp"
#include "ttnn/operations/core/core.hpp"  // for to_layout
#include <tt-logger/tt-logger.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::concat {

ConcatDeviceOperation::program_factory_t ConcatDeviceOperation::select_program_factory(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    if (tensor_args.input_tensors.empty()) {
        TT_FATAL(false, "ConcatDeviceOperation: input_tensors cannot be empty");
    }

    const auto& input_tensors = tensor_args.input_tensors;
    const bool is_sharded = input_tensors[0].is_sharded();

    if (!is_sharded) {
        return program::ConcatProgramFactory{};
    }

    // Sharded cases - determine which specific factory to use
    const bool output_is_sharded = args.output_mem_config.is_sharded();

    if (output_is_sharded) {
        // Sharded-to-sharded (s2s) cases
        if (input_tensors.size() == 2) {
            // Optimized 2-tensor case
            TT_FATAL(
                input_tensors[0].layout() == input_tensors[1].layout(),
                "Expected all input tensors to have the same layout for 2-tensor sharded concat");

            if (input_tensors[0].layout() == Layout::ROW_MAJOR) {
                return program::ConcatS2SRMProgramFactory{};
            }
            return program::ConcatS2STiledProgramFactory{};

        }  // Multi-tensor s2s case
        return program::ConcatS2SMultiProgramFactory{};
    }
    // Sharded-to-interleaved (s2i) case
    return program::ConcatS2IProgramFactory{};
}

void ConcatDeviceOperation::validate_on_program_cache_hit(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    validate_on_program_cache_miss(args, tensor_args);
}

void ConcatDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace tt::constants;

    const auto& input_tensors = tensor_args.input_tensors;
    TT_FATAL(!input_tensors.empty(), "need 1 or more tensors");

    const auto& first_input = input_tensors[0];
    auto shape_first = first_input.padded_shape();
    TT_FATAL(args.dim < shape_first.rank(), "ConcatDeviceOperation dim specified is larger than input tensor rank.");
    shape_first[args.dim] = 0;
    bool shard_first = input_tensors[0].is_sharded();
    bool warn_about_alignment = false;

    for (int i = 0; i < input_tensors.size(); i++) {
        const Tensor& in_ref = input_tensors[i];
        TT_FATAL(in_ref.buffer(), "Operand to concat needs to be allocated in a buffer on device.");
        TT_FATAL(in_ref.device(), "Operand to concat needs to be on device.");
        TT_FATAL(in_ref.device() == first_input.device(), "Operands to concat need to be on the same device.");
        TT_FATAL(in_ref.layout() == first_input.layout(), "All Tensors should have same layouts.");
        TT_FATAL(in_ref.dtype() == first_input.dtype(), "All Tensors should have same dtypes.");
        auto curr_shape = in_ref.padded_shape();
        TT_FATAL(curr_shape.rank() == shape_first.rank(), "Input tensor ranks must be equal");
        curr_shape[args.dim] = 0;
        // last tensor can support without any kernel changes
        if (in_ref.layout() == Layout::TILE) {
            const uint32_t logical_dim = in_ref.logical_shape()[args.dim];
            const uint32_t padded_dim = in_ref.padded_shape()[args.dim];
            if (logical_dim != padded_dim) {
                warn_about_alignment = true;
            }
        }
        TT_FATAL(curr_shape == shape_first, "concat tensors differ in shape across non-concat dimensions.");
        TT_FATAL(in_ref.is_sharded() == shard_first, "All tensors must be sharded or all must be interleaved");
        if (shard_first) {
            TT_FATAL(in_ref.shard_spec().has_value(), "Sharded tensors must have a shard spec.");
            TT_FATAL(
                in_ref.shard_spec().value().grid == first_input.shard_spec().value().grid,
                "Sharded tensors must have the same grid.");
            TT_FATAL(
                in_ref.memory_config().memory_layout() == first_input.memory_config().memory_layout(),
                "Sharded tensors must have the same memory layout.");
            // TODO(jerrysky3): Remove this when we replace the two tensors concat kernel with the general one.
            TT_FATAL(
                input_tensors.size() > 2 || in_ref.memory_config().memory_layout() != TensorMemoryLayout::WIDTH_SHARDED,
                "Width sharded inputs are not supported for two tensors concat yet");
            TT_FATAL(
                in_ref.memory_config().memory_layout() != TensorMemoryLayout::BLOCK_SHARDED,
                "Block sharded inputs are not supported");
        }
    }
    if (warn_about_alignment) {
        log_warning(
            tt::LogOp,
            "ttnn.concat: Tile padding along concatenated dim ({}) is not "
            "directly supported. ttnn.concat will proceed by converting to "
            "row-major then retilizing. This may have adverse performance impacts.",
            args.dim);
    }
    if (shard_first) {
        const auto memory_layout = first_input.memory_config().memory_layout();
        TT_FATAL(
            args.output_mem_config.memory_layout() == memory_layout,
            "Sharded output and inputs must have the same memory layout.");
        TT_FATAL(args.output_mem_config.is_sharded(), "Output must be sharded if input is sharded.");
        TT_FATAL(
            args.output_mem_config.shard_spec().value().grid == first_input.shard_spec().value().grid,
            "Sharded output and inputs must have the same grid.");
        if (args.dim == shape_first.rank() - 1) {
            TT_FATAL(
                memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
                "Only support width concat on height-sharded tensors.");
        } else if (args.dim == shape_first.rank() - 2) {
            TT_FATAL(
                memory_layout == TensorMemoryLayout::WIDTH_SHARDED,
                "Only support height concat on width-sharded tensors.");
        } else {
            TT_FATAL(false, "Only width or height concat on sharded tensors");
        }
        TT_FATAL(
            args.groups == 1 || memory_layout == TensorMemoryLayout::HEIGHT_SHARDED,
            "Groups > 1 is only supported on height-sharded tensors (groups={} and memory_layout={} was provided)",
            args.groups,
            memory_layout);
    }
}

TensorSpec ConcatDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const Tensor& ref_in_tensor = tensor_args.input_tensors.at(0);
    ttnn::Shape shape_out = ref_in_tensor.logical_shape();
    shape_out[args.dim] = 0;
    for (const Tensor& in_ref : tensor_args.input_tensors) {
        ttnn::Shape curr_shape = in_ref.logical_shape();
        shape_out[args.dim] += curr_shape[args.dim];
    }

    return TensorSpec(
        shape_out, TensorLayout(ref_in_tensor.dtype(), PageConfig(ref_in_tensor.layout()), args.output_mem_config));
}

Tensor ConcatDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args) {
    auto output_spec = compute_output_specs(operation_attributes, tensor_args);
    return create_device_tensor(output_spec, tensor_args.input_tensors[0].device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>>
ConcatDeviceOperation::create_op_performance_model(
    const std::vector<Tensor>& input_tensors,
    const std::vector<std::optional<const Tensor>>& /*optional_input_tensors*/,
    std::vector<Tensor>& output_tensors) {
    TT_FATAL(
        !input_tensors.empty(), "ConcatDeviceOperation::create_op_performance_model: input_tensors cannot be empty");
    TT_FATAL(
        !output_tensors.empty(), "ConcatDeviceOperation::create_op_performance_model: output_tensors cannot be empty");

    const auto& input_tensor = input_tensors.at(0);
    const auto& output_tensor = output_tensors.at(0);

    // Use common_tm_bw_model with concat_op=true for concat-specific bandwidth modeling
    int ideal_dev_clock_cycles = common_tm_bw_model(input_tensor, output_tensor, false, 0, false, false, false, true);

    tt::tt_metal::operation::OpPerformanceModelGeneral<std::vector<Tensor>> result(
        input_tensors, output_tensors, ideal_dev_clock_cycles);
    return result;
}
}  // namespace ttnn::operations::data_movement::concat

namespace ttnn::operations::data_movement {

namespace {

using namespace tt::constants;

// Calculate maximum tensors per concat based on runtime args limit
uint32_t calculate_max_tensors_per_concat(const std::vector<Tensor>& input_tensors, const std::int64_t dim) {
    // Runtime args are limited by available L1 kernel config memory.
    // The general limit is 341 uint32_t args (from kernel_types.hpp:max_runtime_args),
    // but concat kernels are compiled with NUM_RUNTIME_ARGS=256.
    //
    // NOTE: The limit applies to the COMBINED total of compile-time + runtime args.
    //
    // Args breakdown for concat_multi_core (interleaved, non-sharded):
    //   Compile-time args:
    //     - src0_cb_index: 1
    //     - num_input_tensors: 1
    //     - page_size_per_tensor[N]: N
    //     - TensorAccessorArgs[N]: N (for interleaved buffers, 1 arg per tensor)
    //   Total compile-time: 2 + 2N
    //
    //   Runtime args per core:
    //     - num_pages_per_core: 1
    //     - curr_tensor: 1
    //     - curr_tensor_id: 1
    //     - src_addr[N]: N
    //     - num_pages_per_block[N]: N
    //     - page_id_per_tensor[N]: N
    //   Total runtime: 3 + 3N
    //
    //   TOTAL ARGS VALIDATED: (2 + 2N) + (3 + 3N) = 5 + 5N
    //
    // Empirical evidence (GitHub issue #22845):
    //   - N=50: SUCCESS (5 + 250 = 255 ≤ 256) ✓
    //   - N=55: FAILURE (5 + 275 = 280 > 256) ✗
    //     Error: "278 unique+common runtime args ... Max allowable is 256"
    //
    // Formula: 5 + 5N ≤ 256  =>  N ≤ 50.2  =>  N_max = 50
    //
    // IMPORTANT: This limit does NOT depend on:
    //   - Tensor shape/size (only number of tensors matters)
    //   - Hardware architecture (Wormhole vs Blackhole use same limit)
    //   - Tensor dtype (bfloat16 vs float32, etc.)
    //
    // It DOES depend on:
    //   - Memory layout (sharded vs interleaved - different kernels)
    //   - Tensor layout (ROW_MAJOR vs TILE - affects TensorAccessorArgs)

    const bool is_sharded = input_tensors[0].is_sharded();

    // Effective kernel limit (empirically determined to be 256, not 341)
    constexpr uint32_t effective_args_limit = 256;

    uint32_t base_args;
    uint32_t args_per_tensor;

    if (is_sharded) {
        // Sharded concat uses different kernels with different arg patterns
        // s2s_concat_multi_core (line 522-523 in concat_program_factory.cpp):
        //   Compile-time: cb_dst_id, page_size, output_stride, num_input_tensors = 4
        //   Runtime per kernel: 4 * num_input_tensors (per reader/writer)
        //
        // Using conservative estimate based on runtime args dominance
        base_args = 4;
        args_per_tensor = 4;

        // Safety margin: reduce by 10% for sharded due to potential additional overhead
        uint32_t theoretical_max = (effective_args_limit - base_args) / args_per_tensor;
        uint32_t safe_max = static_cast<uint32_t>(theoretical_max * 0.9);

        log_debug(
            tt::LogOp, "ttnn.concat: Sharded concat - theoretical_max = {}, safe_max = {}", theoretical_max, safe_max);

        return std::max(2u, safe_max);
    }  // Interleaved concat - precise calculation
    // Formula: 5 + 5N ≤ 256
    const Layout layout = input_tensors[0].layout();
    base_args = 5;
    args_per_tensor = 5;

    uint32_t max_tensors = (effective_args_limit - base_args) / args_per_tensor;

    // Verify our calculation matches empirical evidence
    uint32_t total_args_at_limit = base_args + (args_per_tensor * max_tensors);
    log_debug(
        tt::LogOp,
        "ttnn.concat: Interleaved concat - max_tensors = {}, total_args = {} (limit = {}, layout = {}, dim = {})",
        max_tensors,
        total_args_at_limit,
        effective_args_limit,
        layout,
        dim);
    // Ensure variables are used even if log_debug is compiled out
    (void)total_args_at_limit;
    (void)layout;

    // Should be exactly 50 based on our formula: (256 - 5) / 5 = 50.2 => 50
    TT_FATAL(max_tensors == 50, "Unexpected max_tensors calculation: expected 50, got {}", max_tensors);

    return max_tensors;
}
}  // anonymous namespace

Tensor concat_impl(
    const std::vector<Tensor>& input_tensors,
    const std::int64_t dim,
    const unsigned int groups,
    const MemoryConfig& output_mem_config) {
    TT_FATAL(!input_tensors.empty(), "need 1 or more tensors");
    for (const auto& input_tensor : input_tensors) {
        TT_FATAL(input_tensor.storage_type() == StorageType::DEVICE, "Input tensor must be on device");
    }
    if (input_tensors.size() == 1) {
        // Single tensor case - just ensure it has the correct memory config
        const auto& input = input_tensors[0];
        if (input.memory_config() != output_mem_config) {
            return ttnn::clone(input, std::nullopt, output_mem_config, std::nullopt);
        }
        return input;
    }

    // Handle large number of tensors by splitting into batches
    const uint32_t max_tensors_per_concat = calculate_max_tensors_per_concat(input_tensors, dim);
    if (input_tensors.size() > max_tensors_per_concat) {
        // Split into batches and concat each batch
        std::vector<Tensor> intermediate_results;
        const size_t num_batches = tt::div_up(input_tensors.size(), max_tensors_per_concat);
        intermediate_results.reserve(num_batches);

        log_debug(
            tt::LogOp,
            "ttnn.concat: Processing {} tensors in {} batches of up to {} tensors each",
            input_tensors.size(),
            num_batches,
            max_tensors_per_concat);

        for (size_t i = 0; i < input_tensors.size(); i += max_tensors_per_concat) {
            size_t batch_size =
                std::min(static_cast<size_t>(max_tensors_per_concat), static_cast<size_t>(input_tensors.size() - i));

            // Create batch vector with references to input tensors
            std::vector<Tensor> batch;
            batch.reserve(batch_size);
            for (size_t j = i; j < i + batch_size; ++j) {
                batch.push_back(input_tensors[j]);
            }

            // Recursively concat this batch
            Tensor batch_result = concat_impl(batch, dim, groups, output_mem_config);
            intermediate_results.push_back(std::move(batch_result));

            // Clear batch to release references
            batch.clear();
        }

        // Final concat
        return concat_impl(intermediate_results, dim, groups, output_mem_config);
    }

    uint32_t ref_rank = input_tensors[0].padded_shape().rank();
    uint32_t normalized_dim = input_tensors[0].padded_shape().get_normalized_index(dim);

    if (input_tensors[0].is_sharded()) {
        return ttnn::prim::concat(input_tensors, dim, groups, output_mem_config);
    }
    if (input_tensors[0].layout() == Layout::ROW_MAJOR && normalized_dim == ref_rank - 1) {
        for (const auto& input_tensor : input_tensors) {
            TT_FATAL(
                (input_tensor.padded_shape()[dim] * input_tensor.element_size()) % input_tensor.buffer()->alignment() ==
                    0,
                "Current concat implementation requires aligned last dim when concatting on last dim");
        }
    }
    // Determine target layout by checking all inputs
    // Start with first input's layout, but may need to fall back to ROW_MAJOR
    Layout target_layout = input_tensors[0].layout();

    // Check all inputs - if any ROW_MAJOR input cannot be tiled, use ROW_MAJOR for all
    for (const auto& input_tensor : input_tensors) {
        if (input_tensor.layout() == Layout::ROW_MAJOR) {
            const auto& input_shape = input_tensor.padded_shape();
            if (input_shape.rank() < 2 || input_shape[-2] % TILE_HEIGHT != 0 || input_shape[-1] % TILE_WIDTH != 0) {
                target_layout = Layout::ROW_MAJOR;
                break;
            }
        }
    }

    // Format all inputs to target layout
    std::vector<Tensor> formatted_tensors;
    formatted_tensors.reserve(input_tensors.size());

    for (const auto& input_tensor : input_tensors) {
        if (input_tensor.layout() == target_layout) {
            // Already in target layout
            formatted_tensors.push_back(input_tensor);
        } else {
            formatted_tensors.push_back(ttnn::to_layout(input_tensor, target_layout));
        }
    }

    return ttnn::prim::concat(formatted_tensors, dim, groups, output_mem_config);
}

}  // namespace ttnn::operations::data_movement

namespace ttnn::prim {
ttnn::operations::data_movement::concat::ConcatDeviceOperation::tensor_return_value_t concat(
    const std::vector<Tensor>& input_tensors,
    std::int64_t dim,
    unsigned int groups,
    const tt::tt_metal::MemoryConfig& output_mem_config) {
    using OperationType = ttnn::operations::data_movement::concat::ConcatDeviceOperation;
    uint32_t normalized_dim = input_tensors[0].padded_shape().get_normalized_index(dim);
    return ttnn::device_operation::launch<OperationType>(
        OperationType::operation_attributes_t{
            .dim = normalized_dim,
            .groups = groups,
            .output_mem_config = output_mem_config,
        },
        OperationType::tensor_args_t{.input_tensors = input_tensors});
}
}  // namespace ttnn::prim
