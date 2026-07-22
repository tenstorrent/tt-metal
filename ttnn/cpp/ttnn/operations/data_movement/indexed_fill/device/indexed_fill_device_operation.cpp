// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_device_operation.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/device_operation.hpp"
#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_program_factory.hpp"
#include "ttnn/operations/data_movement/indexed_fill/device/indexed_fill_utils.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

using namespace tt::tt_metal;

namespace {
void check_shard_tile_alignment(const tt::tt_metal::ShardSpec& s, const char* name) {
    TT_FATAL(
        s.shape[0] % tt::constants::TILE_HEIGHT == 0,
        "indexed_fill (TILE): {} shard_spec.shape[0] ({}) must be divisible by TILE_HEIGHT ({})",
        name,
        s.shape[0],
        tt::constants::TILE_HEIGHT);
    TT_FATAL(
        s.shape[1] % tt::constants::TILE_WIDTH == 0,
        "indexed_fill (TILE): {} shard_spec.shape[1] ({}) must be divisible by TILE_WIDTH ({})",
        name,
        s.shape[1],
        tt::constants::TILE_WIDTH);
}
}  // namespace

namespace ttnn::prim {

void IndexedFillDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    const auto& input_tensor_a = tensor_args.input_tensor_a;
    const auto& input_tensor_b = tensor_args.input_tensor_b;
    const auto& batch_ids = tensor_args.batch_id;
    auto input_tensor_a_shape = input_tensor_a.padded_shape();
    auto input_tensor_b_shape = input_tensor_b.padded_shape();

    TT_FATAL(
        input_tensor_a.layout() == Layout::ROW_MAJOR || input_tensor_a.layout() == Layout::TILE,
        "indexed_fill: input_a layout must be ROW_MAJOR or TILE, got {}",
        input_tensor_a.layout());
    TT_FATAL(input_tensor_b.layout() == input_tensor_a.layout(), "Inputs must be same layout");

    // Rank-4 is required: compute_geometry and the shard-local kernel use hardcoded
    // padded_shape()[1] / [2] indexing that assumes [B, H, W, D] layout.
    TT_FATAL(
        input_tensor_a_shape.rank() == 4,
        "indexed_fill: only rank-4 tensors are supported; got rank {}",
        input_tensor_a_shape.rank());

    if (input_tensor_a.layout() == Layout::TILE) {
        TT_FATAL(
            input_tensor_a_shape[-2] % tt::constants::TILE_HEIGHT == 0,
            "indexed_fill (TILE): padded dim[-2] ({}) must be divisible by TILE_HEIGHT ({})",
            input_tensor_a_shape[-2],
            tt::constants::TILE_HEIGHT);
        TT_FATAL(
            input_tensor_a_shape[-1] % tt::constants::TILE_WIDTH == 0,
            "indexed_fill (TILE): padded dim[-1] ({}) must be divisible by TILE_WIDTH ({})",
            input_tensor_a_shape[-1],
            tt::constants::TILE_WIDTH);

        if (input_tensor_a.is_sharded() && input_tensor_a.memory_config().shard_spec().has_value()) {
            check_shard_tile_alignment(*input_tensor_a.memory_config().shard_spec(), "input_a");
        }
        if (input_tensor_b.is_sharded() && input_tensor_b.memory_config().shard_spec().has_value()) {
            check_shard_tile_alignment(*input_tensor_b.memory_config().shard_spec(), "input_b");
        }
        if (args.output_mem_config.is_sharded() && args.output_mem_config.shard_spec().has_value()) {
            check_shard_tile_alignment(*args.output_mem_config.shard_spec(), "output");
        }
    }

    {
        const int64_t rank = static_cast<int64_t>(input_tensor_a_shape.rank());
        for (int64_t d = 0; d < rank; ++d) {
            if (d == args.dim) {
                continue;
            }
            TT_FATAL(
                input_tensor_a_shape[d] == input_tensor_b_shape[d],
                "indexed_fill: input_a and input_b must match on every dim except dim {}; "
                "mismatch at dim {}: input_a.size({})={}, input_b.size({})={}",
                args.dim,
                d,
                d,
                static_cast<uint32_t>(input_tensor_a_shape[d]),
                d,
                static_cast<uint32_t>(input_tensor_b_shape[d]));
        }
        TT_FATAL(
            input_tensor_b_shape[args.dim] == batch_ids.padded_shape()[-1],
            "indexed_fill: input_b.size(dim={})={} must equal the number of batch_ids ({})",
            args.dim,
            static_cast<uint32_t>(input_tensor_b_shape[args.dim]),
            batch_ids.padded_shape()[-1]);
    }
    TT_FATAL(batch_ids.layout() == Layout::ROW_MAJOR, "Batch IDs must be ROW MAJOR");
    TT_FATAL(
        batch_ids.dtype() == DataType::UINT32 || batch_ids.dtype() == DataType::INT32,
        "indexed_fill: batch_id must be UINT32 or INT32, got {}",
        batch_ids.dtype());
    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands to Index Fill need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands to Index Fill need to be allocated in buffers on device!");

    auto require_interleaved_if_not_sharded = [](bool is_sharded, TensorMemoryLayout layout, const char* name) {
        if (!is_sharded) {
            TT_FATAL(
                layout == TensorMemoryLayout::INTERLEAVED,
                "indexed_fill: non-sharded {} must be INTERLEAVED, got {}",
                name,
                layout);
        }
    };
    require_interleaved_if_not_sharded(
        input_tensor_a.is_sharded(), input_tensor_a.memory_config().memory_layout(), "input_a");

    // For WIDTH_SHARDED / BLOCK_SHARDED input_a the shard-local path requires the output to
    // use the same sharding layout (same grid + same shard shape). INTERLEAVED or
    // HEIGHT_SHARDED outputs are handled by the native / generic HEIGHT path.
    if (input_tensor_a.is_sharded() &&
        (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
         input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED)) {
        // Resolve the output memory config so that configs that omit shard_spec (relying on
        // compute_output_specs() to derive one) are expanded before the comparison. This avoids
        // incorrectly rejecting valid callers that pass a shard_spec-less MemoryConfig.
        const auto resolved_out = ttnn::operations::data_movement::indexed_fill::resolve_output_memory_config(
            input_tensor_a, input_tensor_a.padded_shape(), args.output_mem_config);

        TT_FATAL(
            resolved_out.is_sharded() && resolved_out.memory_layout() == input_tensor_a.memory_config().memory_layout(),
            "indexed_fill: WIDTH_SHARDED / BLOCK_SHARDED input_a requires the output to have "
            "the same sharding layout as input_a");

        // If input_a has an explicit shard_spec, verify the grid and shape match. When input_a
        // only has nd_shard_spec the detailed comparison is skipped.
        const bool output_matches =
            !input_tensor_a.memory_config().shard_spec().has_value() || !resolved_out.shard_spec().has_value() ||
            (resolved_out.shard_spec()->grid == input_tensor_a.memory_config().shard_spec()->grid &&
             resolved_out.shard_spec()->shape == input_tensor_a.memory_config().shard_spec()->shape);
        TT_FATAL(
            output_matches,
            "indexed_fill: WIDTH_SHARDED / BLOCK_SHARDED input_a requires the output to have "
            "the same shard grid and shard shape as input_a");

        // For BLOCK_SHARDED the shard-local kernel divides batches evenly across shard rows
        // (total_batches_per_core = B / n_y). Require B to be divisible by n_y so no batch
        // is silently skipped.
        if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED &&
            input_tensor_a.memory_config().shard_spec().has_value()) {
            const uint32_t n_y = input_tensor_a.memory_config().shard_spec()->grid.bounding_box().grid_size().y;
            TT_FATAL(
                n_y > 0 && input_tensor_a_shape[0] % n_y == 0,
                "indexed_fill: BLOCK_SHARDED input_a requires batch dimension B ({}) to be "
                "divisible by the shard-grid height n_y ({})",
                input_tensor_a_shape[0],
                n_y);
        }

        // The shard-local kernel assumes even sharding (shard_ppb is uniform across all cores).
        // Uneven sharding would cause the last core to process a different number of pages,
        // producing wrong results.  Reject early with a clear message.
        TT_FATAL(
            !ttnn::operations::data_movement::indexed_fill::is_uneven(input_tensor_a.tensor_spec()),
            "indexed_fill: WIDTH_SHARDED / BLOCK_SHARDED input_a must have even sharding "
            "(shard shape must divide the padded tensor shape evenly); "
            "the shard shape or tensor shape must be adjusted to make sharding even");
    }

    require_interleaved_if_not_sharded(
        input_tensor_b.is_sharded(), input_tensor_b.memory_config().memory_layout(), "input_b");
    require_interleaved_if_not_sharded(batch_ids.is_sharded(), batch_ids.memory_config().memory_layout(), "batch_id");
    require_interleaved_if_not_sharded(
        args.output_mem_config.is_sharded(), args.output_mem_config.memory_layout(), "output");
}

IndexedFillDeviceOperation::spec_return_value_t IndexedFillDeviceOperation::compute_output_specs(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    using namespace ttnn::operations::data_movement::indexed_fill;

    const auto& input_tensor = tensor_args.input_tensor_a;
    const auto output_shape = input_tensor.logical_shape();
    const auto output_layout = input_tensor.layout();

    const auto resolved_mem_config =
        resolve_output_memory_config(input_tensor, input_tensor.padded_shape(), args.output_mem_config);

    return tt::tt_metal::TensorSpec(
        output_shape, TensorLayout(input_tensor.dtype(), PageConfig(output_layout), resolved_mem_config));
}

IndexedFillDeviceOperation::tensor_return_value_t IndexedFillDeviceOperation::create_output_tensors(
    const operation_attributes_t& args, const tensor_args_t& tensor_args) {
    return create_device_tensor(compute_output_specs(args, tensor_args), tensor_args.input_tensor_a.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<IndexedFillDeviceOperation::tensor_return_value_t>
IndexedFillDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*args*/, const tensor_args_t& tensor_args, tensor_return_value_t& output) {
    const auto& input_tensor = tensor_args.batch_id;
    int ideal_dev_clock_cycles = ttnn::operations::data_movement::common_tm_bw_model(input_tensor, output);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, {output}, ideal_dev_clock_cycles);
    return result;
}

IndexedFillDeviceOperation::tensor_return_value_t indexed_fill(
    const Tensor& batch_id,
    const Tensor& input_tensor_a,
    const Tensor& input_tensor_b,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    int64_t dim) {
    auto worker_grid = ttnn::operations::data_movement::indexed_fill::get_indexed_fill_worker_grid(
        input_tensor_a, input_tensor_b, batch_id, std::optional<tt::tt_metal::MemoryConfig>(output_mem_config));

    return ttnn::device_operation::launch<IndexedFillDeviceOperation>(
        IndexedFillDeviceOperation::operation_attributes_t{
            .output_mem_config = output_mem_config,
            .dim = dim,
            .worker_grid = std::move(worker_grid),
        },
        IndexedFillDeviceOperation::tensor_args_t{
            .batch_id = batch_id,
            .input_tensor_a = input_tensor_a,
            .input_tensor_b = input_tensor_b,
        });
}

}  // namespace ttnn::prim
