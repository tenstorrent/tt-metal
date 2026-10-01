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

    // A WIDTH/BLOCK_SHARDED ROW_MAJOR buffer pages at shard width, but the generic path sizes every
    // page at the full tensor width (generic_aligned_page_size takes the max over input_a, input_b
    // and the output). Whichever of the three is sharded is then addressed with the wrong stride
    // and the result is silently wrong, so the generic path is only safe for ROW_MAJOR when none of
    // them is WIDTH/BLOCK_SHARDED. TILE layout is exempt (a tile is one page either way), as is
    // HEIGHT_SHARDED (its shards span the full width). Making the generic path shard-width aware is
    // the real fix; until then these configurations are rejected rather than silently corrupted.
    //
    // The output always inherits input_a's layout (see compute_output_specs), and resolution only
    // fills in a missing shard_spec, so args.output_mem_config's memory_layout is already final.
    const auto is_row_major_width_or_block_sharded = [](Layout layout, const tt::tt_metal::MemoryConfig& mem_config) {
        return layout == Layout::ROW_MAJOR && (mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED ||
                                               mem_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED);
    };

    if (is_row_major_width_or_block_sharded(input_tensor_a.layout(), input_tensor_a.memory_config()) ||
        is_row_major_width_or_block_sharded(input_tensor_b.layout(), input_tensor_b.memory_config()) ||
        is_row_major_width_or_block_sharded(input_tensor_a.layout(), args.output_mem_config)) {
        TT_FATAL(
            args.dim == 0,
            "indexed_fill: a ROW_MAJOR WIDTH_SHARDED/BLOCK_SHARDED input_a, input_b or output is "
            "only supported for dim=0 (got dim={}); other dims take the generic path, which does "
            "not support ROW_MAJOR sharded page addressing",
            args.dim);

        // create_output_tensors() builds the real output from compute_output_specs(), so going
        // through it here (rather than resolve_output_memory_config()) picks up any shard-shape
        // adjustment TensorLayout makes and keeps this in sync with the factory's path choice.
        const auto resolved_out = IndexedFillDeviceOperation::compute_output_specs(args, tensor_args).memory_config();
        TT_FATAL(
            ttnn::operations::data_movement::indexed_fill::is_shard_local_indexed_fill(
                input_tensor_a.tensor_spec(), input_tensor_b.tensor_spec(), resolved_out),
            "indexed_fill: a ROW_MAJOR WIDTH_SHARDED/BLOCK_SHARDED input_a, input_b or output is "
            "only supported via the shard-local path, and this combination is not eligible for it "
            "(the generic fallback does not support ROW_MAJOR sharded page addressing). "
            "Eligibility requires: input_a and output in L1 with the same sharding layout, grid "
            "and shard shape; ROW_MAJOR shard orientation; even sharding; input_b INTERLEAVED or "
            "WIDTH_SHARDED matching input_a's grid and shard width; and for BLOCK_SHARDED a "
            "rectangular shard grid with B divisible by the grid row count");
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
