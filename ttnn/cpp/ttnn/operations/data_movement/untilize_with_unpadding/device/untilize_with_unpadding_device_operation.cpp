// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_device_operation.hpp"
#include "ttnn/device_operation.hpp"

#include <algorithm>

#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/tensor/tensor_ops.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {
// Number of column shards (KW) a legacy 2D ShardSpec's grid represents, in the orientation-aware
// sense used by the cross-shard-type writer (writer_unary_unpad_cross_sharded.cpp): WIDTH_SHARDED's
// grid is always 1xKW, and BLOCK_SHARDED's grid axes swap which one is the logical column-shard
// axis under COL_MAJOR orientation (matches the convention in compute_output_specs() below and the
// per-core kh/kw derivation in the sharded factory).
uint32_t legacy_2d_shard_column_count(const ShardSpec& shard_spec, TensorMemoryLayout memory_layout) {
    if (memory_layout == TensorMemoryLayout::WIDTH_SHARDED) {
        return shard_spec.grid.num_cores();
    }
    CoreRange bbox = shard_spec.grid.bounding_box();
    uint32_t grid_cols = bbox.end_coord.x - bbox.start_coord.x + 1;
    uint32_t grid_rows = bbox.end_coord.y - bbox.start_coord.y + 1;
    if (shard_spec.orientation != ShardOrientation::ROW_MAJOR) {
        std::swap(grid_cols, grid_rows);
    }
    return grid_cols;
}
}  // namespace

UntilizeWithUnpaddingDeviceOperation::program_factory_t UntilizeWithUnpaddingDeviceOperation::select_program_factory(
    const operation_attributes_t& operation_attributes, const tensor_args_t& input) {
    if (input.layout() == Layout::ROW_MAJOR) {
        return UntilizeWithUnpaddingRowMajorProgramFactory{};
    }
    if (input.memory_config().is_sharded()) {
        TT_FATAL(
            !operation_attributes.sub_core_grids.has_value(),
            "Sharded untilize does not support sub core grid specification");
        if (input.shard_spec().has_value()) {
            return UntilizeWithUnpaddingMultiCoreShardedProgramFactory{};
        }
        return UntilizeWithUnpaddingMultiCoreNDShardedProgramFactory{};
    }
    // Sharded output from interleaved input is only implemented in the row-split multi-core writer
    // (writer_unary_stick_layout_split_rows_multicore.cpp, which uses noc_async_write_sharded to
    // split a row across B/W-sharded output shards). The single-core writer and the block-interleaved
    // writer (used for the wide-row heuristic) haven't been updated for that, so force the row-split
    // factory whenever output is sharded - this takes priority over both use_multicore=false and
    // enough_space_height/the wide-row heuristic below, since neither of those paths' writers can
    // produce correct sharded output.
    if (operation_attributes.output_mem_config.is_sharded()) {
        return UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory{};
    }
    if (!operation_attributes.use_multicore) {
        return UntilizeWithUnpaddingSingleCoreProgramFactory{};
    }
    if (!operation_attributes.enough_space_height) {
        return UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory{};
    }
    const auto& a = input;
    const auto& input_shape = a.padded_shape();
    auto* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid =
        operation_attributes.sub_core_grids.has_value() ? operation_attributes.sub_core_grids.value() : default_grid;

    uint32_t num_blocks = input_shape[-1] == 0 ? 0 : a.physical_volume() / input_shape[-1] / tt::constants::TILE_HEIGHT;
    uint32_t num_tiles_per_row = a.padded_shape()[-1] / tt::constants::TILE_WIDTH;

    uint32_t num_tiles_per_col = a.padded_shape()[-2] / tt::constants::TILE_HEIGHT;

    size_t grid_area = available_grid.num_cores();
    auto [ncores, nblocks_per_core] = compute_ncores(grid_area, num_blocks);
    constexpr uint32_t threshold_row_block = 32;
    if (num_tiles_per_row > threshold_row_block &&
        (num_tiles_per_col > threshold_row_block || num_tiles_per_row > num_tiles_per_col)) {
        uint32_t num_blocks_block =
            (a.padded_shape()[-1] * a.padded_shape()[-2]) / (tt::constants::TILE_HEIGHT * tt::constants::TILE_WIDTH);

        auto ncores_wh = compute_ncores_wh(grid_area, num_blocks_block, num_tiles_per_row, num_tiles_per_col);
        if (ncores < ncores_wh.ncores) {
            return UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory{};
        }
    }
    return UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory{};
}

void UntilizeWithUnpaddingDeviceOperation::validate_on_program_cache_miss(
    const operation_attributes_t& operation_attributes, const tensor_args_t& input) {
    const auto& input_tensor_a = input;

    TT_FATAL(input_tensor_a.storage_type() == StorageType::DEVICE, "Operands need to be on device!");
    TT_FATAL(input_tensor_a.buffer() != nullptr, "Operands need to be allocated in buffers on device!");
    TT_FATAL(
        input_tensor_a.layout() == Layout::TILE || input_tensor_a.layout() == Layout::ROW_MAJOR,
        "Can only untilize tile major or row major data");

    if (input_tensor_a.layout() == Layout::ROW_MAJOR) {
        // ROW_MAJOR input has no tile padding to strip, so this degenerates to a plain unpad/copy -
        // handled by UntilizeWithUnpaddingRowMajorProgramFactory, which reuses ttnn::slice's RM
        // reader/writer kernels. Those kernels already split a logical row across multiple shards
        // via noc_async_{read,write}_sharded when needed (see common/kernels/common.hpp), and address
        // the destination via TensorAccessor (sharding-agnostic), so - unlike the TILE-path writers -
        // sharded RM input/output is not blocked by a row-per-page assumption. The output shard spec
        // is used verbatim from the caller (see compute_output_specs()) since RM shards have no
        // tile-alignment constraint to derive.
        //
        // Rank > 4 is rejected here: the public wrapper's squeeze/unsqueeze composite
        // (build_ndiml_untilize_val / untilize_with_unpadding.cpp) unconditionally replaces
        // output_tensor_end with the full input extents for rank > 4 and reshapes the result back
        // to the original shape, discarding any real truncation request regardless of layout. That
        // is a pre-existing gap for the TILE path too, but this device op previously rejected
        // ROW_MAJOR input outright at every rank, so it never surfaced there; reject rank > 4 until
        // the wrapper is taught to propagate output_tensor_end through the reshape.
        TT_FATAL(
            input_tensor_a.logical_shape().rank() <= 4,
            "ROW_MAJOR input with rank > 4 is not yet supported for untilize_with_unpadding (the rank > 4 "
            "squeeze/unsqueeze wrapper does not yet propagate output_tensor_end)");
        if (input_tensor_a.memory_config().is_sharded()) {
            TT_FATAL(
                input_tensor_a.shard_spec().has_value(),
                "ND-sharded ROW_MAJOR input is not yet supported for untilize_with_unpadding");
        }
        if (operation_attributes.output_mem_config.is_sharded()) {
            TT_FATAL(
                operation_attributes.output_mem_config.shard_spec().has_value(),
                "ND-sharded ROW_MAJOR output is not yet supported for untilize_with_unpadding");
        }
        return;
    }

    TT_FATAL(
        input_tensor_a.physical_volume() % tt::constants::TILE_HW == 0,
        "Input tensor physical volume ({}) must be divisible by TILE_HW ({})",
        input_tensor_a.physical_volume(),
        tt::constants::TILE_HW);

    if (input_tensor_a.memory_config().is_sharded()) {
        if (input_tensor_a.shard_spec().has_value()) {
            TT_FATAL(
                operation_attributes.output_mem_config.memory_layout() != tt::tt_metal::TensorMemoryLayout::ND_SHARDED,
                "Output memory config layout must not be ND_SHARDED when input has a legacy 2d shard_spec");
            if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
                TT_FATAL(
                    input_tensor_a.shard_spec().value().grid.ranges().size() == 1,
                    "Expected single grid range and got {}",
                    input_tensor_a.shard_spec().value().grid.ranges().size());
                if (operation_attributes.output_mem_config.is_sharded()) {
                    // The sharded program factory reads output.shard_spec() independently of the
                    // input's (see out_shard_spec in the sharded factory), the same mechanism
                    // WIDTH_SHARDED->WIDTH_SHARDED sharded output already relies on below - so
                    // BLOCK_SHARDED can reuse it as long as the output stays on the same grid.
                    //
                    // BLOCK_SHARDED -> WIDTH_SHARDED (matching column shard width) is also
                    // supported: the writer (writer_unary_unpad_cross_sharded.cpp) addresses the
                    // destination via TensorAccessor page-id routing (page_id = row*KW + col_shard,
                    // identical for WIDTH and BLOCK), so the executing core need not be the
                    // physically-owning core of the output shard. Currently unbatched-only,
                    // matching the existing restriction on BLOCK_SHARDED -> interleaved below.
                    bool same_type =
                        operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
                    bool cross_to_width =
                        operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;
                    TT_FATAL(
                        same_type || cross_to_width,
                        "Output memory config layout ({}) must be BLOCK_SHARDED (or WIDTH_SHARDED with a "
                        "matching column shard width) when input is BLOCK_SHARDED and output is sharded",
                        operation_attributes.output_mem_config.memory_layout());
                    if (same_type) {
                        // Same-shard-type sharded output binds the output buffer directly as an L1
                        // circular buffer (see the sharded factory's sharded_output_cb_index CB) -
                        // dynamic CBs only work in L1, so DRAM is rejected here. It also copies each
                        // input core's CB straight into the same physical output core, with no
                        // cross-core row remapping, so a batch whose matrices straddle input shard
                        // boundaries after height-unpadding would land on the wrong output core;
                        // unbatched-only until that's implemented (matches the interleaved-output
                        // restriction below).
                        TT_FATAL(
                            operation_attributes.output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1,
                            "BLOCK_SHARDED -> BLOCK_SHARDED output must be in L1; got buffer_type={}",
                            operation_attributes.output_mem_config.buffer_type());
                        TT_FATAL(
                            input_tensor_a.physical_volume() /
                                    (input_tensor_a.padded_shape()[-2] * input_tensor_a.padded_shape()[-1]) ==
                                1,
                            "Can only write unbatched output for BLOCK_SHARDED -> BLOCK_SHARDED");
                    }
                    if (cross_to_width) {
                        TT_FATAL(
                            operation_attributes.output_mem_config.shard_spec().has_value(),
                            "Output memory config is sharded but no shard spec is provided");
                        TT_FATAL(
                            input_tensor_a.shard_spec().value().shape[1] ==
                                operation_attributes.output_mem_config.shard_spec().value().shape[1],
                            "BLOCK_SHARDED -> WIDTH_SHARDED requires matching column shard width; got input={} "
                            "output={}",
                            input_tensor_a.shard_spec().value().shape[1],
                            operation_attributes.output_mem_config.shard_spec().value().shape[1]);
                        // Matching shard WIDTH alone is not enough: TensorSpec permits an output
                        // grid with more cores/capacity than the logical output width actually
                        // needs (e.g. 4 cores of shard width 32 for a logical width of 64 - only 2
                        // of those shards hold real data), and comparing configured GRID counts
                        // (as an earlier version of this check did) doesn't catch that - the writer
                        // still launches every input column and uses its input-derived column
                        // index, so noc_async_write_sharded maps the extra input columns onto
                        // subsequent output rows instead of rejecting them, corrupting data. Compare
                        // the EFFECTIVE output column-shard count (logical output width / shard
                        // width) instead of the grid's configured capacity.
                        uint32_t input_kw = legacy_2d_shard_column_count(
                            input_tensor_a.shard_spec().value(), TensorMemoryLayout::BLOCK_SHARDED);
                        uint32_t output_width = compute_output_specs(operation_attributes, input).padded_shape()[-1];
                        uint32_t output_shard_width =
                            operation_attributes.output_mem_config.shard_spec().value().shape[1];
                        uint32_t effective_output_kw = tt::div_up(output_width, output_shard_width);
                        TT_FATAL(
                            input_kw == effective_output_kw,
                            "BLOCK_SHARDED -> WIDTH_SHARDED requires the output to effectively span the same "
                            "number of column shards as the input ({}), got {} (output width {} / shard width "
                            "{}); width-unpadding that reduces the column-shard count is not yet supported",
                            input_kw,
                            effective_output_kw,
                            output_width,
                            output_shard_width);
                        TT_FATAL(
                            input_tensor_a.physical_volume() /
                                    (input_tensor_a.padded_shape()[-2] * input_tensor_a.padded_shape()[-1]) ==
                                1,
                            "Can only write unbatched output for BLOCK_SHARDED -> WIDTH_SHARDED");
                    }
                } else {
                    TT_FATAL(
                        operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
                        "Output memory config layout must be INTERLEAVED for block sharded input but got {}",
                        operation_attributes.output_mem_config.memory_layout());
                    TT_FATAL(
                        input_tensor_a.physical_volume() /
                                (input_tensor_a.padded_shape()[-2] * input_tensor_a.padded_shape()[-1]) ==
                            1,
                        "Can only write unbatched output interleaved");
                }
            } else if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
                if (operation_attributes.output_mem_config.is_sharded()) {
                    TT_FATAL(
                        operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED,
                        "Output memory config layout must be HEIGHT_SHARDED when output is sharded but got {}",
                        operation_attributes.output_mem_config.memory_layout());
                    // Same-shard-type sharded output binds the output buffer directly as an L1
                    // circular buffer (see the sharded factory's sharded_output_cb_index CB) -
                    // dynamic CBs only work in L1, so DRAM is rejected here.
                    TT_FATAL(
                        operation_attributes.output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1,
                        "HEIGHT_SHARDED -> HEIGHT_SHARDED output must be in L1; got buffer_type={}",
                        operation_attributes.output_mem_config.buffer_type());
                } else {
                    TT_FATAL(
                        operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
                        "Output memory config layout must be INTERLEAVED but got {}",
                        operation_attributes.output_mem_config.memory_layout());
                    // The height-sharded -> interleaved writer walks each core's absolute rows and
                    // maps every row to its (matrix, row-in-matrix), so it handles any batch and any
                    // alignment of matrices to core boundaries (whole matrices per core, a single
                    // matrix split across cores, or a batch whose matrices straddle cores). It only
                    // requires that each shard span the full padded matrix width, which height
                    // sharding always satisfies.
                    TT_FATAL(
                        input_tensor_a.shard_spec().value().shape[1] == input_tensor_a.padded_shape()[-1],
                        "Height-sharded untilize to interleaved output requires the shard width ({}) to equal the "
                        "padded tensor width ({})",
                        input_tensor_a.shard_spec().value().shape[1],
                        input_tensor_a.padded_shape()[-1]);
                }
                // What else?
            } else if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
                auto output_shape = compute_output_specs(operation_attributes, input).padded_shape();
                for (uint32_t i = 0; i < output_shape.rank() - 2; i++) {
                    TT_FATAL(
                        input_tensor_a.padded_shape()[i] == output_shape[i],
                        "Input tensor padded shape[{}] ({}) must equal output shape[{}] ({})",
                        i,
                        input_tensor_a.padded_shape()[i],
                        i,
                        output_shape[i]);
                }
                if (operation_attributes.output_mem_config.is_sharded()) {
                    // WIDTH_SHARDED -> BLOCK_SHARDED (matching column shard width) is supported via
                    // the same TensorAccessor page-id-routed writer as the BLOCK_SHARDED -> WIDTH_SHARDED
                    // direction above (writer_unary_unpad_cross_sharded.cpp) - see the comment there.
                    bool same_type = operation_attributes.output_mem_config.memory_layout() ==
                                     input_tensor_a.memory_config().memory_layout();
                    bool cross_to_block =
                        operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;
                    TT_FATAL(
                        same_type || cross_to_block,
                        "Output memory config layout ({}) must match input tensor memory layout ({}) (or be "
                        "BLOCK_SHARDED with a matching column shard width)",
                        operation_attributes.output_mem_config.memory_layout(),
                        input_tensor_a.memory_config().memory_layout());
                    if (same_type) {
                        // Same-shard-type sharded output binds the output buffer directly as an L1
                        // circular buffer (see the sharded factory's sharded_output_cb_index CB) -
                        // dynamic CBs only work in L1, so DRAM is rejected here.
                        TT_FATAL(
                            operation_attributes.output_mem_config.buffer_type() == tt::tt_metal::BufferType::L1,
                            "WIDTH_SHARDED -> WIDTH_SHARDED output must be in L1; got buffer_type={}",
                            operation_attributes.output_mem_config.buffer_type());
                    }
                    if (cross_to_block) {
                        TT_FATAL(
                            operation_attributes.output_mem_config.shard_spec().has_value(),
                            "Output memory config is sharded but no shard spec is provided");
                        TT_FATAL(
                            input_tensor_a.shard_spec().value().shape[1] ==
                                operation_attributes.output_mem_config.shard_spec().value().shape[1],
                            "WIDTH_SHARDED -> BLOCK_SHARDED requires matching column shard width; got input={} "
                            "output={}",
                            input_tensor_a.shard_spec().value().shape[1],
                            operation_attributes.output_mem_config.shard_spec().value().shape[1]);
                        // See the identical check in the BLOCK_SHARDED -> WIDTH_SHARDED branch above:
                        // matching shard WIDTH alone doesn't guarantee matching column-shard COUNT
                        // once width-unpadding is involved - compare the EFFECTIVE output column-shard
                        // count (logical output width / shard width), not the output grid's configured
                        // capacity (a 4-core BLOCK_SHARDED output with shard width 32 and logical width
                        // 64 only effectively uses 2 of its 4 column shards).
                        uint32_t input_kw = legacy_2d_shard_column_count(
                            input_tensor_a.shard_spec().value(), TensorMemoryLayout::WIDTH_SHARDED);
                        uint32_t output_width = compute_output_specs(operation_attributes, input).padded_shape()[-1];
                        uint32_t output_shard_width =
                            operation_attributes.output_mem_config.shard_spec().value().shape[1];
                        uint32_t effective_output_kw = tt::div_up(output_width, output_shard_width);
                        TT_FATAL(
                            input_kw == effective_output_kw,
                            "WIDTH_SHARDED -> BLOCK_SHARDED requires the output to effectively span the same "
                            "number of column shards as the input ({}), got {} (output width {} / shard width "
                            "{}); width-unpadding that reduces the column-shard count is not yet supported",
                            input_kw,
                            effective_output_kw,
                            output_width,
                            output_shard_width);
                        TT_FATAL(
                            input_tensor_a.physical_volume() /
                                    (input_tensor_a.padded_shape()[-2] * input_tensor_a.padded_shape()[-1]) ==
                                1,
                            "Can only write unbatched output for WIDTH_SHARDED -> BLOCK_SHARDED");
                    } else {
                        TT_FATAL(
                            input_tensor_a.padded_shape()[-1] == output_shape[-1] ||
                                (tt::div_up(output_shape[-1], input_tensor_a.shard_spec().value().shape[1]) ==
                                 input_tensor_a.shard_spec().value().grid.num_cores()),
                            "Input tensor width ({}) must equal output width ({}) or output width / shard width "
                            "must equal num cores",
                            input_tensor_a.padded_shape()[-1],
                            output_shape[-1]);
                    }
                } else {
                    TT_FATAL(
                        operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::INTERLEAVED,
                        "Output memory config layout must be INTERLEAVED but got {}",
                        operation_attributes.output_mem_config.memory_layout());
                    TT_FATAL(
                        input_tensor_a.physical_volume() /
                                (input_tensor_a.padded_shape()[-2] * input_tensor_a.padded_shape()[-1]) ==
                            1,
                        "Can only write unbatched output interleaved");
                    TT_FATAL(
                        input_tensor_a.padded_shape()[-1] - output_shape[-1] <
                            input_tensor_a.shard_spec().value().shape[1],
                        "Input tensor width difference ({}) must be less than shard width ({})",
                        input_tensor_a.padded_shape()[-1] - output_shape[-1],
                        input_tensor_a.shard_spec().value().shape[1]);
                }
            } else {
                TT_THROW("Unsupported sharding scheme");
            }
        } else {
            const auto& nd_spec = input_tensor_a.nd_shard_spec().value();
            uint32_t input_shard_width = nd_spec.shard_shape[-1];
            uint32_t input_shard_height = nd_spec.shard_shape[-2];
            uint32_t tile_width = input_tensor_a.tensor_spec().tile().get_width();
            uint32_t tile_height = input_tensor_a.tensor_spec().tile().get_height();
            TT_FATAL(
                input_shard_width % tile_width == 0,
                "Input shard width {} must be a multiple of tile width",
                input_shard_width);
            TT_FATAL(
                input_shard_height % tile_height == 0,
                "Input shard height {} must be a multiple of tile height",
                input_shard_height);
            if (operation_attributes.output_mem_config.is_sharded()) {
                TT_FATAL(
                    operation_attributes.output_mem_config.shard_spec().has_value() ||
                        operation_attributes.output_mem_config.nd_shard_spec().has_value(),
                    "Output memory config is sharded but no shard spec or nd shard spec is provided");
                uint32_t output_dtype_size = input_tensor_a.element_size();
                if (input_tensor_a.dtype() == tt::tt_metal::DataType::BFLOAT8_B) {
                    output_dtype_size = 2;
                }
                uint32_t output_shard_width;
                if (operation_attributes.output_mem_config.shard_spec().has_value()) {
                    output_shard_width = operation_attributes.output_mem_config.shard_spec().value().shape[1];
                } else {
                    output_shard_width = operation_attributes.output_mem_config.nd_shard_spec().value().shard_shape[-1];
                }
                uint32_t output_row_size_bytes = output_shard_width * output_dtype_size;
                const auto& output_buffer_type = operation_attributes.output_mem_config.buffer_type();
                uint32_t alignment_requirement =
                    input_tensor_a.device()->allocator()->get_alignment(output_buffer_type);
                TT_FATAL(
                    output_row_size_bytes % alignment_requirement == 0,
                    "Output shard row size {} bytes must be aligned to {} bytes buffer alignment requirement",
                    output_row_size_bytes,
                    alignment_requirement);
            }
        }
    } else {
        TT_FATAL(
            input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "Input tensor memory layout must be INTERLEAVED but got {}",
            input_tensor_a.memory_config().memory_layout());
        // Interleaved input can write directly to any 2D-sharded output. The MultiCoreInterleaved
        // writer (writer_unary_stick_layout_split_rows_multicore.cpp) now uses
        // noc_async_write_sharded, which splits a logical row across shards via a per-shard page
        // size override - the same mechanism the ROW_MAJOR-input factory relies on - so
        // WIDTH_SHARDED/BLOCK_SHARDED output (which split a row across multiple cores) is no longer
        // blocked by the old row-per-page assumption. An earlier naive host-side-only relaxation
        // (no kernel change) was hardware-tested and produced wrong data; this is the real fix.
        // The MultiCoreBlockInterleaved writer (used for the very-wide-row heuristic path) has NOT
        // been updated yet, so its dispatch is excluded below.
        if (operation_attributes.output_mem_config.is_sharded()) {
            TT_FATAL(
                operation_attributes.output_mem_config.shard_spec().has_value(),
                "Output memory config is sharded but no shard spec is provided");
        }
    }
}

tt::tt_metal::TensorSpec UntilizeWithUnpaddingDeviceOperation::compute_output_specs(
    const operation_attributes_t& operation_attributes, const tensor_args_t& input) {
    ttsl::SmallVector<uint32_t> out_shape;
    const auto& input_tensor_a = input;
    size_t rank = input_tensor_a.logical_shape().rank();
    out_shape.reserve(rank);
    for (uint32_t i = 0; i < rank; i++) {
        out_shape.push_back(operation_attributes.output_tensor_end[i] + 1);
    }
    Shape output_shape(std::move(out_shape));
    DataType output_dtype = input_tensor_a.dtype() == DataType::BFLOAT8_B ? DataType::BFLOAT16 : input_tensor_a.dtype();
    // ROW_MAJOR output shards have no tile-alignment constraint, so the caller's output_mem_config
    // (including its shard_spec, when sharded) is used as-is via the final fallback return below -
    // skip the TILE-only shard-shape derivations, which round to tile height/width.
    if (input_tensor_a.layout() == Layout::ROW_MAJOR) {
        return TensorSpec(
            output_shape,
            TensorLayout(output_dtype, PageConfig(Layout::ROW_MAJOR), operation_attributes.output_mem_config));
    }
    if (!input_tensor_a.memory_config().is_sharded() && operation_attributes.output_mem_config.is_sharded()) {
        // Interleaved input has no shard spec to inherit a shape from, so derive one the same way
        // the sharded-input "single matrix split across cores" case does below: round per-core
        // extents up to a tile multiple. The caller's shard spec still supplies the core grid
        // (and, for HEIGHT_SHARDED, the width).
        //
        // WIDTH_SHARDED/BLOCK_SHARDED split a logical row across multiple shards/cores; this is now
        // supported because the row-split multi-core writer
        // (writer_unary_stick_layout_split_rows_multicore.cpp) uses noc_async_write_sharded with a
        // per-shard page-size override to split each row's write across shard columns - see
        // select_program_factory(), which forces that factory whenever output is sharded. An earlier
        // naive host-side-only relaxation (no kernel change) was hardware-tested and produced wrong
        // data; this is the real fix.
        ShardSpec shard_spec = operation_attributes.output_mem_config.shard_spec().value();
        uint32_t fused_height = output_shape.volume() / output_shape[-1];
        uint32_t tile_height = input_tensor_a.tensor_spec().tile().get_height();
        uint32_t tile_width = input_tensor_a.tensor_spec().tile().get_width();
        if (operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED) {
            uint32_t num_cores = shard_spec.num_cores();
            shard_spec.shape = {fused_height, tt::round_up(tt::div_up(output_shape[-1], num_cores), tile_width)};
        } else if (operation_attributes.output_mem_config.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            CoreRange bbox = shard_spec.grid.bounding_box();
            uint32_t grid_cols = bbox.end_coord.x - bbox.start_coord.x + 1;
            uint32_t grid_rows = bbox.end_coord.y - bbox.start_coord.y + 1;
            if (shard_spec.orientation != ShardOrientation::ROW_MAJOR) {
                std::swap(grid_cols, grid_rows);
            }
            shard_spec.shape = {
                tt::round_up(tt::div_up(fused_height, grid_rows), tile_height),
                tt::round_up(tt::div_up(output_shape[-1], grid_cols), tile_width)};
        } else {
            uint32_t num_cores = shard_spec.num_cores();
            shard_spec.shape = {tt::round_up(tt::div_up(fused_height, num_cores), tile_height), shard_spec.shape[1]};
        }
        auto mem_config = tt::tt_metal::MemoryConfig(
            operation_attributes.output_mem_config.memory_layout(),
            operation_attributes.output_mem_config.buffer_type(),
            shard_spec);
        return TensorSpec(output_shape, TensorLayout(output_dtype, PageConfig(Layout::ROW_MAJOR), mem_config));
    }
    if (input_tensor_a.memory_config().is_sharded() && operation_attributes.output_mem_config.is_sharded() &&
        input_tensor_a.shard_spec().has_value() &&
        operation_attributes.output_mem_config.memory_layout() != input_tensor_a.memory_config().memory_layout()) {
        // Cross-shard-type (WIDTH_SHARDED <-> BLOCK_SHARDED, matching column shard width - see
        // validate_on_program_cache_miss()). The caller's output shard spec is used verbatim: unlike
        // the same-shard-type derivation below (which reshapes a shard inherited from the input's
        // grid), here the output's grid may differ in shape from the input's, so there is nothing
        // meaningful to derive from the input side.
        return TensorSpec(
            output_shape,
            TensorLayout(output_dtype, PageConfig(Layout::ROW_MAJOR), operation_attributes.output_mem_config));
    }
    if (input_tensor_a.memory_config().is_sharded() && operation_attributes.output_mem_config.is_sharded() &&
        input_tensor_a.shard_spec().has_value()) {
        uint32_t fused_height = output_shape.volume() / output_shape[-1];
        uint32_t num_cores = input_tensor_a.shard_spec().value().num_cores();
        std::array<uint32_t, 2> shard_shape{};
        ShardSpec shard_spec = input_tensor_a.shard_spec().value();
        if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::HEIGHT_SHARDED) {
            const auto tile = input_tensor_a.tensor_spec().tile();
            uint32_t tile_height = tile.get_height();
            // Number of whole padded [H_padded, W] matrices packed into a single core's shard. This
            // must match the sharded writer's `batch` (see the multi-core sharded program factory).
            uint32_t batch = std::max(
                1u,
                (shard_spec.shape[0] * shard_spec.shape[1]) /
                    (input_tensor_a.padded_shape()[-2] * input_tensor_a.padded_shape()[-1]));
            uint32_t shard_idx0;
            if (batch > 1) {
                // Each core holds `batch` full matrices and untilize strips the interior pad rows of
                // every one of them. The writer derives its per-matrix unpadded row count as
                // (output shard height / batch), so the output shard height must be exactly
                // batch * logical_H. Rounding up to a tile multiple here would make that division
                // yield the wrong per-matrix row count and corrupt every matrix past the first.
                shard_idx0 = batch * output_shape[-2];
            } else {
                // A single matrix split across cores: no interior padding, untilize copies each
                // core's shard 1:1, so the output shard height must equal the (tile-aligned) input
                // shard height. See issue #16620.
                shard_idx0 = tt::round_up(tt::div_up(fused_height, num_cores), tile_height);
            }
            shard_shape = {shard_idx0, output_shape[-1]};
        } else if (input_tensor_a.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED) {
            // BLOCK_SHARDED splits along both height and width, so both dims of the shard shrink
            // with unpadding, unlike WIDTH_SHARDED (whole, unsplit height) or HEIGHT_SHARDED (whole,
            // unsplit width) above. Grid is a single CoreRange (enforced in validate()), so its
            // bounding box gives the (rows x cols) shard grid directly.
            const auto tile = input_tensor_a.tensor_spec().tile();
            CoreRange bbox = shard_spec.grid.bounding_box();
            uint32_t grid_cols = bbox.end_coord.x - bbox.start_coord.x + 1;
            uint32_t grid_rows = bbox.end_coord.y - bbox.start_coord.y + 1;
            if (shard_spec.orientation != ShardOrientation::ROW_MAJOR) {
                std::swap(grid_cols, grid_rows);
            }
            shard_shape = {
                tt::round_up(tt::div_up(fused_height, grid_rows), tile.get_height()),
                tt::round_up(tt::div_up(output_shape[-1], grid_cols), tile.get_width())};
        } else {
            shard_shape = {fused_height, shard_spec.shape[1]};
        }
        shard_spec.shape = shard_shape;
        auto mem_config = tt::tt_metal::MemoryConfig(
            input_tensor_a.memory_config().memory_layout(),
            operation_attributes.output_mem_config.buffer_type(),
            shard_spec);

        return tt::tt_metal::TensorSpec(
            output_shape, TensorLayout(output_dtype, PageConfig(Layout::ROW_MAJOR), mem_config));
    }

    return tt::tt_metal::TensorSpec(
        output_shape,
        TensorLayout(output_dtype, PageConfig(Layout::ROW_MAJOR), operation_attributes.output_mem_config));
}

Tensor UntilizeWithUnpaddingDeviceOperation::create_output_tensors(
    const operation_attributes_t& operation_attributes, const tensor_args_t& input) {
    auto output_spec = compute_output_specs(operation_attributes, input);
    return create_device_tensor(output_spec, input.device());
}

tt::tt_metal::operation::OpPerformanceModelGeneral<Tensor>
UntilizeWithUnpaddingDeviceOperation::create_op_performance_model(
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& input,
    tensor_return_value_t& output_tensor) {
    const auto& input_tensor = input;
    uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    uint32_t single_tile_size = tile_width * tile_height * input_tensor.element_size();
    uint32_t num_tiles = std::ceil((float)input_tensor.physical_volume() / (float)single_tile_size);
    int compute_cycles = 0;
    const int max_tiles_per_row = 8;
    const int latency_untilize = 390;      // measured latency for untilize_block
    const int latency_pack_untilize = 80;  // measured latency for pack_untilize_block
    if (std::ceil((float)input_tensor.padded_shape()[-1] / (float)tile_width) <= max_tiles_per_row) {
        compute_cycles = num_tiles * latency_pack_untilize;
    } else {
        compute_cycles = num_tiles * latency_untilize;
    }
    int ideal_dev_clock_cycles =
        operations::data_movement::common_tm_bw_model(input_tensor, output_tensor, false, compute_cycles);
    tt::tt_metal::operation::OpPerformanceModelGeneral<tensor_return_value_t> result(
        {input_tensor}, output_tensor, ideal_dev_clock_cycles);
    return result;
}

Tensor untilize_with_unpadding(
    const Tensor& input_tensor,
    const ttnn::Shape& output_tensor_end,
    const std::optional<tt::tt_metal::MemoryConfig>& output_mem_config,
    bool use_multicore,
    bool fp32_dest_acc_en,
    bool enough_space_height,
    const std::optional<CoreRangeSet>& sub_core_grids) {
    return ttnn::device_operation::launch<UntilizeWithUnpaddingDeviceOperation>(
        UntilizeWithUnpaddingParams{
            .output_tensor_end = output_tensor_end,
            .output_mem_config = output_mem_config.value_or(input_tensor.memory_config()),
            .use_multicore = use_multicore,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .enough_space_height = enough_space_height,
            .sub_core_grids = sub_core_grids},
        input_tensor);
}

}  // namespace ttnn::prim
