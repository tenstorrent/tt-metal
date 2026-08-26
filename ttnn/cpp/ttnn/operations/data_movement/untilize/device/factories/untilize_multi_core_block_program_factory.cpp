// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_multi_core_block_program_factory.hpp"

#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

using ttnn::operations::data_movement::BlockBufferSet;
using ttnn::operations::data_movement::push_buffer_set;
using ttnn::operations::data_movement::union_of;

// The work split plus the two buffer sets derived from it.
//
// `create_descriptor` needs the whole split; the two buffer sets are derived from it here so the
// sizes, the indices, and the core ranges they are built from all come from one place.
struct BlockPlan {
    ttnn::BlockSplitWH split;
    BlockBufferSet full;
    BlockBufferSet cliffrow;
};

BlockPlan make_block_plan(const Tensor& a, uint32_t input_single_tile_size, uint32_t output_single_tile_size) {
    const uint32_t a_tile_width = a.tensor_spec().tile().get_width();
    const uint32_t a_tile_height = a.tensor_spec().tile().get_height();

    IDevice* device = a.device();
    const CoreCoord grid_size = device->compute_with_storage_grid_size();

    const uint32_t num_tiles_per_row = a.padded_shape()[-1] / a_tile_width;
    const uint32_t num_tiles_per_col = a.padded_shape()[-2] / a_tile_height;
    const uint32_t num_blocks = (a.padded_shape()[-1] * a.padded_shape()[-2]) / (a_tile_height * a_tile_width);

    const uint32_t max_l1_size = operations::data_movement::get_max_l1_space(a);
    const uint32_t cb_block_size_limit = max_l1_size / (input_single_tile_size + output_single_tile_size);

    BlockPlan plan;
    plan.split = ttnn::split_blocks_for_tilize_wh(
        grid_size, num_blocks, num_tiles_per_row, num_tiles_per_col, cb_block_size_limit);

    // The work split hands out exactly two block widths, so the op needs exactly two buffer sets:
    //
    //   full     -- `single_sub_block_size` tiles wide: the full-block cores, plus the cliff-*column*
    //               cores (a short column still processes full-width blocks).
    //   cliffrow -- `single_block_size_cliff_row` tiles wide: the cores holding the narrow block at
    //               the end of a row, plus the corner core that is both cliff-row and cliff-column.
    //
    // Each set gets its own indices and its own sizes, so no index is ever re-used at two different
    // sizes. Either set may be empty for a given shape. Untilize has no reader-side
    // DRAM-alignment staging buffer, so `staging_index` stays unset on both sets.
    plan.full = BlockBufferSet{
        .input_index = static_cast<uint8_t>(tt::CBIndex::c_0),
        .output_index = static_cast<uint8_t>(tt::CBIndex::c_16),
        .block_tiles = plan.split.single_sub_block_size,
        .core_ranges = union_of(
            plan.split.core_range, plan.split.has_cliff_col ? plan.split.cliff_col_core_range : CoreRangeSet{}),
    };
    plan.cliffrow = BlockBufferSet{
        .input_index = static_cast<uint8_t>(tt::CBIndex::c_1),
        .output_index = static_cast<uint8_t>(tt::CBIndex::c_17),
        .block_tiles = plan.split.single_block_size_cliff_row,
        .core_ranges = plan.split.has_cliff_row
                           ? union_of(
                                 plan.split.cliff_row_core_range,
                                 plan.split.has_cliff_col ? plan.split.cliff_col_row_core_range : CoreRangeSet{})
                           : CoreRangeSet{},
    };
    return plan;
}

}  // namespace

ProgramDescriptor UntilizeMultiCoreBlockProgramFactory::create_descriptor(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& tensor_return_value) {
    const auto& a = tensor_args.input;
    const Tensor& output = tensor_return_value;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const auto& input_shape = a.padded_shape();
    const uint32_t a_tile_height = a.tensor_spec().tile().get_height();

    const BlockPlan plan = make_block_plan(a, input_single_tile_size, output_single_tile_size);
    const BlockBufferSet& full_set = plan.full;
    const BlockBufferSet& cliffrow_set = plan.cliffrow;
    const auto& [ncores, all_cores, core_range, cliff_row_core_range, cliff_col_core_range, cliff_col_row_core_range, nblocks_per_core, single_block_size, single_block_size_cliff_row, single_block_size_cliff_col, has_cliff_row, has_cliff_col, full_cores_per_row, full_cores_per_col, single_sub_block_size] =
        plan.split;

    // Same grid `make_block_plan` split over -- the runtime-arg loop below walks it in core order.
    const CoreCoord grid_size = a.device()->compute_with_storage_grid_size();

    if (single_sub_block_size > 0 && single_block_size % single_sub_block_size) {
        TT_FATAL(false, "single_block_size is not divided by single_sub_block_size");
    }

    uint32_t total_tiles_per_row =
        (full_cores_per_row * single_block_size) + (has_cliff_row * single_block_size_cliff_row);
    uint32_t row_size_bytes;

    uint32_t el_size = a.element_size();
    if (a.dtype() == DataType::BFLOAT8_B) {
        row_size_bytes = input_shape[-1] * output.element_size();
        el_size = output.element_size();
    } else {
        row_size_bytes = input_shape[-1] * a.element_size();
    }

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    ProgramDescriptor desc;

    for (const BlockBufferSet* set : {&full_set, &cliffrow_set}) {
        if (set->empty()) {
            continue;
        }
        TT_FATAL(
            set->block_tiles > 0,
            "Buffer set on cores {} has a zero block width; its buffers would be empty",
            set->core_ranges.str());
        push_buffer_set(
            desc,
            *set,
            input_single_tile_size,
            output_single_tile_size,
            input_cb_data_format,
            output_cb_data_format,
            /*dram_alignment=*/0,
            a_tile_height);
    }

    // reader
    uint32_t num_tiles_2d = a.padded_shape()[-1] * a.padded_shape()[-2] / TILE_HW;

    auto log_shape = output.logical_shape();
    uint32_t third_dim = 1;
    if (log_shape.rank() == 3) {
        third_dim = log_shape[-3];
    } else if (log_shape.rank() >= 4) {
        third_dim = log_shape[-3] * log_shape[-4];
    }

    uint32_t total_num_rows = output.logical_shape()[-2];

    // One reader and one writer per buffer set, each over that set's cores and bound to that set's
    // indices. A set's cores are exactly the cores whose block width its buffers are sized for, so
    // every writer instance's contiguous walk from `get_read_ptr()` stays inside a buffer that is an
    // exact multiple of the block it drains.
    auto make_reader_kernel = [&](const BlockBufferSet& set) {
        std::vector<uint32_t> reader_compile_time_args = {
            num_tiles_2d, third_dim, total_tiles_per_row, set.input_index};
        TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

        KernelDescriptor reader_desc;
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_wh_multicore.cpp";
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = set.core_ranges;
        reader_desc.compile_time_args = std::move(reader_compile_time_args);
        reader_desc.config = ReaderConfigDescriptor{};
        return reader_desc;
    };

    auto make_writer_kernel = [&](const BlockBufferSet& set) {
        std::vector<uint32_t> writer_ct_args = {
            total_num_rows, third_dim, TILE_HEIGHT, row_size_bytes, set.output_index};
        TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

        KernelDescriptor writer_desc;
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
            "writer_unary_stick_layout_wh_multicore.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = set.core_ranges;
        writer_desc.compile_time_args = std::move(writer_ct_args);
        writer_desc.config = WriterConfigDescriptor{};
        return writer_desc;
    };

    KernelDescriptor full_reader_desc = make_reader_kernel(full_set);
    KernelDescriptor full_writer_desc = make_writer_kernel(full_set);
    KernelDescriptor cliffrow_reader_desc = make_reader_kernel(cliffrow_set);
    KernelDescriptor cliffrow_writer_desc = make_writer_kernel(cliffrow_set);

    // compute
    uint32_t single_sub_block_size_wh = single_block_size * single_block_size / single_sub_block_size;
    uint32_t single_sub_block_size_cliff_col_wh =
        single_block_size_cliff_col * single_block_size / single_sub_block_size;

    std::vector<std::pair<std::string, std::string>> compute_kernel_defines;
    if (input_cb_data_format == tt::DataFormat::Int32 || input_cb_data_format == tt::DataFormat::UInt32 ||
        input_cb_data_format == tt::DataFormat::Float32) {
        compute_kernel_defines.emplace_back("DST_ACCUM_MODE", "1");
    }
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[full_set.input_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[cliffrow_set.input_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    const std::string compute_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_wh.cpp";

    // The compute kernel stays split per region -- each region has its own block *count* -- but each
    // instance binds the buffer set matching its cores' block *width*. The region's block-width CTA
    // (the second one) must equal that set's `block_tiles`, since it is the page count the kernel
    // waits on and pops; the assertion below keeps the two from drifting apart.
    auto make_compute_kernel =
        [&](const CoreRangeSet& cores, const BlockBufferSet& set, uint32_t block_size_col, uint32_t block_size_row) {
            TT_FATAL(
                block_size_row == set.block_tiles,
                "Compute on cores {} expects a block width of {} tiles but its buffers hold {}",
                cores.str(),
                block_size_row,
                set.block_tiles);
            KernelDescriptor cd;
            cd.kernel_source = compute_kernel_path;
            cd.source_type = KernelDescriptor::SourceType::FILE_PATH;
            cd.core_ranges = cores;
            cd.compile_time_args = {block_size_col, block_size_row, third_dim, set.input_index, set.output_index};
            cd.defines = compute_kernel_defines;
            cd.config = ComputeConfigDescriptor{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
            };
            return cd;
        };

    std::vector<KernelDescriptor> compute_kernels;
    compute_kernels.reserve(4);
    if (!core_range.empty()) {
        compute_kernels.push_back(
            make_compute_kernel(core_range, full_set, single_sub_block_size_wh, single_sub_block_size));
    }
    if (has_cliff_col && has_cliff_row) {
        compute_kernels.push_back(make_compute_kernel(
            cliff_col_row_core_range, cliffrow_set, single_block_size_cliff_col, single_block_size_cliff_row));
    }
    if (has_cliff_row) {
        compute_kernels.push_back(
            make_compute_kernel(cliff_row_core_range, cliffrow_set, single_block_size, single_block_size_cliff_row));
    }
    if (has_cliff_col) {
        compute_kernels.push_back(make_compute_kernel(
            cliff_col_core_range, full_set, single_sub_block_size_cliff_col_wh, single_sub_block_size));
    }

    // RUNTIME ARGS
    const auto& cores = grid_to_cores(ncores, grid_size.x, grid_size.y, true);
    uint32_t start_row_id = 0;
    uint32_t start_column_id = 0;
    uint32_t tile_start_id = 0;
    uint32_t single_block_size_row_arg;
    uint32_t single_block_size_col_arg;
    uint32_t single_sub_block_size_row_arg;

    uint32_t total_row_cores = full_cores_per_row;
    if (has_cliff_row) {
        total_row_cores++;
    }
    uint32_t cores_col_count = 1;

    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];

        if (has_cliff_col && has_cliff_row && i == ncores - 1) {
            single_block_size_row_arg = single_block_size_cliff_row;
            single_block_size_col_arg = single_block_size_cliff_col;
            single_sub_block_size_row_arg = single_block_size_cliff_row;

        } else if (has_cliff_row && i != 0 && ((i + 1) % (full_cores_per_row + 1)) == 0) {
            single_block_size_row_arg = single_block_size_cliff_row;
            single_block_size_col_arg = single_block_size;
            single_sub_block_size_row_arg = single_block_size_cliff_row;

        } else if (i < total_row_cores * full_cores_per_col) {
            single_block_size_row_arg = single_block_size;
            single_block_size_col_arg = single_block_size;
            single_sub_block_size_row_arg = single_sub_block_size;

        } else {
            single_block_size_row_arg = single_block_size;
            single_block_size_col_arg = single_block_size_cliff_col;
            single_sub_block_size_row_arg = single_sub_block_size;
        }

        // Route this core's args to the reader/writer instance for its buffer set.
        const bool is_cliff_row_core = !cliffrow_set.empty() && cliffrow_set.core_ranges.contains(core);
        const BlockBufferSet& set = is_cliff_row_core ? cliffrow_set : full_set;
        KernelDescriptor& reader_desc = is_cliff_row_core ? cliffrow_reader_desc : full_reader_desc;
        KernelDescriptor& writer_desc = is_cliff_row_core ? cliffrow_writer_desc : full_writer_desc;
        TT_FATAL(
            single_sub_block_size_row_arg == set.block_tiles,
            "Core {} is fed a sub-block of {} tiles but the buffers on it hold {}. The work split "
            "assigned this core a block width that disagrees with its runtime args",
            core.str(),
            single_sub_block_size_row_arg,
            set.block_tiles);

        // reader runtime args — Buffer* slot auto-registers as a BufferBinding so the
        // framework patches addresses on cache hits. This factory defines no
        // override_runtime_arguments, so `resolve_bindings` walks every kernel's bindings and both
        // pairs refresh on their own.
        reader_desc.emplace_runtime_args(
            core, {src0_buffer, tile_start_id, single_block_size_row_arg, single_block_size_col_arg});

        // writer runtime args
        writer_desc.emplace_runtime_args(
            core,
            {dst_buffer,
             TILE_WIDTH * el_size * single_block_size_row_arg,
             start_row_id,
             start_column_id,
             single_block_size_row_arg,
             single_block_size_col_arg,
             TILE_WIDTH * el_size * single_sub_block_size_row_arg,
             single_sub_block_size_row_arg});

        uint32_t end_column_id = start_column_id + (single_block_size_row_arg * TILE_WIDTH * el_size);
        start_column_id = end_column_id % row_size_bytes;
        if (end_column_id % row_size_bytes == 0 && end_column_id != 0) {
            start_row_id += single_block_size_col_arg * TILE_HEIGHT;
        }

        if (start_column_id == 0) {
            tile_start_id = cores_col_count * single_block_size_col_arg * total_tiles_per_row;
            cores_col_count++;
        } else {
            tile_start_id += single_block_size_row_arg;
        }
    }

    // Push each non-empty set's reader and writer.
    if (!full_set.empty()) {
        desc.kernels.push_back(std::move(full_reader_desc));
        desc.kernels.push_back(std::move(full_writer_desc));
    }
    if (!cliffrow_set.empty()) {
        desc.kernels.push_back(std::move(cliffrow_reader_desc));
        desc.kernels.push_back(std::move(cliffrow_writer_desc));
    }
    for (auto& cd : compute_kernels) {
        desc.kernels.push_back(std::move(cd));
    }

    return desc;
}

}  // namespace ttnn::prim
