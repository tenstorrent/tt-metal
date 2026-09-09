// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_block_interleaved_program_factory.hpp"

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

using ttnn::operations::data_movement::BlockBufferSet;
using ttnn::operations::data_movement::BlockCoreOrder;
using ttnn::operations::data_movement::BlockDirection;
using ttnn::operations::data_movement::BlockPlan;
using ttnn::operations::data_movement::buffer_set_for_core;
using ttnn::operations::data_movement::make_block_plan;
using ttnn::operations::data_movement::push_buffer_set;

}  // namespace

tt::tt_metal::ProgramDescriptor UntilizeWithUnpaddingMultiCoreBlockInterleavedProgramFactory::create_descriptor(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    ProgramDescriptor desc;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const auto& input_shape = a.padded_shape();
    const auto& output_shape = output.padded_shape();
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    // ColumnMajor: this factory's runtime-arg loop walks corerange_to_cores(available_grid), which
    // is the order the CoreRangeSet work-split overload assigns in. Untilize direction, so the
    // split follows the *input* padded shape -- the output here is the unpadded one.
    const BlockPlan plan = make_block_plan(
        BlockDirection::Untilize,
        BlockCoreOrder::ColumnMajor,
        a,
        output,
        input_single_tile_size,
        output_single_tile_size,
        TILE_HEIGHT,
        TILE_WIDTH,
        sub_core_grids);
    const BlockBufferSet& full_set = plan.full;
    const BlockBufferSet& cliffrow_set = plan.cliffrow;
    const auto& [ncores, all_cores, core_range, cliff_row_core_range, cliff_col_core_range, cliff_col_row_core_range, nblocks_per_core, single_block_size, single_block_size_cliff_row, single_block_size_cliff_col, has_cliff_row, has_cliff_col, full_cores_per_row, full_cores_per_col, single_sub_block_size] =
        plan.split;

    if (single_sub_block_size > 0 && single_block_size % single_sub_block_size) {
        TT_FATAL(false, "single_block_size is not divided by single_sub_block_size");
    }

    uint32_t total_tiles_per_row =
        (full_cores_per_row * single_block_size) + (has_cliff_row * single_block_size_cliff_row);
    uint32_t padded_row_size_bytes;
    uint32_t unpadded_row_size_bytes;

    uint32_t el_size;
    if (a.dtype() == DataType::BFLOAT8_B) {
        padded_row_size_bytes = input_shape[-1] * output.element_size();
        unpadded_row_size_bytes = output_shape[-1] * output.element_size();
        el_size = output.element_size();
    } else {
        padded_row_size_bytes = input_shape[-1] * a.element_size();
        unpadded_row_size_bytes = output_shape[-1] * a.element_size();
        el_size = a.element_size();
    }

    // One buffer set per block width, each on its own disjoint cores. This replaces the legacy
    // layout of one (input, output) index pair re-used at two different sizes across four regions.
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
            TILE_HEIGHT);
    }

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

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
            total_num_rows, third_dim, TILE_HEIGHT, unpadded_row_size_bytes, set.output_index};
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
    KernelDescriptor::Defines compute_kernel_defines;
    if (input_cb_data_format == tt::DataFormat::Int32 || input_cb_data_format == tt::DataFormat::UInt32 ||
        input_cb_data_format == tt::DataFormat::Float32) {
        compute_kernel_defines.emplace_back("DST_ACCUM_MODE", "1");
    }

    const std::string compute_kernel_path(
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize_wh.cpp");

    // The compute kernel stays split per region -- each region has its own block *count* -- but each
    // instance binds the buffer set matching its cores' block *width*. The region's block-width CTA
    // (the second one) must equal that set's `block_tiles`, since it is the page count the kernel
    // waits on and pops; the assertion below keeps the two from drifting apart.
    auto push_compute =
        [&](const CoreRangeSet& cr, const BlockBufferSet& set, uint32_t block_size_col, uint32_t block_size_row) {
            TT_FATAL(
                block_size_row == set.block_tiles,
                "Compute on cores {} expects a block width of {} tiles but its buffers hold {}",
                cr.str(),
                block_size_row,
                set.block_tiles);
            // fp32 unpack is marked for exactly the buffer this kernel reads. Marking both sets'
            // indices would set it on `cliffrow_set.input_index` even when that set is empty -- an
            // operand with no CB on any core -- and would do so in the full-set kernels too.
            std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
                NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
            if (fp32_dest_acc_en) {
                unpack_to_dest_mode[set.input_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
            }

            KernelDescriptor compute_desc;
            compute_desc.kernel_source = compute_kernel_path;
            compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
            compute_desc.core_ranges = cr;
            compute_desc.compile_time_args = {
                block_size_col, block_size_row, third_dim, set.input_index, set.output_index};
            compute_desc.defines = compute_kernel_defines;
            compute_desc.config = ComputeConfigDescriptor{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
            };
            desc.kernels.push_back(std::move(compute_desc));
        };

    if (!core_range.empty()) {
        push_compute(core_range, full_set, single_sub_block_size_wh, single_sub_block_size);
    }
    if (has_cliff_col && has_cliff_row) {
        push_compute(cliff_col_row_core_range, cliffrow_set, single_block_size_cliff_col, single_block_size_cliff_row);
    }
    if (has_cliff_row) {
        push_compute(cliff_row_core_range, cliffrow_set, single_block_size, single_block_size_cliff_row);
    }
    if (has_cliff_col) {
        push_compute(cliff_col_core_range, full_set, single_sub_block_size_cliff_col_wh, single_sub_block_size);
    }

    // RUNTIME ARGS
    const auto& cores = corerange_to_cores(available_grid);
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

        // Route this core's args to the reader/writer instance for its buffer set. Membership is read
        // from the work split's own core assignment rather than re-derived from the branch above, so
        // the args and the buffers they drive can never disagree about which set a core is in. The
        // assertion then checks the one thing that must hold: the set's buffers are sized for exactly
        // the sub-block width being passed here, which is what keeps the writer's contiguous walk
        // inside its buffer.
        const BlockBufferSet& set = buffer_set_for_core(plan, core);
        const bool is_cliff_row_core = (&set == &cliffrow_set);
        KernelDescriptor& reader_desc = is_cliff_row_core ? cliffrow_reader_desc : full_reader_desc;
        KernelDescriptor& writer_desc = is_cliff_row_core ? cliffrow_writer_desc : full_writer_desc;
        TT_FATAL(
            single_sub_block_size_row_arg == set.block_tiles,
            "Core {} is fed a sub-block of {} tiles but the buffers on it hold {}. The work split "
            "assigned this core a block width that disagrees with its runtime args",
            core.str(),
            single_sub_block_size_row_arg,
            set.block_tiles);

        // reader runtime args — the Buffer* slot auto-registers as a BufferBinding. This factory
        // defines no override_runtime_arguments, so `resolve_bindings` walks every kernel's bindings
        // and both pairs refresh their addresses on a cache hit on their own.
        reader_desc.emplace_runtime_args(
            core, {src0_buffer, tile_start_id, single_block_size_row_arg, single_block_size_col_arg});

        //  writer runtime args
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
        start_column_id = end_column_id % padded_row_size_bytes;
        if (end_column_id % padded_row_size_bytes == 0 && end_column_id != 0) {
            start_row_id += single_block_size_col_arg * TILE_HEIGHT;
        }

        if (start_column_id == 0) {
            tile_start_id = cores_col_count * single_block_size_col_arg * total_tiles_per_row;
            cores_col_count++;
        } else {
            tile_start_id += single_block_size_row_arg;
        }
    }

    // Insert each non-empty set's reader+writer at the beginning, so the dataflow kernels precede
    // the compute kernels, in full-then-cliffrow order.
    if (!cliffrow_set.empty()) {
        desc.kernels.insert(desc.kernels.begin(), std::move(cliffrow_writer_desc));
        desc.kernels.insert(desc.kernels.begin(), std::move(cliffrow_reader_desc));
    }
    if (!full_set.empty()) {
        desc.kernels.insert(desc.kernels.begin(), std::move(full_writer_desc));
        desc.kernels.insert(desc.kernels.begin(), std::move(full_reader_desc));
    }

    return desc;
}

}  // namespace ttnn::prim
