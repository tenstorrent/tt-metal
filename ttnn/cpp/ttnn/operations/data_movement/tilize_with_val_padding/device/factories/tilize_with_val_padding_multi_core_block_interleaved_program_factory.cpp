// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_with_val_padding_multi_core_block_interleaved_program_factory.hpp"

#include <cmath>

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/data_movement/tilize_with_val_padding/device/factories/tilize_with_val_padding_factory_helper.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// One set of buffers, sized for a single block width.
//
// The work split gives cores one of two block widths (the full width, and a narrower cliff-row
// width), and a block's width fixes the size of the buffers that carry it. That size is a
// *correctness* property, not a performance one: the reader fills a whole block through one raw
// linear write starting at `get_write_ptr()`, and `cb_push_back` requires the producer to write
// contiguously — it only wraps when the write pointer lands exactly on `fifo_limit`
// ("producer always writes into contiguous memory, it cannot wrap"). A buffer whose size is not
// an exact multiple of the block pushed into it therefore overruns into its neighbour rather than
// wrapping. So each block width gets its **own** set of buffers, with its own indices, sized for
// that width — rather than one set of indices re-used at different sizes on disjoint cores.
//
// The two sets live on disjoint core ranges, so L1 usage is unchanged: a core allocates only the
// set belonging to its own block width.
struct BlockBufferSet {
    uint8_t staging_index;  // per-row DRAM-alignment staging buffer (reader-private scratchpad)
    uint8_t input_index;    // row-major block the reader fills and compute tilizes
    uint8_t output_index;   // tilized block compute produces and the writer drains
    uint32_t block_tiles;   // block width in tiles — the page count of input/output
    CoreRangeSet core_ranges;

    bool empty() const { return core_ranges.empty(); }
};

// Union of two core ranges, either of which may be empty.
CoreRangeSet union_of(const CoreRangeSet& a, const CoreRangeSet& b) {
    if (a.empty()) {
        return b;
    }
    if (b.empty()) {
        return a;
    }
    return a.merge(b);
}

// Append the three CBDescriptors for one buffer set.
void push_buffer_set(
    ProgramDescriptor& desc,
    const BlockBufferSet& set,
    uint32_t input_single_tile_size,
    uint32_t output_single_tile_size,
    tt::DataFormat input_cb_data_format,
    tt::DataFormat output_cb_data_format,
    uint32_t dram_alignment) {
    // c_1 is a per-row staging buffer used by the reader when the DRAM source row and the
    // L1 destination have different alignment offsets: the reader rounds the source address
    // down to a dram_alignment boundary, issues one noc_async_read of (row_bytes + dram_alignment)
    // into this buffer, then copies the correctly-offset slice into the input buffer.
    //   row_bytes  = TILE_WIDTH * elt_size * block_tiles  (one row of a block)
    //              = input_single_tile_size / TILE_HEIGHT * block_tiles
    //   + dram_alignment    : tail bytes from rounding the DRAM read down to alignment
    //   + dram_alignment    : headroom for aligning the L1 write pointer up to dram_alignment
    //                         (get_write_ptr only guarantees L1 alignment, not DRAM alignment)
    uint32_t input_row_bytes = input_single_tile_size / TILE_HEIGHT;
    uint32_t temp_cb_size = input_row_bytes * set.block_tiles + 2 * dram_alignment;
    desc.cbs.push_back(CBDescriptor{
        .total_size = temp_cb_size,
        .core_ranges = set.core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = set.staging_index,
            .data_format = input_cb_data_format,
            .page_size = temp_cb_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = set.block_tiles * input_single_tile_size,
        .core_ranges = set.core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = set.input_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
    });
    desc.cbs.push_back(CBDescriptor{
        .total_size = set.block_tiles * output_single_tile_size,
        .core_ranges = set.core_ranges,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = set.output_index,
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
    });
}

}  // namespace

ProgramDescriptor TilizeWithValPaddingMultiCoreBlockInterleavedFactory::create_descriptor(
    const TilizeWithValPaddingParams& operation_attributes, const Tensor& input_tensor, Tensor& tensor_return_value) {
    const Tensor& a = input_tensor;
    const Tensor& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B;

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    uint32_t max_l1_size = operations::data_movement::get_max_l1_space(a);
    uint32_t num_tiles_per_col = output.padded_shape()[-2] / TILE_HEIGHT;
    uint32_t num_tiles_per_row = output.padded_shape()[-1] / TILE_WIDTH;
    uint32_t num_blocks = (output.padded_shape()[-1] * output.padded_shape()[-2]) / (TILE_HEIGHT * TILE_WIDTH);
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    // Fold the staging buffer (bytes/tile + fixed) into the limit or the region overruns L1.
    uint32_t staging_bytes_per_tile = input_single_tile_size / TILE_HEIGHT;
    uint32_t fixed_staging_bytes = 2 * dram_alignment;
    uint32_t budget_for_tiles = (max_l1_size > fixed_staging_bytes) ? (max_l1_size - fixed_staging_bytes) : 0;
    uint32_t bytes_per_tile_pair = input_single_tile_size + output_single_tile_size + staging_bytes_per_tile;
    uint32_t cb_block_size_limit = (bytes_per_tile_pair == 0) ? 0 : budget_for_tiles / bytes_per_tile_pair;

    auto
        [ncores,
         all_cores,
         core_range,
         cliff_row_core_range,
         cliff_col_core_range,
         cliff_col_row_core_range,
         nblocks_per_core,
         single_block_size,
         single_block_size_cliff_row,
         single_block_size_cliff_col,
         has_cliff_row,
         has_cliff_col,
         full_cores_per_row,
         full_cores_per_col,
         single_sub_block_size] =
            ttnn::split_blocks_for_tilize_wh(
                available_grid, num_blocks, num_tiles_per_row, num_tiles_per_col, cb_block_size_limit);

    if (single_sub_block_size > 0 && single_block_size % single_sub_block_size) {
        TT_FATAL(false, "single_block_size is not divided by single_sub_block_size");
    }

    uint32_t total_tiles_per_row =
        (full_cores_per_row * single_block_size) + (has_cliff_row * single_block_size_cliff_row);

    uint32_t unpadded_row_size_bytes = a.padded_shape()[-1] * a.element_size();     // Assuming bfloat16 dataformat
    uint32_t padded_row_size_bytes = output.padded_shape()[-1] * a.element_size();  // Assuming bfloat16 dataformat

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    ProgramDescriptor desc;

    // The work split hands out exactly two block widths, so the op needs exactly two buffer sets:
    //
    //   full     — `single_sub_block_size` tiles wide: the full-block cores, plus the cliff-*column*
    //              cores (a short column still processes full-width blocks).
    //   cliffrow — `single_block_size_cliff_row` tiles wide: the cores holding the narrow block at
    //              the end of a row, plus the corner core that is both cliff-row and cliff-column.
    //
    // Each set gets its own indices and its own sizes, so no index is ever re-used at two
    // different sizes. Either set may be empty for a given shape.
    const BlockBufferSet full_set{
        .staging_index = static_cast<uint8_t>(tt::CBIndex::c_1),
        .input_index = static_cast<uint8_t>(tt::CBIndex::c_0),
        .output_index = static_cast<uint8_t>(tt::CBIndex::c_16),
        .block_tiles = single_sub_block_size,
        .core_ranges = union_of(core_range, has_cliff_col ? cliff_col_core_range : CoreRangeSet{}),
    };
    const BlockBufferSet cliffrow_set{
        .staging_index = static_cast<uint8_t>(tt::CBIndex::c_3),
        .input_index = static_cast<uint8_t>(tt::CBIndex::c_2),
        .output_index = static_cast<uint8_t>(tt::CBIndex::c_17),
        .block_tiles = single_block_size_cliff_row,
        .core_ranges = has_cliff_row
                           ? union_of(cliff_row_core_range, has_cliff_col ? cliff_col_row_core_range : CoreRangeSet{})
                           : CoreRangeSet{},
    };

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
            dram_alignment);
    }

    // reader
    uint32_t packed_pad_value = detail::get_packed_value(a, operation_attributes.pad_value);
    // log2(TILE_WIDTH * data_format_size_in_bytes)

    uint32_t num_tiles_2d = output.padded_shape()[-1] * output.padded_shape()[-2] / TILE_HW;

    auto log_shape = output.logical_shape();
    uint32_t third_dim = 1;
    if (log_shape.rank() == 3) {
        third_dim = log_shape[-3];
    } else if (log_shape.rank() >= 4) {
        third_dim = log_shape[-3] * log_shape[-4];
    }

    uint32_t tile_height = output.tensor_spec().tile().get_height();

    uint32_t total_num_rows = a.logical_shape()[-2];

    if (output.padded_shape()[-2] > tt::round_up(total_num_rows, tile_height)) {
        total_num_rows = output.padded_shape()[-2];
    }

    // One reader and one writer per buffer set, each over that set's cores and bound to that
    // set's indices. A set's cores are exactly the cores whose block width its buffers are sized
    // for, so every instance's raw block write lands in a buffer that is an exact multiple of it.
    auto make_reader_kernel = [&](const BlockBufferSet& set) {
        std::vector<uint32_t> reader_compile_time_args = {
            total_num_rows,
            third_dim,
            tile_height,
            a.element_size(),
            unpadded_row_size_bytes,
            dram_alignment,
            set.input_index,
            set.staging_index};
        TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

        KernelDescriptor reader_desc;
        reader_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/tilize_with_val_padding/device/kernels/dataflow/"
            "reader_unary_pad_multicore_both_dims.cpp";
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = set.core_ranges;
        reader_desc.compile_time_args = std::move(reader_compile_time_args);
        reader_desc.config = ReaderConfigDescriptor{};
        return reader_desc;
    };

    auto make_writer_kernel = [&](const BlockBufferSet& set) {
        std::vector<uint32_t> writer_compile_time_args = {
            set.output_index, num_tiles_2d, third_dim, total_tiles_per_row};
        TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

        KernelDescriptor writer_desc;
        writer_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id_wh.cpp";
        writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        writer_desc.core_ranges = set.core_ranges;
        writer_desc.compile_time_args = std::move(writer_compile_time_args);
        writer_desc.config = WriterConfigDescriptor{};
        return writer_desc;
    };

    KernelDescriptor full_reader_desc = make_reader_kernel(full_set);
    KernelDescriptor full_writer_desc = make_writer_kernel(full_set);
    KernelDescriptor cliffrow_reader_desc = make_reader_kernel(cliffrow_set);
    KernelDescriptor cliffrow_writer_desc = make_writer_kernel(cliffrow_set);

    // compute
    uint32_t single_sub_block_wh = single_block_size * single_block_size / single_sub_block_size;
    uint32_t single_sub_block_cliff_col_wh = single_block_size_cliff_col * single_block_size / single_sub_block_size;

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_llk_acc) {
        unpack_to_dest_mode[full_set.input_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[cliffrow_set.input_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    const std::string compute_kernel_path =
        "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/compute/tilize_wh.cpp";

    // The compute kernel stays split per region — each region has its own block *count* — but each
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
            cd.config = ComputeConfigDescriptor{
                .fp32_dest_acc_en = fp32_llk_acc,
                .unpack_to_dest_mode = unpack_to_dest_mode,
            };
            return cd;
        };

    std::vector<KernelDescriptor> compute_kernels;
    compute_kernels.reserve(4);
    if (!core_range.empty()) {
        compute_kernels.push_back(
            make_compute_kernel(core_range, full_set, single_sub_block_wh, single_sub_block_size));
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
        compute_kernels.push_back(
            make_compute_kernel(cliff_col_core_range, full_set, single_sub_block_cliff_col_wh, single_sub_block_size));
    }

    // RUNTIME ARGS
    const auto cores = corerange_to_cores(available_grid);
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

        // Route this core's args to the reader/writer instance for its buffer set. Membership is
        // read from the work split's own core assignment rather than re-derived from the branch
        // above, so the args and the buffers they drive can never disagree about which set a core
        // is in. The assertion then checks the one thing that must hold: the set's buffers are
        // sized for exactly the sub-block width being passed here, which is what keeps the
        // reader's raw block write inside its buffer.
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

        // reader runtime args — Buffer* slot auto-registers as a BufferBinding, which is what
        // refreshes addresses on a cache hit (this factory declares no override_runtime_arguments,
        // so `resolve_bindings` walks every kernel's bindings, extra pairs included).
        reader_desc.emplace_runtime_args(
            core,
            {src0_buffer,
             packed_pad_value,
             TILE_WIDTH * a.element_size() * single_block_size_row_arg,
             start_row_id,
             start_column_id,
             single_block_size_row_arg,
             single_block_size_col_arg,
             TILE_WIDTH * a.element_size() * single_sub_block_size_row_arg,
             single_sub_block_size_row_arg});

        // writer runtime args
        writer_desc.emplace_runtime_args(
            core, {dst_buffer, tile_start_id, single_block_size_row_arg, single_block_size_col_arg});

        uint32_t end_column_id = start_column_id + (single_block_size_row_arg * TILE_WIDTH * a.element_size());
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

    // One (reader, writer) pair per non-empty set, ahead of the compute kernels.
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
