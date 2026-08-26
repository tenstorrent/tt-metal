// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tilize_multi_core_block_program_factory.hpp"
#include "ttnn/operations/data_movement/tilize/device/tilize_device_operation.hpp"

#include <tt-metalium/experimental/program_descriptor_patching.hpp>

#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include "ttnn/operations/data_movement/common/common.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

// Runtime-arg widths of this factory's DM kernels, used by the cache-hit hook to confirm it is
// patching a reader/writer and not something else. Keep in step with the emplace_runtime_args
// calls below.
constexpr size_t kReaderRuntimeArgCount = 9;
constexpr size_t kWriterRuntimeArgCount = 4;

using ttnn::operations::data_movement::BlockBufferSet;
using ttnn::operations::data_movement::BlockPlan;
using ttnn::operations::data_movement::buffer_set_for_core;
using ttnn::operations::data_movement::make_block_plan;
using ttnn::operations::data_movement::push_buffer_set;

}  // namespace

ProgramDescriptor TilizeMultiCoreBlockProgramFactory::create_descriptor(
    const TilizeParams& operation_attributes, const TilizeInputs& tensor_args, Tensor& tensor_return_value) {
    const auto& a = tensor_args.input_tensor;
    const Tensor& output = tensor_return_value;
    const uint32_t tile_width = operation_attributes.tile.get_width();
    const uint32_t tile_height = operation_attributes.tile.get_height();

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = operation_attributes.tile.get_tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = operation_attributes.tile.get_tile_size(output_cb_data_format);

    // UInt8 requires fp32 dest acc on Blackhole: hardware promotes 8-bit integers to 32-bit in
    // dest but keeps them as integers (not float), so the output CB stays as UInt8 (not Float32).
    bool fp32_llk_acc = a.dtype() == DataType::FLOAT32 || a.dtype() == DataType::FP8_E4M3 ||
                        output.dtype() == DataType::FP8_E4M3 || output.dtype() == DataType::BFLOAT8_B ||
                        a.dtype() == DataType::UINT8;

    const uint32_t tile_hw = operation_attributes.tile.get_tile_hw();
    const uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();

    const BlockPlan plan = make_block_plan(
        a,
        output,
        input_single_tile_size,
        output_single_tile_size,
        tile_height,
        tile_width,
        operation_attributes.sub_core_grids);
    const BlockBufferSet& full_set = plan.full;
    const BlockBufferSet& cliffrow_set = plan.cliffrow;
    const auto& [ncores, all_cores, core_range, cliff_row_core_range, cliff_col_core_range, cliff_col_row_core_range, nblocks_per_core, single_block_size, single_block_size_cliff_row, single_block_size_cliff_col, has_cliff_row, has_cliff_col, full_cores_per_row, full_cores_per_col, single_sub_block_size] =
        plan.split;

    // Same grid `make_block_plan` split over — the runtime-arg loop below walks it in core order.
    const CoreCoord grid_size = a.device()->compute_with_storage_grid_size();
    const CoreRangeSet available_grid = operation_attributes.sub_core_grids.has_value()
                                            ? operation_attributes.sub_core_grids.value()
                                            : CoreRangeSet(CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1}));

    if (single_sub_block_size > 0 && single_block_size % single_sub_block_size) {
        TT_FATAL(false, "single_block_size is not divided by single_sub_block_size");
    }

    uint32_t total_tiles_per_row =
        (full_cores_per_row * single_block_size) + (has_cliff_row * single_block_size_cliff_row);

    uint32_t row_size_bytes = a.padded_shape()[-1] * a.element_size();  // Assuming bfloat16 dataformat

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_FATAL(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    const TileDescriptor tile_descriptor(operation_attributes.tile);

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
            dram_alignment,
            tile_height,
            tile_descriptor);
    }

    // reader
    uint32_t num_tiles_2d = output.padded_shape()[-1] * output.padded_shape()[-2] / tile_hw;

    auto log_shape = output.logical_shape();
    uint32_t third_dim = 1;
    if (log_shape.rank() == 3) {
        third_dim = log_shape[-3];
    } else if (log_shape.rank() >= 4) {
        third_dim = log_shape[-3] * log_shape[-4];
    }

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
            row_size_bytes,
            dram_alignment,
            set.input_index,
            *set.staging_index};
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
    // UInt8 uses 32-bit dest as integer (not float): do not enable FP32 unpack-to-dest mode.
    if (fp32_llk_acc && a.dtype() != DataType::UINT8) {
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

        // Route this core's args to the reader/writer instance for its buffer set. Membership is
        // read from the work split's own core assignment rather than re-derived from the branch
        // above, so the args and the buffers they drive can never disagree about which set a core
        // is in. The assertion then checks the one thing that must hold: the set's buffers are
        // sized for exactly the sub-block width being passed here, which is what keeps the
        // reader's raw block write inside its buffer.
        const BlockBufferSet& set = buffer_set_for_core(plan, core);
        const bool is_cliff_row_core = &set == &cliffrow_set;
        KernelDescriptor& reader_desc = is_cliff_row_core ? cliffrow_reader_desc : full_reader_desc;
        KernelDescriptor& writer_desc = is_cliff_row_core ? cliffrow_writer_desc : full_writer_desc;
        TT_FATAL(
            single_sub_block_size_row_arg == set.block_tiles,
            "Core {} is fed a sub-block of {} tiles but the buffers on it hold {}. The work split "
            "assigned this core a block width that disagrees with its runtime args",
            core.str(),
            single_sub_block_size_row_arg,
            set.block_tiles);

        // reader runtime args — slot 0 carries the input buffer. Note the `Buffer*` here does NOT
        // self-refresh: this factory defines override_runtime_arguments, and the adapter then skips
        // automatic binding resolution entirely, so the explicit slot-0 patch in that hook is the
        // only thing that re-points addresses on a cache hit.
        reader_desc.emplace_runtime_args(
            core,
            {src0_buffer,
             std::uint32_t{0},
             tile_width * a.element_size() * single_block_size_row_arg,
             start_row_id,
             start_column_id,
             single_block_size_row_arg,
             single_block_size_col_arg,
             tile_width * a.element_size() * single_sub_block_size_row_arg,
             single_sub_block_size_row_arg});

        // writer runtime args
        writer_desc.emplace_runtime_args(
            core, {dst_buffer, tile_start_id, single_block_size_row_arg, single_block_size_col_arg});

        uint32_t end_column_id = start_column_id + (single_block_size_row_arg * tile_width * a.element_size());
        start_column_id = end_column_id % row_size_bytes;
        if (end_column_id % row_size_bytes == 0 && end_column_id != 0) {
            start_row_id += single_block_size_col_arg * tile_height;
        }

        if (start_column_id == 0) {
            tile_start_id = cores_col_count * single_block_size_col_arg * total_tiles_per_row;
            cores_col_count++;
        } else {
            tile_start_id += single_block_size_row_arg;
        }
    }

    // One (reader, writer) pair per non-empty buffer set. Every pair's slot 0 has to be re-pointed
    // on a program-cache hit: because this factory defines override_runtime_arguments, the adapter
    // skips automatic binding resolution, so the `Buffer*` in slot 0 does not refresh on its own
    // and an unpatched pair keeps the address from the call that populated the cache — quietly
    // reading or writing the wrong buffer.
    //
    // Collect the handles as they are assigned rather than letting the hook infer them from the
    // push order. The hook cannot re-derive any of this: the block split folds in
    // `get_max_l1_space`, which reads live L1 occupancy (`lowest_occupied_compute_l1_address`) — a
    // value the program cache does not key on, so a fresh split on a later hit can disagree with
    // the split baked into the cached program.
    std::vector<uint32_t> dm_kernel_handles;
    dm_kernel_handles.reserve(4);
    const auto push_pair = [&desc, &dm_kernel_handles](KernelDescriptor&& reader, KernelDescriptor&& writer) {
        dm_kernel_handles.push_back(static_cast<uint32_t>(desc.kernels.size()));
        desc.kernels.push_back(std::move(reader));
        dm_kernel_handles.push_back(static_cast<uint32_t>(desc.kernels.size()));
        desc.kernels.push_back(std::move(writer));
    };
    if (!full_set.empty()) {
        push_pair(std::move(full_reader_desc), std::move(full_writer_desc));
    }
    if (!cliffrow_set.empty()) {
        push_pair(std::move(cliffrow_reader_desc), std::move(cliffrow_writer_desc));
    }

    // Hand the hook the handles it must patch, riding on the first reader's common args:
    // {num_pairs, reader0, writer0, [reader1, writer1]}. Host-side metadata only — the reader
    // kernel reads no common args — and it keeps the fact with the program it describes.
    TT_FATAL(!desc.kernels.empty(), "Block tilize emitted no kernels");
    TT_FATAL(
        dm_kernel_handles.size() == 2 * plan.num_dm_pairs(),
        "Recorded {} DM kernel handles for {} pairs",
        dm_kernel_handles.size(),
        plan.num_dm_pairs());
    KernelDescriptor::RTArgList dm_kernel_metadata;
    dm_kernel_metadata.push_back(plan.num_dm_pairs());
    for (uint32_t handle : dm_kernel_handles) {
        dm_kernel_metadata.push_back(handle);
    }
    desc.kernels.front().emplace_common_runtime_args(dm_kernel_metadata);

    for (auto& cd : compute_kernels) {
        desc.kernels.push_back(std::move(cd));
    }

    return desc;
}

void TilizeMultiCoreBlockProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const TilizeParams& /*operation_attributes*/,
    const TilizeInputs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    // Every shape-derived arg is keyed, so only the reader/writer buffer addresses move on a hit.
    //
    // This factory emits one (reader, writer) pair *per buffer set*, so there may be two pairs to
    // re-point rather than one. Both must be patched: an unpatched pair keeps the address from the
    // call that populated the cache, which on a later hit with new tensor storage reads or writes
    // the wrong buffer with nothing to flag it. The count comes from the program itself — see the
    // note in create_descriptor for why it must not be re-derived here.
    // create_descriptor recorded {num_pairs, reader0, writer0, [reader1, writer1]} here, so the
    // handles are read rather than inferred from the push order — nothing in this hook assumes the
    // DM kernels come first or sit at consecutive indices.
    auto& dm_kernel_metadata = tt::tt_metal::GetCommonRuntimeArgs(program, 0);
    const uint32_t num_pairs = dm_kernel_metadata[0];
    TT_FATAL(
        num_pairs == 1 || num_pairs == 2,
        "Block tilize recorded {} reader/writer pairs; the work split yields at most two block widths, "
        "so this should be 1 or 2",
        num_pairs);
    TT_FATAL(
        dm_kernel_metadata.size() == 1 + 2 * num_pairs,
        "Block tilize recorded {} metadata args for {} pairs; expected {}",
        dm_kernel_metadata.size(),
        num_pairs,
        1 + 2 * num_pairs);

    const uint32_t src_addr = tensor_args.input_tensor.buffer()->address();
    const uint32_t dst_addr = tensor_return_value.buffer()->address();

    // Belt and braces on top of the recorded handles: confirm each one really is the reader or
    // writer before writing an address into its slot 0. Putting a buffer address into a compute
    // kernel's args would be exactly the silent corruption this hook exists to prevent. A reader
    // carries kReaderRuntimeArgCount args and a writer kWriterRuntimeArgCount, while this
    // factory's compute kernels carry none, so the arg width identifies the role.
    const auto patch_dm_kernel = [&program](
                                     uint32_t kernel_idx, uint32_t addr, size_t expected_args, const char* role) {
        for (const auto& col : tt::tt_metal::GetRuntimeArgs(program, kernel_idx)) {
            for (const auto& args : col) {
                if (args.size() == 0) {
                    continue;  // core outside this kernel's range
                }
                TT_FATAL(
                    args.size() == expected_args,
                    "Kernel {} should be the {} (buffer address at slot 0) but carries {} runtime args, "
                    "not {}. The reader/writer-pairs-first kernel order this hook relies on has changed",
                    kernel_idx,
                    role,
                    args.size(),
                    expected_args);
            }
        }
        patch_tilize_kernel_slot0(program, kernel_idx, addr);
    };

    for (uint32_t pair = 0; pair < num_pairs; ++pair) {
        patch_dm_kernel(dm_kernel_metadata[1 + 2 * pair], src_addr, kReaderRuntimeArgCount, "input reader");
        patch_dm_kernel(dm_kernel_metadata[2 + 2 * pair], dst_addr, kWriterRuntimeArgCount, "output writer");
    }
}

}  // namespace ttnn::prim
