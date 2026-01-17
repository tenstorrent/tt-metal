// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_interleaved_program_factory.hpp"
#include "untilize_with_unpadding_multi_core_block_interleaved_program_factory.hpp"

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory::cached_program_t
UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory::create(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    bool use_pack_untilize = operation_attributes.use_pack_untilize;
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;

    tt::tt_metal::Program program{};

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

    uint32_t num_blocks = input_shape[-1] == 0 ? 0 : a.physical_volume() / input_shape[-1] / TILE_HEIGHT;
    uint32_t num_tiles_per_row = a.padded_shape()[-1] / TILE_WIDTH;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, num_blocks);

    bool has_cliff = !core_range_cliff.empty();

    uint32_t padded_row_size_bytes;
    uint32_t unpadded_row_size_bytes;

    if (a.dtype() == DataType::BFLOAT8_B) {
        padded_row_size_bytes = input_shape[-1] * output.element_size();
        unpadded_row_size_bytes = output_shape[-1] * output.element_size();
    } else {
        padded_row_size_bytes = input_shape[-1] * a.element_size();
        unpadded_row_size_bytes = output_shape[-1] * a.element_size();
    }

    create_cb(tt::CBIndex::c_0, program, all_cores, input_single_tile_size, num_tiles_per_row, input_cb_data_format);
    create_cb(tt::CBIndex::c_16, program, all_cores, output_single_tile_size, num_tiles_per_row, output_cb_data_format);

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    /** reader
     */

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);
    KernelHandle unary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_compile_time_args));

    /** writer
     */
    std::vector<uint32_t> writer_ct_args = {
        (input_cb_data_format == tt::DataFormat::Float32 or input_cb_data_format == tt::DataFormat::UInt32 or
         input_cb_data_format == tt::DataFormat::Int32),
        unpadded_row_size_bytes};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
    KernelHandle unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize_with_unpadding/device/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_multicore.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    /** compute
     */
    std::map<std::string, std::string> compute_kernel_defines;
    if (input_cb_data_format == tt::DataFormat::Int32 || input_cb_data_format == tt::DataFormat::UInt32 ||
        input_cb_data_format == tt::DataFormat::Float32) {
        compute_kernel_defines["DST_ACCUM_MODE"] = "1";
    }
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[tt::CBIndex::c_0] = UnpackToDestMode::UnpackToDestFp32;
    }
    std::string compute_kernel(
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp");
    if (!use_pack_untilize || a.dtype() == DataType::UINT16 ||
        (input_cb_data_format == tt::DataFormat::Float32 && num_tiles_per_row > MAX_PACK_UNTILIZE_WIDTH)) {
        compute_kernel = "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp";
        unpack_to_dest_mode[tt::CBIndex::c_0] =
            UnpackToDestMode::Default;  // TODO: We need SFPU untilize for FP32 (#30400, #33795)
    }

    if (!core_range.empty()) {
        CreateKernel(
            program,
            compute_kernel,
            core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .compile_args = {nblocks_per_core, num_tiles_per_row, tt::CBIndex::c_0, tt::CBIndex::c_16},
                .defines = compute_kernel_defines});
    }
    if (has_cliff) {
        CreateKernel(
            program,
            compute_kernel,
            core_range_cliff,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .compile_args = {nblocks_per_core_cliff, num_tiles_per_row, tt::CBIndex::c_0, tt::CBIndex::c_16},
                .defines = compute_kernel_defines});
    }

    uint32_t tile_height = output.tensor_spec().tile().get_height();
    auto core_assignments = ttnn::distribute_work(
        output_shape, input_shape, ncores, nblocks_per_core, has_cliff, nblocks_per_core_cliff, tile_height);

    uint32_t tile_start_id = 0;
    uint32_t row_start_id = 0;

    const auto& cores = corerange_to_cores(available_grid);
    for (uint32_t i = 0; i < ncores; ++i) {
        const auto& core = cores[i];
        const std::vector<BlockRep>& assignment = core_assignments.at(i);

        // writer runtime args
        std::vector<uint32_t> writer_rt_args = {
            dst_buffer->address(),
            padded_row_size_bytes,
            row_start_id,
            static_cast<unsigned int>(assignment.size()),
        };

        uint32_t nblocks_per_core = 0;

        BlockRep ref_el = assignment[0];
        uint32_t count_repeated = 0;  // will be incremented in first iteration of the loop
        for (const auto& el : assignment) {
            nblocks_per_core += el.block_count();
            row_start_id += el.data_row_count();
            if (compare_assignments(ref_el, el)) {
                count_repeated++;
            } else {
                // push back information for previous elements
                writer_rt_args.push_back(ref_el.n_data);
                writer_rt_args.push_back(ref_el.n_mixed);
                writer_rt_args.push_back(ref_el.n_pads);
                writer_rt_args.push_back(ref_el.times);
                writer_rt_args.push_back(count_repeated);
                // Set up assignment for this element
                ref_el = el;
                count_repeated = 1;
            }
        }
        writer_rt_args.push_back(ref_el.n_data);
        writer_rt_args.push_back(ref_el.n_mixed);
        writer_rt_args.push_back(ref_el.n_pads);
        writer_rt_args.push_back(ref_el.times);
        writer_rt_args.push_back(count_repeated);

        uint32_t num_tiles_per_core = num_tiles_per_row * nblocks_per_core;

        // reader runtime args
        const std::array reader_rt_args = {src0_buffer->address(), num_tiles_per_core, tile_start_id};

        SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);

        tile_start_id += num_tiles_per_core;
    }

    return cached_program_t{
        std::move(program),
        shared_variables_t{
            .reader_kernel_id = unary_reader_kernel_id,
            .writer_kernel_id = unary_writer_kernel_id,
            .cores = cores,
            .ncores = ncores}};
}

void UntilizeWithUnpaddingMultiCoreInterleavedProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const UntilizeWithUnpaddingParams& /*operation_attributes*/,
    const Tensor& input,
    const Tensor& output) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;
    const auto& ncores = shared_vars.ncores;
    const auto& cores = shared_vars.cores;
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();
    auto& reader_runtime_args_by_core = GetRuntimeArgs(program, shared_vars.reader_kernel_id);
    auto& writer_runtime_args_by_core = GetRuntimeArgs(program, shared_vars.writer_kernel_id);

    for (uint32_t i = 0; i < ncores; ++i) {
        const CoreCoord& core = cores[i];
        {
            auto& runtime_args = reader_runtime_args_by_core[core.x][core.y];
            runtime_args[0] = src_buffer->address();
        }
        {
            auto& runtime_args = writer_runtime_args_by_core[core.x][core.y];
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::prim
