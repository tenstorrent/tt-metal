// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "untilize_multi_core_parallelize_column_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

UntilizeMultiCoreParallelizeColumnProgramFactory::cached_program_t
UntilizeMultiCoreParallelizeColumnProgramFactory::create(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    const UntilizeTensorReturnValue& tensor_return_value) {
    tt::tt_metal::Program program{};

    const auto& a = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& use_pack_untilize = operation_attributes.use_pack_untilize;
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    IDevice* device = a.device();

    auto grid_size = device->compute_with_storage_grid_size();

    uint32_t ntiles = a.physical_volume() / TILE_HW;
    uint32_t ncores_x = grid_size.x;
    uint32_t ncores_y = grid_size.y;
    // uint32_t ncores_x = 2;

    ncores_x = untilize_helper::get_largest_divisor(ntiles, ncores_x);
    // uint32_t ncores_y = 1;
    ncores_y = untilize_helper::get_largest_divisor(ntiles, ncores_y, ncores_x);

    TT_ASSERT(ntiles % (ncores_x * ncores_y) == 0);
    uint32_t ntiles_per_block = ntiles / (ncores_x * ncores_y);

    // TODO increase block size to increase untilize performance, currently each untilize block is a single tile
    // uint32_t max_l1_size = a.device()->l1_size_per_core() / 2 -
    // a.device()->allocator()->get_base_allocator_addr(HalMemType::L1); uint32_t max_tiles =
    // (max_l1_size / (input_single_tile_size + output_single_tile_size))/2;  // 2 CBs, double buffering each
    uint32_t max_tiles = 1;

    uint32_t stick_s = a.padded_shape()[-1];
    uint32_t ntiles_per_row = stick_s / TILE_WIDTH;
    uint32_t stick_size = stick_s * output.element_size();
    uint32_t ntiles_per_column = ntiles / ntiles_per_row;
    uint32_t starting_tile = ntiles_per_block;
    if (ntiles_per_row > max_tiles) {
        starting_tile = max_tiles;
    }
    ntiles_per_block = untilize_helper::get_largest_divisor(ntiles_per_row, starting_tile);
    TT_ASSERT(
        ntiles_per_row % ntiles_per_block == 0 and ntiles_per_block >= 1 and ntiles_per_block <= ntiles_per_row and
        ntiles % ntiles_per_block == 0);

    uint32_t nblocks = (ntiles / ntiles_per_block);

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(CoreCoord(ncores_x, ncores_y), nblocks);

    bool row_major = true;
    std::vector<CoreCoord> cores_with_rtargs;

    uint32_t num_input_tiles = ntiles_per_block * 2;
    auto [src0_cb_index, cb_src0] = create_cb(
        tt::CBIndex::c_0, program, all_cores, input_single_tile_size, num_input_tiles, input_cb_data_format, nullptr);

    uint32_t num_output_tiles = ntiles_per_block * 2;
    auto [output_cb_index, cb_output] = create_cb(
        tt::CBIndex::c_16,
        program,
        all_cores,
        output_single_tile_size,
        num_output_tiles,
        output_cb_data_format,
        nullptr);

    Buffer* src0_buffer = a.buffer();
    Buffer* dst_buffer = output.buffer();
    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*src0_buffer).append_to(reader_ct_args);

    auto unary_reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_ct_args));

    std::vector<uint32_t> writer_ct_args = {stick_size};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

    auto unary_writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
        "writer_unary_stick_layout_split_rows_interleaved_parallel_columns.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    /** compute
     */
    std::vector<uint32_t> compute_args = {
        (uint32_t)nblocks_per_core,  // per_core_block_cnt
        (uint32_t)ntiles_per_block,  // per_block_ntiles
        (uint32_t)src0_cb_index,
        (uint32_t)output_cb_index};
    std::vector<uint32_t> compute_args_cliff = {
        (uint32_t)nblocks_per_core_cliff,
        (uint32_t)ntiles_per_block,  // per_block_ntiles
        (uint32_t)src0_cb_index,
        (uint32_t)output_cb_index};

    std::map<std::string, std::string> compute_kernel_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_kernel_defines["DST_ACCUM_MODE"] = "1";
    }
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (fp32_dest_acc_en) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }
    std::string compute_kernel(
        "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/pack_untilize.cpp");
    if (!use_pack_untilize || a.dtype() == DataType::UINT16 ||
        (a.dtype() == DataType::FLOAT32 && ntiles_per_block > MAX_PACK_UNTILIZE_WIDTH)) {
        log_debug(tt::LogOp, "Using slow untilize.");
        compute_kernel =
            std::string("ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/compute/untilize.cpp");
        unpack_to_dest_mode[src0_cb_index] =
            UnpackToDestMode::Default;  // TODO: We need SFPU untilize for FP32 (#30400, #33795)
    } else {
        log_debug(tt::LogOp, "Using fast pack untilize.");
    }
    if (!core_range.ranges().empty()) {
        CreateKernel(
            program,
            compute_kernel,
            core_range,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .compile_args = compute_args,
                .defines = compute_kernel_defines});
    }
    if (!core_range_cliff.ranges().empty()) {
        CreateKernel(
            program,
            compute_kernel,
            core_range_cliff,
            ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .compile_args = compute_args_cliff,
                .defines = compute_kernel_defines});
    }

    uint32_t ncores_full = ncores;
    auto full_cores = all_cores;
    if (nblocks_per_core_cliff > 0 && nblocks_per_core_cliff < nblocks_per_core) {
        // unequal case with cliff
        ncores_full -= 1;
        full_cores = core_range;
    }
    uint32_t tile_start_id = 0;
    uint32_t offset_within_stick = 0;
    auto cores = grid_to_cores(ncores_x * ncores_y, ncores_x, ncores_y, row_major);

    auto nsticks_per_core = ntiles_per_column * TILE_HEIGHT;

    for (auto core : cores) {
        if (!full_cores.contains(core)) {
            continue;
        }
        // reader runtime args
        auto ntiles_per_core = ntiles_per_block * nblocks_per_core;
        const std::array reader_rt_args = {
            src0_buffer->address(),  // src_addr
            ntiles_per_core,         // ntiles
            tile_start_id            // start_id
        };

        const std::array writer_rt_args = {
            dst_buffer->address(),               // dst_addr
            nsticks_per_core,                    // nsticks
            ntiles_per_core,                     // ntiles_per_core
            TILE_WIDTH * output.element_size(),  // tile_width_size
            std::uint32_t{0},                    // start stick id = 0, since parallelizing on height
            offset_within_stick};

        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);
        cores_with_rtargs.push_back(core);
        tile_start_id += ntiles_per_core;
        offset_within_stick += ntiles_per_core * TILE_WIDTH * output.element_size();
    }
    if (ncores_full < ncores) {
        // last core is the cliff core with nblocks_per_core_cliff blocks
        CoreCoord core = row_major ? CoreCoord{ncores_full % ncores_x, ncores_full / ncores_x}
                                   : CoreCoord{ncores_full / ncores_y, ncores_full % ncores_y};
        // reader runtime args
        auto ntiles_per_core_cliff = ntiles_per_block * nblocks_per_core_cliff;
        const std::array reader_rt_args = {
            src0_buffer->address(),  // src_addr
            ntiles_per_core_cliff,   // ntiles
            tile_start_id            // start_id
        };

        const std::array writer_rt_args = {
            dst_buffer->address(),               // dst_addr
            nsticks_per_core,                    // nsticks
            stick_size,                          // block_size_nbytes
            ntiles_per_core_cliff,               // ntiles_per_core
            TILE_WIDTH * output.element_size(),  // tile_width_size
            std::uint32_t{0},                    // start stick id = 0, since parallelizing on height
            offset_within_stick};
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, writer_rt_args);
    }

    return UntilizeMultiCoreParallelizeColumnProgramFactory::cached_program_t{
        std::move(program), {unary_reader_kernel_id, unary_writer_kernel_id, cb_src0, cb_output, cores_with_rtargs}};
}

void UntilizeMultiCoreParallelizeColumnProgramFactory::override_runtime_arguments(
    UntilizeMultiCoreParallelizeColumnProgramFactory::cached_program_t& cached_program,
    const UntilizeOperationAttributes& /*operation_attributes*/,
    const UntilizeTensorArgs& tensor_args,
    const UntilizeTensorReturnValue& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& cores_with_rtargs = cached_program.shared_variables.cores_with_runtime_args;
    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = tensor_return_value.buffer();
    {
        auto& runtime_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
        for (const CoreCoord& core : cores_with_rtargs) {
            auto& runtime_args = runtime_args_by_core[core.x][core.y];
            runtime_args[0] = src_buffer->address();
        }
    }

    {
        auto& runtime_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
        for (const CoreCoord& core : cores_with_rtargs) {
            auto& runtime_args = runtime_args_by_core[core.x][core.y];
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::prim
