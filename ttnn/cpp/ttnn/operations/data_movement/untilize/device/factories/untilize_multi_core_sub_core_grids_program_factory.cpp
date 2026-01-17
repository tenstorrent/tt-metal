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
#include "untilize_multi_core_sub_core_grids_program_factory.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

UntilizeMultiCoreSubCoreGridsProgramFactory::cached_program_t UntilizeMultiCoreSubCoreGridsProgramFactory::create(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    const UntilizeTensorReturnValue& tensor_return_value) {
    tt::tt_metal::Program program{};

    const auto& a = tensor_args.input;
    const auto& output = tensor_return_value;
    const auto& sub_core_grids = operation_attributes.sub_core_grids.value();
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    const auto& use_pack_untilize = operation_attributes.use_pack_untilize;

    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    uint32_t ntiles = a.physical_volume() / TILE_HW;
    uint32_t ncores = sub_core_grids.num_cores();
    for (uint32_t core_id = ncores; core_id >= 1; core_id--) {
        if (ntiles % ncores == 0) {
            break;
        }
        ncores--;
    }

    TT_ASSERT(ntiles % (ncores) == 0);

    uint32_t max_tiles = 1;
    uint32_t ntiles_per_block = ntiles / ncores;
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

    auto cores = corerange_to_cores(sub_core_grids, ncores, true);
    auto all_cores = num_cores_to_corerangeset_in_subcoregrids(cores[0], ncores, sub_core_grids, true);
    uint32_t nblocks_per_core = nblocks / ncores;

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

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_ct_args));

    std::vector<uint32_t> writer_ct_args = {stick_size};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

    auto writer_kernel_id = CreateKernel(
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

    CreateKernel(
        program,
        compute_kernel,
        all_cores,
        ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .compile_args = compute_args,
            .defines = compute_kernel_defines});

    uint32_t tile_start_id = 0;
    uint32_t offset_within_stick = 0;

    auto nsticks_per_core = ntiles_per_column * TILE_HEIGHT;

    for (auto core : cores) {
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

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_rt_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_rt_args);
        cores_with_rtargs.push_back(core);
        tile_start_id += ntiles_per_core;
        offset_within_stick += ntiles_per_core * TILE_WIDTH * output.element_size();
    }

    return UntilizeMultiCoreSubCoreGridsProgramFactory::cached_program_t{
        std::move(program), {reader_kernel_id, writer_kernel_id, cb_src0, cb_output, cores_with_rtargs}};
}

void UntilizeMultiCoreSubCoreGridsProgramFactory::override_runtime_arguments(
    UntilizeMultiCoreSubCoreGridsProgramFactory::cached_program_t& cached_program,
    const UntilizeOperationAttributes& /*operation_attributes*/,
    const UntilizeTensorArgs& tensor_args,
    const UntilizeTensorReturnValue& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& cores_with_rtargs = cached_program.shared_variables.cores_with_rtargs;

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
