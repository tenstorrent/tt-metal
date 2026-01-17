// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::prim {

using namespace tt::constants;

TypecastProgramFactory::cached_program_t TypecastProgramFactory::create(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& input_dtype = args.input_dtype;
    const auto& output_dtype = args.output_dtype;

    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size_output = tt::tile_size(cb_data_format_output);

    uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;

    tt::tt_metal::IDevice* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = tt::CBIndex::c_2;
    uint32_t num_output_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * single_tile_size_output, {{output_cb_index, cb_data_format_output}})
            .set_page_size(output_cb_index, single_tile_size_output);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle typecast_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle typecast_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        num_tiles_per_core_group_1,  // per_core_block_cnt
        1,                           // per_core_block_size
        src0_cb_index,
        output_cb_index};

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    bool math_approx_mode = false;

    std::map<std::string, std::string> unary_defines;
    unary_defines["TYPECAST_LLK_INIT"] = fmt::format(
        "typecast_tile_init<{0}u, {1}u>",
        static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
        static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));
    unary_defines["TYPECAST_LLK"] = fmt::format(
        "typecast_tile<{0}u, {1}u>",
        static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
        static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));

    const auto* path = "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp";

    tt::tt_metal::CreateKernel(
        program,
        path,
        core_group_1,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = args.fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = args.bfp8_pack_precise,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args_group_1,
            .defines = unary_defines});

    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            num_tiles_per_core_group_2,  // per_core_block_cnt
            1,                           // per_core_block_size
            src0_cb_index,
            output_cb_index};

        tt::tt_metal::CreateKernel(
            program,
            path,
            core_group_2,
            tt::tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = args.fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .bfp8_pack_precise = args.bfp8_pack_precise,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args_group_2,
                .defines = unary_defines});
    }

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        tt::tt_metal::SetRuntimeArgs(
            program, typecast_reader_kernel_id, core, {src_buffer->address(), num_tiles_per_core, num_tiles_written});

        tt::tt_metal::SetRuntimeArgs(
            program, typecast_writer_kernel_id, core, {dst_buffer->address(), num_tiles_per_core, num_tiles_written});
        num_tiles_written += num_tiles_per_core;
    }

    return cached_program_t{
        std::move(program), {typecast_reader_kernel_id, typecast_writer_kernel_id, num_cores, num_cores_y}};
}

void TypecastProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TypecastParams& /*operation_attributes*/,
    const TypecastInputs& tensor_args,
    Tensor& output) {
    auto& typecast_reader_kernel_id = cached_program.shared_variables.typecast_reader_kernel_id;
    auto& typecast_writer_kernel_id = cached_program.shared_variables.typecast_writer_kernel_id;
    const uint32_t num_cores = cached_program.shared_variables.num_cores;
    const uint32_t num_cores_y = cached_program.shared_variables.num_cores_y;

    auto& program = cached_program.program;

    const auto& input = tensor_args.input;
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(program, typecast_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, typecast_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

// For sub_core_grids
TypecastSubgridProgramFactory::cached_program_t TypecastSubgridProgramFactory::create(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& input_dtype = args.input_dtype;
    const auto& output_dtype = args.output_dtype;
    const auto& sub_core_grids = args.sub_core_grids;

    TT_FATAL(sub_core_grids.has_value(), "sub_core_grids cannot be null");

    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size_output = tt::tile_size(cb_data_format_output);

    uint32_t ntiles = input.physical_volume() / tt::constants::TILE_HW;
    uint32_t ncores = sub_core_grids->num_cores();

    TT_FATAL(ncores != 0, "number of cores cannot be 0");

    for (uint32_t core_id = ncores; core_id >= 1; core_id--) {
        if (ntiles % ncores == 0) {
            break;
        }
        ncores--;
    }
    TT_FATAL(
        (ntiles % (ncores) == 0), "{} num of tiles are not split uniformly across {} num of cores", ntiles, ncores);

    auto cores = corerange_to_cores(sub_core_grids.value(), ncores, true);
    auto all_cores = num_cores_to_corerangeset_in_subcoregrids(cores[0], ncores, sub_core_grids.value(), true);
    if (ncores == 1) {
        all_cores = ttnn::CoreRangeSet(ttnn::CoreRange(cores[0]));
    }

    uint32_t ntiles_per_block = ntiles / ncores;
    uint32_t nblocks = (ntiles / ntiles_per_block);
    uint32_t nblocks_per_core = nblocks / ncores;

    std::vector<CoreCoord> cores_with_rtargs;

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = ntiles_per_block * 2;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t output_cb_index = tt::CBIndex::c_2;
    uint32_t num_output_tiles = ntiles_per_block * 2;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * single_tile_size_output, {{output_cb_index, cb_data_format_output}})
            .set_page_size(output_cb_index, single_tile_size_output);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle typecast_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle typecast_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_kernel_args = {
        static_cast<uint32_t>(nblocks_per_core),  // per_core_block_cnt
        static_cast<uint32_t>(ntiles_per_block),  // per_block_ntiles // per_core_block_size
        src0_cb_index,
        output_cb_index};

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    bool math_approx_mode = false;

    std::map<std::string, std::string> unary_defines;
    unary_defines["TYPECAST_LLK_INIT"] = fmt::format(
        "typecast_tile_init<{0}u, {1}u>",
        static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
        static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));
    unary_defines["TYPECAST_LLK"] = fmt::format(
        "typecast_tile<{0}u, {1}u>",
        static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
        static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));

    const auto* path = "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp";

    tt::tt_metal::CreateKernel(
        program,
        path,
        all_cores,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = args.fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = args.bfp8_pack_precise,
            .math_approx_mode = math_approx_mode,
            .compile_args = compute_kernel_args,
            .defines = unary_defines});

    uint32_t tile_start_id = 0;
    auto ntiles_per_core = ntiles_per_block * nblocks_per_core;

    for (auto core : cores) {
        tt::tt_metal::SetRuntimeArgs(
            program, typecast_reader_kernel_id, core, {src_buffer->address(), ntiles_per_core, tile_start_id});

        tt::tt_metal::SetRuntimeArgs(
            program, typecast_writer_kernel_id, core, {dst_buffer->address(), ntiles_per_core, tile_start_id});

        cores_with_rtargs.push_back(core);
        tile_start_id += ntiles_per_core;
    }

    return cached_program_t{
        std::move(program), {typecast_reader_kernel_id, typecast_writer_kernel_id, cores_with_rtargs}};
}

void TypecastSubgridProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TypecastParams& /*operation_attributes*/,
    const TypecastInputs& tensor_args,
    Tensor& output) {
    auto& typecast_reader_kernel_id = cached_program.shared_variables.typecast_reader_kernel_id;
    auto& typecast_writer_kernel_id = cached_program.shared_variables.typecast_writer_kernel_id;
    auto& cores_with_rtargs = cached_program.shared_variables.cores_with_rtargs;

    auto& program = cached_program.program;

    const auto& input = tensor_args.input;
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    {
        auto& runtime_args_by_core = GetRuntimeArgs(program, typecast_reader_kernel_id);
        for (const CoreCoord& core : cores_with_rtargs) {
            auto& runtime_args = runtime_args_by_core[core.x][core.y];
            runtime_args[0] = src_buffer->address();
        }
    }

    {
        auto& runtime_args_by_core = GetRuntimeArgs(program, typecast_writer_kernel_id);
        for (const CoreCoord& core : cores_with_rtargs) {
            auto& runtime_args = runtime_args_by_core[core.x][core.y];
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::prim
