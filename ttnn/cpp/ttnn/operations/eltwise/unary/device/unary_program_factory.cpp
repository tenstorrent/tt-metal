// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>

#include "unary_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/eltwise/unary/common/unary_op_utils.hpp"

namespace ttnn::operations::unary::program {

static const std::string compute_root = "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/compute/";

using namespace tt::constants;

UnaryProgramFactory::cached_program_t UnaryProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& ops_chain = args.op_chain;
    uint32_t packed_scalar1 = 0u;
    uint32_t packed_scalar2 = 0u;
    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size_output = tt::tile_size(cb_data_format_output);

    // Get number of pages (tiles for TILE layout, rows for ROW_MAJOR layout)
    const uint32_t num_pages = input.buffer()->num_pages();
    const bool is_row_major = input.layout() == Layout::ROW_MAJOR;

    tt::tt_metal::IDevice* device = input.device();

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_pages);
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    uint32_t num_input_tiles = 2;
    // For bitcast, use output format for input CB to avoid unpacker conversion
    // This ensures raw bit copying without conversion
    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    // Set CB page size correctly based on layout (tile size for tile layout, buffer page size for row-major)
    const uint32_t input_cb_page_size = is_row_major ? src_buffer->page_size() : single_tile_size;
    const uint32_t output_cb_page_size = is_row_major ? dst_buffer->page_size() : single_tile_size_output;

    tt::DataFormat cb_data_format_for_input =
        (ops_chain[0].type() == UnaryOpType::BITCAST) ? cb_data_format_output : cb_data_format;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            num_input_tiles * input_cb_page_size, {{src0_cb_index, cb_data_format_for_input}})
            .set_page_size(src0_cb_index, input_cb_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t tmp0_cb_index = tt::CBIndex::c_1;  // temporary buffer for intermediate results
    if (ops_chain[0].type() == UnaryOpType::HARDSHRINK || ops_chain[0].type() == UnaryOpType::CBRT ||
        ops_chain[0].type() == UnaryOpType::LOGIT) {
        tt::tt_metal::CircularBufferConfig cb_tmp0_config =
            tt::tt_metal::CircularBufferConfig(num_input_tiles * input_cb_page_size, {{tmp0_cb_index, cb_data_format}})
                .set_page_size(tmp0_cb_index, input_cb_page_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_tmp0_config);
    }

    uint32_t output_cb_index = tt::CBIndex::c_2;
    uint32_t num_output_tiles = 2;
    tt::tt_metal::CircularBufferConfig cb_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_tiles * output_cb_page_size, {{output_cb_index, cb_data_format_output}})
            .set_page_size(output_cb_index, output_cb_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(output_cb_index)};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        num_pages_per_core_group_1,  // per_core_block_cnt
        1,                           // per_core_block_size
    };

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[tmp0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    bool math_approx_mode = std::all_of(
        args.op_chain.begin(), args.op_chain.end(), [](const auto& u) { return utils::get_op_approx_mode(u.type()); });
    std::map<std::string, std::string> unary_defines = utils::get_block_defines(args.op_chain, "0", "0", input.dtype());

    if (input.dtype() == DataType::FLOAT32) {
        unary_defines["INP_FLOAT32"] = "1";
    } else if (input.dtype() == DataType::INT32) {
        unary_defines["INP_INT32"] = "1";
    } else if (input.dtype() == DataType::UINT32) {
        unary_defines["INP_UINT32"] = "1";
    } else {
        unary_defines["INP_FLOAT"] = "1";
    }

    if (!ops_chain[0].empty()) {
        switch (ops_chain[0].type()) {
            case UnaryOpType::HARDSHRINK:
                packed_scalar1 = utils::pack_scalar_runtime_arg(ops_chain[0], 0, input.dtype());
                break;
            case UnaryOpType::WHERE_TSS:
                packed_scalar1 = utils::pack_scalar_runtime_arg(ops_chain[0], 0, input.dtype());
                packed_scalar2 = utils::pack_scalar_runtime_arg(ops_chain[0], 1, input.dtype());
                break;
            case UnaryOpType::LOGIT: {
                float value1 = *ops_chain[0].get_param_if<float>(0);
                float value2 = 1.0f - value1;
                packed_scalar1 = utils::pack_scalar_runtime_arg_impl(value1, input.dtype());
                packed_scalar2 = utils::pack_scalar_runtime_arg_impl(value2, input.dtype());
                if (value1 > 0.5f) {
                    unary_defines["WHERE"] = "where_tile";
                    unary_defines["CLAMP"] = "clamp_tile";
                } else if (value1 >= 0.0f) {
                    unary_defines["CLAMP"] = "clamp_tile";
                }
                break;
            }
            default: break;
        }
    }

    auto path = fmt::format("{}/{}", compute_root, utils::get_compute_kernel_path(ops_chain[0].type(), input.dtype()));

    auto eltwise_unary_kernel_group_1_id = tt::tt_metal::CreateKernel(
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

    auto eltwise_unary_kernel_group_2_id = 0;
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            num_pages_per_core_group_2,  // per_core_block_cnt
            1,                           // per_core_block_size
        };

        eltwise_unary_kernel_group_2_id = tt::tt_metal::CreateKernel(
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

    for (uint32_t i = 0, num_pages_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_pages_per_core = 0;
        auto kernel_id = eltwise_unary_kernel_group_1_id;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
            kernel_id = eltwise_unary_kernel_group_2_id;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        tt::tt_metal::SetRuntimeArgs(
            program, unary_reader_kernel_id, core, {src_buffer->address(), num_pages_per_core, num_pages_written});

        tt::tt_metal::SetRuntimeArgs(
            program, unary_writer_kernel_id, core, {dst_buffer->address(), num_pages_per_core, num_pages_written});

        tt::tt_metal::SetRuntimeArgs(program, kernel_id, core, {packed_scalar1, packed_scalar2});
        num_pages_written += num_pages_per_core;
    }

    return cached_program_t{
        std::move(program), {unary_reader_kernel_id, unary_writer_kernel_id, num_cores, num_cores_y}};
}

void UnaryProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& output) {
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    const uint32_t num_cores = cached_program.shared_variables.num_cores;
    const uint32_t num_cores_y = cached_program.shared_variables.num_cores_y;

    auto& program = cached_program.program;

    const auto& input = tensor_args.input;
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    for (uint32_t i = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

// Sub Core Grids : should be fused later with UnaryProgramFactory directing all_device_cores to use cores from
// sub_core_grids and update the override args accordingly after adding cores as rtargs
UnarySubCoreGridProgramFactory::cached_program_t UnarySubCoreGridProgramFactory::create(
    const operation_attributes_t& args, const tensor_args_t& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& ops_chain = args.op_chain;
    uint32_t packed_scalar1 = 0u;
    uint32_t packed_scalar2 = 0u;
    const auto& sub_core_grids = args.sub_core_grids;

    TT_FATAL(sub_core_grids.has_value(), "sub_core_grids cannot be null");

    tt::tt_metal::Program program{};

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size_output = tt::tile_size(cb_data_format_output);

    uint32_t num_tiles = input.physical_volume() / tt::constants::TILE_HW;

    // TODO:
    // auto compute_with_storage_grid_size = sub_core_grids.value();
    // auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
    //     tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    uint32_t ncores = sub_core_grids->num_cores();

    TT_FATAL(ncores != 0, "number of cores cannot be 0");

    for (uint32_t core_id = ncores; core_id >= 1; core_id--) {
        if (num_tiles % ncores == 0) {
            break;
        }
        ncores--;
    }
    TT_FATAL(
        (num_tiles % (ncores) == 0),
        "{} num of tiles are not split uniformly across {} num of cores",
        num_tiles,
        ncores);

    auto cores = corerange_to_cores(sub_core_grids.value(), ncores, true);
    auto all_cores = num_cores_to_corerangeset_in_subcoregrids(cores[0], ncores, sub_core_grids.value(), true);
    if (ncores == 1) {
        all_cores = ttnn::CoreRangeSet(ttnn::CoreRange(cores[0]));
    }

    uint32_t ntiles_per_block = num_tiles / ncores;
    uint32_t nblocks = (num_tiles / ntiles_per_block);
    uint32_t nblocks_per_core = nblocks / ncores;
    std::vector<CoreCoord> cores_with_rtargs;

    uint32_t num_input_tiles = ntiles_per_block * 2;

    uint32_t src0_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    uint32_t tmp0_cb_index = tt::CBIndex::c_1;  // temporary buffer for intermediate results
    if (ops_chain[0].type() == UnaryOpType::HARDSHRINK || ops_chain[0].type() == UnaryOpType::CBRT) {
        tt::tt_metal::CircularBufferConfig cb_tmp0_config =
            tt::tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{tmp0_cb_index, cb_data_format}})
                .set_page_size(tmp0_cb_index, single_tile_size);
        tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_tmp0_config);
    }

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
    std::vector<uint32_t> writer_compile_time_args = {output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_kernel_args = {
        (uint32_t)nblocks_per_core,  // per_core_block_cnt
        (uint32_t)ntiles_per_block,  // per_block_num_tiles // per_core_block_size
    };

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[tmp0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    bool math_approx_mode = std::all_of(
        args.op_chain.begin(), args.op_chain.end(), [](const auto& u) { return utils::get_op_approx_mode(u.type()); });
    std::map<std::string, std::string> unary_defines = utils::get_block_defines(args.op_chain, "0", "0", input.dtype());

    if (input.dtype() == DataType::FLOAT32) {
        unary_defines["INP_FLOAT32"] = "1";
    } else if (input.dtype() == DataType::INT32) {
        unary_defines["INP_INT32"] = "1";
    } else if (input.dtype() == DataType::UINT32) {
        unary_defines["INP_UINT32"] = "1";
    } else {
        unary_defines["INP_FLOAT"] = "1";
    }

    if (!ops_chain[0].empty()) {
        switch (ops_chain[0].type()) {
            case UnaryOpType::HARDSHRINK:
                packed_scalar1 = utils::pack_scalar_runtime_arg(ops_chain[0], 0, input.dtype());
                break;
            case UnaryOpType::WHERE_TSS:
                packed_scalar1 = utils::pack_scalar_runtime_arg(ops_chain[0], 0, input.dtype());
                packed_scalar2 = utils::pack_scalar_runtime_arg(ops_chain[0], 1, input.dtype());
                break;
            default: break;
        }
    }

    auto path = fmt::format("{}/{}", compute_root, utils::get_compute_kernel_path(ops_chain[0].type(), input.dtype()));

    auto eltwise_unary_kernel_id = tt::tt_metal::CreateKernel(
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
            program, unary_reader_kernel_id, core, {src_buffer->address(), ntiles_per_core, tile_start_id});

        tt::tt_metal::SetRuntimeArgs(
            program, unary_writer_kernel_id, core, {dst_buffer->address(), ntiles_per_core, tile_start_id});

        tt::tt_metal::SetRuntimeArgs(program, eltwise_unary_kernel_id, core, {packed_scalar1, packed_scalar2});

        cores_with_rtargs.push_back(core);
        tile_start_id += ntiles_per_core;
    }

    return cached_program_t{std::move(program), {unary_reader_kernel_id, unary_writer_kernel_id, cores_with_rtargs}};
}

void UnarySubCoreGridProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& output) {
    auto& unary_reader_kernel_id = cached_program.shared_variables.unary_reader_kernel_id;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;
    auto& cores_with_rtargs = cached_program.shared_variables.cores_with_rtargs;

    auto& program = cached_program.program;

    const auto& input = tensor_args.input;
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    {
        auto& runtime_args_by_core = GetRuntimeArgs(program, unary_reader_kernel_id);
        for (const CoreCoord& core : cores_with_rtargs) {
            auto& runtime_args = runtime_args_by_core[core.x][core.y];
            runtime_args[0] = src_buffer->address();
        }
    }

    {
        auto& runtime_args_by_core = GetRuntimeArgs(program, unary_writer_kernel_id);
        for (const CoreCoord& core : cores_with_rtargs) {
            auto& runtime_args = runtime_args_by_core[core.x][core.y];
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::operations::unary::program
