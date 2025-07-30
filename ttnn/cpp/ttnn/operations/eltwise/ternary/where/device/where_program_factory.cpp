// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "where_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/cb_utils.hpp"
#include <cmath>

namespace ttnn::operations::ternary {
WhereDeviceOperation::WhereProgramFactory::cached_program_t WhereDeviceOperation::WhereProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& [predicate_tensor, value_true_tensor, value_false_tensor, optional_output_tensor] = tensor_args;

    auto program = CreateProgram();

    auto* device = predicate_tensor.device();

    auto predicate_data_format = datatype_to_dataformat_converter(predicate_tensor.dtype());
    auto value_true_data_format = datatype_to_dataformat_converter(value_true_tensor.dtype());
    auto value_false_data_format = datatype_to_dataformat_converter(value_false_tensor.dtype());
    auto output_data_format = datatype_to_dataformat_converter(output.dtype());

    // auto predicate_data_format = datatype_to_dataformat_converter(
    //     (predicate_tensor.dtype() == DataType::BFLOAT16) ? DataType::UINT16 : predicate_tensor.dtype());
    // auto value_true_data_format = datatype_to_dataformat_converter(
    //     (value_true_tensor.dtype() == DataType::BFLOAT16) ? DataType::UINT16 : value_true_tensor.dtype());
    // auto value_false_data_format = datatype_to_dataformat_converter(
    //     (value_false_tensor.dtype() == DataType::BFLOAT16) ? DataType::UINT16 : value_false_tensor.dtype());
    // auto output_data_format =
    //     datatype_to_dataformat_converter((output.dtype() == DataType::BFLOAT16) ? DataType::UINT16 : output.dtype());

    uint32_t predicate_single_tile_size = tt_metal::detail::TileSize(predicate_data_format);
    uint32_t value_true_single_tile_size = tt_metal::detail::TileSize(value_true_data_format);
    uint32_t value_false_single_tile_size = tt_metal::detail::TileSize(value_false_data_format);
    uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_data_format);

    uint32_t num_output_tiles = output.physical_volume() / output.tensor_spec().tile().get_tile_hw();

    // we parallelize the computation across the output tiles
    // constexpr bool row_major = true;
    // auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    // uint32_t num_cores_x = compute_with_storage_grid_size.x;
    // uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // auto all_device_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_output_tiles);

    // Number of tiles to store per input CB (double buffer)
    constexpr uint32_t num_tiles_per_cb = 2;

    // Input buffers
    auto [predicate_tensor_cb, predicate_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_0,
        program,
        all_cores,
        predicate_single_tile_size,
        num_tiles_per_cb,
        predicate_data_format);  // predicate_tensor
    auto [value_true_tensor_cb, value_true_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_1,
        program,
        all_cores,
        value_true_single_tile_size,
        num_tiles_per_cb,
        value_true_data_format);  // value_true_tensor
    auto [value_false_tensor_cb, value_false_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_2,
        program,
        all_cores,
        value_false_single_tile_size,
        num_tiles_per_cb,
        value_false_data_format);  // value_false_tensor

    // Output buffer
    auto [output_tensor_cb, output_tensor_cb_handle] = create_cb(
        tt::CBIndex::c_3,
        program,
        all_cores,
        output_single_tile_size,
        num_tiles_per_cb,
        output_data_format);  // output

    auto predicate_is_dram =
        static_cast<uint32_t>(predicate_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    auto value_true_is_dram =
        static_cast<uint32_t>(value_true_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    auto value_false_is_dram =
        static_cast<uint32_t>(value_false_tensor.buffer()->buffer_type() == tt_metal::BufferType::DRAM);
    auto output_is_dram = static_cast<uint32_t>(output.buffer()->buffer_type() == tt_metal::BufferType::DRAM);

    // READER KERNEL
    auto reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/ternary/where/device/kernels/dataflow/ternary_reader_nobcast_ttt.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(
            {predicate_is_dram,
             predicate_tensor_cb,
             value_true_is_dram,
             value_true_tensor_cb,
             value_false_is_dram,
             value_false_tensor_cb}));

    // WRITER KERNEL
    auto writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig({output_tensor_cb, output_is_dram}));

    // COMPUTE KERNEL
    bool fp32_dest_acc_en = output_data_format == tt::DataFormat::UInt32 ||
                            output_data_format == tt::DataFormat::Int32 ||
                            output_data_format == tt::DataFormat::Float32;

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    unpack_to_dest_mode[tt::CBIndex::c_0] = (predicate_tensor.dtype() == DataType::FLOAT32)
                                                ? UnpackToDestMode::UnpackToDestFp32
                                                : UnpackToDestMode::Default;
    unpack_to_dest_mode[tt::CBIndex::c_1] = (value_true_tensor.dtype() == DataType::FLOAT32)
                                                ? UnpackToDestMode::UnpackToDestFp32
                                                : UnpackToDestMode::Default;
    unpack_to_dest_mode[tt::CBIndex::c_2] = (value_false_tensor.dtype() == DataType::FLOAT32)
                                                ? UnpackToDestMode::UnpackToDestFp32
                                                : UnpackToDestMode::Default;
    unpack_to_dest_mode[tt::CBIndex::c_3] =
        (output.dtype() == DataType::FLOAT32) ? UnpackToDestMode::UnpackToDestFp32 : UnpackToDestMode::Default;

    // constexpr uint32_t num_tiles_per_cycle = 1;  // we produce 1 output tile per read-compute-write cycle

    std::vector<uint32_t> compute_kernel_args = {
        num_tiles_per_core_group_1,
        1,
    };

    std::map<std::string, std::string> kernel_defines;
    kernel_defines["WHERE_LLK"] = "where_tile";

    if (predicate_tensor.dtype() == DataType::FLOAT32) {
        kernel_defines["WHERE_LLK"] = "where_fp32_tile";
    } else if (predicate_tensor.dtype() == DataType::INT32) {
        kernel_defines["WHERE_LLK"] = "where_int32_tile";
    }

    // auto compute_kernel_id = tt_metal::CreateKernel(
    //     program,
    //     "ttnn/cpp/ttnn/operations/eltwise/ternary/where/device/kernels/compute/where_sfpu_no_bcast_ttt.cpp",
    //     all_device_cores,
    //     tt_metal::ComputeConfig{
    //         .fp32_dest_acc_en = fp32_dest_acc_en,
    //         .unpack_to_dest_mode = unpack_to_dest_mode,
    //         .compile_args = compute_kernel_args,
    //         .defines = kernel_defines});

    auto path = "ttnn/cpp/ttnn/operations/eltwise/ternary/where/device/kernels/compute/where_sfpu_no_bcast_ttt.cpp";

    auto eltwise_unary_kernel_group_1_id = tt::tt_metal::CreateKernel(
        program,
        path,
        core_group_1,
        tt::tt_metal::ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .compile_args = compute_kernel_args,
            .defines = kernel_defines});

    auto eltwise_unary_kernel_group_2_id = 0;
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            num_tiles_per_core_group_2,  // per_core_block_cnt
            1,
        };

        eltwise_unary_kernel_group_2_id = tt::tt_metal::CreateKernel(
            program,
            path,
            core_group_2,
            tt::tt_metal::ComputeConfig{
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .compile_args = compute_kernel_args_group_2,
                .defines = kernel_defines});
    }

    // auto set_runtime_args = [](Program& program, KernelHandle kernel_id, CoreCoord core, auto&& args) {
    //     tt_metal::SetRuntimeArgs(program, kernel_id, core, args);
    // };

    // CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
    //     program,
    //     reader_kernel_id,
    //     writer_kernel_id,
    //     compute_kernel_id,
    //     compute_with_storage_grid_size,
    //     operation_attributes,
    //     tensor_args,
    //     output,
    //     set_runtime_args);

    // return {
    //     std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id, compute_with_storage_grid_size}};
    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_tiles_per_core = 0;
        auto kernel_id = eltwise_unary_kernel_group_1_id;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
            kernel_id = eltwise_unary_kernel_group_2_id;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        tt::tt_metal::SetRuntimeArgs(
            program,
            reader_kernel_id,
            core,
            {predicate_tensor.buffer()->address(),
             value_true_tensor.buffer()->address(),
             value_false_tensor.buffer()->address(),
             num_tiles_per_core,
             num_tiles_written});

        tt::tt_metal::SetRuntimeArgs(
            program, writer_kernel_id, core, {output.buffer()->address(), num_tiles_per_core, num_tiles_written});

        tt::tt_metal::SetRuntimeArgs(program, kernel_id, core, {});

        num_tiles_written += num_tiles_per_core;
    }

    return cached_program_t{std::move(program), {reader_kernel_id, writer_kernel_id, num_cores, num_cores_y}};
}

void WhereDeviceOperation::WhereProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    // auto update_args =
    //     [](tt::tt_metal::Program& program, tt::tt_metal::KernelHandle kernel_id, CoreCoord core, auto&& args) {
    //         auto& all_args = GetRuntimeArgs(program, kernel_id);
    //         auto& core_args = all_args.at(core.x).at(core.y);
    //         std::copy(args.begin(), args.end(), core_args.data());
    //     };

    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const uint32_t num_cores = cached_program.shared_variables.num_cores;
    const uint32_t num_cores_y = cached_program.shared_variables.num_cores_y;

    auto& program = cached_program.program;

    // const auto& input = tensor_args.input;
    // auto src_buffer = input.buffer();
    // auto dst_buffer = output.buffer();
    const auto& [predicate_tensor, value_true_tensor, value_false_tensor, optional_output_tensor] = tensor_args;

    for (uint32_t i = 0, num_tiles_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = predicate_tensor.buffer()->address();
            runtime_args[1] = value_true_tensor.buffer()->address();
            runtime_args[2] = value_false_tensor.buffer()->address();
        }

        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = output.buffer()->address();
        }
    }

    // CMAKE_UNIQUE_NAMESPACE::set_or_update_runtime_arguments(
    //     cached_program.program,
    //     cached_program.shared_variables.reader_kernel_id,
    //     cached_program.shared_variables.writer_kernel_id,
    //     // cached_program.shared_variables.compute_kernel_id,
    //     // cached_program.shared_variables.compute_with_storage_grid_size,
    //     operation_attributes,
    //     tensor_args,
    //     output,
    //     update_args);
}

}  // namespace ttnn::operations::ternary
