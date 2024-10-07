// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include "common/constants.hpp"
#include "full_like_device_operation.hpp"
#include "host_api.hpp"
#include "impl/buffers/circular_buffer_types.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include "ttnn/tensor/types.hpp"
#include "tt_metal/common/work_split.hpp"

namespace ttnn::operations::full_like {

using namespace tt;
using namespace tt::tt_metal;
using namespace tt::constants;


union datatype {
    uint32_t u32;
    float f32;
} u;

FullLikeOperation::ProgramFactory::cached_program_t FullLikeOperation::ProgramFactory::create(
    const operation_attributes_t &operation_attributes,
    const tensor_args_t &tensor_args,
    tensor_return_value_t &output_tensor) {

    output_tensor.print();

    auto input = tensor_args.input;
    auto fill_value = operation_attributes.fill_value;
    uint32_t fill_value_u;
    if (std::holds_alternative<int>(fill_value)) {
        fill_value_u = static_cast<uint32_t>(std::get<int>(fill_value));
    } else if (std::holds_alternative<float>(fill_value)) {
        fill_value_u = static_cast<uint32_t>(std::get<float>(fill_value));
    }
    DataType dtype{operation_attributes.dtype};
    Layout layout{operation_attributes.layout};
    Device *device = input.device();
    MemoryConfig memory_config{operation_attributes.memory_config};

    auto output = output_tensor;

    auto num_tiles = compute_volume(input.legacy_shape()) / TILE_HW;

    // tt::DataFormat cb_data_format;
    // if (dtype == DataType::UINT8) {
    //     cb_data_format = tt::DataFormat::UInt8;
    // } else if (dtype == DataType::UINT16) {
    //     cb_data_format = tt::DataFormat::UInt16;
    // } else if (dtype == DataType::UINT32) {
    //     cb_data_format = tt::DataFormat::UInt32;
    // } else if (dtype == DataType::FLOAT32) {
    //     cb_data_format = tt::DataFormat::Float32;
    // } else if (dtype == DataType::BFLOAT16) {
    //     cb_data_format = tt::DataFormat::Float16_b;
    // }

    Program program{};

    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(dtype);
    uint32_t single_tile_size = tt::tt_metal::detail::TileSize(data_format);

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_tiles);

    constexpr CB cb_fill_value = CB::c_intermed0;

    const uint32_t cb_fill_value_num_tiles = 1;

    auto src_buffer = input.buffer();
    const bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> reader_compile_time_args = {
        (uint32_t) src_is_dram,
    };
    tt::tt_metal::CircularBufferConfig src_cb_config = tt::tt_metal::CircularBufferConfig(
        num_tiles * single_tile_size, {{tt::CB::c_intermed0, data_format}}
    ).set_page_size(tt::CB::c_intermed0, single_tile_size);
    auto cb_src = tt::tt_metal::CreateCircularBuffer(program, all_cores, src_cb_config);


    auto dst_buffer = output.buffer();
    bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t) dst_is_dram
    };
    tt::tt_metal::CircularBufferConfig dst_cb_config = tt::tt_metal::CircularBufferConfig(
        num_tiles * single_tile_size, {{tt::CB::c_out0, data_format}}
    ).set_page_size(tt::CB::c_out0, single_tile_size);
    auto cb_dst = tt::tt_metal::CreateCircularBuffer(program, all_cores, dst_cb_config);


    /* READER/WRTIER KERNEL */
    auto reader_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/full_like/device/kernels/reader_full_like.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    auto writer_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/full_like/device/kernels/writer_full_like.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    uint32_t tiles_offset = 0;
    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord core(i / num_cores_y, i % num_cores_y);

        uint32_t num_tiles_per_core = 0;
        if (core_group_1.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.core_coord_in_core_ranges(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        SetRuntimeArgs(program, reader_id, core, {fill_value_u});
        SetRuntimeArgs(
            program,
            writer_id,
            core,
            {
                output.buffer()->address(),
                num_tiles_per_core,
                tiles_offset,
                fill_value_u
            });

        tiles_offset += num_tiles_per_core;
    }

    // auto cb_src = tt::tt_metal::CreateCircularBuffer(program, core, src_cb_config);

    // Program program{};
    // tt::tt_metal::CommandQueue &cq = device->command_queue();
    // CoreRange core({0, 0}, {0, 0});

    // tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(dtype);
    // uint32_t single_tile_size = tt::tt_metal::detail::TileSize(data_format);

    // auto src_buffer = input.buffer();
    // auto dst_buffer = output.buffer();

    // auto src_buffer_addr = src_buffer->address();
    // auto dst_buffer_addr = dst_buffer->address();


    // auto num_tiles = compute_volume(input.legacy_shape()) / TILE_HW;

    // const bool src_is_dram = src_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    // std::vector<uint32_t> reader_compile_time_args = {
    //     (uint32_t) src_is_dram,
    // };
    // tt::tt_metal::CircularBufferConfig src_cb_config = tt::tt_metal::CircularBufferConfig(
    //     num_tiles * single_tile_size, {{tt::CB::c_in0, data_format}}
    // ).set_page_size(tt::CB::c_in0, single_tile_size);


    // auto cb_src = tt::tt_metal::CreateCircularBuffer(program, core, src_cb_config);
    // bool dst_is_dram = dst_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    // std::vector<uint32_t> writer_compile_time_args = {
    //     (std::uint32_t) dst_is_dram
    // };
    // tt::tt_metal::CircularBufferConfig dst_cb_config = tt::tt_metal::CircularBufferConfig(
    //     num_tiles * single_tile_size, {{tt::CB::c_out0, data_format}}
    // ).set_page_size(tt::CB::c_out0, single_tile_size);
    // auto cb_dst = tt::tt_metal::CreateCircularBuffer(program, core, dst_cb_config);

    // auto reader_id = tt::tt_metal::CreateKernel(
    //     program,
    //     "ttnn/cpp/ttnn/operations/full_like/device/kernels/reader_full_like.cpp",
    //     core,
    //     tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    // auto writer_id = tt::tt_metal::CreateKernel(
    //     program,
    //     "ttnn/cpp/ttnn/operations/full_like/device/kernels/writer_full_like.cpp",
    //     core,
    //     tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));



    // vector<uint32_t> compute_args = {};
    // auto full_like_single_core_kernel_id = tt::tt_metal::CreateKernel(
    //     program,
    //     "dmm",
    //     all_cores,
    //     tt::tt_metal::ComputeConfig{
    //         .math_fidelity = MathFidelity::HiFi4,
    //         .fp32_dest_acc_en = true,
    //         .compile_args = compute_args
    //     }
    // );


    // uint32_t tiles_offset = 0;
    // for (uint32_t i = 0; i < num_cores; i++) {
    //     const CoreCoord core(i / num_cores_y, i % num_cores_y);

    //     uint32_t num_tiles_per_core = 0;
    //     if (core_group_1.core_coord_in_core_ranges(core)) {
    //         num_tiles_per_core = num_tiles_per_core_group_1;
    //     } else if (core_group_2.core_coord_in_core_ranges(core)) {
    //         num_tiles_per_core = num_tiles_per_core_group_2;
    //     } else {
    //         TT_ASSERT(false, "Core not in specified core ranges");
    //     }

    //     SetRuntimeArgs(program, reader_id, core, {src_buffer_addr, value.u32, num_tiles});
    //     SetRuntimeArgs(
    //         program,
    //         writer_id,
    //         core,
    //         {
    //             dst_buffer_addr,
    //             num_tiles_per_core,
    //             tiles_offset,
    //         });

    //     tiles_offset += num_tiles_per_core;
    // }

    return {std::move(program), { writer_id}};
}

void FullLikeOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& unary_writer_kernel_id = cached_program.shared_variables.unary_writer_kernel_id;

    const auto& input = tensor_args.input;
    auto& output_tensor = tensor_return_value;

    auto src_buffer = input.buffer();
    auto dst_buffer = output_tensor.buffer();

    {
        auto& runtime_args = tt::tt_metal::GetRuntimeArgs(program, unary_writer_kernel_id, CoreCoord{0, 0});
        runtime_args[0] = dst_buffer->address();
    }
}

}
