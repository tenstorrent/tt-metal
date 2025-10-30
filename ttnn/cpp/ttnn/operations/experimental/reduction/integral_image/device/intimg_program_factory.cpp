// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "intimg_device_operation.hpp"
#include "intimg_program_factory.hpp"
#include "intimg_work_split.hpp"

#include "tt-metalium/base_types.hpp"
#include "tt-metalium/circular_buffer_config.hpp"
#include "tt-metalium/constants.hpp"
#include "tt-metalium/host_api.hpp"
#include "tt-metalium/kernel_types.hpp"
#include "ttnn/tensor/types.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include <iostream>

namespace ttnn::operations::experimental::reduction {

constexpr uint32_t SEMAPHORES_STARTING_VALUE = 0;
constexpr uint32_t TILES_PER_CB = 8;

IntImgProgramFactory::cached_program_t IntImgProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    std::cout << "PFPFPFPFPFPFPFPFPF" << std::endl;
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input_tensor{tensor_args.input_tensor};
    auto& output_tensor{tensor_return_value};
    const auto& input_shape{input_tensor.padded_shape()};

    Program program{};

    auto input_buffer{input_tensor.buffer()};
    auto zero_tile_buffer{tensor_args.zero_tile_tensor.buffer()};
    auto output_buffer{output_tensor.buffer()};

    const auto dst_cb_data_format{datatype_to_dataformat_converter(input_tensor.dtype())};
    const bool fp32_dest_acc_en{
        (dst_cb_data_format == DataFormat::Float32) || (dst_cb_data_format == DataFormat::Int32) ||
        (dst_cb_data_format == DataFormat::UInt32)};

    const auto tile_spec = input_tensor.tensor_spec().tile();

    const auto total_core_range_set =
        generate_max_core_range_set(input_shape);  // expected to be no more than 4 rows and 5 columns!
    std::cout << "TOTAL_CORE_RANGE_SET::: " << total_core_range_set.str() << std::endl;
    // TT_FATAL(
    //     !total_core_range_set.empty(),
    //     "the total core range set calculated by the input shape {} is empty, cannot proceed further with program
    //     creation", input_shape
    // );
    // const CoreCoord core_mesh_size = total_core_range_set.bounding_box().end_coord;
    // const auto per_core_set_work_split = split_intimg_work_to_cores(input_shape, core_mesh_size);

    create_cb(program, input_tensor.dtype(), IntImgCB::START, total_core_range_set, TILES_PER_CB);
    create_cb(program, input_tensor.dtype(), IntImgCB::INPUT, total_core_range_set, TILES_PER_CB);
    create_cb(program, input_tensor.dtype(), IntImgCB::ACC, total_core_range_set, TILES_PER_CB);
    create_cb(
        program, input_tensor.dtype(), IntImgCB::BEFORE_ADDER_PROPAGATION_STAGE, total_core_range_set, TILES_PER_CB);
    create_cb(program, input_tensor.dtype(), IntImgCB::OUTPUT, total_core_range_set, TILES_PER_CB);
    create_cb(program, input_tensor.dtype(), IntImgCB::TO_BOT_STAGE_TILE, total_core_range_set, TILES_PER_CB);
    create_cb(
        program,
        input_tensor.dtype(),
        IntImgCB::FROM_TOP_STAGE_TILE,
        total_core_range_set,
        TILES_PER_CB);  // TODO(jbbieniekTT): obligatory?
    create_cb(program, input_tensor.dtype(), IntImgCB::AXIS_3_BUFFER_0, total_core_range_set, TILES_PER_CB);
    create_cb(program, input_tensor.dtype(), IntImgCB::AXIS_3_BUFFER_1, total_core_range_set, TILES_PER_CB);

    const uint32_t tile_width = tile_spec.get_width();
    const uint32_t tile_height = tile_spec.get_height();

    const uint32_t num_tiles_along_channels = (input_shape[3] + tile_width - 1) / tile_width;
    const uint32_t num_tiles_along_height = (input_shape[2] + tile_height - 1) / tile_height;

    const uint32_t top_semaphore_id = CreateSemaphore(program, total_core_range_set, SEMAPHORES_STARTING_VALUE);
    const uint32_t bot_semaphore_id = CreateSemaphore(program, total_core_range_set, SEMAPHORES_STARTING_VALUE);

    const std::vector<uint32_t> compute_compile_time_args{
        static_cast<uint32_t>(IntImgCB::START),
        static_cast<uint32_t>(IntImgCB::INPUT),
        static_cast<uint32_t>(IntImgCB::ACC),
        static_cast<uint32_t>(IntImgCB::BEFORE_ADDER_PROPAGATION_STAGE),
        static_cast<uint32_t>(IntImgCB::OUTPUT),
        static_cast<uint32_t>(IntImgCB::TO_BOT_STAGE_TILE),
        static_cast<uint32_t>(IntImgCB::FROM_TOP_STAGE_TILE),
        static_cast<uint32_t>(IntImgCB::AXIS_3_BUFFER_0),
        static_cast<uint32_t>(IntImgCB::AXIS_3_BUFFER_1),
        tile_height,
        tile_width,
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
        num_tiles_along_channels,
        num_tiles_along_height,
        top_semaphore_id,  // for communication with upper cores
        bot_semaphore_id   // for communication with lower cores
    };
    std::vector<uint32_t> dataflow_compile_time_args{compute_compile_time_args};
    tt::tt_metal::TensorAccessorArgs(input_buffer).append_to(dataflow_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(zero_tile_buffer).append_to(dataflow_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(dataflow_compile_time_args);

    const ReaderDataMovementConfig reader_config{dataflow_compile_time_args};
    const ComputeConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .math_approx_mode = false,
        .compile_args = compute_compile_time_args,
        .defines = {}};
    const WriterDataMovementConfig writer_config{dataflow_compile_time_args};

    auto reader_kernel_id{create_kernel(program, KERNEL_PATHS[0], total_core_range_set, reader_config)};
    auto compute_kernel_id{create_kernel(program, KERNEL_PATHS[1], total_core_range_set, compute_config)};
    auto writer_kernel_id{create_kernel(program, KERNEL_PATHS[2], total_core_range_set, writer_config)};

    const CoreCoord core_mesh_size = total_core_range_set.bounding_box().end_coord;
    const auto per_core_set_work_split = split_intimg_work_to_cores(input_shape, core_mesh_size);

    set_runtime_args(
        program,
        reader_kernel_id,
        compute_kernel_id,
        writer_kernel_id,
        per_core_set_work_split,
        input_buffer->address(),
        zero_tile_buffer->address(),
        output_buffer->address());

    return {
        std::move(program),
        {.reader_kernel_id = reader_kernel_id,
         .compute_kernel_id = compute_kernel_id,
         .writer_kernel_id = writer_kernel_id}};
}

void IntImgProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    const auto& compute_kernel_id = cached_program.shared_variables.compute_kernel_id;
    const auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;

    auto input_buffer_address = tensor_args.input_tensor.buffer()->address();
    auto zero_tile_buffer_address = tensor_args.zero_tile_tensor.buffer()->address();
    auto output_buffer_address = tensor_return_value.buffer()->address();
    const auto& input_shape = tensor_args.input_tensor.padded_shape();

    const auto total_core_range_set =
        generate_max_core_range_set(input_shape);  // expected to be no more than 4 rows and 5 columns!
    TT_FATAL(
        !total_core_range_set.empty(),
        "the total core range set calculated by the input shape {} is empty, cannot proceed further with program "
        "creation",
        input_shape);
    const CoreCoord core_mesh_size = total_core_range_set.bounding_box().end_coord;
    const auto per_core_set_work_split = intimg::common::split_intimg_work_to_cores(input_shape, core_mesh_size);

    std::cout << "CCCCCCCCC" << std::endl;
    set_runtime_args(
        program,
        reader_kernel_id,
        compute_kernel_id,
        writer_kernel_id,
        per_core_set_work_split,
        input_buffer_address,
        zero_tile_buffer_address,
        output_buffer_address);
}

CBHandle IntImgProgramFactory::create_cb(
    Program& program,
    const DataType& dtype,
    const IntImgCB& intimg_cb,
    const CoreRangeSet& core_range_set,
    const uint32_t& num_tiles) {
    const uint32_t cb_id{static_cast<uint32_t>(intimg_cb)};
    const auto cb_data_format{datatype_to_dataformat_converter(dtype)};
    const uint32_t single_tile_size{tt::tile_size(cb_data_format)};
    const auto cb_config{CircularBufferConfig{num_tiles * single_tile_size, {{cb_id, cb_data_format}}}.set_page_size(
        cb_id, single_tile_size)};
    return CreateCircularBuffer(program, core_range_set, cb_config);
}

KernelHandle IntImgProgramFactory::create_kernel(
    Program& program,
    const char* kernel_path,
    const CoreRangeSet& core_range_set,
    const std::variant<DataMovementConfig, ComputeConfig, EthernetConfig>& config,
    const std::vector<uint32_t>& runtime_args) {
    auto kernel_id{CreateKernel(program, kernel_path, core_range_set, config)};

    SetRuntimeArgs(program, kernel_id, core_range_set, runtime_args);

    return kernel_id;
}

void IntImgProgramFactory::set_runtime_args(
    Program& program,
    KernelHandle reader_kernel_id,
    KernelHandle compute_kernel_id,
    KernelHandle writer_kernel_id,
    const IntImgPerCoreSetWorkSplit& per_core_set_work_split,
    uint32_t input_buffer_address,
    uint32_t zero_tile_buffer_address,
    uint32_t output_buffer_address) {
    for (const auto& [core_set_name, core_set_work_def] : make_intimg_work_map(per_core_set_work_split)) {
        bool core_set_engaged = core_set_work_def.first;
        if (core_set_engaged) {
            const auto& engaged_core_set = core_set_work_def.second.first;
            const auto& per_core_set_work_def = core_set_work_def.second.second;
            const auto bounding_box = engaged_core_set.bounding_box();
            const auto start_x = bounding_box.start_coord.x;
            const auto end_x = bounding_box.end_coord.x;
            const auto start_y = bounding_box.start_coord.y;
            const auto end_y = bounding_box.end_coord.y;
            for (uint32_t x = start_x; x < end_x; ++x) {
                for (uint32_t y = start_y; y < end_y; ++y) {
                    const CoreCoord core_coord{x, y};

                    const uint32_t starting_row_chunk_for_core_set_along_channels =
                        per_core_set_work_def.starting_row_chunk_per_core_set_along_channels +
                        (x - start_x) * per_core_set_work_def.row_chunks_per_core_along_channels;
                    const uint32_t starting_row_chunk_for_core_set_along_height =
                        per_core_set_work_def.starting_row_chunk_per_core_set_along_height +
                        (y - start_y) * per_core_set_work_def.row_chunks_per_core_along_channels;

                    std::vector<uint32_t> runtime_args{
                        input_buffer_address,
                        zero_tile_buffer_address,
                        output_buffer_address,
                        starting_row_chunk_for_core_set_along_channels,
                        per_core_set_work_def.row_chunks_per_core_along_channels,
                        starting_row_chunk_for_core_set_along_height,
                        per_core_set_work_def.row_chunks_per_core_along_height};

                    SetRuntimeArgs(program, reader_kernel_id, core_coord, runtime_args);
                    SetRuntimeArgs(program, compute_kernel_id, core_coord, runtime_args);
                    SetRuntimeArgs(program, writer_kernel_id, core_coord, runtime_args);
                }
            }
        }
    }
}

}  // namespace ttnn::operations::experimental::reduction
