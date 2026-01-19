// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "fill_pad_program_factory.hpp"
#include "ttnn/operations/core/core.hpp"
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/ccl/sharding_addrgen_helper.hpp"

namespace ttnn::prim {

using namespace ttnn::operations::data_movement;

FillPadProgramFactory::cached_program_t FillPadProgramFactory::create(
    const FillPadParams& operation_attributes, const FillPadInputs& tensor_args, Tensor& /*tensor_return_value*/) {
    const Tensor& input_tensor = tensor_args.input;
    const float fill_value = operation_attributes.fill_value;
    tt::tt_metal::IDevice* device = input_tensor.device();
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    const tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());

    tt::tt_metal::Buffer* tens_buffer = input_tensor.buffer();
    TT_ASSERT(tens_buffer != nullptr, "Input buffer should be allocated on device!");

    const uint32_t input_element_size_bytes = detail::data_type_to_size.at(input_tensor.dtype());
    const uint32_t cb_page_size = (input_element_size_bytes * tt::constants::FACE_HEIGHT) + sizeof(uint16_t);
    const uint32_t height = input_tensor.logical_shape()[-2];
    const uint32_t width = input_tensor.logical_shape()[-1];

    const uint32_t problem_size = input_tensor.logical_shape()[-3];

    const auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    const uint32_t num_cores_x = compute_with_storage_grid_size.x;
    const uint32_t num_cores_y = compute_with_storage_grid_size.y;

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, problem_size);
    const uint32_t g1_numcores = core_group_1.num_cores();

    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    const tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(cb_page_size * 2, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, cb_page_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores, cb_src0_config);

    const bool src_is_dram = tens_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    // pack bf16 vals
    uint32_t packed_fill_value = static_cast<std::uint32_t>(fill_value);
    if (input_tensor.dtype() == DataType::BFLOAT16) {
        packed_fill_value = pack_two_bfloat16_into_uint32({bfloat16(fill_value), bfloat16(fill_value)});
    } else if (input_tensor.dtype() == DataType::UINT16) {
        packed_fill_value = pack_two_uint16_into_uint32({fill_value, fill_value});
    } else if (input_tensor.dtype() == DataType::FLOAT32) {
        packed_fill_value = std::bit_cast<uint32_t>(fill_value);
    }

    const uint32_t padded_height = tt::div_up(height, tt::constants::TILE_HEIGHT) * tt::constants::TILE_HEIGHT;
    const uint32_t padded_width = tt::div_up(width, tt::constants::TILE_HEIGHT) * tt::constants::TILE_HEIGHT;
    const uint32_t tiles_per_2d_tensor =
        padded_height / tt::constants::TILE_HEIGHT * padded_width / tt::constants::TILE_HEIGHT;
    const uint32_t tiles_per_tile_row = padded_width / tt::constants::TILE_HEIGHT;

    const bool sharded = input_tensor.memory_config().memory_layout() != TensorMemoryLayout::INTERLEAVED;

    std::vector<uint32_t> writer_compile_time_args = {
        static_cast<std::uint32_t>(src0_cb_index),
        static_cast<std::uint32_t>(src_is_dram),
        static_cast<std::uint32_t>(packed_fill_value),
        static_cast<std::uint32_t>(input_element_size_bytes),
        static_cast<std::uint32_t>(height),
        static_cast<std::uint32_t>(width),
        static_cast<std::uint32_t>(padded_height),
        static_cast<std::uint32_t>(padded_width),
        static_cast<std::uint32_t>(tiles_per_2d_tensor),
        static_cast<std::uint32_t>(tiles_per_tile_row),
        static_cast<std::uint32_t>(tt::constants::TILE_HEIGHT),
        static_cast<std::uint32_t>(tt::constants::FACE_HEIGHT)};

    std::map<std::string, std::string> compute_defines;
    if (sharded) {
        shard_builder::extend_sharding_compile_time_args(input_tensor, writer_compile_time_args);
        compute_defines["SHARDED"] = "1";
    } else {
        tt::tt_metal::TensorAccessorArgs(*tens_buffer).append_to(writer_compile_time_args);
    }

    const tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/fill_pad/device/kernels/dataflow/fill_pad_writer.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(
            writer_compile_time_args, compute_defines));  // writer only for in-place operation

    const std::vector<CoreCoord> cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, false);
    std::vector<uint32_t> writer_runtime_args = {
        static_cast<std::uint32_t>(tens_buffer->address()),
        static_cast<std::uint32_t>(cb_page_size),
        static_cast<std::uint32_t>(0),
        static_cast<std::uint32_t>(0)};

    uint32_t tile_offset = 0;
    for (uint32_t i = 0; i < cores.size(); ++i) {
        const CoreCoord& core = cores[i];
        const uint32_t local_num_2d_tensors =
            i < g1_numcores ? num_blocks_per_core_group_1 : num_blocks_per_core_group_2;
        {
            writer_runtime_args[2] = tile_offset;
            writer_runtime_args[3] = local_num_2d_tensors;
            if (sharded) {
                shard_builder::extend_sharding_run_time_args(input_tensor, writer_runtime_args);
            }
            tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
        }

        tile_offset += local_num_2d_tensors * tiles_per_2d_tensor;
    }

    return cached_program_t{
        std::move(program), shared_variables_t{.writer_kernel_id = writer_kernel_id, .cores = cores}};
}

void FillPadProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const FillPadParams& /*operation_attributes*/,
    const FillPadInputs& tensor_args,
    Tensor& /*tensor_return_value*/) {
    const Tensor& input_tensor = tensor_args.input;
    tt::tt_metal::Buffer* tens_buffer = input_tensor.buffer();

    tt::tt_metal::Program& program = cached_program.program;
    const tt::tt_metal::KernelHandle writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;

    auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id);

    for (const auto& core : cores) {
        auto& runtime_args = writer_runtime_args[core.x][core.y];
        runtime_args[0] = tens_buffer->address();
    }
}

}  // namespace ttnn::prim
