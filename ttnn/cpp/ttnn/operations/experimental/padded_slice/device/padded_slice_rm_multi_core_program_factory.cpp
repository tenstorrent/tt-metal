// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "hostdevcommon/kernel_structs.h"
#include "optional"
#include "tt-metalium/assert.hpp"
#include <tt-logger/tt-logger.hpp>
#include "tt-metalium/math.hpp"
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/data_movement/slice/device/slice_op.hpp"
#include "ttnn/operations/math.hpp"
#include <algorithm>
#include <cstdint>
#include <ranges>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <vector>

#include "padded_slice_rm_multi_core_program_factory.hpp"
#include "padded_slice_op.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::experimental::detail {

extern uint32_t get_num_cores_channels_from_sharded_tensor(const Tensor& tensor);

tt::tt_metal::operation::ProgramWithCallbacks padded_slice_rm_multi_core(
    const Tensor& a, Tensor& output, const ttnn::Shape& output_tensor_start, const ttnn::Shape& output_tensor_end) {
    const ttnn::Shape output_shape = output.logical_shape();
    ttnn::Shape actual_output_shape = output_tensor_end;
    for (int i = 0; i < output_shape.rank(); i++) {
        actual_output_shape[i] = output_tensor_end[i] - output_tensor_start[i];
    }

    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();

    // This should allocate a DRAM buffer on the device
    tt::tt_metal::IDevice* device = a.device();

    uint32_t num_unpadded_sticks = output.physical_volume() / output.padded_shape()[-1];

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    tt::tt_metal::Buffer* src0_buffer = a.buffer();

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    TT_FATAL(output.is_sharded(), "Output Tensor must be sharded.");
    auto output_shard_spec = output.shard_spec().value();

    bool is_block_sharded = output.memory_config().memory_layout() == TensorMemoryLayout::BLOCK_SHARDED;

    uint32_t output_row_size_bytes = output_shard_spec.shape[1] * output.element_size();

    CoreRangeSet total_cores = output.shard_spec().value().grid;
    bool rm_orientation = output_shard_spec.orientation == ShardOrientation::ROW_MAJOR;

    std::vector<CoreCoord> iter_cores = corerange_to_cores(total_cores, std::nullopt, rm_orientation);
    uint32_t num_cores_total = total_cores.num_cores();

    uint32_t num_cores_channels = get_num_cores_channels_from_sharded_tensor(output);

    bool pad_output_row = false;

    TT_FATAL(
        a.logical_shape()[3] % num_cores_channels == 0,
        "Input tensor should be divisible by number of cores in channel dimension");
    uint32_t input_row_size_bytes = a.logical_shape()[-1] * a.element_size();
    input_row_size_bytes = input_row_size_bytes / num_cores_channels;

    tt::tt_metal::Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    bool src0_is_dram = src0_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    TT_FATAL(
        dst_buffer->buffer_type() == tt::tt_metal::BufferType::L1,
        "Output buffer should be L1 for padded_slice operation with tiled inputs");

    uint32_t src_stick_size = input_row_size_bytes;
    uint32_t dst_stick_size = output_row_size_bytes;

    uint32_t src0_cb_index = 0;
    uint32_t temp_pad_cb_index = 1;
    uint32_t max_read_size = 4096;

    auto src_buffer_alignment = a.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto dst_buffer_alignment = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM
                                    ? ::hal::get_dram_alignment()
                                    : ::hal::get_l1_alignment();
    auto alignment = std::max(src_buffer_alignment, dst_buffer_alignment);

    // if begins is not aligned then we need to pad the cb size, so that we can read from the nearest aligned address
    uint32_t begins_bytes = output_tensor_start[-1] * a.element_size();
    uint32_t misalignment = begins_bytes % src_buffer_alignment;

    if (misalignment != 0) {
        alignment *= 2;
    }
    uint32_t cb_page_size = tt::round_up(output_row_size_bytes, alignment);

    CBHandle cb_src0;
    uint32_t num_output_sticks_per_core = output_shard_spec.shape[0];
    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(
            num_output_sticks_per_core * output_row_size_bytes, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, output_row_size_bytes)
            .set_globally_allocated_address(*output.buffer());
    cb_src0 = tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_src0_config);
    if (output_row_size_bytes > input_row_size_bytes) {
        pad_output_row = true;
        tt::tt_metal::CircularBufferConfig cb_temp_pad_config =
            tt::tt_metal::CircularBufferConfig(1 * output_row_size_bytes, {{temp_pad_cb_index, cb_data_format}})
                .set_page_size(temp_pad_cb_index, output_row_size_bytes);
        tt::tt_metal::CreateCircularBuffer(program, total_cores, cb_temp_pad_config);
    }

    std::vector<uint32_t> writer_compile_time_args_vec = {(std::uint32_t)src0_cb_index};

    std::vector<uint32_t> reader_compile_time_args_vec = {(std::uint32_t)src0_is_dram, misalignment};
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/padded_slice/device/kernels/dataflow/"
        "padded_slice_reader_rm_interleaved_start_id.cpp",
        total_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args_vec));

    tt::tt_metal::KernelHandle unary_writer_kernel_id;
    if (pad_output_row) {
        writer_compile_time_args_vec.push_back((std::uint32_t)temp_pad_cb_index);
        unary_writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/experimental/padded_slice/device/kernels/dataflow/"
            "writer_unary_sharded_padded_rm.cpp",
            total_cores,
            tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args_vec));
    } else {
        unary_writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/data_movement/sharded/device/kernels/dataflow/writer_unary_sharded.cpp",
            total_cores,
            tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args_vec));
    }

    auto all_runtime_args = get_padded_slice_runtime_args_rm_sharded_output(
        a, output, output_tensor_start, actual_output_shape, iter_cores, max_read_size);

    uint32_t i = 0;
    for (const auto& core : iter_cores) {
        tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);
        tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second);
        i++;
    }

    auto override_runtime_args_callback = [unary_reader_kernel_id,
                                           unary_writer_kernel_id,
                                           output_tensor_start,
                                           actual_output_shape,
                                           compute_with_storage_grid_size,
                                           max_read_size,
                                           iter_cores,
                                           cb_src0](
                                              const void* operation,
                                              Program& program,
                                              const std::vector<Tensor>& input_tensors,
                                              const std::vector<std::optional<const Tensor>>&,
                                              const std::vector<Tensor>& output_tensors) {
        const auto& src_tensor = input_tensors.at(0);
        auto dst_tensor = output_tensors.at(0);
        TT_FATAL(dst_tensor.is_sharded(), "Output tensor must be sharded");
        UpdateDynamicCircularBufferAddress(program, cb_src0, *dst_tensor.buffer());

        auto all_runtime_args = get_padded_slice_runtime_args_rm_sharded_output(
            src_tensor, dst_tensor, output_tensor_start, actual_output_shape, iter_cores, max_read_size);

        uint32_t i = 0;
        for (const auto& core : iter_cores) {
            tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, all_runtime_args[i].first);
            tt::tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, core, all_runtime_args[i].second);
            i++;
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_args_callback};
}

}  // namespace ttnn::operations::experimental::detail
