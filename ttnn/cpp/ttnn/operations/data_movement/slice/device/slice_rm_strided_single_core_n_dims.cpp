// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "slice_rm_strided_single_core_n_dims.hpp"
#include "optional"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::data_movement::detail {

operation::ProgramWithCallbacks slice_rm_strided_single_core_n_dims(
    const Tensor& a,
    Tensor& output,
    const ttnn::Shape& output_tensor_start,
    const ttnn::Shape& output_tensor_end,
    const ttnn::Shape& step) {
    // TODO: multi core implementation - work division is not trivial as we need to determine the N/C/H/W start and end
    // points for each split, and base that off stride
    tt::tt_metal::Program program = tt::tt_metal::CreateProgram();
    const auto& output_shape = output.padded_shape();
    const auto& input_shape = a.padded_shape();
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t src_is_dram = a.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;
    uint32_t dst_is_dram = output.buffer()->buffer_type() == tt::tt_metal::BufferType::DRAM ? 1 : 0;

    uint32_t page_size_output = dst_is_dram ? tt::round_up(output_shape[-1] * a.element_size(), TILE_WIDTH)
                                            : tt::round_up(output_shape[-1] * a.element_size(), TILE_WIDTH / 2);
    uint32_t page_size_input = src_is_dram ? tt::round_up(input_shape[-1] * a.element_size(), TILE_WIDTH)
                                           : tt::round_up(input_shape[-1] * a.element_size(), TILE_WIDTH / 2);

    tt::tt_metal::CircularBufferConfig cb_src0_config =
        tt::tt_metal::CircularBufferConfig(1 * page_size_input, {{tt::CBIndex::c_0, cb_data_format}})
            .set_page_size(tt::CBIndex::c_0, page_size_input);

    tt::tt_metal::CircularBufferConfig cb_dst0_config =
        tt::tt_metal::CircularBufferConfig(2 * page_size_output, {{tt::CBIndex::c_24, cb_data_format}})
            .set_page_size(tt::CBIndex::c_24, page_size_output);

    CoreRange core({0, 0}, {0, 0});
    auto cb_input_tensor = tt::tt_metal::CreateCircularBuffer(program, core, cb_src0_config);
    auto cb_output_tensor = tt::tt_metal::CreateCircularBuffer(program, core, cb_dst0_config);

    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "strided_slice_reader_rm_interleaved_nd.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig({
            src_is_dram,
            (uint32_t)page_size_input,
            (uint32_t)input_shape.rank(),
        }

                                               ));

    tt::tt_metal::KernelHandle unary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/strided_slice_writer_rm_interleaved.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig({
            dst_is_dram,
            (uint32_t)page_size_output,
        }));

    std::vector<uint32_t> reader_runtime_args;
    reader_runtime_args.reserve(1 + (4 * input_shape.rank()));
    reader_runtime_args.push_back(a.buffer()->address());

    reader_runtime_args.insert(reader_runtime_args.end(), input_shape.cbegin(), input_shape.cend());
    reader_runtime_args.insert(reader_runtime_args.end(), output_tensor_start.cbegin(), output_tensor_start.cend());
    reader_runtime_args.insert(reader_runtime_args.end(), output_tensor_end.cbegin(), output_tensor_end.cend());
    reader_runtime_args.insert(reader_runtime_args.end(), step.cbegin(), step.cend());

    tt::tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, core, reader_runtime_args);

    uint32_t pages = output.physical_volume() / output_shape[-1];
    tt::tt_metal::SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        {
            output.buffer()->address(),
            pages,
        });

    auto override_runtime_arguments_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>& optional_input_tensors,
            const std::vector<Tensor>& output_tensors) {
            auto input_buffer = input_tensors.at(0).buffer();
            auto output_buffer = output_tensors.at(0).buffer();

        CoreCoord core = {0, 0};

        {
            auto& reader_runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            reader_runtime_args[0] = input_buffer->address();

            auto& writer_runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            writer_runtime_args[0] = output_buffer->address();
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace ttnn::operations::data_movement::detail