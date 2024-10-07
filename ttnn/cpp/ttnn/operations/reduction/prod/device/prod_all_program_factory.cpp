// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/prod/device/prod_op_all.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"


namespace tt {
using namespace constants;
namespace operations {
namespace primary {

    operation::ProgramWithCallbacks prod_single_core(const Tensor &a, const Tensor& output)
    {

        auto program = tt::tt_metal::CreateProgram();

        CoreRange core({0, 0}, {0, 0});

        tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.get_dtype());
        uint32_t single_tile_size = tt_metal::detail::TileSize(cb_data_format);

        uint32_t num_tiles = a.volume() / TILE_HW;

        // This should allocate a DRAM buffer on the device
        tt_metal::Device *device = a.device();

        uint32_t src0_cb_index = 0;
        uint32_t num_input_tiles = 2;
        tt_metal::CircularBufferConfig cb_src0_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{src0_cb_index, cb_data_format}})
            .set_page_size(src0_cb_index, single_tile_size);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        tt_metal::CircularBufferConfig cb_inter_config = tt_metal::CircularBufferConfig(num_input_tiles * single_tile_size, {{tt::CB::c_intermed0, cb_data_format}})
            .set_page_size(tt::CB::c_intermed0, single_tile_size);
        auto cb_interm = tt_metal::CreateCircularBuffer(program, core, cb_inter_config);

        uint32_t output_cb_index = 16; // output operands start at index 16
        uint32_t num_output_tiles = 2;
        tt_metal::CircularBufferConfig cb_output_config = tt_metal::CircularBufferConfig(num_output_tiles * single_tile_size, {{output_cb_index, cb_data_format}})
            .set_page_size(output_cb_index, single_tile_size);
        auto cb_output = tt_metal::CreateCircularBuffer(program, core, cb_output_config);

        auto src_buffer = a.buffer();
        auto dst_buffer = output.buffer();

        bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
        std::vector<uint32_t> reader_compile_time_args = {(uint32_t)src_is_dram};
        bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
        std::vector<uint32_t> writer_compile_time_args = {
            (std::uint32_t) output_cb_index,
            (std::uint32_t) dst_is_dram
        };

        tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
            core,
            tt_metal::ReaderDataMovementConfig{reader_compile_time_args});

        tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
            core,
            tt_metal::WriterDataMovementConfig{writer_compile_time_args});

        vector<uint32_t> compute_kernel_args = {
            num_tiles, // per_core_block_cnt
            1 // per_core_block_size
        };

        bool fp32_dest_acc_en = false;
        bool math_approx_mode = true;
        auto eltwise_unary_kernel_id = tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/prod/device/kernels/compute/prod_all.cpp",
            core,
            tt_metal::ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = fp32_dest_acc_en,
                .math_approx_mode = math_approx_mode,
                .compile_args = compute_kernel_args
            }
    );

    SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {
            src_buffer->address(),
            num_tiles, 0
        }
    );

    SetRuntimeArgs(
        program,
        unary_writer_kernel_id,
        core,
        {
            dst_buffer->address(),
            num_tiles, 0
        }
    );

    auto override_runtime_args_callback = [unary_reader_kernel_id, unary_writer_kernel_id](
        const ProgramHandle program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto src_buffer = input_buffers.at(0);

        auto dst_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }

        {
            auto &runtime_args = GetRuntimeArgs(program, unary_writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    };

    return {program, override_runtime_args_callback};
}

}  // namespace primary
}  // namespace operations
}  // namespace tt
