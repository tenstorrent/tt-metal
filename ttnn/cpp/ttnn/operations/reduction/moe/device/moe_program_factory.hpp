// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_log.h"

namespace ttnn::operations::reduction::detail {

operation::ProgramWithCallbacks moe_single_core_interleaved(const Tensor &input_tensor, const Tensor &topk_mask_tensor, const Tensor &expert_mask_tensor, const uint16_t k, Tensor &out_tensor) {
    tt::tt_metal::Program program{};
    CoreRange core({0, 0}, {0, 0});

    bool fp32_dest_acc_en = false;
    tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.get_dtype());
    tt::DataFormat topk_mask_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(topk_mask_tensor.get_dtype());
    tt::DataFormat expert_mask_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(expert_mask_tensor.get_dtype());
    tt::DataFormat out_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(out_tensor.get_dtype());
    tt::DataFormat scalar_df = tt::DataFormat::Float16_b;
    tt::DataFormat im_df = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;

    uint32_t input_tile_size = tile_size(input_cb_data_format);
    uint32_t topk_mask_tile_size = tile_size(topk_mask_cb_data_format);
    uint32_t expert_mask_tile_size = tile_size(expert_mask_cb_data_format);
    uint32_t out_tile_size = tile_size(out_cb_data_format);
    uint32_t scalar_tile_size = tt_metal::detail::TileSize(scalar_df);
    uint32_t index_tile_size = tt_metal::detail::TileSize(tt::DataFormat::UInt16);
    uint32_t value_tile_size = tt_metal::detail::TileSize(tt::DataFormat::Float16_b);

    auto input_buffer = input_tensor.buffer();
    auto topk_mask_buffer = topk_mask_tensor.buffer();
    auto expert_mask_buffer = expert_mask_tensor.buffer();
    auto out_buffer = out_tensor.buffer();

    bool input_is_dram = input_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool topk_mask_is_dram = topk_mask_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool expert_mask_is_dram = expert_mask_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool out_is_dram = out_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    uint32_t num_input_tiles = input_tensor.volume()/TILE_HW;
    uint32_t num_out_tiles = out_tensor.volume()/TILE_HW;
    uint32_t scale_tiles = 1;

    auto input_shape = input_tensor.get_legacy_shape();
    uint32_t Ht = (input_shape[0]*input_shape[1]*input_shape[2])/TILE_HEIGHT;
    uint32_t Wt = input_shape[3]/TILE_WIDTH;
    // for streaming in input
    uint32_t num_cb_unit = 2;
    uint32_t cb_in_units = 2 * num_cb_unit;

    // INPUT CBs
    // Two tiles are loaded in for topk_local_sort at a time, and we double buffer to avoid stalls, so allocate four tiles of space
    uint32_t input_cb_index = tt::CB::c_in0;
    tt::tt_metal::CircularBufferConfig input_cb_config = tt::tt_metal::CircularBufferConfig(
        cb_in_units * input_tile_size, {{input_cb_index, input_cb_data_format}})
		.set_page_size(input_cb_index, input_tile_size);
    auto cb_input_tensor = tt::tt_metal::CreateCircularBuffer(program, core, input_cb_config);

    uint32_t topk_mask_cb_index = tt::CB::c_in1;
    tt::tt_metal::CircularBufferConfig topk_mask_cb_config = tt::tt_metal::CircularBufferConfig(
        cb_in_units * topk_mask_tile_size, {{topk_mask_cb_index, topk_mask_cb_data_format}})
        .set_page_size(topk_mask_cb_index, topk_mask_tile_size);
    auto cb_topk_mask_tensor = tt::tt_metal::CreateCircularBuffer(program, core, topk_mask_cb_config);

    uint32_t expert_mask_cb_index = tt::CB::c_in2;
    tt::tt_metal::CircularBufferConfig expert_mask_cb_config = tt::tt_metal::CircularBufferConfig(
        cb_in_units * expert_mask_tile_size, {{expert_mask_cb_index, expert_mask_cb_data_format}})
        .set_page_size(expert_mask_cb_index, expert_mask_tile_size);
    auto cb_expert_mask_tensor = tt::tt_metal::CreateCircularBuffer(program, core, expert_mask_cb_config);

    // identity scale input
    uint32_t scale_cb_index = tt::CB::c_in3;
    tt::tt_metal::CircularBufferConfig scale_cb_config = tt::tt_metal::CircularBufferConfig(scale_tiles * scalar_tile_size, {{scale_cb_index, scalar_df}}).set_page_size(scale_cb_index, scalar_tile_size);
    auto scale_cb_tensor = tt::tt_metal::CreateCircularBuffer(program, core, scale_cb_config);

    // TOP K CBs
    // Two tiles are loaded in for topk_local_sort at a time, and we double buffer to avoid stalls, so allocate four tiles of space
    // This CB carries the indices that are created in the reader kernel
    uint32_t index_cb_index = tt::CB::c_intermed0_0;
    tt::tt_metal::CircularBufferConfig index_input_intermed0_config = tt::tt_metal::CircularBufferConfig(
        cb_in_units * index_tile_size, {{index_cb_index, index_cb_data_format}})
		.set_page_size(index_cb_index, index_tile_size);
    auto cb_index_tensor = tt::tt_metal::CreateCircularBuffer(program, core, index_input_intermed0_config);

    // Single buffered circular buffer that holds the transposed input tiles
    uint32_t input_transposed_cb_index = tt::CB::c_intermed0_1;
    tt::tt_metal::CircularBufferConfig input_transposed_cb_config = tt::tt_metal::CircularBufferConfig(
         Wt * value_tile_size, {{input_transposed_cb_index, input_cb_data_format}})
		.set_page_size(input_transposed_cb_index, input_tile_size);
    auto cb_input_transposed_tiles = tt::tt_metal::CreateCircularBuffer(program, core, input_transposed_cb_config);

    // Single buffered circular buffer that holds the transposed index tiles
    uint32_t index_transposed_cb_index = tt::CB::c_intermed0_2;
    tt::tt_metal::CircularBufferConfig index_transposed_cb_config = tt::tt_metal::CircularBufferConfig(
         Wt * index_tile_size, {{index_transposed_cb_index, index_cb_data_format}})
		.set_page_size(index_transposed_cb_index, index_tile_size);
    auto cb_index_transposed_tiles = tt::tt_metal::CreateCircularBuffer(program, core, index_transposed_cb_config);

    // topk values
    uint32_t values_cb_index = tt::CB::c_intermed1_0;
    tt::tt_metal::CircularBufferConfig values_cb_config = tt::tt_metal::CircularBufferConfig(
        num_cb_unit * value_tile_size, {{values_cb_index, value_cb_data_format}})
        .set_page_size(values_cb_index, value_tile_size);
    auto cb_values_tensor = tt::tt_metal::CreateCircularBuffer(program, core, values_cb_config);

    // topk indices
    uint32_t output_ind_cb_index = tt::CB::c_intermed1_1;
    tt::tt_metal::CircularBufferConfig output_ind_cb_config = tt::tt_metal::CircularBufferConfig(
        num_cb_unit * index_tile_size, {{output_ind_cb_index, index_cb_data_format}})
        .set_page_size(output_ind_cb_index, index_tile_size);
    auto cb_output_ind_tensor = tt::tt_metal::CreateCircularBuffer(program, core, output_ind_cb_config);

    // OUTPUT CBs
    uint32_t out_cb_index = tt::CB::c_out0;
    auto c_out0_config = CircularBufferConfig(out0_t * out_tile_size, {{out_cb_index, out_cb_data_format}}).set_page_size(out_cb_index, out_tile_size);
    auto cb_out0_id = CreateCircularBuffer( program, core, c_out0_config );

    std::vector<uint32_t> reader_compile_time_args = {
                                                        input_cb_index,
                                                        index_cb_index,
                                                        topk_mask_cb_index,
                                                        expert_mask_cb_index,
                                                        (uint32_t)input_is_dram,
                                                        (uint32_t)topk_mask_is_dram,
                                                        (uint32_t)expert_mask_is_dram,
                                                        Ht,
                                                        Wt};
    tt::tt_metal::KernelHandle unary_reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/reader_create_index_tensor.cpp",
        core,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));


    SetRuntimeArgs(
        program,
        unary_reader_kernel_id,
        core,
        {
            input_buffer->address(),
            topk_mask_buffer->address(),
            expert_mask_buffer->address(),
        }
    );

    std::vector<uint32_t> writer_compile_time_args = {out_cb_index
                                                        (std::uint32_t) out_is_dram,
                                                        Ht,
                                                        k};
    tt::tt_metal::KernelHandle binary_writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/dataflow/writer_binary_interleaved.cpp",
        core,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    SetRuntimeArgs(
        program,
        binary_writer_kernel_id,
        core,
        {
            out_buffer->address(),

        }
    );

    std::vector<uint32_t> compute_args = {
                                        input_cb_index,
                                        topk_mask_cb_index,
                                        expert_mask_cb_index,
                                        scale_cb_index,
                                        index_cb_index,
                                        input_transposed_cb_index,
                                        index_transposed_cb_index,
                                        values_cb_index,
                                        output_ind_cb_index,
                                        out_cb_index,
                                        Ht,
                                        Wt,
                                        k,
                                        (std::uint32_t) std::log2(k),
                                        (std::uint32_t) std::log2(Wt),
                                        };

    tt::tt_metal::KernelHandle moe_compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/moe/device/kernels/compute/moe.cpp",
        core,
        tt::tt_metal::ComputeConfig{.compile_args = compute_args}
    );


    auto override_runtime_args_callback = [unary_reader_kernel_id, binary_writer_kernel_id](
        const Program &program,
        const std::vector<Buffer*>& input_buffers,
        const std::vector<Buffer*>& output_buffers
    ) {

        auto input_buffer = input_buffers.at(0);
        auto output_buffer = output_buffers.at(0);

        CoreCoord core = {0, 0};

        {
            auto &reader_runtime_args = GetRuntimeArgs(program, unary_reader_kernel_id, core);
            reader_runtime_args[0] = input_buffer->address();

            auto &writer_runtime_args = GetRuntimeArgs(program, binary_writer_kernel_id, core);
            writer_runtime_args[0] = output_buffer->address();
        }

    };

    return {std::move(program), override_runtime_args_callback};
}
} // namespace ttnn::operations::reduction::detail
