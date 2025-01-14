// SPDX-FileCopyrightText: © 2024 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tt_metal/host_api.hpp"
#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"
#include "tt_log.h"
#include "tt_metal/common/math.hpp"
#include "ttnn/operation.hpp"
#include <algorithm>
namespace ttnn::operations::reduction::detail {

operation::ProgramWithCallbacks sampling_multicore_interleaved(
    const Tensor& input_values_tensor,
    const Tensor& input_indices_tensor,
    const std::vector<uint16_t>& k,
    const std::vector<uint16_t>& p,
    Tensor& output_tensor) {
    using namespace tt::constants;
    tt::tt_metal::Program program{};

    tt::DataFormat input_values_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(input_values_tensor.get_dtype());
    tt::DataFormat input_indices_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(input_indices_tensor.get_dtype());
    tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_tensor.get_dtype());
    tt::DataFormat index_cb_data_format = tt::DataFormat::UInt16;

    uint32_t input_values_tile_size = tile_size(input_values_cb_data_format);
    uint32_t input_indices_tile_size = tile_size(input_indices_cb_data_format);
    uint32_t index_tile_size = tile_size(index_cb_data_format);

    auto input_values_buffer = input_values_tensor.buffer();
    auto input_indices_buffer = input_indices_tensor.buffer();
    auto output_buffer = output_tensor.buffer();

    bool input_values_is_dram = input_values_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool input_indices_is_dram = input_indices_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;
    bool output_is_dram = output_buffer->buffer_type() == tt::tt_metal::BufferType::DRAM;

    uint32_t num_input_values_tiles = input_values_tensor.volume() / TILE_HW;
    uint32_t num_input_indices_tiles = input_indices_tensor.volume() / TILE_HW;
    auto device = input_values_tensor.device();

    auto input_shape = input_values_tensor.get_legacy_shape();
    uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / TILE_HEIGHT;
    uint32_t Wt = input_shape[3] / TILE_WIDTH;
    const auto& num_cores = Ht * TILE_HEIGHT;
    CoreRangeSet core_grid = CoreRangeSet(std::vector{CoreRange({0, 0}, {8 - 1, 4 - 1})});
    const auto cores = corerange_to_cores(core_grid, 32, true);

    // for streaming in input
    uint32_t num_cb_unit = 2;
    uint32_t cb_in_units = 2 * num_cb_unit;

    // Two tiles are loaded in for sampling_local_sort at a time, and we double buffer to avoid stalls, so allocate
    // four tiles of space
    // TODO: In theory if we have enough memory we could allocate 2*Wt tiles to reduce stalls
    uint32_t input_values_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig input_values_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_in_units * input_values_tile_size, {{input_values_cb_index, input_values_cb_data_format}})
            .set_page_size(input_values_cb_index, input_values_tile_size);
    auto cb_input_values_tensor = tt::tt_metal::CreateCircularBuffer(program, core_grid, input_values_cb_config);

    uint32_t input_indices_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig input_indices_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_in_units * input_indices_tile_size, {{input_indices_cb_index, input_indices_cb_data_format}})
            .set_page_size(input_indices_cb_index, input_indices_tile_size);
    auto cb_input_indices_tensor = tt::tt_metal::CreateCircularBuffer(program, core_grid, input_indices_cb_config);

    uint32_t index_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig index_input_intermed0_config =
        tt::tt_metal::CircularBufferConfig(cb_in_units * index_tile_size, {{index_cb_index, index_cb_data_format}})
            .set_page_size(index_cb_index, index_tile_size);
    auto cb_index_tensor = tt::tt_metal::CreateCircularBuffer(program, core_grid, index_input_intermed0_config);

    // identity scale input
    tt::DataFormat scalar_df = tt::DataFormat::Float16_b;
    uint32_t scale_tiles = 1;
    uint32_t scalar_tile_size = tile_size(scalar_df);
    uint32_t scale_cb_index = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig scale_cb_config =
        tt::tt_metal::CircularBufferConfig(scale_tiles * scalar_tile_size, {{scale_cb_index, scalar_df}})
            .set_page_size(scale_cb_index, scalar_tile_size);
    auto scale_cb_tensor = tt::tt_metal::CreateCircularBuffer(program, core_grid, scale_cb_config);

    uint32_t topk_mask_cb_index = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig topk_mask_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_in_units * input_values_tile_size, {{topk_mask_cb_index, input_values_cb_data_format}})
            .set_page_size(topk_mask_cb_index, input_values_tile_size);
    auto cb_topk_mask_tensor = tt::tt_metal::CreateCircularBuffer(program, core_grid, topk_mask_cb_config);

    // Compute kernel CBs
    // // Single buffered circular buffer that holds the transposed input tiles
    uint32_t input_transposed_cb_index = tt::CBIndex::c_24;
    tt::tt_metal::CircularBufferConfig input_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt * input_values_tile_size, {{input_transposed_cb_index, input_values_cb_data_format}})
            .set_page_size(input_transposed_cb_index, input_values_tile_size);
    auto cb_input_transposed_tiles = tt::tt_metal::CreateCircularBuffer(program, core_grid, input_transposed_cb_config);

    // // Single buffered circular buffer that holds the transposed index tiles
    uint32_t index_transposed_cb_index = tt::CBIndex::c_25;
    tt::tt_metal::CircularBufferConfig index_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(Wt * index_tile_size, {{index_transposed_cb_index, index_cb_data_format}})
            .set_page_size(index_transposed_cb_index, index_tile_size);
    auto cb_index_transposed_tiles = tt::tt_metal::CreateCircularBuffer(program, core_grid, index_transposed_cb_config);

    // // Output sampling values
    uint32_t values_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig values_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_cb_unit * input_values_tile_size, {{values_cb_index, input_values_cb_data_format}})
            .set_page_size(values_cb_index, input_values_tile_size);
    auto cb_values_tensor = tt::tt_metal::CreateCircularBuffer(program, core_grid, values_cb_config);

    // // Output local indices
    uint32_t output_ind_cb_index = tt::CBIndex::c_17;
    tt::tt_metal::CircularBufferConfig output_ind_cb_config =
        tt::tt_metal::CircularBufferConfig(num_cb_unit * index_tile_size, {{output_ind_cb_index, index_cb_data_format}})
            .set_page_size(output_ind_cb_index, index_tile_size);
    auto output_ind_cb_tensor = tt::tt_metal::CreateCircularBuffer(program, core_grid, output_ind_cb_config);

    uint32_t num_out_tiles =
        Ht * round_up_to_mul32(static_cast<std::uint32_t>(*std::max_element(k.begin(), k.end()))) / TILE_WIDTH;
    uint32_t cb_cur_max_index = tt::CBIndex::c_29;
    tt::tt_metal::CircularBufferConfig cb_cur_max_config =
        tt::tt_metal::CircularBufferConfig(
            num_out_tiles * input_values_tile_size, {{cb_cur_max_index, input_values_cb_data_format}})
            .set_page_size(cb_cur_max_index, input_values_tile_size);
    auto cb_cur_max_tensor = tt::tt_metal::CreateCircularBuffer(program, core_grid, cb_cur_max_config);

    uint32_t cb_cur_sum_index = tt::CBIndex::c_30;
    tt::tt_metal::CircularBufferConfig cb_cur_sum_config =
        tt::tt_metal::CircularBufferConfig(
            num_out_tiles * input_values_tile_size, {{cb_cur_sum_index, input_values_cb_data_format}})
            .set_page_size(cb_cur_sum_index, input_values_tile_size);
    auto cb_cur_sum_tensor = tt::tt_metal::CreateCircularBuffer(program, core_grid, cb_cur_sum_config);

    // RM CBs for sampling

    // values
    uint32_t num_out0_units = 32;
    uint32_t values_rm_unit_size = input_values_tensor.element_size();
    uint32_t aligned_values_rm_unit_size = num_out0_units * values_rm_unit_size;
    uint32_t values_rm_cb_index = tt::CBIndex::c_14;
    tt::tt_metal::CircularBufferConfig values_rm_cb_config =
        tt::tt_metal::CircularBufferConfig(
            aligned_values_rm_unit_size, {{values_rm_cb_index, input_values_cb_data_format}})
            .set_page_size(values_rm_cb_index, aligned_values_rm_unit_size);
    auto cb_values_rm_tensor = tt::tt_metal::CreateCircularBuffer(program, core_grid, values_rm_cb_config);

    // indices
    uint32_t indices_rm_unit_size = 2;  // uint16 datum size
    uint32_t aligned_indices_rm_unit_size = num_out0_units * indices_rm_unit_size;
    uint32_t indices_rm_cb_index = tt::CBIndex::c_15;
    tt::tt_metal::CircularBufferConfig indices_rm_cb_config =
        tt::tt_metal::CircularBufferConfig(aligned_indices_rm_unit_size, {{indices_rm_cb_index, index_cb_data_format}})
            .set_page_size(indices_rm_cb_index, aligned_indices_rm_unit_size);
    auto cb_indices_rm_tensor = tt::tt_metal::CreateCircularBuffer(program, core_grid, indices_rm_cb_config);

    // final indices
    uint32_t final_indices_rm_unit_size = input_indices_tensor.element_size();
    uint32_t aligned_final_indices_rm_unit_size = num_out0_units * final_indices_rm_unit_size;
    uint32_t final_indices_rm_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::CircularBufferConfig final_indices_rm_cb_config =
        tt::tt_metal::CircularBufferConfig(
            aligned_final_indices_rm_unit_size, {{final_indices_rm_cb_index, input_indices_cb_data_format}})
            .set_page_size(final_indices_rm_cb_index, aligned_final_indices_rm_unit_size);
    auto cb_final_indices_rm_tensor =
        tt::tt_metal::CreateCircularBuffer(program, core_grid, final_indices_rm_cb_config);

    // // Output sampling indices
    uint32_t output_unit_size = output_tensor.element_size();
    uint32_t aligned_out0_unit_size = num_out0_units * output_unit_size;
    uint32_t output_cb_index = tt::CBIndex::c_17;
    tt::tt_metal::CircularBufferConfig output_cb_config =
        tt::tt_metal::CircularBufferConfig(aligned_out0_unit_size, {{output_ind_cb_index, index_cb_data_format}})
            .set_page_size(output_ind_cb_index, aligned_out0_unit_size);
    auto cb_output_tensor = tt::tt_metal::CreateCircularBuffer(program, core_grid, output_cb_config);

    std::vector<uint32_t> reader_compile_time_args = {
        input_values_cb_index,
        input_indices_cb_index,
        index_cb_index,
        (uint32_t)input_values_is_dram,
        (uint32_t)input_indices_is_dram,
        Ht,
        Wt};
    tt::tt_metal::KernelHandle reader_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/reader_values_indices_tensor.cpp",
        core_grid,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args));

    SetRuntimeArgs(
        program,
        reader_kernel_id,
        core_grid,
        {
            input_values_buffer->address(),
            input_indices_buffer->address(),
        });

    bfloat16 bfloat_identity_scalar = bfloat16(1.0f);
    uint32_t packed_identity_scalar = pack_two_bfloat16_into_uint32({bfloat_identity_scalar, bfloat_identity_scalar});
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)output_is_dram,
        output_cb_index,
        topk_mask_cb_index,
        scale_cb_index,
        packed_identity_scalar,
        final_indices_rm_cb_index,
        values_rm_cb_index,
        indices_rm_cb_index,
        aligned_values_rm_unit_size,
        aligned_indices_rm_unit_size,
        aligned_final_indices_rm_unit_size,
        aligned_out0_unit_size,
        Wt,
        0};
    tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/writer_interleaved.cpp",
        core_grid,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

    std::vector<uint32_t> compute_args = {
        input_values_cb_index,
        input_indices_cb_index,
        index_cb_index,
        input_transposed_cb_index,
        index_transposed_cb_index,
        values_cb_index,
        output_ind_cb_index,
        topk_mask_cb_index,
        scale_cb_index,
        cb_cur_max_index,
        cb_cur_sum_index,
        values_rm_cb_index,
        indices_rm_cb_index,
        final_indices_rm_cb_index,
        Ht,
        Wt,
        (std::uint32_t)std::log2(Wt),
    };
    tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/topk/device/kernels/compute/sampling.cpp",
        core_grid,
        tt::tt_metal::ComputeConfig{.compile_args = compute_args});

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        tt::tt_metal::SetRuntimeArgs(
            program, writer_kernel_id, core, {output_buffer->address(), k[i], i, round_up_to_mul32(k[i])});

        tt::tt_metal::SetRuntimeArgs(
            program,
            compute_kernel_id,
            core,
            {round_up_to_mul32(k[i]), (std::uint32_t)std::log2(round_up_to_mul32(k[i]))});
    }

    auto override_runtime_args_callback = [reader_kernel_id, writer_kernel_id, cores](
                                              const Program& program,
                                              const std::vector<Buffer*>& input_buffers,
                                              const std::vector<Buffer*>& output_buffers) {
        auto input_values_buffer = input_buffers.at(0);
        auto input_indices_buffer = input_buffers.at(1);
        auto output_buffer = output_buffers.at(0);

        for (const auto& core : cores) {
            auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            reader_runtime_args[0] = input_values_buffer->address();
            reader_runtime_args[1] = input_indices_buffer->address();

            auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            writer_runtime_args[0] = output_buffer->address();
        }
    };

    return {std::move(program), override_runtime_args_callback};
}

}  // namespace ttnn::operations::reduction::detail
