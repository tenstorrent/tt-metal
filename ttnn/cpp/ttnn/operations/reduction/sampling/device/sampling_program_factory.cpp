// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/reduction/sampling/device/sampling_program_factory.hpp"

#include <algorithm>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operation.hpp"

namespace ttnn::operations::reduction::sampling::program {

SamplingProgramFactory::cached_program_t SamplingProgramFactory::create(
    const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& output_tensor) {
    using namespace tt::constants;

    const auto& input_values_tensor = tensor_args.input_values;
    const auto& input_indices_tensor = tensor_args.input_indices;
    const auto& k = tensor_args.k;
    const auto& p = tensor_args.p;
    const auto& temp = tensor_args.temp;

    const auto& seed = operation_attributes.seed;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    tt::tt_metal::Program program{};
    uint32_t random_seed = 0;

    tt::DataFormat input_values_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(input_values_tensor.dtype());
    tt::DataFormat input_indices_cb_data_format =
        tt::tt_metal::datatype_to_dataformat_converter(input_indices_tensor.dtype());
    tt::DataFormat index_cb_data_format = tt::DataFormat::UInt16;
    tt::DataFormat k_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(k.dtype());
    tt::DataFormat p_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(p.dtype());
    tt::DataFormat temp_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(temp.dtype());

    uint32_t input_values_tile_size = tile_size(input_values_cb_data_format);
    uint32_t index_tile_size = tile_size(index_cb_data_format);

    auto* input_values_buffer = input_values_tensor.buffer();
    auto* input_indices_buffer = input_indices_tensor.buffer();
    auto* k_buffer = k.buffer();
    auto* p_buffer = p.buffer();
    auto* temp_buffer = temp.buffer();
    auto* output_buffer = output_tensor.buffer();

    auto* device = input_values_tensor.device();

    auto input_shape = input_values_tensor.logical_shape();
    uint32_t Ht = (input_shape[0] * input_shape[1] * input_shape[2]) / TILE_HEIGHT;
    uint32_t Wt = input_shape[3] / TILE_WIDTH;
    auto num_cores = Ht * TILE_HEIGHT;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    CoreRangeSet core_grid = tt::tt_metal::num_cores_to_corerangeset(num_cores, compute_with_storage_grid_size, true);

    if (sub_core_grids.has_value()) {
        core_grid = sub_core_grids.value();
    }
    auto cores = corerange_to_cores(core_grid, num_cores, true);

    if (seed.has_value()) {
        random_seed = seed.value();
    }

    // for streaming in input
    uint32_t num_cb_unit = 2;
    uint32_t cb_in_units = 2 * num_cb_unit;

    // Two tiles are loaded in for sampling_local_sort at a time, and we double buffer to avoid stalls, so allocate four
    // tiles of space
    uint32_t input_values_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig input_values_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_in_units * input_values_tile_size, {{input_values_cb_index, input_values_cb_data_format}})
            .set_page_size(input_values_cb_index, input_values_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, input_values_cb_config);

    uint32_t index_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig index_input_intermed0_config =
        tt::tt_metal::CircularBufferConfig(cb_in_units * index_tile_size, {{index_cb_index, index_cb_data_format}})
            .set_page_size(index_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, index_input_intermed0_config);

    // identity scale input
    tt::DataFormat scalar_df = tt::DataFormat::Float16_b;
    uint32_t scale_tiles = 1;
    uint32_t scalar_tile_size = tile_size(scalar_df);
    uint32_t scale_cb_index = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig scale_cb_config =
        tt::tt_metal::CircularBufferConfig(scale_tiles * scalar_tile_size, {{scale_cb_index, scalar_df}})
            .set_page_size(scale_cb_index, scalar_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, scale_cb_config);

    uint32_t topk_mask_cb_index = tt::CBIndex::c_4;
    tt::tt_metal::CircularBufferConfig topk_mask_cb_config =
        tt::tt_metal::CircularBufferConfig(
            cb_in_units * input_values_tile_size, {{topk_mask_cb_index, input_values_cb_data_format}})
            .set_page_size(topk_mask_cb_index, input_values_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, topk_mask_cb_config);

    // Compute kernel CBs
    // // Single buffered circular buffer that holds the transposed input tiles
    uint32_t input_transposed_cb_index = tt::CBIndex::c_5;
    tt::tt_metal::CircularBufferConfig input_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Wt * input_values_tile_size, {{input_transposed_cb_index, input_values_cb_data_format}})
            .set_page_size(input_transposed_cb_index, input_values_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, input_transposed_cb_config);

    // // Single buffered circular buffer that holds the transposed index tiles
    uint32_t index_transposed_cb_index = tt::CBIndex::c_6;
    tt::tt_metal::CircularBufferConfig index_transposed_cb_config =
        tt::tt_metal::CircularBufferConfig(Wt * index_tile_size, {{index_transposed_cb_index, index_cb_data_format}})
            .set_page_size(index_transposed_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, index_transposed_cb_config);

    // // Output sampling values
    uint32_t values_cb_index = tt::CBIndex::c_7;
    tt::tt_metal::CircularBufferConfig values_cb_config =
        tt::tt_metal::CircularBufferConfig(
            num_cb_unit * input_values_tile_size, {{values_cb_index, input_values_cb_data_format}})
            .set_page_size(values_cb_index, input_values_tile_size);

    tt::tt_metal::CreateCircularBuffer(program, core_grid, values_cb_config);

    uint32_t cb_local_vals_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_local_vals_config =
        tt::tt_metal::CircularBufferConfig(
            num_cb_unit * input_values_tile_size, {{cb_local_vals_index, input_values_cb_data_format}})
            .set_page_size(cb_local_vals_index, input_values_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, cb_local_vals_config);

    // // Output local indices
    uint32_t output_ind_cb_index = tt::CBIndex::c_8;
    tt::tt_metal::CircularBufferConfig output_ind_cb_config =
        tt::tt_metal::CircularBufferConfig(num_cb_unit * index_tile_size, {{output_ind_cb_index, index_cb_data_format}})
            .set_page_size(output_ind_cb_index, index_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, output_ind_cb_config);

    uint32_t num_out_tiles = Ht;
    uint32_t cb_cur_max_index = tt::CBIndex::c_9;
    tt::tt_metal::CircularBufferConfig cb_cur_max_config =
        tt::tt_metal::CircularBufferConfig(
            num_out_tiles * input_values_tile_size, {{cb_cur_max_index, input_values_cb_data_format}})
            .set_page_size(cb_cur_max_index, input_values_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, cb_cur_max_config);

    uint32_t cb_cur_sum_index = tt::CBIndex::c_10;
    tt::tt_metal::CircularBufferConfig cb_cur_sum_config =
        tt::tt_metal::CircularBufferConfig(
            num_out_tiles * input_values_tile_size, {{cb_cur_sum_index, input_values_cb_data_format}})
            .set_page_size(cb_cur_sum_index, input_values_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, cb_cur_sum_config);

    // RM CBs for sampling

    // random number
    const uint32_t rand_tile_size = tile_size(tt::DataFormat::Float16_b);
    constexpr uint32_t rand_tile_index = tt::CBIndex::c_11;
    tt::tt_metal::CircularBufferConfig cb_rand_config =
        tt::tt_metal::CircularBufferConfig(rand_tile_size, {{rand_tile_index, tt::DataFormat::Float16_b}})
            .set_page_size(rand_tile_index, rand_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, cb_rand_config);

    // final indices
    uint32_t final_indices_rm_unit_size = input_indices_tensor.element_size();  // 4 for int32
    uint32_t aligned_final_indices_rm_unit_size = Wt * TILE_WIDTH * final_indices_rm_unit_size;
    uint32_t final_indices_rm_cb_index = tt::CBIndex::c_12;
    tt::tt_metal::CircularBufferConfig final_indices_rm_cb_config =
        tt::tt_metal::CircularBufferConfig(
            Ht * TILE_HEIGHT * aligned_final_indices_rm_unit_size,
            {{final_indices_rm_cb_index, input_indices_cb_data_format}})
            .set_page_size(final_indices_rm_cb_index, aligned_final_indices_rm_unit_size);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, final_indices_rm_cb_config);

    // // Output sampling indices
    uint32_t output_unit_size = output_tensor.element_size();
    uint32_t aligned_out0_unit_size = Ht * TILE_HEIGHT * output_unit_size;
    uint32_t output_cb_index = tt::CBIndex::c_13;
    tt::tt_metal::CircularBufferConfig output_cb_config =
        tt::tt_metal::CircularBufferConfig(aligned_out0_unit_size, {{output_cb_index, index_cb_data_format}})
            .set_page_size(output_cb_index, aligned_out0_unit_size);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, output_cb_config);

    // Add k and p circular buffers
    const uint32_t uint32_bytes = 4;
    const uint32_t bf16_bytes = 2;
    uint32_t k_cb_index = tt::CBIndex::c_14;
    // Increase buffer size to accommodate all cores to avoid unaligned NOC reads
    uint32_t k_chunk_size = num_cores * uint32_bytes;
    tt::tt_metal::CircularBufferConfig k_cb_config =
        tt::tt_metal::CircularBufferConfig(k_chunk_size, {{k_cb_index, k_cb_data_format}})
            .set_page_size(k_cb_index, k_chunk_size);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, k_cb_config);

    uint32_t p_cb_index = tt::CBIndex::c_15;
    // Increase buffer size to accommodate all cores to avoid unaligned NOC reads
    uint32_t p_chunk_size = num_cores * bf16_bytes;
    tt::tt_metal::CircularBufferConfig p_cb_config =
        tt::tt_metal::CircularBufferConfig(p_chunk_size, {{p_cb_index, p_cb_data_format}})
            .set_page_size(p_cb_index, p_chunk_size);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, p_cb_config);

    // Add temp circular buffer
    uint32_t temp_cb_index = tt::CBIndex::c_16;
    // Increase buffer size to accommodate all cores to avoid unaligned NOC reads
    uint32_t temp_chunk_size = num_cores * bf16_bytes;
    tt::tt_metal::CircularBufferConfig temp_cb_config =
        tt::tt_metal::CircularBufferConfig(temp_chunk_size, {{temp_cb_index, temp_cb_data_format}})
            .set_page_size(temp_cb_index, temp_chunk_size);
    tt::tt_metal::CreateCircularBuffer(program, core_grid, temp_cb_config);

    std::vector<uint32_t> reader_compile_time_args = {
        input_values_cb_index, final_indices_rm_cb_index, index_cb_index, Ht, Wt, aligned_final_indices_rm_unit_size};
    tt::tt_metal::TensorAccessorArgs(input_values_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(input_indices_buffer).append_to(reader_compile_time_args);
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

    std::vector<tt::tt_metal::KernelHandle> writer_kernel_ids;
    writer_kernel_ids.reserve(cores.size());

    std::vector<tt::tt_metal::KernelHandle> compute_kernel_ids;
    compute_kernel_ids.reserve(cores.size());

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];

        std::vector<uint32_t> writer_compile_time_args;
        tt::tt_metal::TensorAccessorArgs(output_buffer).append_to(writer_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(temp_buffer).append_to(writer_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(k_buffer).append_to(writer_compile_time_args);
        tt::tt_metal::TensorAccessorArgs(p_buffer).append_to(writer_compile_time_args);
        writer_compile_time_args.insert(
            writer_compile_time_args.end(),
            {
                output_cb_index,
                topk_mask_cb_index,
                scale_cb_index,
                packed_identity_scalar,
                final_indices_rm_cb_index,
                cb_local_vals_index,
                output_ind_cb_index,
                aligned_final_indices_rm_unit_size,
                aligned_out0_unit_size,
                rand_tile_index,
                k_cb_index,
                p_cb_index,
                temp_cb_index,
                i,
                TILE_WIDTH,
                num_cores,
            });
        tt::tt_metal::KernelHandle writer_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/dataflow/writer_interleaved.cpp",
            core,
            tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args));

        tt::tt_metal::SetRuntimeArgs(
            program,
            writer_kernel_id,
            core,
            {output_buffer->address(), temp_buffer->address(), k_buffer->address(), p_buffer->address()});

        writer_kernel_ids.push_back(writer_kernel_id);

        std::vector<uint32_t> compute_args = {
            input_values_cb_index,
            index_cb_index,
            input_transposed_cb_index,
            index_transposed_cb_index,
            values_cb_index,
            output_ind_cb_index,
            topk_mask_cb_index,
            scale_cb_index,
            cb_cur_max_index,
            cb_cur_sum_index,
            Ht,
            Wt,
            (std::uint32_t)std::log2(Wt),
            rand_tile_index,
            random_seed,
            cb_local_vals_index,
            temp_cb_index};

        tt::tt_metal::KernelHandle compute_kernel_id = tt::tt_metal::CreateKernel(
            program,
            "ttnn/cpp/ttnn/operations/reduction/sampling/device/kernels/compute/sampling.cpp",
            core,
            tt::tt_metal::ComputeConfig{.compile_args = compute_args});

        compute_kernel_ids.push_back(compute_kernel_id);
    }

    return cached_program_t{std::move(program), {reader_kernel_id, writer_kernel_ids, cores}};
}

void SamplingProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    Tensor& tensor_return_value) {
    auto& program = cached_program.program;
    auto& shared_vars = cached_program.shared_variables;
    auto& reader_kernel_id = shared_vars.reader_kernel_id;
    auto& writer_kernel_ids = shared_vars.writer_kernel_ids;
    auto& cores = shared_vars.cores;

    auto* input_values_buffer = tensor_args.input_values.buffer();
    auto* input_indices_buffer = tensor_args.input_indices.buffer();
    auto* k_buffer = tensor_args.k.buffer();
    auto* p_buffer = tensor_args.p.buffer();
    auto* temp_buffer = tensor_args.temp.buffer();
    auto* output_buffer = tensor_return_value.buffer();

    for (uint32_t i = 0; i < cores.size(); ++i) {
        const auto& core = cores[i];
        auto& reader_runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
        reader_runtime_args[0] = input_values_buffer->address();
        reader_runtime_args[1] = input_indices_buffer->address();

        auto& writer_runtime_args = GetRuntimeArgs(program, writer_kernel_ids.at(i), core);
        writer_runtime_args[0] = output_buffer->address();
        writer_runtime_args[1] = temp_buffer->address();
        writer_runtime_args[2] = k_buffer->address();
        writer_runtime_args[3] = p_buffer->address();
    }
}

}  // namespace ttnn::operations::reduction::sampling::program
