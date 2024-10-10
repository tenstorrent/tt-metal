// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include "rotary_embedding_llama_program_factory.hpp"
#include "tt_metal/common/work_split.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

using namespace tt::constants;

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks rotary_embedding_llama_multi_core(
    const Tensor &input,
    const Tensor &cos,
    const Tensor &sin,
    const Tensor &trans_mat,
    Tensor &output,
    ttnn::DeviceComputeKernelConfig compute_kernel_config
) {
    Program program{};

    const tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.get_dtype());
    const uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);

    const tt::DataFormat cos_cb_data_format = tt_metal::datatype_to_dataformat_converter(cos.get_dtype());
    const uint32_t cos_single_tile_size = tt_metal::detail::TileSize(cos_cb_data_format);

    const tt::DataFormat sin_cb_data_format = tt_metal::datatype_to_dataformat_converter(sin.get_dtype());
    const uint32_t sin_single_tile_size = tt_metal::detail::TileSize(sin_cb_data_format);

    const tt::DataFormat trans_mat_cb_data_format = tt_metal::datatype_to_dataformat_converter(trans_mat.get_dtype());
    const uint32_t trans_mat_single_tile_size = tt_metal::detail::TileSize(trans_mat_cb_data_format);

    const tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.get_dtype());
    const uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    const uint32_t batch = input.get_padded_shape()[0];
    const uint32_t n_heads = input.get_padded_shape()[1];
    const uint32_t seq_len_t = input.get_padded_shape()[2] / TILE_HEIGHT;
    const uint32_t head_dim_t = input.get_padded_shape()[3] / TILE_WIDTH;

    const uint32_t Wbytes = input.get_padded_shape()[-1] * sizeof(bfloat16);

    tt_metal::Device *device = input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    CoreRange all_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    bool in_sharded = input.shard_spec().has_value();
    bool out_sharded = output.shard_spec().has_value();
    std::optional<ShardSpec> shard_spec = in_sharded ? input.shard_spec() : output.shard_spec();

    const uint32_t num_input_tiles = 2 * head_dim_t;
    const uint32_t num_output_tiles = num_input_tiles;

    bool row_major = true;

    // Parallelization
    const uint32_t num_cores = num_cores_x * num_cores_y;
    const uint32_t batch_parallel_factor = std::min(batch, num_cores);
    const uint32_t seq_parallel_factor = std::min(num_cores / batch_parallel_factor, seq_len_t);
    const uint32_t batch_per_core = (batch + batch_parallel_factor - 1) / batch_parallel_factor;
    const uint32_t seq_per_core = (seq_len_t + seq_parallel_factor - 1) / seq_parallel_factor;

    const uint32_t num_sin_cos_rows_per_core = (seq_len_t + seq_parallel_factor - 1) / seq_parallel_factor;
    const uint32_t num_rows_per_core = num_sin_cos_rows_per_core * n_heads;

    uint32_t num_cos_sin_tiles = 2 * head_dim_t * num_sin_cos_rows_per_core;

    uint32_t input_cb_num_tiles = num_sin_cos_rows_per_core * num_input_tiles;

    const bool use_reload_impl = num_rows_per_core > 8;
    if (use_reload_impl) {
        // Do reload implementation of kernel to reduce buffer sizes
        // Only size CBs to double buffer head_dim_t tiles for all inputs
        input_cb_num_tiles = num_input_tiles;
        num_cos_sin_tiles = num_input_tiles;
    }


    uint32_t input_cb_index = CB::c_in0;
    tt_metal::CircularBufferConfig cb_input_config =
        tt_metal::CircularBufferConfig(
            input_cb_num_tiles * input_single_tile_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, input_single_tile_size);
    auto cb_input = tt_metal::CreateCircularBuffer(program, all_cores, cb_input_config);

    uint32_t cos_cb_index = CB::c_in1;
    tt_metal::CircularBufferConfig cb_cos_config =
        tt_metal::CircularBufferConfig(num_cos_sin_tiles * cos_single_tile_size, {{cos_cb_index, cos_cb_data_format}})
            .set_page_size(cos_cb_index, cos_single_tile_size);
    auto cb_cos = tt_metal::CreateCircularBuffer(program, all_cores, cb_cos_config);

    uint32_t sin_cb_index = CB::c_in2;
    tt_metal::CircularBufferConfig cb_sin_config =
        tt_metal::CircularBufferConfig(num_cos_sin_tiles * sin_single_tile_size, {{sin_cb_index, sin_cb_data_format}})
            .set_page_size(sin_cb_index, sin_single_tile_size);
    auto cb_sin = tt_metal::CreateCircularBuffer(program, all_cores, cb_sin_config);

    uint32_t trans_mat_cb_index = CB::c_in3;
    // We only take one tile of trans_mat
    uint32_t num_trans_mat_tiles = 1;
    tt_metal::CircularBufferConfig cb_trans_mat_config =
        tt_metal::CircularBufferConfig(num_trans_mat_tiles * trans_mat_single_tile_size, {{trans_mat_cb_index, trans_mat_cb_data_format}})
            .set_page_size(trans_mat_cb_index, trans_mat_single_tile_size);
    auto cb_trans_mat = tt_metal::CreateCircularBuffer(program, all_cores, cb_trans_mat_config);

    uint32_t num_interm_tiles = head_dim_t;
    uint32_t rotated_input_interm_cb_index = CB::c_intermed0;
    tt_metal::CircularBufferConfig cb_rotated_input_interm_config =
        tt_metal::CircularBufferConfig(
            num_interm_tiles * input_single_tile_size, {{rotated_input_interm_cb_index, input_cb_data_format}})
            .set_page_size(rotated_input_interm_cb_index, input_single_tile_size);
    auto cb_rotated_input_interm = tt_metal::CreateCircularBuffer(program, all_cores, cb_rotated_input_interm_config);

    uint32_t cos_interm_cb_index = CB::c_intermed1;
    tt_metal::CircularBufferConfig cb_cos_interm_config =
        tt_metal::CircularBufferConfig(
            num_interm_tiles * cos_single_tile_size, {{cos_interm_cb_index, cos_cb_data_format}})
            .set_page_size(cos_interm_cb_index, cos_single_tile_size);
    auto cb_cos_interm = tt_metal::CreateCircularBuffer(program, all_cores, cb_cos_interm_config);

    uint32_t sin_interm_cb_index = CB::c_intermed2;
    tt_metal::CircularBufferConfig cb_sin_interm_config =
        tt_metal::CircularBufferConfig(
            num_interm_tiles * sin_single_tile_size, {{sin_interm_cb_index, sin_cb_data_format}})
            .set_page_size(sin_interm_cb_index, sin_single_tile_size);
    auto cb_sin_interm = tt_metal::CreateCircularBuffer(program, all_cores, cb_sin_interm_config);

    uint32_t output_cb_index = CB::c_out0;  // output operands start at index 16
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(
            num_output_tiles * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    std::map<string, string> kernel_defines;
    kernel_defines["RELOAD_IMPL"] = use_reload_impl ? "1" : "0";

    auto src_buffer = input.buffer();
    auto cos_buffer = cos.buffer();
    auto sin_buffer = sin.buffer();
    auto trans_mat_buffer = trans_mat.buffer();
    auto dst_buffer = output.buffer();

    bool src_is_dram = src_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool cos_is_dram = cos_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool sin_is_dram = sin_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    bool trans_mat_is_dram = trans_mat_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;



    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)cos_cb_index,
        (std::uint32_t)sin_cb_index,
        (std::uint32_t)trans_mat_cb_index,
        (std::uint32_t)src_is_dram,
        (std::uint32_t)cos_is_dram,
        (std::uint32_t)sin_is_dram,
        (std::uint32_t)trans_mat_is_dram,
        (std::uint32_t)n_heads,
        (std::uint32_t)seq_len_t,
        (std::uint32_t)head_dim_t,
    };
    bool dst_is_dram = dst_buffer->buffer_type() == tt_metal::BufferType::DRAM ? 1 : 0;
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)output_cb_index,
        (std::uint32_t)dst_is_dram,
        (std::uint32_t)n_heads,
        (std::uint32_t)head_dim_t,
        (std::uint32_t)seq_len_t,
    };

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/dataflow/reader_rotary_embedding_llama_interleaved_start_id.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/dataflow/writer_rotary_embedding_llama_interleaved_start_id.cpp",
        all_cores,
        tt_metal::WriterDataMovementConfig(writer_compile_time_args, kernel_defines));

    std::vector<uint32_t> compute_kernel_args = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)cos_cb_index,
        (std::uint32_t)sin_cb_index,
        (std::uint32_t)trans_mat_cb_index,
        (std::uint32_t)rotated_input_interm_cb_index,
        (std::uint32_t)cos_interm_cb_index,
        (std::uint32_t)sin_interm_cb_index,
        (std::uint32_t)output_cb_index,
        (std::uint32_t)head_dim_t,
        (std::uint32_t)n_heads,
        };

    auto rotary_embedding_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/rotary_embedding_llama.cpp",
        all_cores,
        tt_metal::ComputeConfig{.math_fidelity=math_fidelity, .fp32_dest_acc_en=fp32_dest_acc_en, .compile_args = compute_kernel_args, .defines = kernel_defines});

    const auto &cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    /*
        Overall loop iterations: # total cores
    */

    std::vector<uint32_t> default_reader_args = {
        src_buffer->address(),
        cos_buffer->address(),
        sin_buffer->address(),
        trans_mat_buffer->address(),
        0,
        0,
        0,
        0
    };

    std::vector<uint32_t> default_writer_args = {
        dst_buffer->address(),
        0,
        0,
        0,
        0
    };

    std::vector<uint32_t> default_compute_args = {
        0,
        0,
        0,
        0
    };

    std::vector< std::vector<uint32_t> > unary_reader_args = {cores.size(), default_reader_args};
    std::vector< std::vector<uint32_t> > unary_writer_args = {cores.size(), default_writer_args};
    std::vector< std::vector<uint32_t> > unary_compute_args = {cores.size(), default_compute_args};

    uint32_t num_active_cores = 0;

    for (uint32_t batch_parallel = 0; batch_parallel < batch_parallel_factor; batch_parallel++) {
        for (uint32_t seq_parallel = 0; seq_parallel < seq_parallel_factor; seq_parallel++) {
            uint32_t core_idx = batch_parallel * seq_parallel_factor + seq_parallel;
            const CoreCoord &core = cores.at(core_idx);
            uint32_t start_batch = batch_parallel * batch_per_core;
            uint32_t end_batch = std::min(start_batch + batch_per_core, batch);
            uint32_t start_seq = seq_parallel * seq_per_core;
            uint32_t end_seq = std::min(start_seq + seq_per_core, seq_len_t);

            if (start_seq >= seq_len_t || start_batch >= batch) {
                // Important to skip cores which have no work to do, otherwise they will wait
                // on cos/sin data which will never arrive.
                continue;
            }
            tt::log_debug(tt::LogTest, "core: {}, start_batch: {}, end_batch: {}, start_seq: {}, end_seq: {}",
                         core_idx, start_batch, end_batch, start_seq, end_seq);


            // Reader runtime args
            auto& reader_rt_args = unary_reader_args[core_idx];
            reader_rt_args[4] = start_batch;
            reader_rt_args[5] = end_batch;
            reader_rt_args[6] = start_seq;
            reader_rt_args[7] = end_seq;

            // Writer runtime args
            auto& writer_rt_args = unary_writer_args[core_idx];
            writer_rt_args[1] = start_batch;
            writer_rt_args[2] = end_batch;
            writer_rt_args[3] = start_seq;
            writer_rt_args[4] = end_seq;

            // Compute runtime args
            auto& compute_rt_args = unary_compute_args[core_idx];
            compute_rt_args[0] = start_batch;
            compute_rt_args[1] = end_batch;
            compute_rt_args[2] = start_seq;
            compute_rt_args[3] = end_seq;

            num_active_cores = core_idx + 1;

        }
    }

    tt_metal::SetRuntimeArgs(program, unary_reader_kernel_id, cores, unary_reader_args);
    tt_metal::SetRuntimeArgs(program, unary_writer_kernel_id, cores, unary_writer_args);
    tt_metal::SetRuntimeArgs(program, rotary_embedding_kernel_id, cores, unary_compute_args);

    auto override_runtime_arguments_callback = [unary_reader_kernel_id,
                                                unary_writer_kernel_id,
                                                cores,
                                                num_active_cores](
                                                   const void *operation,
                                                   const Program &program,
                                                   const std::vector<Tensor> &input_tensors,
                                                   const std::vector<std::optional<const Tensor>> &,
                                                   const std::vector<Tensor> &output_tensors) {

        auto src_buffer = input_tensors.at(0).buffer();
        auto cos_buffer = input_tensors.at(1).buffer();
        auto sin_buffer = input_tensors.at(2).buffer();
        auto trans_mat_buffer = input_tensors.at(3).buffer();

        auto dst_buffer = output_tensors.at(0).buffer();

        auto &cached_reader_args = GetRuntimeArgs(program, unary_reader_kernel_id);
        auto &cached_writer_args = GetRuntimeArgs(program, unary_writer_kernel_id);

        for (uint32_t i = 0; i < num_active_cores; ++i) {
            const CoreCoord &core = cores.at(i);
            {
                auto& runtime_args = cached_reader_args.at(core.x).at(core.y);
                runtime_args[0] = src_buffer->address();
                runtime_args[1] = cos_buffer->address();
                runtime_args[2] = sin_buffer->address();
                runtime_args[3] = trans_mat_buffer->address();
            }

            {
                auto& runtime_args = cached_writer_args.at(core.x).at(core.y);
                runtime_args[0] = dst_buffer->address();
            }
        }
    };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
