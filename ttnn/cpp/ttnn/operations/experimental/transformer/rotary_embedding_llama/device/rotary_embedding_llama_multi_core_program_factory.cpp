// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_multi_core_program_factory.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::transformer::rotary_embedding_llama::program {

RotaryEmbeddingLlamaMultiCore::cached_program_t RotaryEmbeddingLlamaMultiCore::create(
    const RotaryEmbeddingLlamaParams& operation_attributes,
    const RotaryEmbeddingLlamaInputs& tensor_args,
    tt::tt_metal::Tensor& output) {
    using namespace tt::constants;
    using namespace tt::tt_metal;
    using namespace tt;

    const auto& input = tensor_args.input_tensor;
    const auto& cos = tensor_args.cos_cache;
    const auto& sin = tensor_args.sin_cache;
    const auto& trans_mat = tensor_args.trans_mat;

    Program program{};

    const tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    const tt::DataFormat cos_cb_data_format = tt_metal::datatype_to_dataformat_converter(cos.dtype());
    const uint32_t cos_single_tile_size = tt::tile_size(cos_cb_data_format);

    const tt::DataFormat sin_cb_data_format = tt_metal::datatype_to_dataformat_converter(sin.dtype());
    const uint32_t sin_single_tile_size = tt::tile_size(sin_cb_data_format);

    const tt::DataFormat trans_mat_cb_data_format = tt_metal::datatype_to_dataformat_converter(trans_mat.dtype());
    const uint32_t trans_mat_single_tile_size = tt::tile_size(trans_mat_cb_data_format);

    const tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const uint32_t batch = input.padded_shape()[0];
    const uint32_t n_heads = input.padded_shape()[1];
    const uint32_t seq_len_t = input.padded_shape()[2] / TILE_HEIGHT;
    const uint32_t head_dim_t = input.padded_shape()[3] / TILE_WIDTH;

    // Flag for whether or not sin/cos vary per head. If false, they will be broadcasted across heads.
    const bool freq_per_head = cos.padded_shape()[1] == n_heads;

    tt_metal::IDevice* device = input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    uint32_t num_cores_x = compute_with_storage_grid_size.x;
    uint32_t num_cores_y = compute_with_storage_grid_size.y;

    CoreRange all_cores = CoreRange({0, 0}, {num_cores_x - 1, num_cores_y - 1});

    bool in_sharded = input.shard_spec().has_value();
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

    // Reload implementation is used if sequence length is larger than some heuristic threshold where
    // the buffer size will be too large or if sin/cos are not broadcasted across heads.
    const bool use_reload_impl = num_rows_per_core > 8 || freq_per_head;
    if (use_reload_impl) {
        // Only size CBs to double buffer head_dim_t tiles for all inputs
        input_cb_num_tiles = num_input_tiles;
        num_cos_sin_tiles = num_input_tiles;
    }

    uint32_t input_cb_index = CBIndex::c_0;
    tt_metal::CircularBufferConfig cb_input_config =
        tt_metal::CircularBufferConfig(
            input_cb_num_tiles * input_single_tile_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, input_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_input_config);

    uint32_t cos_cb_index = CBIndex::c_1;
    tt_metal::CircularBufferConfig cb_cos_config =
        tt_metal::CircularBufferConfig(num_cos_sin_tiles * cos_single_tile_size, {{cos_cb_index, cos_cb_data_format}})
            .set_page_size(cos_cb_index, cos_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_cos_config);

    uint32_t sin_cb_index = CBIndex::c_2;
    tt_metal::CircularBufferConfig cb_sin_config =
        tt_metal::CircularBufferConfig(num_cos_sin_tiles * sin_single_tile_size, {{sin_cb_index, sin_cb_data_format}})
            .set_page_size(sin_cb_index, sin_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_sin_config);

    uint32_t trans_mat_cb_index = CBIndex::c_3;
    // We only take one tile of trans_mat
    uint32_t num_trans_mat_tiles = 1;
    tt_metal::CircularBufferConfig cb_trans_mat_config =
        tt_metal::CircularBufferConfig(
            num_trans_mat_tiles * trans_mat_single_tile_size, {{trans_mat_cb_index, trans_mat_cb_data_format}})
            .set_page_size(trans_mat_cb_index, trans_mat_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_trans_mat_config);

    uint32_t num_interm_tiles = head_dim_t;
    uint32_t rotated_input_interm_cb_index = CBIndex::c_24;
    tt_metal::CircularBufferConfig cb_rotated_input_interm_config =
        tt_metal::CircularBufferConfig(
            num_interm_tiles * input_single_tile_size, {{rotated_input_interm_cb_index, input_cb_data_format}})
            .set_page_size(rotated_input_interm_cb_index, input_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_rotated_input_interm_config);

    uint32_t cos_interm_cb_index = CBIndex::c_25;
    tt_metal::CircularBufferConfig cb_cos_interm_config =
        tt_metal::CircularBufferConfig(
            num_interm_tiles * cos_single_tile_size, {{cos_interm_cb_index, cos_cb_data_format}})
            .set_page_size(cos_interm_cb_index, cos_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_cos_interm_config);

    uint32_t sin_interm_cb_index = CBIndex::c_26;
    tt_metal::CircularBufferConfig cb_sin_interm_config =
        tt_metal::CircularBufferConfig(
            num_interm_tiles * sin_single_tile_size, {{sin_interm_cb_index, sin_cb_data_format}})
            .set_page_size(sin_interm_cb_index, sin_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_sin_interm_config);

    uint32_t output_cb_index = CBIndex::c_16;  // output operands start at index 16
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(
            num_output_tiles * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    std::map<std::string, std::string> kernel_defines;
    kernel_defines["RELOAD_IMPL"] = use_reload_impl ? "1" : "0";

    auto* src_buffer = input.buffer();
    auto* cos_buffer = cos.buffer();
    auto* sin_buffer = sin.buffer();
    auto* trans_mat_buffer = trans_mat.buffer();
    auto* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)cos_cb_index,
        (std::uint32_t)sin_cb_index,
        (std::uint32_t)trans_mat_cb_index,
        (std::uint32_t)n_heads,
        (std::uint32_t)seq_len_t,
        (std::uint32_t)head_dim_t,
        (std::uint32_t)freq_per_head,
    };
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(cos_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(sin_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(trans_mat_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)output_cb_index,
        (std::uint32_t)n_heads,
        (std::uint32_t)head_dim_t,
        (std::uint32_t)seq_len_t,
    };
    tt::tt_metal::TensorAccessorArgs(dst_buffer).append_to(writer_compile_time_args);

    tt_metal::KernelHandle unary_reader_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/dataflow/"
        "reader_rotary_embedding_llama_interleaved_start_id.cpp",
        all_cores,
        tt_metal::ReaderDataMovementConfig(reader_compile_time_args, kernel_defines));

    tt_metal::KernelHandle unary_writer_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/dataflow/"
        "writer_rotary_embedding_llama_interleaved_start_id.cpp",
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
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/"
        "rotary_embedding_llama.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_kernel_args,
            .defines = kernel_defines});

    const auto& cores = grid_to_cores(num_cores, num_cores_x, num_cores_y, row_major);

    /*
        Overall loop iterations: # total cores
    */

    std::vector<uint32_t> default_reader_args = {
        src_buffer->address(), cos_buffer->address(), sin_buffer->address(), trans_mat_buffer->address(), 0, 0, 0, 0};

    std::vector<uint32_t> default_writer_args = {dst_buffer->address(), 0, 0, 0, 0};

    std::vector<uint32_t> default_compute_args = {0, 0, 0, 0};

    std::vector<std::vector<uint32_t>> unary_reader_args = {cores.size(), default_reader_args};
    std::vector<std::vector<uint32_t>> unary_writer_args = {cores.size(), default_writer_args};
    std::vector<std::vector<uint32_t>> unary_compute_args = {cores.size(), default_compute_args};

    uint32_t num_active_cores = 0;

    for (uint32_t batch_parallel = 0; batch_parallel < batch_parallel_factor; batch_parallel++) {
        for (uint32_t seq_parallel = 0; seq_parallel < seq_parallel_factor; seq_parallel++) {
            uint32_t core_idx = (batch_parallel * seq_parallel_factor) + seq_parallel;
            uint32_t start_batch = batch_parallel * batch_per_core;
            uint32_t end_batch = std::min(start_batch + batch_per_core, batch);
            uint32_t start_seq = seq_parallel * seq_per_core;
            uint32_t end_seq = std::min(start_seq + seq_per_core, seq_len_t);

            if (start_seq >= seq_len_t || start_batch >= batch) {
                // Important to skip cores which have no work to do, otherwise they will wait
                // on cos/sin data which will never arrive.
                continue;
            }
            log_debug(
                tt::LogTest,
                "core: {}, start_batch: {}, end_batch: {}, start_seq: {}, end_seq: {}",
                core_idx,
                start_batch,
                end_batch,
                start_seq,
                end_seq);

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

    RotaryEmbeddingLlamaMultiCore::shared_variables_t shared_variables;
    shared_variables.unary_reader_kernel_id = unary_reader_kernel_id;
    shared_variables.unary_writer_kernel_id = unary_writer_kernel_id;
    shared_variables.rotary_embedding_kernel_id = rotary_embedding_kernel_id;
    shared_variables.cores = cores;
    shared_variables.num_active_cores = num_active_cores;

    return {std::move(program), std::move(shared_variables)};
}

void RotaryEmbeddingLlamaMultiCore::override_runtime_arguments(
    cached_program_t& cached_program,
    const RotaryEmbeddingLlamaParams& /*operation_attributes*/,
    const RotaryEmbeddingLlamaInputs& tensor_args,
    tt::tt_metal::Tensor& output) {
    using namespace tt::constants;
    using namespace tt::tt_metal;

    auto& program = cached_program.program;
    auto& shared_variables = cached_program.shared_variables;

    auto* src_buffer = tensor_args.input_tensor.buffer();
    auto* cos_buffer = tensor_args.cos_cache.buffer();
    auto* sin_buffer = tensor_args.sin_cache.buffer();
    auto* trans_mat_buffer = tensor_args.trans_mat.buffer();
    auto* dst_buffer = output.buffer();

    auto& cached_reader_args = GetRuntimeArgs(program, shared_variables.unary_reader_kernel_id);
    auto& cached_writer_args = GetRuntimeArgs(program, shared_variables.unary_writer_kernel_id);

    const auto& cores = shared_variables.cores;
    uint32_t num_active_cores = shared_variables.num_active_cores;

    for (uint32_t i = 0; i < num_active_cores; ++i) {
        const CoreCoord& core = cores.at(i);
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
}

}  // namespace ttnn::operations::experimental::transformer::rotary_embedding_llama::program
