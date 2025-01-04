// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <vector>
#include "rotary_embedding_llama_fused_qk_program_factory.hpp"
#include "tt_metal/common/work_split.hpp"

#include "tt_metal/common/constants.hpp"
#include "tt_metal/detail/util.hpp"
#include "tt_metal/host_api.hpp"

namespace tt {

namespace tt_metal {

operation::ProgramWithCallbacks rotary_embedding_llama_fused_qk_multi_core_sharded(
    const Tensor& q_input,
    const Tensor& k_input,
    const Tensor& cos,
    const Tensor& sin,
    const Tensor& trans_mat,
    Tensor& q_output,
    Tensor& k_output,
    ttnn::DeviceComputeKernelConfig compute_kernel_config) {
    Program program{};

    const tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(q_input.get_dtype());
    const uint32_t input_single_tile_size = tt_metal::detail::TileSize(input_cb_data_format);

    const tt::DataFormat cos_cb_data_format = tt_metal::datatype_to_dataformat_converter(cos.get_dtype());
    const uint32_t cos_single_tile_size = tt_metal::detail::TileSize(cos_cb_data_format);

    const tt::DataFormat sin_cb_data_format = tt_metal::datatype_to_dataformat_converter(sin.get_dtype());
    const uint32_t sin_single_tile_size = tt_metal::detail::TileSize(sin_cb_data_format);

    const tt::DataFormat trans_mat_cb_data_format = tt_metal::datatype_to_dataformat_converter(trans_mat.get_dtype());
    const uint32_t trans_mat_single_tile_size = tt_metal::detail::TileSize(trans_mat_cb_data_format);

    const tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(q_output.get_dtype());
    const uint32_t output_single_tile_size = tt_metal::detail::TileSize(output_cb_data_format);

    std::optional<ShardSpec> q_shard_spec = q_input.shard_spec();
    std::optional<ShardSpec> k_shard_spec = k_input.shard_spec();
    std::optional<ShardSpec> cos_sin_shard_spec = cos.shard_spec();

    const uint32_t batch = q_input.get_padded_shape()[1];
    const uint32_t q_n_heads_t = q_shard_spec->shape[0] / constants::TILE_HEIGHT;
    const uint32_t k_n_heads_t = k_shard_spec->shape[0] / constants::TILE_HEIGHT;

    const uint32_t head_dim_t = q_shard_spec->shape[1] / constants::TILE_WIDTH;

    tt_metal::Device* device = q_input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), compute_kernel_config);

    CoreRangeSet q_cores = q_shard_spec->grid;

    CoreRangeSet k_cores = k_shard_spec->grid;

    CoreRangeSet all_cores = cos_sin_shard_spec->grid;

    const uint32_t num_q_input_tiles = q_n_heads_t * head_dim_t;
    const uint32_t num_q_output_tiles = num_q_input_tiles;

    const uint32_t num_k_input_tiles = k_n_heads_t * head_dim_t;
    const uint32_t num_k_output_tiles = num_k_input_tiles;

    // Parallelization

    const uint32_t batch_per_core = 1;  // TODO: To make general, add support for batch_per_core > 1

    const uint32_t num_sin_cos_rows_per_core = batch_per_core;
    uint32_t num_cos_sin_tiles = head_dim_t * num_sin_cos_rows_per_core;

    // Set up the CBs
    auto q_src_buffer = q_input.buffer();
    auto k_src_buffer = k_input.buffer();
    auto cos_buffer = cos.buffer();
    auto sin_buffer = sin.buffer();
    auto trans_mat_buffer = trans_mat.buffer();
    auto q_dst_buffer = q_output.buffer();
    auto k_dst_buffer = k_output.buffer();

    uint32_t q_input_cb_index = CBIndex::c_0;
    tt_metal::CircularBufferConfig cb_q_input_config =
        tt_metal::CircularBufferConfig(
            num_q_input_tiles * input_single_tile_size, {{q_input_cb_index, input_cb_data_format}})
            .set_page_size(q_input_cb_index, input_single_tile_size)
            .set_globally_allocated_address(*q_src_buffer);
    auto cb_q_input = tt_metal::CreateCircularBuffer(program, q_cores, cb_q_input_config);

    uint32_t k_input_cb_index = CBIndex::c_1;
    tt_metal::CircularBufferConfig cb_k_input_config =
        tt_metal::CircularBufferConfig(
            num_k_input_tiles * input_single_tile_size, {{k_input_cb_index, input_cb_data_format}})
            .set_page_size(k_input_cb_index, input_single_tile_size)
            .set_globally_allocated_address(*k_src_buffer);
    auto cb_k_input = tt_metal::CreateCircularBuffer(program, k_cores, cb_k_input_config);

    uint32_t cos_cb_index = CBIndex::c_2;
    tt_metal::CircularBufferConfig cb_cos_config =
        tt_metal::CircularBufferConfig(num_cos_sin_tiles * cos_single_tile_size, {{cos_cb_index, cos_cb_data_format}})
            .set_page_size(cos_cb_index, cos_single_tile_size)
            .set_globally_allocated_address(*cos_buffer);
    auto cb_cos = tt_metal::CreateCircularBuffer(program, all_cores, cb_cos_config);

    uint32_t sin_cb_index = CBIndex::c_3;
    tt_metal::CircularBufferConfig cb_sin_config =
        tt_metal::CircularBufferConfig(num_cos_sin_tiles * sin_single_tile_size, {{sin_cb_index, sin_cb_data_format}})
            .set_page_size(sin_cb_index, sin_single_tile_size)
            .set_globally_allocated_address(*sin_buffer);
    auto cb_sin = tt_metal::CreateCircularBuffer(program, all_cores, cb_sin_config);

    uint32_t trans_mat_cb_index = CBIndex::c_4;
    // We only take one tile of trans_mat
    uint32_t num_trans_mat_tiles = 1;
    tt_metal::CircularBufferConfig cb_trans_mat_config =
        tt_metal::CircularBufferConfig(
            num_trans_mat_tiles * trans_mat_single_tile_size, {{trans_mat_cb_index, trans_mat_cb_data_format}})
            .set_page_size(trans_mat_cb_index, trans_mat_single_tile_size)
            .set_globally_allocated_address(*trans_mat_buffer);
    auto cb_trans_mat = tt_metal::CreateCircularBuffer(program, all_cores, cb_trans_mat_config);

    uint32_t num_interm_tiles = head_dim_t;
    uint32_t rotated_input_interm_cb_index = CBIndex::c_24;
    tt_metal::CircularBufferConfig cb_rotated_input_interm_config =
        tt_metal::CircularBufferConfig(
            num_interm_tiles * input_single_tile_size, {{rotated_input_interm_cb_index, input_cb_data_format}})
            .set_page_size(rotated_input_interm_cb_index, input_single_tile_size);
    auto cb_rotated_input_interm = tt_metal::CreateCircularBuffer(program, all_cores, cb_rotated_input_interm_config);

    uint32_t cos_interm_cb_index = CBIndex::c_25;
    tt_metal::CircularBufferConfig cb_cos_interm_config =
        tt_metal::CircularBufferConfig(
            num_interm_tiles * input_single_tile_size, {{cos_interm_cb_index, cos_cb_data_format}})
            .set_page_size(cos_interm_cb_index, cos_single_tile_size);
    auto cb_cos_interm = tt_metal::CreateCircularBuffer(program, all_cores, cb_cos_interm_config);

    uint32_t sin_interm_cb_index = CBIndex::c_26;
    tt_metal::CircularBufferConfig cb_sin_interm_config =
        tt_metal::CircularBufferConfig(
            num_interm_tiles * input_single_tile_size, {{sin_interm_cb_index, sin_cb_data_format}})
            .set_page_size(sin_interm_cb_index, sin_single_tile_size);
    auto cb_sin_interm = tt_metal::CreateCircularBuffer(program, all_cores, cb_sin_interm_config);

    uint32_t q_output_cb_index = CBIndex::c_16;  // output operands start at index 16
    tt_metal::CircularBufferConfig cb_q_output_config =
        tt_metal::CircularBufferConfig(
            num_q_output_tiles * output_single_tile_size, {{q_output_cb_index, output_cb_data_format}})
            .set_page_size(q_output_cb_index, output_single_tile_size)
            .set_globally_allocated_address(*q_dst_buffer);
    auto cb_q_output = tt_metal::CreateCircularBuffer(program, q_cores, cb_q_output_config);
    uint32_t k_output_cb_index = CBIndex::c_17;  // output operands start at index 17
    tt_metal::CircularBufferConfig cb_k_output_config =
        tt_metal::CircularBufferConfig(
            num_k_output_tiles * output_single_tile_size, {{k_output_cb_index, output_cb_data_format}})
            .set_page_size(k_output_cb_index, output_single_tile_size)
            .set_globally_allocated_address(*k_dst_buffer);
    auto cb_k_output = tt_metal::CreateCircularBuffer(program, k_cores, cb_k_output_config);

    // Set up the kernel
    std::vector<uint32_t> q_compute_kernel_args = {
        q_input_cb_index,
        cos_cb_index,
        sin_cb_index,
        trans_mat_cb_index,
        rotated_input_interm_cb_index,
        cos_interm_cb_index,
        sin_interm_cb_index,
        q_output_cb_index,
        head_dim_t,
        q_n_heads_t,
    };

    auto q_rotary_embedding_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/"
        "rotary_embedding_llama_sharded.cpp",
        q_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = q_compute_kernel_args});

    std::vector<uint32_t> k_compute_kernel_args = {
        k_input_cb_index,
        cos_cb_index,
        sin_cb_index,
        trans_mat_cb_index,
        rotated_input_interm_cb_index,
        cos_interm_cb_index,
        sin_interm_cb_index,
        k_output_cb_index,
        head_dim_t,
        k_n_heads_t,
    };

    auto k_rotary_embedding_kernel_id = tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/"
        "rotary_embedding_llama_sharded.cpp",
        k_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = k_compute_kernel_args});

    auto override_runtime_arguments_callback =
        [cb_q_input, cb_k_input, cb_cos, cb_sin, cb_trans_mat, cb_q_output, cb_k_output](
            const void* operation,
            Program& program,
            const std::vector<Tensor>& input_tensors,
            const std::vector<std::optional<const Tensor>>&,
            const std::vector<Tensor>& output_tensors) {
            auto q_src_buffer = input_tensors.at(0).buffer();
            auto k_src_buffer = input_tensors.at(1).buffer();
            auto cos_buffer = input_tensors.at(2).buffer();
            auto sin_buffer = input_tensors.at(3).buffer();
            auto trans_mat_buffer = input_tensors.at(4).buffer();
            auto q_dst_buffer = output_tensors.at(0).buffer();
            auto k_dst_buffer = output_tensors.at(1).buffer();

            // Update the CB globally allocated addresses here
            UpdateDynamicCircularBufferAddress(program, cb_q_input, *q_src_buffer);
            UpdateDynamicCircularBufferAddress(program, cb_k_input, *k_src_buffer);
            UpdateDynamicCircularBufferAddress(program, cb_cos, *cos_buffer);
            UpdateDynamicCircularBufferAddress(program, cb_sin, *sin_buffer);
            UpdateDynamicCircularBufferAddress(program, cb_trans_mat, *trans_mat_buffer);
            UpdateDynamicCircularBufferAddress(program, cb_q_output, *q_dst_buffer);
            UpdateDynamicCircularBufferAddress(program, cb_k_output, *k_dst_buffer);
        };

    return {.program = std::move(program), .override_runtime_arguments_callback = override_runtime_arguments_callback};
}

}  // namespace tt_metal

}  // namespace tt
