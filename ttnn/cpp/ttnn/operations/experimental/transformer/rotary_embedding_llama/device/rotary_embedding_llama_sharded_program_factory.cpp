// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_sharded_program_factory.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::experimental::transformer::rotary_embedding_llama::program {

RotaryEmbeddingLlamaMultiCoreSharded::cached_program_t RotaryEmbeddingLlamaMultiCoreSharded::create(
    const RotaryEmbeddingLlamaParams& operation_attributes,
    const RotaryEmbeddingLlamaInputs& tensor_args,
    tt::tt_metal::Tensor& output) {
    using namespace tt::constants;
    using namespace tt;
    using namespace tt::tt_metal;

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

    bool in_sharded = input.shard_spec().has_value();
    std::optional<ShardSpec> shard_spec = in_sharded ? input.shard_spec() : output.shard_spec();

    const uint32_t batch = input.padded_shape()[1];
    const uint32_t n_heads_t = shard_spec->shape[0] / constants::TILE_HEIGHT;
    const uint32_t head_dim_t = shard_spec->shape[1] / constants::TILE_WIDTH;

    tt_metal::IDevice* device = input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    CoreRange all_cores = shard_spec->grid.bounding_box();
    uint32_t num_cores_x = all_cores.grid_size().x;
    uint32_t num_cores_y = all_cores.grid_size().y;

    const uint32_t num_input_tiles = n_heads_t * head_dim_t;
    const uint32_t num_output_tiles = num_input_tiles;

    // Parallelization
    const uint32_t num_cores = num_cores_x * num_cores_y;
    const uint32_t batch_parallel_factor = std::min(batch, num_cores);
    const uint32_t batch_per_core = (batch + batch_parallel_factor - 1) /
                                    batch_parallel_factor;  // TODO: To make general, add support for batch_per_core > 1

    const uint32_t num_sin_cos_rows_per_core = batch_per_core;
    uint32_t num_cos_sin_tiles = head_dim_t * num_sin_cos_rows_per_core;

    // Set up the CBs
    auto* src_buffer = input.buffer();
    auto* cos_buffer = cos.buffer();
    auto* sin_buffer = sin.buffer();
    auto* trans_mat_buffer = trans_mat.buffer();
    auto* dst_buffer = output.buffer();

    uint32_t input_cb_index = CBIndex::c_0;
    tt_metal::CircularBufferConfig cb_input_config =
        tt_metal::CircularBufferConfig(
            num_input_tiles * input_single_tile_size, {{input_cb_index, input_cb_data_format}})
            .set_page_size(input_cb_index, input_single_tile_size)
            .set_globally_allocated_address(*src_buffer);
    auto cb_input = tt_metal::CreateCircularBuffer(program, all_cores, cb_input_config);

    uint32_t cos_cb_index = CBIndex::c_1;
    tt_metal::CircularBufferConfig cb_cos_config =
        tt_metal::CircularBufferConfig(num_cos_sin_tiles * cos_single_tile_size, {{cos_cb_index, cos_cb_data_format}})
            .set_page_size(cos_cb_index, cos_single_tile_size)
            .set_globally_allocated_address(*cos_buffer);
    auto cb_cos = tt_metal::CreateCircularBuffer(program, all_cores, cb_cos_config);

    uint32_t sin_cb_index = CBIndex::c_2;
    tt_metal::CircularBufferConfig cb_sin_config =
        tt_metal::CircularBufferConfig(num_cos_sin_tiles * sin_single_tile_size, {{sin_cb_index, sin_cb_data_format}})
            .set_page_size(sin_cb_index, sin_single_tile_size)
            .set_globally_allocated_address(*sin_buffer);
    auto cb_sin = tt_metal::CreateCircularBuffer(program, all_cores, cb_sin_config);

    uint32_t trans_mat_cb_index = CBIndex::c_3;
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
    tt_metal::CreateCircularBuffer(program, all_cores, cb_rotated_input_interm_config);

    uint32_t cos_interm_cb_index = CBIndex::c_25;
    tt_metal::CircularBufferConfig cb_cos_interm_config =
        tt_metal::CircularBufferConfig(
            num_interm_tiles * input_single_tile_size, {{cos_interm_cb_index, cos_cb_data_format}})
            .set_page_size(cos_interm_cb_index, cos_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_cos_interm_config);

    uint32_t sin_interm_cb_index = CBIndex::c_26;
    tt_metal::CircularBufferConfig cb_sin_interm_config =
        tt_metal::CircularBufferConfig(
            num_interm_tiles * input_single_tile_size, {{sin_interm_cb_index, sin_cb_data_format}})
            .set_page_size(sin_interm_cb_index, sin_single_tile_size);
    tt_metal::CreateCircularBuffer(program, all_cores, cb_sin_interm_config);

    uint32_t output_cb_index = CBIndex::c_16;  // output operands start at index 16
    tt_metal::CircularBufferConfig cb_output_config =
        tt_metal::CircularBufferConfig(
            num_output_tiles * output_single_tile_size, {{output_cb_index, output_cb_data_format}})
            .set_page_size(output_cb_index, output_single_tile_size)
            .set_globally_allocated_address(*dst_buffer);
    auto cb_output = tt_metal::CreateCircularBuffer(program, all_cores, cb_output_config);

    // Set up the kernel
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
        (std::uint32_t)n_heads_t,
    };

    tt_metal::CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama/device/kernels/compute/"
        "rotary_embedding_llama_sharded.cpp",
        all_cores,
        tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_kernel_args});

    RotaryEmbeddingLlamaMultiCoreSharded::shared_variables_t shared_variables;
    shared_variables.cb_input = cb_input;
    shared_variables.cb_cos = cb_cos;
    shared_variables.cb_sin = cb_sin;
    shared_variables.cb_trans_mat = cb_trans_mat;
    shared_variables.cb_output = cb_output;

    return {std::move(program), std::move(shared_variables)};
}

void RotaryEmbeddingLlamaMultiCoreSharded::override_runtime_arguments(
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

    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_input, *src_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_cos, *cos_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_sin, *sin_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_trans_mat, *trans_mat_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_output, *dst_buffer);
}

}  // namespace ttnn::operations::experimental::transformer::rotary_embedding_llama::program
