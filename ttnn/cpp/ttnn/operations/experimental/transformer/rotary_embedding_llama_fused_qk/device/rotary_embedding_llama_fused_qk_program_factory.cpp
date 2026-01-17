// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_fused_qk_device_operation_types.hpp"
#include "rotary_embedding_llama_fused_qk_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>

namespace ttnn::experimental::prim {

using namespace tt::constants;

RotaryEmbeddingLlamaFusedQKProgramFactory::cached_program_t RotaryEmbeddingLlamaFusedQKProgramFactory::create(
    const RotaryEmbeddingLlamaFusedQkParams& operation_attributes,
    const RotaryEmbeddingLlamaFusedQkInputs& tensor_args,
    RotaryEmbeddingLlamaFusedQkResult& tensor_return_value) {
    Program program{};

    const auto& q_input = tensor_args.q_input;
    const auto& k_input = tensor_args.k_input;
    const auto& cos = tensor_args.cos;
    const auto& sin = tensor_args.sin;
    const auto& trans_mat = tensor_args.trans_mat;
    auto& q_output = std::get<0>(tensor_return_value);
    auto& k_output = std::get<1>(tensor_return_value);

    const tt::DataFormat input_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(q_input.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    const tt::DataFormat cos_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(cos.dtype());
    const uint32_t cos_single_tile_size = tt::tile_size(cos_cb_data_format);

    const tt::DataFormat sin_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(sin.dtype());
    const uint32_t sin_single_tile_size = tt::tile_size(sin_cb_data_format);

    const tt::DataFormat trans_mat_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(trans_mat.dtype());
    const uint32_t trans_mat_single_tile_size = tt::tile_size(trans_mat_cb_data_format);

    const tt::DataFormat output_cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(q_output.dtype());
    const uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const std::optional<tt::tt_metal::ShardSpec>& q_shard_spec = q_input.shard_spec();
    const std::optional<tt::tt_metal::ShardSpec>& k_shard_spec = k_input.shard_spec();
    const std::optional<tt::tt_metal::ShardSpec>& cos_sin_shard_spec = cos.shard_spec();

    const uint32_t q_n_heads_t =
        operation_attributes.row_major_QK ? 1 : q_shard_spec->shape[0] / tt::constants::TILE_HEIGHT;
    const uint32_t k_n_heads_t =
        operation_attributes.row_major_QK ? 1 : k_shard_spec->shape[0] / tt::constants::TILE_HEIGHT;

    const uint32_t head_dim_t =
        operation_attributes.row_major_QK ? 1 : q_shard_spec->shape[1] / tt::constants::TILE_WIDTH;

    tt::tt_metal::IDevice* device = q_input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    CoreRangeSet q_cores = q_shard_spec->grid;

    CoreRangeSet k_cores = k_shard_spec->grid;

    CoreRangeSet all_cores = cos_sin_shard_spec->grid;
    CoreRangeSet all_cores_bb = all_cores.bounding_box();
    CoreRangeSet unused_cores = all_cores_bb.subtract(all_cores);

    const uint32_t num_q_input_tiles = q_n_heads_t * head_dim_t;
    const uint32_t num_q_output_tiles = num_q_input_tiles;

    const uint32_t num_k_input_tiles = k_n_heads_t * head_dim_t;
    const uint32_t num_k_output_tiles = num_k_input_tiles;

    // Parallelization

    const uint32_t batch_per_core = 1;  // TODO: To make general, add support for batch_per_core > 1

    const uint32_t num_sin_cos_rows_per_core = batch_per_core;
    uint32_t num_cos_sin_tiles = head_dim_t * num_sin_cos_rows_per_core;

    // Set up the CBs
    auto* q_src_buffer = q_input.buffer();
    auto* k_src_buffer = k_input.buffer();
    auto* cos_buffer = cos.buffer();
    auto* sin_buffer = sin.buffer();
    auto* trans_mat_buffer = trans_mat.buffer();
    auto* q_dst_buffer = q_output.buffer();
    auto* k_dst_buffer = k_output.buffer();

    uint32_t q_input_cb_index = tt::CBIndex::c_0;
    tt::tt_metal::CircularBufferConfig cb_q_input_config =
        tt::tt_metal::CircularBufferConfig(
            num_q_input_tiles * input_single_tile_size, {{q_input_cb_index, input_cb_data_format}})
            .set_page_size(q_input_cb_index, input_single_tile_size)
            .set_globally_allocated_address(*q_src_buffer);
    auto cb_q_input = tt::tt_metal::CreateCircularBuffer(program, all_cores_bb, cb_q_input_config);

    uint32_t k_input_cb_index = tt::CBIndex::c_1;
    tt::tt_metal::CircularBufferConfig cb_k_input_config =
        tt::tt_metal::CircularBufferConfig(
            num_k_input_tiles * input_single_tile_size, {{k_input_cb_index, input_cb_data_format}})
            .set_page_size(k_input_cb_index, input_single_tile_size)
            .set_globally_allocated_address(*k_src_buffer);
    auto cb_k_input = tt::tt_metal::CreateCircularBuffer(program, all_cores_bb, cb_k_input_config);

    uint32_t cos_cb_index = tt::CBIndex::c_2;
    tt::tt_metal::CircularBufferConfig cb_cos_config =
        tt::tt_metal::CircularBufferConfig(
            num_cos_sin_tiles * cos_single_tile_size, {{cos_cb_index, cos_cb_data_format}})
            .set_page_size(cos_cb_index, cos_single_tile_size)
            .set_globally_allocated_address(*cos_buffer);
    auto cb_cos = tt::tt_metal::CreateCircularBuffer(program, all_cores_bb, cb_cos_config);

    uint32_t sin_cb_index = tt::CBIndex::c_3;
    tt::tt_metal::CircularBufferConfig cb_sin_config =
        tt::tt_metal::CircularBufferConfig(
            num_cos_sin_tiles * sin_single_tile_size, {{sin_cb_index, sin_cb_data_format}})
            .set_page_size(sin_cb_index, sin_single_tile_size)
            .set_globally_allocated_address(*sin_buffer);
    auto cb_sin = tt::tt_metal::CreateCircularBuffer(program, all_cores_bb, cb_sin_config);

    uint32_t trans_mat_cb_index = tt::CBIndex::c_4;
    // We only take one tile of trans_mat
    uint32_t num_trans_mat_tiles = 1;
    tt::tt_metal::CircularBufferConfig cb_trans_mat_config =
        tt::tt_metal::CircularBufferConfig(
            num_trans_mat_tiles * trans_mat_single_tile_size, {{trans_mat_cb_index, trans_mat_cb_data_format}})
            .set_page_size(trans_mat_cb_index, trans_mat_single_tile_size)
            .set_globally_allocated_address(*trans_mat_buffer);
    auto cb_trans_mat = tt::tt_metal::CreateCircularBuffer(program, all_cores_bb, cb_trans_mat_config);

    uint32_t num_interm_tiles = head_dim_t;
    uint32_t rotated_input_interm_cb_index = tt::CBIndex::c_24;
    tt::tt_metal::CircularBufferConfig cb_rotated_input_interm_config =
        tt::tt_metal::CircularBufferConfig(
            num_interm_tiles * input_single_tile_size, {{rotated_input_interm_cb_index, input_cb_data_format}})
            .set_page_size(rotated_input_interm_cb_index, input_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_bb, cb_rotated_input_interm_config);

    uint32_t cos_interm_cb_index = tt::CBIndex::c_25;
    tt::tt_metal::CircularBufferConfig cb_cos_interm_config =
        tt::tt_metal::CircularBufferConfig(
            num_interm_tiles * input_single_tile_size, {{cos_interm_cb_index, cos_cb_data_format}})
            .set_page_size(cos_interm_cb_index, cos_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_bb, cb_cos_interm_config);

    uint32_t sin_interm_cb_index = tt::CBIndex::c_26;
    tt::tt_metal::CircularBufferConfig cb_sin_interm_config =
        tt::tt_metal::CircularBufferConfig(
            num_interm_tiles * input_single_tile_size, {{sin_interm_cb_index, sin_cb_data_format}})
            .set_page_size(sin_interm_cb_index, sin_single_tile_size);
    tt::tt_metal::CreateCircularBuffer(program, all_cores_bb, cb_sin_interm_config);

    uint32_t q_output_cb_index = tt::CBIndex::c_16;  // output operands start at index 16
    tt::tt_metal::CircularBufferConfig cb_q_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_q_output_tiles * output_single_tile_size, {{q_output_cb_index, output_cb_data_format}})
            .set_page_size(q_output_cb_index, output_single_tile_size)
            .set_globally_allocated_address(*q_dst_buffer);
    auto cb_q_output = tt::tt_metal::CreateCircularBuffer(program, all_cores_bb, cb_q_output_config);
    uint32_t k_output_cb_index = tt::CBIndex::c_17;  // output operands start at index 17
    tt::tt_metal::CircularBufferConfig cb_k_output_config =
        tt::tt_metal::CircularBufferConfig(
            num_k_output_tiles * output_single_tile_size, {{k_output_cb_index, output_cb_data_format}})
            .set_page_size(k_output_cb_index, output_single_tile_size)
            .set_globally_allocated_address(*k_dst_buffer);
    auto cb_k_output = tt::tt_metal::CreateCircularBuffer(program, all_cores_bb, cb_k_output_config);

    // Set up the kernel
    std::vector<uint32_t> compute_kernel_args = {
        q_input_cb_index,
        q_output_cb_index,
        q_n_heads_t,
        k_input_cb_index,
        k_output_cb_index,
        k_n_heads_t,
        head_dim_t,

        cos_cb_index,
        sin_cb_index,
        trans_mat_cb_index,

        rotated_input_interm_cb_index,
        cos_interm_cb_index,
        sin_interm_cb_index,
    };
    const std::string compute_kernel_path =
        operation_attributes.row_major_QK
            ? "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/"
              "compute/rotary_embedding_llama_sharded_row_major.cpp"
            : "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_llama_fused_qk/device/kernels/"
              "compute/rotary_embedding_llama_sharded.cpp";
    auto rotary_embedding_kernel_id = tt::tt_metal::CreateKernel(
        program,
        compute_kernel_path,
        all_cores_bb,
        tt::tt_metal::ComputeConfig{
            .math_fidelity = math_fidelity, .fp32_dest_acc_en = fp32_dest_acc_en, .compile_args = compute_kernel_args});

    // Runtime args to differentiate between q, k or no work groups
    // TODO: Turn off unused compute cores? (technically, it doesn't matter since only compute kernel)
    // Running into code size issues on TRISC2 with profiler turned on; need to reduce stack size by 4B
    // constexpr bool has_work = true;
    constexpr bool is_q = true;  // If not q, must be k
    tt::tt_metal::SetRuntimeArgs(program, rotary_embedding_kernel_id, q_cores, {is_q});
    tt::tt_metal::SetRuntimeArgs(program, rotary_embedding_kernel_id, k_cores, {!is_q});
    // tt::tt_metal::SetRuntimeArgs(program, rotary_embedding_kernel_id, unused_cores, {!has_work});

    RotaryEmbeddingLlamaFusedQKSharedVariables shared_variables{
        .cb_q_input = cb_q_input,
        .cb_k_input = cb_k_input,
        .cb_cos = cb_cos,
        .cb_sin = cb_sin,
        .cb_trans_mat = cb_trans_mat,
        .cb_q_output = cb_q_output,
        .cb_k_output = cb_k_output};

    // NOLINTNEXTLINE(performance-move-const-arg): CachedProgram ctor requires rvalue reference
    return cached_program_t{std::move(program), std::move(shared_variables)};
}

void RotaryEmbeddingLlamaFusedQKProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const RotaryEmbeddingLlamaFusedQkParams& /*operation_attributes*/,
    const RotaryEmbeddingLlamaFusedQkInputs& tensor_args,
    RotaryEmbeddingLlamaFusedQkResult& tensor_return_value) {
    auto& program = cached_program.program;
    const auto& shared_variables = cached_program.shared_variables;

    auto* q_src_buffer = tensor_args.q_input.buffer();
    auto* k_src_buffer = tensor_args.k_input.buffer();
    auto* cos_buffer = tensor_args.cos.buffer();
    auto* sin_buffer = tensor_args.sin.buffer();
    auto* trans_mat_buffer = tensor_args.trans_mat.buffer();
    auto* q_dst_buffer = std::get<0>(tensor_return_value).buffer();
    auto* k_dst_buffer = std::get<1>(tensor_return_value).buffer();

    // Update the CB globally allocated addresses here
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_q_input, *q_src_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_k_input, *k_src_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_cos, *cos_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_sin, *sin_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_trans_mat, *trans_mat_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_q_output, *q_dst_buffer);
    UpdateDynamicCircularBufferAddress(program, shared_variables.cb_k_output, *k_dst_buffer);
}

}  // namespace ttnn::experimental::prim
