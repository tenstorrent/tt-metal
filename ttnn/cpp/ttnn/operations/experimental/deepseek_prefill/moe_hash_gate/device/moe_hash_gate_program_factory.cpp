// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_hash_gate_device_operation.hpp"
#include <bit>
#include <algorithm>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_align.hpp>
#include "ttnn/operations/cb_utils.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::moe_hash_gate {

MoeHashGateDeviceOperation::ProgramFactory::cached_program_t MoeHashGateDeviceOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& scores = tensor_args.scores;
    const auto& input_ids = tensor_args.input_ids;
    const auto& tid2eid = tensor_args.tid2eid;
    auto& output_weights = tensor_return_value[0];
    auto& output_indices = tensor_return_value[1];

    TT_FATAL(output_weights.dtype() == DataType::BFLOAT16, "Output weights tensor must be BFLOAT16");
    TT_FATAL(output_weights.layout() == Layout::TILE, "Output weights tensor must be TILE layout");
    TT_FATAL(output_indices.dtype() == DataType::UINT16, "Output indices tensor must be UINT16");
    TT_FATAL(output_indices.layout() == Layout::TILE, "Output indices tensor must be TILE layout");

    auto* device = scores.device();
    TT_FATAL(device != nullptr, "Device must be non-null");
    tt::tt_metal::Program program{};

    auto grid = device->compute_with_storage_grid_size();
    auto num_tiles = scores.buffer()->num_pages();
    uint32_t tile_width = scores.tensor_spec().page_config().get_tile().get_width();
    uint32_t tile_height = scores.tensor_spec().page_config().get_tile().get_height();
    auto width_tiles = scores.padded_shape()[-1] / tile_width;
    auto height_tiles = num_tiles / width_tiles;
    uint32_t experts = scores.logical_shape()[-1];
    uint32_t tokens = scores.logical_shape().volume() / experts;
    uint32_t seq_len = scores.logical_shape()[-2];

    log_debug(tt::LogOp, "moe_hash_gate height_tiles: {} width_tiles: {}", height_tiles, width_tiles);

    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_height_tiles_per_core_group_1,
         num_height_tiles_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid, height_tiles);

    uint32_t remainder_tokens_per_tile = seq_len % tile_height == 0 ? tile_height : seq_len % tile_height;
    uint32_t seq_len_tiles = tt::div_up(seq_len, tile_height);
    uint32_t n_activated_expert_tiles = tt::div_up(operation_attributes.n_activated_experts, 32);

    auto scores_data_format = tt::tt_metal::datatype_to_dataformat_converter(scores.dtype());
    auto weights_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_weights.dtype());
    auto indices_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_indices.dtype());

    uint32_t scores_page_size = scores.buffer()->page_size();
    uint32_t weights_page_size = output_weights.buffer()->page_size();
    uint32_t indices_page_size = output_indices.buffer()->page_size();
    uint32_t input_ids_page_size = input_ids.buffer()->page_size();
    uint32_t tid2eid_page_size = tid2eid.buffer()->page_size();
    // Each per-token tid2eid row lookup is a separate DRAM->L1 NoC read; the destination L1 offset
    // must be DRAM-aligned or the read silently drops (odd rows landing at 32B offsets read as zero).
    // Stride the scratch by the aligned row size so every token's destination is aligned.
    uint32_t dram_alignment = tt::tt_metal::hal::get_dram_alignment();
    uint32_t tid2eid_row_stride = tt::align(tid2eid_page_size, dram_alignment);

    // --- Circular buffers (only the activation/normalize/scale + fused-lookup subset) ---
    auto cb_in_scores = tt::CBIndex::c_0;
    auto cb_input_ids = tt::CBIndex::c_1;
    auto cb_out_weights = tt::CBIndex::c_2;
    auto cb_out_indices = tt::CBIndex::c_3;
    auto cb_sigmoid_scores = tt::CBIndex::c_4;
    auto cb_reduce_intermediate = tt::CBIndex::c_5;
    auto cb_reduce_ones_scalar = tt::CBIndex::c_6;
    auto cb_epsilon_scalar = tt::CBIndex::c_7;
    auto cb_route_scale_scalar = tt::CBIndex::c_8;
    auto cb_normalized_scores = tt::CBIndex::c_9;
    auto cb_reciprocal_sums = tt::CBIndex::c_10;
    auto cb_gathered_sigmoid = tt::CBIndex::c_11;
    auto cb_padding_config = tt::CBIndex::c_12;
    auto cb_tid2eid_row = tt::CBIndex::c_13;

    tt::tt_metal::create_cb(cb_in_scores, program, all_cores, scores_page_size, 2 * width_tiles, scores_data_format);
    tt::tt_metal::create_cb(
        cb_out_weights, program, all_cores, weights_page_size, 2 * n_activated_expert_tiles, weights_data_format);
    tt::tt_metal::create_cb(
        cb_out_indices, program, all_cores, indices_page_size, 2 * n_activated_expert_tiles, indices_data_format);
    tt::tt_metal::create_cb(cb_sigmoid_scores, program, all_cores, scores_page_size, width_tiles, scores_data_format);
    tt::tt_metal::create_cb(
        cb_reduce_intermediate, program, all_cores, scores_page_size, 2 * n_activated_expert_tiles, scores_data_format);
    tt::tt_metal::create_cb(cb_reduce_ones_scalar, program, all_cores, scores_page_size, 1, scores_data_format);
    tt::tt_metal::create_cb(cb_epsilon_scalar, program, all_cores, scores_page_size, 1, scores_data_format);
    tt::tt_metal::create_cb(cb_route_scale_scalar, program, all_cores, scores_page_size, 1, scores_data_format);
    tt::tt_metal::create_cb(
        cb_normalized_scores, program, all_cores, scores_page_size, 2 * n_activated_expert_tiles, scores_data_format);
    tt::tt_metal::create_cb(
        cb_reciprocal_sums, program, all_cores, scores_page_size, 2 * n_activated_expert_tiles, scores_data_format);
    tt::tt_metal::create_cb(
        cb_gathered_sigmoid, program, all_cores, scores_page_size, 2 * n_activated_expert_tiles, scores_data_format);

    // input_ids: one ROW_MAJOR page per height tile (tile_height uint32 token ids).
    tt::tt_metal::create_cb(cb_input_ids, program, all_cores, input_ids_page_size, 2, tt::DataFormat::UInt32);
    // tid2eid scratch: hold all tile_height looked-up rows for a tile before assembling the index tile.
    // Rows are strided by the DRAM-aligned size so each per-token read lands at an aligned destination.
    tt::tt_metal::create_cb(
        cb_tid2eid_row, program, all_cores, tile_height * tid2eid_row_stride, 1, tt::DataFormat::UInt16);

    // Scratch CB for the optional [num_real_tokens, pad_side] padding config row (see writer). When no
    // padding config is supplied we fall back to the output_indices buffer purely to size the CB.
    auto* padding_config_buffer =
        tensor_args.padding_config.has_value() ? tensor_args.padding_config->buffer() : output_indices.buffer();
    uint32_t padding_config_page_size = static_cast<uint32_t>(padding_config_buffer->aligned_page_size());
    tt::tt_metal::create_cb(cb_padding_config, program, all_cores, padding_config_page_size, 1, tt::DataFormat::UInt32);

    // --- Reader: logits -> cb_in_scores; tid2eid[input_ids] -> cb_out_indices ---
    std::unordered_map<std::string, uint32_t> reader_named_compile_time_args = {
        {"cb_in_scores", cb_in_scores},
        {"cb_out_indices", cb_out_indices},
        {"cb_input_ids", cb_input_ids},
        {"cb_tid2eid_row", cb_tid2eid_row},
        {"width_tiles", width_tiles},
        {"scores_page_size", scores_page_size},
        {"input_ids_page_size", input_ids_page_size},
        {"tid2eid_page_size", tid2eid_page_size},
        {"tid2eid_row_stride", tid2eid_row_stride},
        {"n_activated_experts", operation_attributes.n_activated_experts},
        {"n_activated_expert_tiles", n_activated_expert_tiles},
        {"tile_height", tile_height},
        {"seq_len_tiles", seq_len_tiles},
        {"remainder_tokens_per_tile", remainder_tokens_per_tile},
    };

    std::vector<uint32_t> reader_compile_time_args = {};
    tt::tt_metal::TensorAccessorArgs(scores.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(input_ids.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(tid2eid.buffer()).append_to(reader_compile_time_args);

    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_hash_gate/device/kernels/dataflow/"
        "reader_moe_hash_gate.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, {}, reader_named_compile_time_args));

    // --- Compute: apply_score_func -> normalize_scores -> scale ---
    std::unordered_map<std::string, uint32_t> compute_named_compile_time_args = {
        {"cb_in_scores", cb_in_scores},
        {"cb_sigmoid_scores", cb_sigmoid_scores},
        {"cb_out_weights", cb_out_weights},
        {"width_tiles", width_tiles},
        {"cb_reduce_intermediate", cb_reduce_intermediate},
        {"cb_reduce_ones_scalar", cb_reduce_ones_scalar},
        {"cb_epsilon_scalar", cb_epsilon_scalar},
        {"cb_route_scale_scalar", cb_route_scale_scalar},
        {"cb_normalized_scores", cb_normalized_scores},
        {"cb_reciprocal_sums", cb_reciprocal_sums},
        {"cb_gathered_sigmoid", cb_gathered_sigmoid},
        {"score_func", static_cast<uint32_t>(operation_attributes.score_func)},
    };

    std::vector<uint32_t> compute_compile_time_args = {};
    bool fp32_dest_acc_en = true;
    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_hash_gate/device/kernels/compute/"
        "moe_hash_gate.cpp",
        all_cores,
        ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_compile_time_args,
            .named_compile_args = compute_named_compile_time_args});

    // --- Writer: gather + sentinel patch + write ---
    std::unordered_map<std::string, uint32_t> writer_named_compile_time_args = {
        {"cb_out_weights", cb_out_weights},
        {"cb_out_indices", cb_out_indices},
        {"cb_sigmoid_scores", cb_sigmoid_scores},
        {"cb_gathered_sigmoid", cb_gathered_sigmoid},
        {"cb_reduce_ones_scalar", cb_reduce_ones_scalar},
        {"cb_epsilon_scalar", cb_epsilon_scalar},
        {"cb_route_scale_scalar", cb_route_scale_scalar},
        {"cb_padding_config", cb_padding_config},
        {"scores_page_size", scores_page_size},
        {"weights_page_size", weights_page_size},
        {"indices_page_size", indices_page_size},
        {"experts", experts},
        {"width_tiles", width_tiles},
        {"tile_height", tile_height},
        {"tokens", tokens},
        {"n_activated_experts", operation_attributes.n_activated_experts},
        {"n_activated_expert_tiles", n_activated_expert_tiles},
        {"packed_one_scalar", std::bit_cast<uint32_t>(1.0f)},
        {"packed_epsilon", std::bit_cast<uint32_t>(operation_attributes.epsilon)},
        {"packed_route_scale", std::bit_cast<uint32_t>(operation_attributes.route_scale)},
        {"seq_len_tiles", seq_len_tiles},
        {"remainder_tokens_per_tile", remainder_tokens_per_tile},
    };

    std::vector<uint32_t> writer_compile_time_args = {};
    tt::tt_metal::TensorAccessorArgs(output_weights.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_indices.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(padding_config_buffer).append_to(writer_compile_time_args);

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_hash_gate/device/kernels/dataflow/"
        "writer_moe_hash_gate.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, {}, writer_named_compile_time_args));

    std::vector<uint32_t> reader_runtime_args = {
        scores.buffer()->address(), input_ids.buffer()->address(), tid2eid.buffer()->address(), 0, 0};
    std::vector<uint32_t> compute_runtime_args = {0, 0};
    std::vector<uint32_t> writer_runtime_args = {
        output_weights.buffer()->address(),
        output_indices.buffer()->address(),
        0,
        0,
        tensor_args.padding_config.has_value() ? padding_config_buffer->address() : 0};

    uint32_t start_height_tile = 0;
    uint32_t end_height_tile = 0;
    auto cores = corerange_to_cores(all_cores, std::nullopt);
    for (const auto& core : cores) {
        uint32_t workload_per_core = 0;
        if (core_group_1.contains(core)) {
            workload_per_core = num_height_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            workload_per_core = num_height_tiles_per_core_group_2;
        } else {
            workload_per_core = 0;
        }
        start_height_tile = end_height_tile;
        end_height_tile = start_height_tile + workload_per_core;

        reader_runtime_args[3] = start_height_tile;
        reader_runtime_args[4] = end_height_tile;

        compute_runtime_args[0] = start_height_tile;
        compute_runtime_args[1] = end_height_tile;

        writer_runtime_args[2] = start_height_tile;
        writer_runtime_args[3] = end_height_tile;

        tt::tt_metal::SetRuntimeArgs(program, reader_kernel_id, core, reader_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, compute_kernel_id, core, compute_runtime_args);
        tt::tt_metal::SetRuntimeArgs(program, writer_kernel_id, core, writer_runtime_args);
    }

    return {std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id, cores}};
}

void MoeHashGateDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& /*operation_attributes*/,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    auto& program = cached_program.program;
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    auto& cores = cached_program.shared_variables.cores;
    for (const auto& core : cores) {
        auto& reader_runtime_args = tt::tt_metal::GetRuntimeArgs(program, reader_kernel_id, core);
        reader_runtime_args[0] = tensor_args.scores.buffer()->address();
        reader_runtime_args[1] = tensor_args.input_ids.buffer()->address();
        reader_runtime_args[2] = tensor_args.tid2eid.buffer()->address();
        auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
        writer_runtime_args[0] = tensor_return_value[0].buffer()->address();
        writer_runtime_args[1] = tensor_return_value[1].buffer()->address();
        writer_runtime_args[4] =
            tensor_args.padding_config.has_value() ? tensor_args.padding_config->buffer()->address() : 0;
    }
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_hash_gate
