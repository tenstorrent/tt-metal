// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_grouped_gate_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"

namespace ttnn::operations::experimental::reduction {

DeepseekGroupedGateDeviceOperation::ProgramFactory::cached_program_t
DeepseekGroupedGateDeviceOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& scores = tensor_args.scores;
    const auto& bias = tensor_args.bias;
    auto& output_weights = tensor_return_value[0];
    auto& output_indices = tensor_return_value[1];

    TT_FATAL(output_weights.dtype() == DataType::BFLOAT16, "Output weights tensor must be BFLOAT16");
    TT_FATAL(output_weights.layout() == Layout::TILE, "Output weights tensor must be TILE layout");
    TT_FATAL(output_indices.dtype() == DataType::UINT16, "Output indices tensor must be UINT16");
    TT_FATAL(output_indices.layout() == Layout::TILE, "Output indices tensor must be TILE layout");

    auto* device = scores.device();
    TT_FATAL(device != nullptr, "Device must be non-null");
    // Create program
    tt::tt_metal::Program program{};

    auto grid = device->compute_with_storage_grid_size();
    auto num_tiles = scores.buffer()->num_pages();
    uint32_t tile_width = scores.tensor_spec().page_config().get_tile().get_width();
    uint32_t tile_height = scores.tensor_spec().page_config().get_tile().get_height();
    auto width_tiles = scores.padded_shape()[-1] / scores.tensor_spec().page_config().get_tile().get_width();
    auto height_tiles = num_tiles / width_tiles;
    uint32_t experts = scores.logical_shape()[-1];
    uint32_t tokens = scores.logical_shape().volume() / experts;
    uint32_t seq_len = scores.logical_shape()[-2];

    log_debug(tt::LogOp, "height_tiles: {} width_tiles: {}", height_tiles, width_tiles);

    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_height_tiles_per_core_group_1,
         num_height_tiles_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid, height_tiles);

    uint32_t remainder_tokens_per_tile = seq_len % tile_height == 0 ? tile_height : seq_len % tile_height;

    // Input/output circular buffers
    auto cb_in_scores = tt::CBIndex::c_0;
    auto cb_in_bias = tt::CBIndex::c_1;
    auto cb_out_weights = tt::CBIndex::c_2;
    auto cb_out_indices = tt::CBIndex::c_3;

    auto scores_data_format = tt::tt_metal::datatype_to_dataformat_converter(scores.dtype());
    auto bias_data_format = tt::tt_metal::datatype_to_dataformat_converter(bias.dtype());
    auto weights_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_weights.dtype());
    auto indices_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_indices.dtype());

    uint32_t n_activated_expert_tiles = tt::div_up(operation_attributes.n_activated_experts, 32);
    // Deadlock prevention for bias CB:
    //   - Bias needs width_tiles capacity because:
    //       * The add_bias stage does not consume bias until ALL sigmoid tiles are ready.
    //       * The reader pushes scores and bias together.
    //   - If the bias CB is too small, the reader can block, causing a deadlock.
    tt::tt_metal::create_cb(
        cb_in_scores, program, all_cores, scores.buffer()->page_size(), 2 * width_tiles, scores_data_format);
    tt::tt_metal::create_cb(
        cb_in_bias, program, all_cores, bias.buffer()->page_size(), 2 * width_tiles, bias_data_format);
    tt::tt_metal::create_cb(
        cb_out_weights,
        program,
        all_cores,
        output_weights.buffer()->page_size(),
        2 * n_activated_expert_tiles,
        weights_data_format);
    tt::tt_metal::create_cb(
        cb_out_indices,
        program,
        all_cores,
        output_indices.buffer()->page_size(),
        2 * n_activated_expert_tiles,
        indices_data_format);

    // Sigmoid output and biased scores CBs
    // Note: cb_sigmoid_scores needs width_tiles capacity since add_bias waits for all tiles at once
    // and writer also needs all tiles. Don't double-buffer - it causes L1 memory pressure.
    auto cb_sigmoid_scores = tt::CBIndex::c_4;
    auto cb_biased_scores = tt::CBIndex::c_5;
    tt::tt_metal::create_cb(
        cb_sigmoid_scores, program, all_cores, scores.buffer()->page_size(), width_tiles, scores_data_format);
    tt::tt_metal::create_cb(
        cb_biased_scores, program, all_cores, scores.buffer()->page_size(), width_tiles, scores_data_format);

    // Per-group sorting CBs
    // cb_sorted_group_scores is consumed one tile at a time by writer's generate_summed_experts_tiles
    // cb_sorted_expert_indices_temp is used transiently and popped immediately in process_and_sort_tiles
    // cb_expert_index_template needs all width_tiles for generate_winning_group_tiles
    auto cb_sorted_group_scores = tt::CBIndex::c_6;
    auto cb_sorted_expert_indices_temp = tt::CBIndex::c_7;
    auto cb_expert_index_template = tt::CBIndex::c_8;
    tt::tt_metal::create_cb(
        cb_sorted_group_scores, program, all_cores, scores.buffer()->page_size(), 2, scores_data_format);
    tt::tt_metal::create_cb(
        cb_sorted_expert_indices_temp, program, all_cores, scores.buffer()->page_size(), 2, tt::DataFormat::UInt16);
    tt::tt_metal::create_cb(
        cb_expert_index_template,
        program,
        all_cores,
        scores.buffer()->page_size(),
        width_tiles,
        tt::DataFormat::UInt16);

    // Group selection CBs
    uint32_t num_group_tiles = tt::div_up(operation_attributes.n_groups, 32);
    auto cb_group_index_template = tt::CBIndex::c_9;
    auto cb_group_summed_scores = tt::CBIndex::c_10;
    auto cb_top_experts_per_group = tt::CBIndex::c_11;
    auto cb_sorted_group_order = tt::CBIndex::c_12;
    tt::tt_metal::create_cb(
        cb_group_index_template,
        program,
        all_cores,
        scores.buffer()->page_size(),
        num_group_tiles,
        tt::DataFormat::UInt16);
    tt::tt_metal::create_cb(
        cb_top_experts_per_group,
        program,
        all_cores,
        scores.buffer()->page_size(),
        operation_attributes.summed_experts_per_group,
        scores_data_format);
    tt::tt_metal::create_cb(
        cb_group_summed_scores, program, all_cores, scores.buffer()->page_size(), num_group_tiles, scores_data_format);
    tt::tt_metal::create_cb(
        cb_sorted_group_order,
        program,
        all_cores,
        output_indices.buffer()->page_size(),
        num_group_tiles,
        tt::DataFormat::UInt16);

    // Winning group processing CBs
    auto cb_winning_group_scores = tt::CBIndex::c_13;
    auto cb_winning_group_indices = tt::CBIndex::c_14;
    tt::tt_metal::create_cb(
        cb_winning_group_scores,
        program,
        all_cores,
        output_weights.buffer()->page_size(),
        operation_attributes.topk_groups,
        scores_data_format);
    tt::tt_metal::create_cb(
        cb_winning_group_indices,
        program,
        all_cores,
        output_indices.buffer()->page_size(),
        operation_attributes.topk_groups,
        tt::DataFormat::UInt16);

    // Final topk intermediate CBs
    auto cb_reduce_intermediate = tt::CBIndex::c_15;
    auto cb_final_indices_transposed = tt::CBIndex::c_16;
    tt::tt_metal::create_cb(
        cb_reduce_intermediate,
        program,
        all_cores,
        scores.buffer()->page_size(),
        2 * n_activated_expert_tiles,
        scores_data_format);
    tt::tt_metal::create_cb(
        cb_final_indices_transposed,
        program,
        all_cores,
        output_indices.buffer()->page_size(),
        2 * n_activated_expert_tiles,
        tt::DataFormat::UInt16);

    // Normalization scalar CBs
    auto cb_reduce_ones_scalar = tt::CBIndex::c_17;
    tt::tt_metal::create_cb(
        cb_reduce_ones_scalar, program, all_cores, scores.buffer()->page_size(), 1, scores_data_format);

    auto cb_epsilon_scalar = tt::CBIndex::c_18;
    tt::tt_metal::create_cb(cb_epsilon_scalar, program, all_cores, scores.buffer()->page_size(), 1, scores_data_format);

    auto cb_route_scale_scalar = tt::CBIndex::c_19;
    tt::tt_metal::create_cb(
        cb_route_scale_scalar, program, all_cores, scores.buffer()->page_size(), 1, scores_data_format);

    // Normalization intermediate CBs
    auto cb_normalized_scores = tt::CBIndex::c_20;
    tt::tt_metal::create_cb(
        cb_normalized_scores,
        program,
        all_cores,
        scores.buffer()->page_size(),
        2 * n_activated_expert_tiles,
        scores_data_format);

    auto cb_reciprocal_sums = tt::CBIndex::c_21;
    tt::tt_metal::create_cb(
        cb_reciprocal_sums,
        program,
        all_cores,
        scores.buffer()->page_size(),
        2 * n_activated_expert_tiles,
        scores_data_format);

    // Gather CB for selected expert sigmoid scores
    auto cb_gathered_sigmoid = tt::CBIndex::c_22;
    tt::tt_metal::create_cb(
        cb_gathered_sigmoid,
        program,
        all_cores,
        scores.buffer()->page_size(),
        2 * n_activated_expert_tiles,
        scores_data_format);

    // Reader kernel compile time arguments
    std::unordered_map<std::string, uint32_t> reader_named_compile_time_args = {
        {"cb_in_scores", cb_in_scores},
        {"cb_in_bias", cb_in_bias},
        {"cb_route_scale_scalar", cb_route_scale_scalar},
        {"width_tiles", width_tiles},
        {"scores_page_size", scores.buffer()->page_size()},
        {"bias_page_size", bias.buffer()->page_size()},
    };

    std::vector<uint32_t> reader_compile_time_args = {};
    tt::tt_metal::TensorAccessorArgs(scores.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(bias.buffer()).append_to(reader_compile_time_args);

    // Reader kernel
    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/kernels/dataflow/"
        "reader_deepseek_grouped_gate.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, {}, reader_named_compile_time_args));

    // Compute kernel compile time arguments
    std::unordered_map<std::string, uint32_t> compute_named_compile_time_args = {
        {"cb_in_scores", cb_in_scores},
        {"cb_in_bias", cb_in_bias},
        {"cb_sigmoid_scores", cb_sigmoid_scores},
        {"cb_biased_scores", cb_biased_scores},
        {"cb_out_weights", cb_out_weights},
        {"cb_out_indices", cb_out_indices},
        {"cb_group_index_template", cb_group_index_template},
        {"cb_group_summed_scores", cb_group_summed_scores},
        {"cb_top_experts_per_group", cb_top_experts_per_group},
        {"cb_sorted_group_order", cb_sorted_group_order},
        {"width_tiles", width_tiles},
        {"scores_page_size", scores.buffer()->page_size()},
        {"bias_page_size", bias.buffer()->page_size()},
        {"weights_page_size", output_weights.buffer()->page_size()},
        {"indices_page_size", output_indices.buffer()->page_size()},
        {"cb_sorted_group_scores", cb_sorted_group_scores},
        {"cb_sorted_expert_indices_temp", cb_sorted_expert_indices_temp},
        {"cb_expert_index_template", cb_expert_index_template},
        {"group_size", experts / operation_attributes.n_groups},
        {"log_group_size", std::log2(experts / operation_attributes.n_groups)},
        {"summed_experts_per_group", operation_attributes.summed_experts_per_group},
        {"topk_groups", operation_attributes.topk_groups},
        {"n_groups", operation_attributes.n_groups},
        {"log_topk_groups", std::log2(operation_attributes.topk_groups)},
        {"log_n_groups", std::log2(operation_attributes.n_groups)},
        {"cb_winning_group_scores", cb_winning_group_scores},
        {"cb_winning_group_indices", cb_winning_group_indices},
        {"num_group_tiles", num_group_tiles},
        {"n_activated_experts", operation_attributes.n_activated_experts},
        {"n_activated_expert_tiles", n_activated_expert_tiles},
        {"log_n_activated_experts", std::log2(operation_attributes.n_activated_experts)},
        {"cb_reduce_intermediate", cb_reduce_intermediate},
        {"cb_final_indices_transposed", cb_final_indices_transposed},
        {"cb_reduce_ones_scalar", cb_reduce_ones_scalar},
        {"cb_epsilon_scalar", cb_epsilon_scalar},
        {"cb_route_scale_scalar", cb_route_scale_scalar},
        {"cb_normalized_scores", cb_normalized_scores},
        {"cb_reciprocal_sums", cb_reciprocal_sums},
        {"cb_gathered_sigmoid", cb_gathered_sigmoid},
    };

    std::vector<uint32_t> compute_compile_time_args = {};

    // Compute kernel
    bool fp32_dest_acc_en = false;  // Needed for topK to work with uint16_t indices
    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/kernels/compute/"
        "deepseek_grouped_gate.cpp",
        all_cores,
        ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_compile_time_args,
            .named_compile_args = compute_named_compile_time_args});

    // Writer kernel

    // Writer kernel compile time arguments
    std::unordered_map<std::string, uint32_t> writer_named_compile_time_args = {
        {"cb_out_weights", cb_out_weights},
        {"cb_out_indices", cb_out_indices},
        {"cb_expert_index_template", cb_expert_index_template},
        {"cb_group_index_template", cb_group_index_template},
        {"cb_top_experts_per_group", cb_top_experts_per_group},
        {"cb_gathered_sigmoid", cb_gathered_sigmoid},
        {"cb_sorted_group_scores", cb_sorted_group_scores},
        {"weights_page_size", output_weights.buffer()->page_size()},
        {"indices_page_size", output_indices.buffer()->page_size()},
        {"experts", experts},
        {"width_tiles", width_tiles},
        {"tile_width", tile_width},
        {"tile_height", tile_height},
        {"tokens", tokens},
        {"topk_groups", operation_attributes.topk_groups},
        {"n_groups", operation_attributes.n_groups},
        {"summed_experts_per_group", operation_attributes.summed_experts_per_group},
        {"cb_winning_group_scores", cb_winning_group_scores},
        {"cb_winning_group_indices", cb_winning_group_indices},
        {"num_group_tiles", num_group_tiles},
        {"cb_sorted_group_order", cb_sorted_group_order},
        {"cb_in_scores", cb_in_scores},
        {"cb_sigmoid_scores", cb_sigmoid_scores},
        {"cb_biased_scores", cb_biased_scores},
        {"cb_reduce_ones_scalar", cb_reduce_ones_scalar},
        {"n_activated_experts", operation_attributes.n_activated_experts},
        {"packed_one_scalar", static_cast<uint32_t>(std::bit_cast<uint16_t>(bfloat16(1.0f))) << 16},
        {"packed_epsilon",
         static_cast<uint32_t>(std::bit_cast<uint16_t>(bfloat16(operation_attributes.epsilon))) << 16},
        {"packed_route_scale",
         static_cast<uint32_t>(std::bit_cast<uint16_t>(bfloat16(operation_attributes.route_scale))) << 16},
        {"cb_epsilon_scalar", cb_epsilon_scalar},
        {"cb_route_scale_scalar", cb_route_scale_scalar},
        {"seq_len_tiles", tt::div_up(seq_len, tile_height)},
        {"remainder_tokens_per_tile", remainder_tokens_per_tile},
        {"n_activated_expert_tiles", n_activated_expert_tiles},
    };

    std::vector<uint32_t> writer_compile_time_args = {};
    tt::tt_metal::TensorAccessorArgs(output_weights.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(output_indices.buffer()).append_to(writer_compile_time_args);

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/kernels/dataflow/"
        "writer_deepseek_grouped_gate.cpp",
        all_cores,
        tt::tt_metal::WriterDataMovementConfig(writer_compile_time_args, {}, writer_named_compile_time_args));

    std::vector<uint32_t> reader_runtime_args = {scores.buffer()->address(), bias.buffer()->address(), 0, 0};
    std::vector<uint32_t> compute_runtime_args = {0, 0};
    std::vector<uint32_t> writer_runtime_args = {
        output_weights.buffer()->address(), output_indices.buffer()->address(), 0, 0};

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
            // no-op
            workload_per_core = 0;
        }
        start_height_tile = end_height_tile;
        end_height_tile = start_height_tile + workload_per_core;

        reader_runtime_args[2] = start_height_tile;
        reader_runtime_args[3] = end_height_tile;

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

void DeepseekGroupedGateDeviceOperation::ProgramFactory::override_runtime_arguments(
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
        reader_runtime_args[1] = tensor_args.bias.buffer()->address();
        auto& writer_runtime_args = tt::tt_metal::GetRuntimeArgs(program, writer_kernel_id, core);
        writer_runtime_args[0] = tensor_return_value[0].buffer()->address();
        writer_runtime_args[1] = tensor_return_value[1].buffer()->address();
    }
}

}  // namespace ttnn::operations::experimental::reduction
