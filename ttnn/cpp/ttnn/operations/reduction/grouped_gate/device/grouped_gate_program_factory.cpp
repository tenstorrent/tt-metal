// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_gate_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/cb_utils.hpp"

namespace ttnn::operations::reduction {

GroupedGateDeviceOperation::ProgramFactory::cached_program_t GroupedGateDeviceOperation::ProgramFactory::create(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& scores = tensor_args.scores;
    const auto& bias = tensor_args.bias;
    auto& output_weights = tensor_return_value[0];
    auto& output_indices = tensor_return_value[1];

    TT_FATAL(scores.dtype() == DataType::BFLOAT16, "Scores tensor must be BFLOAT16");
    TT_FATAL(scores.layout() == Layout::TILE, "Scores tensor must be TILE layout");
    TT_FATAL(bias.dtype() == DataType::BFLOAT16, "Bias tensor must be BFLOAT16");
    TT_FATAL(bias.layout() == Layout::TILE, "Bias tensor must be TILE layout");
    TT_FATAL(output_weights.dtype() == DataType::BFLOAT16, "Output weights tensor must be BFLOAT16");
    TT_FATAL(output_weights.layout() == Layout::TILE, "Output weights tensor must be TILE layout");
    TT_FATAL(output_indices.dtype() == DataType::UINT16, "Output indices tensor must be UINT16");
    TT_FATAL(output_indices.layout() == Layout::TILE, "Output indices tensor must be TILE layout");

    auto device = scores.device();
    TT_FATAL(device != nullptr, "Device must be non-null");
    // Create program
    tt::tt_metal::Program program{};

    // auto grid = device->compute_with_storage_grid_size();
    auto grid = CoreCoord(1, 1);
    auto num_tiles = scores.buffer()->num_pages();
    log_info(tt::LogAlways, "num_tiles: {}", num_tiles);
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

    auto scores_cb_index = tt::CBIndex::c_0;
    auto bias_cb_index = tt::CBIndex::c_1;
    auto weights_cb_index = tt::CBIndex::c_2;
    auto indices_cb_index = tt::CBIndex::c_3;
    // input/output circular buffers
    auto scores_data_format = tt::tt_metal::datatype_to_dataformat_converter(scores.dtype());
    auto bias_data_format = tt::tt_metal::datatype_to_dataformat_converter(bias.dtype());
    auto weights_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_weights.dtype());
    auto indices_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_indices.dtype());

    uint32_t n_activated_expert_tiles = tt::div_up(operation_attributes.n_activated_experts, 32);
    // Scores are streamed one tile at a time (double-buffered with 2 tiles)
    // Bias needs width_tiles capacity because add_bias doesn't consume bias until ALL sigmoid tiles are ready,
    // but reader pushes scores+bias together - if bias CB is too small, reader blocks causing deadlock
    tt::tt_metal::create_cb(scores_cb_index, program, all_cores, scores.buffer()->page_size(), 2, scores_data_format);
    tt::tt_metal::create_cb(
        bias_cb_index, program, all_cores, bias.buffer()->page_size(), width_tiles, bias_data_format);
    tt::tt_metal::create_cb(
        weights_cb_index,
        program,
        all_cores,
        output_weights.buffer()->page_size(),
        2 * n_activated_expert_tiles,
        weights_data_format);
    tt::tt_metal::create_cb(
        indices_cb_index,
        program,
        all_cores,
        output_indices.buffer()->page_size(),
        2 * n_activated_expert_tiles,
        indices_data_format);

    // sigmoid input + add bias block CBs
    // Note: sigmoid_input needs width_tiles capacity since add_bias waits for all tiles at once
    // and writer also needs all tiles. Don't double-buffer - it causes L1 memory pressure.
    auto sigmoid_input_cb_index = tt::CBIndex::c_4;
    auto add_bias_cb_index = tt::CBIndex::c_5;
    tt::tt_metal::create_cb(
        sigmoid_input_cb_index, program, all_cores, scores.buffer()->page_size(), width_tiles, scores_data_format);
    tt::tt_metal::create_cb(
        add_bias_cb_index, program, all_cores, scores.buffer()->page_size(), width_tiles, scores_data_format);

    // topk intermediate CBs
    // topk_input_cb is consumed one tile at a time by writer's generate_summed_experts_tiles
    // topk_index_cb is used transiently and popped immediately in process_and_sort_tiles
    // topk_index_creation_cb needs all width_tiles for generate_winning_group_tiles
    auto topk_input_cb_index = tt::CBIndex::c_6;
    auto topk_index_cb_index = tt::CBIndex::c_7;
    auto topk_index_creation_cb_index = tt::CBIndex::c_8;
    tt::tt_metal::create_cb(
        topk_input_cb_index, program, all_cores, scores.buffer()->page_size(), 2, scores_data_format);
    tt::tt_metal::create_cb(
        topk_index_cb_index, program, all_cores, scores.buffer()->page_size(), 2, tt::DataFormat::UInt16);
    tt::tt_metal::create_cb(
        topk_index_creation_cb_index,
        program,
        all_cores,
        scores.buffer()->page_size(),
        width_tiles,
        tt::DataFormat::UInt16);

    // Generated group indices, intermediate CB for (summed_experts_per_group - 1) CBs, and one for the final summed
    // scores
    uint32_t num_group_tiles = tt::div_up(operation_attributes.n_groups, 32);
    auto group_indices_cb_index = tt::CBIndex::c_9;
    auto group_scores_cb_index = tt::CBIndex::c_10;
    auto summed_experts_cb_index = tt::CBIndex::c_11;
    auto sorted_group_indices_cb_index = tt::CBIndex::c_12;
    tt::tt_metal::create_cb(
        group_indices_cb_index,
        program,
        all_cores,
        scores.buffer()->page_size(),
        num_group_tiles,
        tt::DataFormat::UInt16);
    tt::tt_metal::create_cb(
        summed_experts_cb_index,
        program,
        all_cores,
        scores.buffer()->page_size(),
        operation_attributes.summed_experts_per_group,
        scores_data_format);
    tt::tt_metal::create_cb(
        group_scores_cb_index, program, all_cores, scores.buffer()->page_size(), num_group_tiles, scores_data_format);
    tt::tt_metal::create_cb(
        sorted_group_indices_cb_index,
        program,
        all_cores,
        output_indices.buffer()->page_size(),
        num_group_tiles,
        tt::DataFormat::UInt16);

    // Expert scores and indices in the winning groups
    auto winning_group_scores_cb_index = tt::CBIndex::c_13;
    auto winning_group_indices_cb_index = tt::CBIndex::c_14;
    tt::tt_metal::create_cb(
        winning_group_scores_cb_index,
        program,
        all_cores,
        output_weights.buffer()->page_size(),
        operation_attributes.topk_groups,
        scores_data_format);
    tt::tt_metal::create_cb(
        winning_group_indices_cb_index,
        program,
        all_cores,
        output_indices.buffer()->page_size(),
        operation_attributes.topk_groups,
        tt::DataFormat::UInt16);

    // IntermediateCBs for sorting the expert scores and indices in the winning groups
    auto intermediate_local_sort_cb_index = tt::CBIndex::c_15;
    auto intermediate_local_sort_indices_cb_index = tt::CBIndex::c_16;
    tt::tt_metal::create_cb(
        intermediate_local_sort_cb_index,
        program,
        all_cores,
        scores.buffer()->page_size(),
        2 * n_activated_expert_tiles,
        scores_data_format);
    tt::tt_metal::create_cb(
        intermediate_local_sort_indices_cb_index,
        program,
        all_cores,
        output_indices.buffer()->page_size(),
        2 * n_activated_expert_tiles,
        tt::DataFormat::UInt16);

    // Pre-normalized scores in the winning groups
    auto pre_normalized_scores_cb_index = tt::CBIndex::c_17;
    tt::tt_metal::create_cb(
        pre_normalized_scores_cb_index,
        program,
        all_cores,
        output_weights.buffer()->page_size(),
        2 * n_activated_expert_tiles,
        scores_data_format);

    auto reduce_scalar_cb_index = tt::CBIndex::c_18;
    tt::tt_metal::create_cb(
        reduce_scalar_cb_index, program, all_cores, scores.buffer()->page_size(), 1, scores_data_format);

    auto epsilon_cb_index = tt::CBIndex::c_19;
    tt::tt_metal::create_cb(epsilon_cb_index, program, all_cores, scores.buffer()->page_size(), 1, scores_data_format);

    auto scales_cb_index = tt::CBIndex::c_20;
    tt::tt_metal::create_cb(scales_cb_index, program, all_cores, scores.buffer()->page_size(), 1, scores_data_format);

    auto normalized_cb_index = tt::CBIndex::c_21;
    tt::tt_metal::create_cb(
        normalized_cb_index,
        program,
        all_cores,
        scores.buffer()->page_size(),
        2 * n_activated_expert_tiles,
        scores_data_format);

    auto transpose_cb_index = tt::CBIndex::c_22;
    tt::tt_metal::create_cb(
        transpose_cb_index,
        program,
        all_cores,
        scores.buffer()->page_size(),
        2 * n_activated_expert_tiles,
        scores_data_format);

    auto normalized_transpose_cb_index = tt::CBIndex::c_23;
    tt::tt_metal::create_cb(
        normalized_transpose_cb_index,
        program,
        all_cores,
        scores.buffer()->page_size(),
        2 * n_activated_expert_tiles,
        scores_data_format);

    auto post_sort_transpose_cb_index = tt::CBIndex::c_24;
    tt::tt_metal::create_cb(
        post_sort_transpose_cb_index,
        program,
        all_cores,
        scores.buffer()->page_size(),
        2 * n_activated_expert_tiles,
        scores_data_format);

    // Reader kernel compile time arguments
    std::unordered_map<std::string, uint32_t> reader_named_compile_time_args = {
        {"scores_cb_index", scores_cb_index},
        {"bias_cb_index", bias_cb_index},
        {"scales_cb_index", scales_cb_index},
        {"width_tiles", width_tiles},
        {"scores_page_size", scores.buffer()->page_size()},
        {"bias_page_size", bias.buffer()->page_size()},
    };

    std::vector<uint32_t> reader_compile_time_args = {};
    TensorAccessorArgs(scores.buffer()).append_to(reader_compile_time_args);
    TensorAccessorArgs(bias.buffer()).append_to(reader_compile_time_args);

    // Reader kernel
    auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/grouped_gate/device/kernels/dataflow/reader_grouped_gate.cpp",
        all_cores,
        tt::tt_metal::ReaderDataMovementConfig(reader_compile_time_args, {}, reader_named_compile_time_args));

    // Compute kernel compile time arguments
    std::unordered_map<std::string, uint32_t> compute_named_compile_time_args = {
        {"scores_cb_index", scores_cb_index},
        {"bias_cb_index", bias_cb_index},
        {"sigmoid_input_cb_index", sigmoid_input_cb_index},
        {"add_bias_cb_index", add_bias_cb_index},
        {"weights_cb_index", weights_cb_index},
        {"indices_cb_index", indices_cb_index},
        {"group_indices_cb_index", group_indices_cb_index},
        {"group_scores_cb_index", group_scores_cb_index},
        {"summed_experts_cb_index", summed_experts_cb_index},
        {"sorted_group_indices_cb_index", sorted_group_indices_cb_index},
        {"width_tiles", width_tiles},
        {"scores_page_size", scores.buffer()->page_size()},
        {"bias_page_size", bias.buffer()->page_size()},
        {"weights_page_size", output_weights.buffer()->page_size()},
        {"indices_page_size", output_indices.buffer()->page_size()},
        {"topk_input_cb_index", topk_input_cb_index},
        {"topk_index_cb_index", topk_index_cb_index},
        {"topk_index_creation_cb_index", topk_index_creation_cb_index},
        {"group_size", experts / operation_attributes.n_groups},
        {"log_group_size", std::log2(experts / operation_attributes.n_groups)},
        {"summed_experts_per_group", operation_attributes.summed_experts_per_group},
        {"topk_groups", operation_attributes.topk_groups},
        {"n_groups", operation_attributes.n_groups},
        {"log_topk_groups", std::log2(operation_attributes.topk_groups)},
        {"log_n_groups", std::log2(operation_attributes.n_groups)},
        {"winning_group_scores_cb_index", winning_group_scores_cb_index},
        {"winning_group_indices_cb_index", winning_group_indices_cb_index},
        {"num_group_tiles", num_group_tiles},
        {"n_activated_experts", operation_attributes.n_activated_experts},
        {"n_activated_expert_tiles", n_activated_expert_tiles},
        {"log_n_activated_experts", std::log2(operation_attributes.n_activated_experts)},
        {"intermediate_local_sort_cb_index", intermediate_local_sort_cb_index},
        {"intermediate_local_sort_indices_cb_index", intermediate_local_sort_indices_cb_index},
        {"pre_normalized_scores_cb_index", pre_normalized_scores_cb_index},
        {"reduce_scalar_cb_index", reduce_scalar_cb_index},
        {"epsilon_cb_index", epsilon_cb_index},
        {"scales_cb_index", scales_cb_index},
        {"normalized_cb_index", normalized_cb_index},
        {"transpose_cb_index", transpose_cb_index},
        {"normalized_transpose_cb_index", normalized_transpose_cb_index},
        {"post_sort_transpose_cb_index", post_sort_transpose_cb_index},
    };

    std::vector<uint32_t> compute_compile_time_args = {};

    // Compute kernel
    bool fp32_dest_acc_en =
        false;  // has to be false otherwise transpose_wh_tile of the index tiles corrupts dest reg 1 bfp16 somehow?
    auto compute_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/grouped_gate/device/kernels/compute/grouped_gate.cpp",
        all_cores,
        ComputeConfig{
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .compile_args = compute_compile_time_args,
            .named_compile_args = compute_named_compile_time_args});

    // Writer kernel

    // Writer kernel compile time arguments
    std::unordered_map<std::string, uint32_t> writer_named_compile_time_args = {
        {"weights_cb_index", weights_cb_index},
        {"indices_cb_index", indices_cb_index},
        {"topk_index_creation_cb_index", topk_index_creation_cb_index},
        {"group_indices_cb_index", group_indices_cb_index},
        {"summed_experts_cb_index", summed_experts_cb_index},
        {"topk_input_cb_index", topk_input_cb_index},
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
        {"winning_group_scores_cb_index", winning_group_scores_cb_index},
        {"winning_group_indices_cb_index", winning_group_indices_cb_index},
        {"num_group_tiles", num_group_tiles},
        {"sorted_group_indices_cb_index", sorted_group_indices_cb_index},
        {"scores_cb_index", scores_cb_index},
        {"sigmoid_input_cb_index", sigmoid_input_cb_index},
        {"reduce_scalar_cb_index", reduce_scalar_cb_index},
        {"n_activated_experts", operation_attributes.n_activated_experts},
        {"packed_one_scalar", static_cast<uint32_t>(std::bit_cast<uint16_t>(bfloat16(1.0f))) << 16},
        {"packed_epsilon",
         static_cast<uint32_t>(std::bit_cast<uint16_t>(bfloat16(operation_attributes.epsilon))) << 16},
        {"packed_route_scale",
         static_cast<uint32_t>(std::bit_cast<uint16_t>(bfloat16(operation_attributes.route_scale))) << 16},
        {"epsilon_cb_index", epsilon_cb_index},
        {"scales_cb_index", scales_cb_index},
        {"seq_len_tiles", tt::div_up(seq_len, tile_height)},
        {"remainder_tokens_per_tile", remainder_tokens_per_tile},
    };

    std::vector<uint32_t> writer_compile_time_args = {};
    TensorAccessorArgs(output_weights.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(output_indices.buffer()).append_to(writer_compile_time_args);

    auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/reduction/grouped_gate/device/kernels/dataflow/writer_grouped_gate.cpp",
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

void GroupedGateDeviceOperation::ProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    // Placeholder for runtime argument override logic
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

}  // namespace ttnn::operations::reduction
