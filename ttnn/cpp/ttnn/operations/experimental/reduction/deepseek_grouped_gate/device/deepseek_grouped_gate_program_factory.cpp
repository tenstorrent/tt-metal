// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "deepseek_grouped_gate_device_operation.hpp"

#include <bit>
#include <map>
#include <string>
#include <vector>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::reduction {

tt::tt_metal::ProgramDescriptor DeepseekGroupedGateDeviceOperation::ProgramFactory::create_descriptor(
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

    const uint32_t scores_page_size = static_cast<uint32_t>(scores.buffer()->page_size());
    const uint32_t bias_page_size = static_cast<uint32_t>(bias.buffer()->page_size());
    const uint32_t weights_page_size = static_cast<uint32_t>(output_weights.buffer()->page_size());
    const uint32_t indices_page_size = static_cast<uint32_t>(output_indices.buffer()->page_size());

    uint32_t n_activated_expert_tiles = tt::div_up(operation_attributes.n_activated_experts, 32);

    ProgramDescriptor desc;

    // Helper lambda to build a CBDescriptor with a single format descriptor.
    auto add_cb = [&desc, &all_cores](
                      uint32_t cb_id, uint32_t page_size, uint32_t cb_num_tiles, tt::DataFormat format) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = cb_num_tiles * page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_id), .data_format = format, .page_size = page_size}}}});
    };

    // Deadlock prevention for bias CB:
    //   - Bias needs width_tiles capacity because:
    //       * The add_bias stage does not consume bias until ALL sigmoid tiles are ready.
    //       * The reader pushes scores and bias together.
    //   - If the bias CB is too small, the reader can block, causing a deadlock.
    add_cb(cb_in_scores, scores_page_size, 2 * width_tiles, scores_data_format);
    add_cb(cb_in_bias, bias_page_size, 2 * width_tiles, bias_data_format);
    add_cb(cb_out_weights, weights_page_size, 2 * n_activated_expert_tiles, weights_data_format);
    add_cb(cb_out_indices, indices_page_size, 2 * n_activated_expert_tiles, indices_data_format);

    // Sigmoid output and biased scores CBs
    // Note: cb_sigmoid_scores needs width_tiles capacity since add_bias waits for all tiles at once
    // and writer also needs all tiles. Don't double-buffer - it causes L1 memory pressure.
    auto cb_sigmoid_scores = tt::CBIndex::c_4;
    auto cb_biased_scores = tt::CBIndex::c_5;
    add_cb(cb_sigmoid_scores, scores_page_size, width_tiles, scores_data_format);
    add_cb(cb_biased_scores, scores_page_size, width_tiles, scores_data_format);

    // Per-group sorting CBs
    // cb_sorted_group_scores is consumed one tile at a time by writer's generate_summed_experts_tiles
    // cb_sorted_expert_indices_temp is used transiently and popped immediately in process_and_sort_tiles
    // cb_expert_index_template needs all width_tiles for generate_winning_group_tiles
    auto cb_sorted_group_scores = tt::CBIndex::c_6;
    auto cb_sorted_expert_indices_temp = tt::CBIndex::c_7;
    auto cb_expert_index_template = tt::CBIndex::c_8;
    add_cb(cb_sorted_group_scores, scores_page_size, 2, scores_data_format);
    add_cb(cb_sorted_expert_indices_temp, scores_page_size, 2, tt::DataFormat::UInt16);
    add_cb(cb_expert_index_template, scores_page_size, width_tiles, tt::DataFormat::UInt16);

    // Group selection CBs
    uint32_t num_group_tiles = tt::div_up(operation_attributes.n_groups, 32);
    auto cb_group_index_template = tt::CBIndex::c_9;
    auto cb_group_summed_scores = tt::CBIndex::c_10;
    auto cb_top_experts_per_group = tt::CBIndex::c_11;
    auto cb_sorted_group_order = tt::CBIndex::c_12;
    add_cb(cb_group_index_template, scores_page_size, num_group_tiles, tt::DataFormat::UInt16);
    add_cb(
        cb_top_experts_per_group, scores_page_size, operation_attributes.summed_experts_per_group, scores_data_format);
    add_cb(cb_group_summed_scores, scores_page_size, num_group_tiles, scores_data_format);
    add_cb(cb_sorted_group_order, indices_page_size, num_group_tiles, tt::DataFormat::UInt16);

    // Winning group processing CBs
    auto cb_winning_group_scores = tt::CBIndex::c_13;
    auto cb_winning_group_indices = tt::CBIndex::c_14;
    add_cb(cb_winning_group_scores, weights_page_size, operation_attributes.topk_groups, scores_data_format);
    add_cb(cb_winning_group_indices, indices_page_size, operation_attributes.topk_groups, tt::DataFormat::UInt16);

    // Final topk intermediate CBs
    auto cb_reduce_intermediate = tt::CBIndex::c_15;
    auto cb_final_indices_transposed = tt::CBIndex::c_16;
    add_cb(cb_reduce_intermediate, scores_page_size, 2 * n_activated_expert_tiles, scores_data_format);
    add_cb(cb_final_indices_transposed, indices_page_size, 2 * n_activated_expert_tiles, tt::DataFormat::UInt16);

    // Normalization scalar CBs
    auto cb_reduce_ones_scalar = tt::CBIndex::c_17;
    add_cb(cb_reduce_ones_scalar, scores_page_size, 1, scores_data_format);

    auto cb_epsilon_scalar = tt::CBIndex::c_18;
    add_cb(cb_epsilon_scalar, scores_page_size, 1, scores_data_format);

    auto cb_route_scale_scalar = tt::CBIndex::c_19;
    add_cb(cb_route_scale_scalar, scores_page_size, 1, scores_data_format);

    // Normalization intermediate CBs
    auto cb_normalized_scores = tt::CBIndex::c_20;
    add_cb(cb_normalized_scores, scores_page_size, 2 * n_activated_expert_tiles, scores_data_format);

    auto cb_reciprocal_sums = tt::CBIndex::c_21;
    add_cb(cb_reciprocal_sums, scores_page_size, 2 * n_activated_expert_tiles, scores_data_format);

    // Gather CB for selected expert sigmoid scores
    auto cb_gathered_sigmoid = tt::CBIndex::c_22;
    add_cb(cb_gathered_sigmoid, scores_page_size, 2 * n_activated_expert_tiles, scores_data_format);

    // Reader kernel compile time arguments
    std::map<std::string, uint32_t> reader_named_compile_time_args = {
        {"cb_in_scores", cb_in_scores},
        {"cb_in_bias", cb_in_bias},
        {"cb_route_scale_scalar", cb_route_scale_scalar},
        {"width_tiles", width_tiles},
        {"scores_page_size", scores_page_size},
        {"bias_page_size", bias_page_size},
    };

    std::vector<uint32_t> reader_compile_time_args = {};
    tt::tt_metal::TensorAccessorArgs(scores.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(bias.buffer()).append_to(reader_compile_time_args);

    // Compute kernel compile time arguments
    std::map<std::string, uint32_t> compute_named_compile_time_args = {
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
        {"scores_page_size", scores_page_size},
        {"bias_page_size", bias_page_size},
        {"weights_page_size", weights_page_size},
        {"indices_page_size", indices_page_size},
        {"cb_sorted_group_scores", cb_sorted_group_scores},
        {"cb_sorted_expert_indices_temp", cb_sorted_expert_indices_temp},
        {"cb_expert_index_template", cb_expert_index_template},
        {"group_size", experts / operation_attributes.n_groups},
        {"log_group_size", static_cast<uint32_t>(std::countr_zero(experts / operation_attributes.n_groups))},
        {"summed_experts_per_group", operation_attributes.summed_experts_per_group},
        {"topk_groups", operation_attributes.topk_groups},
        {"n_groups", operation_attributes.n_groups},
        {"log_topk_groups", static_cast<uint32_t>(std::countr_zero(operation_attributes.topk_groups))},
        {"log_n_groups", static_cast<uint32_t>(std::countr_zero(operation_attributes.n_groups))},
        {"cb_winning_group_scores", cb_winning_group_scores},
        {"cb_winning_group_indices", cb_winning_group_indices},
        {"num_group_tiles", num_group_tiles},
        {"n_activated_experts", operation_attributes.n_activated_experts},
        {"n_activated_expert_tiles", n_activated_expert_tiles},
        {"log_n_activated_experts", static_cast<uint32_t>(std::countr_zero(operation_attributes.n_activated_experts))},
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

    // Writer kernel compile time arguments
    std::map<std::string, uint32_t> writer_named_compile_time_args = {
        {"cb_out_weights", cb_out_weights},
        {"cb_out_indices", cb_out_indices},
        {"cb_expert_index_template", cb_expert_index_template},
        {"cb_group_index_template", cb_group_index_template},
        {"cb_top_experts_per_group", cb_top_experts_per_group},
        {"cb_gathered_sigmoid", cb_gathered_sigmoid},
        {"cb_sorted_group_scores", cb_sorted_group_scores},
        {"weights_page_size", weights_page_size},
        {"indices_page_size", indices_page_size},
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

    ////////////////////////////////////////////////////////////////////////////
    //                      Build kernels
    ////////////////////////////////////////////////////////////////////////////

    // Reader kernel
    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/kernels/dataflow/"
        "reader_deepseek_grouped_gate.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.named_compile_time_args = KernelDescriptor::NamedCompileTimeArgs(
        reader_named_compile_time_args.begin(), reader_named_compile_time_args.end());
    reader_kernel_desc.config = ReaderConfigDescriptor{};

    // Writer kernel
    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/kernels/dataflow/"
        "writer_deepseek_grouped_gate.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_kernel_desc.named_compile_time_args = KernelDescriptor::NamedCompileTimeArgs(
        writer_named_compile_time_args.begin(), writer_named_compile_time_args.end());
    writer_kernel_desc.config = WriterConfigDescriptor{};

    // Compute kernel
    bool fp32_dest_acc_en = false;  // Needed for topK to work with uint16_t indices
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/reduction/deepseek_grouped_gate/device/kernels/compute/"
        "deepseek_grouped_gate.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = all_cores;
    compute_kernel_desc.compile_time_args = std::move(compute_compile_time_args);
    compute_kernel_desc.named_compile_time_args = KernelDescriptor::NamedCompileTimeArgs(
        compute_named_compile_time_args.begin(), compute_named_compile_time_args.end());
    compute_kernel_desc.config = ComputeConfigDescriptor{.fp32_dest_acc_en = fp32_dest_acc_en};

    // Build runtime args per core
    auto cores = corerange_to_cores(all_cores, std::nullopt);
    reader_kernel_desc.runtime_args.reserve(cores.size());
    writer_kernel_desc.runtime_args.reserve(cores.size());
    compute_kernel_desc.runtime_args.reserve(cores.size());

    uint32_t start_height_tile = 0;
    uint32_t end_height_tile = 0;
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

        reader_kernel_desc.emplace_runtime_args(
            core, {scores.buffer(), bias.buffer(), start_height_tile, end_height_tile});
        compute_kernel_desc.emplace_runtime_args(core, {start_height_tile, end_height_tile});
        writer_kernel_desc.emplace_runtime_args(
            core, {output_weights.buffer(), output_indices.buffer(), start_height_tile, end_height_tile});
    }

    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));
    desc.kernels.push_back(std::move(compute_kernel_desc));

    return desc;
}

}  // namespace ttnn::operations::experimental::reduction
