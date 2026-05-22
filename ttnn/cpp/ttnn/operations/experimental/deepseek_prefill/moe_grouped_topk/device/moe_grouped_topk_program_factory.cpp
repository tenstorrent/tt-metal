// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_grouped_topk_device_operation.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include "ttnn/operations/cb_utils.hpp"

#include <bit>
#include <cmath>

namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk {

tt::tt_metal::ProgramDescriptor MoeGroupedTopkDeviceOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    ProgramDescriptor desc;

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

    auto cb_in_scores = tt::CBIndex::c_0;
    auto cb_in_bias = tt::CBIndex::c_1;
    auto cb_out_weights = tt::CBIndex::c_2;
    auto cb_out_indices = tt::CBIndex::c_3;

    auto scores_data_format = tt::tt_metal::datatype_to_dataformat_converter(scores.dtype());
    auto bias_data_format = tt::tt_metal::datatype_to_dataformat_converter(bias.dtype());
    auto weights_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_weights.dtype());
    auto indices_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_indices.dtype());

    uint32_t n_activated_expert_tiles = tt::div_up(operation_attributes.n_activated_experts, 32);
    uint32_t uint16_page_size = output_indices.buffer()->page_size();

    auto add_cb = [&](uint8_t index, uint32_t num_pages, uint32_t page_size, tt::DataFormat data_format) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_pages * page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = index,
                .data_format = data_format,
                .page_size = page_size,
            }}},
        });
    };

    add_cb(static_cast<uint8_t>(cb_in_scores), 2 * width_tiles, scores.buffer()->page_size(), scores_data_format);
    add_cb(static_cast<uint8_t>(cb_in_bias), 2 * width_tiles, bias.buffer()->page_size(), bias_data_format);
    add_cb(
        static_cast<uint8_t>(cb_out_weights),
        2 * n_activated_expert_tiles,
        output_weights.buffer()->page_size(),
        weights_data_format);
    add_cb(
        static_cast<uint8_t>(cb_out_indices),
        2 * n_activated_expert_tiles,
        output_indices.buffer()->page_size(),
        indices_data_format);

    auto cb_sigmoid_scores = tt::CBIndex::c_4;
    auto cb_biased_scores = tt::CBIndex::c_5;
    add_cb(static_cast<uint8_t>(cb_sigmoid_scores), width_tiles, scores.buffer()->page_size(), scores_data_format);
    add_cb(static_cast<uint8_t>(cb_biased_scores), width_tiles, scores.buffer()->page_size(), scores_data_format);

    auto cb_sorted_group_scores = tt::CBIndex::c_6;
    auto cb_sorted_expert_indices_temp = tt::CBIndex::c_7;
    auto cb_expert_index_template = tt::CBIndex::c_8;
    add_cb(static_cast<uint8_t>(cb_sorted_group_scores), 2, scores.buffer()->page_size(), scores_data_format);
    add_cb(static_cast<uint8_t>(cb_sorted_expert_indices_temp), 2, uint16_page_size, tt::DataFormat::UInt16);
    add_cb(static_cast<uint8_t>(cb_expert_index_template), width_tiles, uint16_page_size, tt::DataFormat::UInt16);

    uint32_t num_group_tiles = tt::div_up(operation_attributes.n_groups, 32);
    auto cb_group_index_template = tt::CBIndex::c_9;
    auto cb_group_summed_scores = tt::CBIndex::c_10;
    auto cb_top_experts_per_group = tt::CBIndex::c_11;
    auto cb_sorted_group_order = tt::CBIndex::c_12;
    add_cb(static_cast<uint8_t>(cb_group_index_template), num_group_tiles, uint16_page_size, tt::DataFormat::UInt16);
    add_cb(
        static_cast<uint8_t>(cb_top_experts_per_group),
        operation_attributes.summed_experts_per_group,
        scores.buffer()->page_size(),
        scores_data_format);
    add_cb(
        static_cast<uint8_t>(cb_group_summed_scores),
        num_group_tiles,
        scores.buffer()->page_size(),
        scores_data_format);
    add_cb(
        static_cast<uint8_t>(cb_sorted_group_order),
        num_group_tiles,
        output_indices.buffer()->page_size(),
        tt::DataFormat::UInt16);

    auto cb_winning_group_scores = tt::CBIndex::c_13;
    auto cb_winning_group_indices = tt::CBIndex::c_14;
    add_cb(
        static_cast<uint8_t>(cb_winning_group_scores),
        operation_attributes.topk_groups,
        scores.buffer()->page_size(),
        scores_data_format);
    add_cb(
        static_cast<uint8_t>(cb_winning_group_indices),
        operation_attributes.topk_groups,
        output_indices.buffer()->page_size(),
        tt::DataFormat::UInt16);

    auto cb_reduce_intermediate = tt::CBIndex::c_15;
    auto cb_final_indices_transposed = tt::CBIndex::c_16;
    add_cb(
        static_cast<uint8_t>(cb_reduce_intermediate),
        2 * n_activated_expert_tiles,
        scores.buffer()->page_size(),
        scores_data_format);
    add_cb(
        static_cast<uint8_t>(cb_final_indices_transposed),
        2 * n_activated_expert_tiles,
        output_indices.buffer()->page_size(),
        tt::DataFormat::UInt16);

    auto cb_reduce_ones_scalar = tt::CBIndex::c_17;
    add_cb(static_cast<uint8_t>(cb_reduce_ones_scalar), 1, scores.buffer()->page_size(), scores_data_format);

    auto cb_epsilon_scalar = tt::CBIndex::c_18;
    add_cb(static_cast<uint8_t>(cb_epsilon_scalar), 1, scores.buffer()->page_size(), scores_data_format);

    auto cb_route_scale_scalar = tt::CBIndex::c_19;
    add_cb(static_cast<uint8_t>(cb_route_scale_scalar), 1, scores.buffer()->page_size(), scores_data_format);

    auto cb_normalized_scores = tt::CBIndex::c_20;
    add_cb(
        static_cast<uint8_t>(cb_normalized_scores),
        2 * n_activated_expert_tiles,
        scores.buffer()->page_size(),
        scores_data_format);

    auto cb_reciprocal_sums = tt::CBIndex::c_21;
    add_cb(
        static_cast<uint8_t>(cb_reciprocal_sums),
        2 * n_activated_expert_tiles,
        scores.buffer()->page_size(),
        scores_data_format);

    auto cb_gathered_sigmoid = tt::CBIndex::c_22;
    add_cb(
        static_cast<uint8_t>(cb_gathered_sigmoid),
        2 * n_activated_expert_tiles,
        scores.buffer()->page_size(),
        scores_data_format);

    KernelDescriptor::NamedCompileTimeArgs reader_named_compile_time_args = {
        {"cb_in_scores", cb_in_scores},
        {"cb_in_bias", cb_in_bias},
        {"cb_route_scale_scalar", cb_route_scale_scalar},
        {"width_tiles", width_tiles},
        {"scores_page_size", scores.buffer()->page_size()},
        {"bias_page_size", bias.buffer()->page_size()},
    };

    std::vector<uint32_t> reader_compile_time_args = {};
    tt::tt_metal::TensorAccessorArgs(*scores.buffer()).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*bias.buffer()).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/kernels/dataflow/"
        "reader_moe_grouped_topk.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.named_compile_time_args = std::move(reader_named_compile_time_args);
    reader_desc.config = ReaderDataMovementConfig{};

    KernelDescriptor::NamedCompileTimeArgs compute_named_compile_time_args = {
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
        {"log_group_size", static_cast<uint32_t>(std::log2(experts / operation_attributes.n_groups))},
        {"summed_experts_per_group", operation_attributes.summed_experts_per_group},
        {"topk_groups", operation_attributes.topk_groups},
        {"n_groups", operation_attributes.n_groups},
        {"log_topk_groups", static_cast<uint32_t>(std::log2(operation_attributes.topk_groups))},
        {"log_n_groups", static_cast<uint32_t>(std::log2(operation_attributes.n_groups))},
        {"cb_winning_group_scores", cb_winning_group_scores},
        {"cb_winning_group_indices", cb_winning_group_indices},
        {"num_group_tiles", num_group_tiles},
        {"n_activated_experts", operation_attributes.n_activated_experts},
        {"n_activated_expert_tiles", n_activated_expert_tiles},
        {"log_n_activated_experts", static_cast<uint32_t>(std::log2(operation_attributes.n_activated_experts))},
        {"cb_reduce_intermediate", cb_reduce_intermediate},
        {"cb_final_indices_transposed", cb_final_indices_transposed},
        {"cb_reduce_ones_scalar", cb_reduce_ones_scalar},
        {"cb_epsilon_scalar", cb_epsilon_scalar},
        {"cb_route_scale_scalar", cb_route_scale_scalar},
        {"cb_normalized_scores", cb_normalized_scores},
        {"cb_reciprocal_sums", cb_reciprocal_sums},
        {"cb_gathered_sigmoid", cb_gathered_sigmoid},
        {"stable_sort", static_cast<uint32_t>(operation_attributes.stable_sort)},
    };

    bool fp32_dest_acc_en = true;
    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/kernels/compute/"
        "moe_grouped_topk.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.named_compile_time_args = std::move(compute_named_compile_time_args);
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    KernelDescriptor::NamedCompileTimeArgs writer_named_compile_time_args = {
        {"cb_out_weights", cb_out_weights},
        {"cb_out_indices", cb_out_indices},
        {"cb_expert_index_template", cb_expert_index_template},
        {"cb_group_index_template", cb_group_index_template},
        {"cb_top_experts_per_group", cb_top_experts_per_group},
        {"cb_gathered_sigmoid", cb_gathered_sigmoid},
        {"cb_sorted_group_scores", cb_sorted_group_scores},
        {"scores_page_size", scores.buffer()->page_size()},
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
        {"packed_one_scalar", std::bit_cast<uint32_t>(1.0f)},
        {"packed_epsilon", std::bit_cast<uint32_t>(operation_attributes.epsilon)},
        {"packed_route_scale", std::bit_cast<uint32_t>(operation_attributes.route_scale)},
        {"cb_epsilon_scalar", cb_epsilon_scalar},
        {"cb_route_scale_scalar", cb_route_scale_scalar},
        {"seq_len_tiles", tt::div_up(seq_len, tile_height)},
        {"remainder_tokens_per_tile", remainder_tokens_per_tile},
        {"n_activated_expert_tiles", n_activated_expert_tiles},
    };

    std::vector<uint32_t> writer_compile_time_args = {};
    tt::tt_metal::TensorAccessorArgs(*output_weights.buffer()).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*output_indices.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/kernels/dataflow/"
        "writer_moe_grouped_topk.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.named_compile_time_args = std::move(writer_named_compile_time_args);
    writer_desc.config = WriterDataMovementConfig{};

    // ---- Per-core runtime args ----
    auto cores = corerange_to_cores(all_cores, std::nullopt);
    reader_desc.runtime_args.reserve(cores.size());
    compute_desc.runtime_args.reserve(cores.size());
    writer_desc.runtime_args.reserve(cores.size());

    uint32_t start_height_tile = 0;
    uint32_t end_height_tile = 0;
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

        reader_desc.runtime_args.emplace_back(
            core,
            std::vector<uint32_t>{
                scores.buffer()->address(), bias.buffer()->address(), start_height_tile, end_height_tile});
        compute_desc.runtime_args.emplace_back(core, std::vector<uint32_t>{start_height_tile, end_height_tile});
        writer_desc.runtime_args.emplace_back(
            core,
            std::vector<uint32_t>{
                output_weights.buffer()->address(),
                output_indices.buffer()->address(),
                start_height_tile,
                end_height_tile});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk
