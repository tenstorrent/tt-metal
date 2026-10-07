// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_grouped_topk_device_operation.hpp"

#include <bit>

#include <tt-metalium/circular_buffer_constants.h>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk {

namespace {

// floor(log2(value)). std::bit_width stays exact on powers of two; std::log2 can
// round just below the integer, and a uint32 truncation then drops the result by one.
uint32_t floor_log2(uint32_t value) {
    TT_FATAL(value > 0, "log2 input must be positive, got {}", value);
    return static_cast<uint32_t>(std::bit_width(value) - 1);
}

// ceil(log2(value)). width_tiles is not always a power of two (for example 384 experts).
uint32_t ceil_log2(uint32_t value) {
    TT_FATAL(value > 0, "log2 input must be positive, got {}", value);
    return value == 1 ? 0u : static_cast<uint32_t>(std::bit_width(value - 1u));
}

}  // namespace

tt::tt_metal::ProgramDescriptor MoeGroupedTopkDeviceOperation::ProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& scores = tensor_args.scores;
    const auto& bias = tensor_args.bias;
    auto& output_weights = tensor_return_value[0];
    auto& output_indices = tensor_return_value[1];

    // Test-only debug output.
    const bool dump_biased = tensor_args.biased_scores.has_value();

    TT_FATAL(output_weights.dtype() == DataType::BFLOAT16, "Output weights tensor must be BFLOAT16");
    const bool row_major_weights = output_weights.layout() == Layout::ROW_MAJOR;
    TT_FATAL(output_indices.dtype() == DataType::UINT16, "Output indices tensor must be UINT16");
    TT_FATAL(output_indices.layout() == Layout::TILE, "Output indices tensor must be TILE layout");

    auto* device = scores.device();
    TT_FATAL(device != nullptr, "Device must be non-null");
    ProgramDescriptor desc;

    auto grid = device->compute_with_storage_grid_size();
    auto* scores_buffer = scores.buffer();
    auto* bias_buffer = bias.buffer();
    auto* weights_buffer = output_weights.buffer();
    auto* indices_buffer = output_indices.buffer();
    TT_FATAL(scores_buffer != nullptr, "scores buffer must be allocated on device");
    TT_FATAL(bias_buffer != nullptr, "bias buffer must be allocated on device");
    TT_FATAL(weights_buffer != nullptr, "output weights buffer must be allocated on device");
    TT_FATAL(indices_buffer != nullptr, "output indices buffer must be allocated on device");

    auto num_tiles = scores_buffer->num_pages();
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

    // The whole gate pipeline computes in fp32, so every intermediate score CB is fp32 regardless of
    // the input dtype.
    const auto compute_data_format = tt::DataFormat::Float32;
    const uint32_t compute_page_size = tt::tile_size(compute_data_format);

    uint32_t n_activated_expert_tiles = tt::div_up(operation_attributes.n_activated_experts, 32);
    // Template-side index tiles are uint32 so they reach 32-bit DEST as plain integers; outputs stay uint16.
    const uint32_t uint32_page_size = tt::tile_size(tt::DataFormat::UInt32);
    const uint32_t scores_page_size = scores_buffer->page_size();
    const uint32_t bias_page_size = bias_buffer->page_size();
    const uint32_t weights_page_size = weights_buffer->page_size();
    const uint32_t indices_page_size = indices_buffer->page_size();

    auto add_cb = [&](uint32_t cb_idx, uint32_t page_size, uint32_t num_pages, tt::DataFormat data_format) {
        desc.cbs.push_back(CBDescriptor{
            .total_size = page_size * num_pages,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_idx),
                .data_format = data_format,
                .page_size = page_size,
            }}},
        });
    };

    add_cb(cb_in_scores, scores_page_size, 2 * width_tiles, scores_data_format);
    add_cb(cb_in_bias, bias_page_size, 2 * width_tiles, bias_data_format);

    // fp32 upcast targets for the raw inputs (identity copy when the input is already fp32).
    auto cb_scores_fp32 = tt::CBIndex::c_24;
    auto cb_bias_fp32 = tt::CBIndex::c_25;
    add_cb(cb_scores_fp32, compute_page_size, 2 * width_tiles, compute_data_format);
    add_cb(cb_bias_fp32, compute_page_size, 2 * width_tiles, compute_data_format);
    add_cb(cb_out_weights, tt::tile_size(weights_data_format), 2 * n_activated_expert_tiles, weights_data_format);
    add_cb(cb_out_indices, indices_page_size, 2 * n_activated_expert_tiles, indices_data_format);

    auto cb_sigmoid_scores = tt::CBIndex::c_4;
    auto cb_biased_scores = tt::CBIndex::c_5;
    add_cb(cb_sigmoid_scores, compute_page_size, width_tiles, compute_data_format);
    add_cb(cb_biased_scores, compute_page_size, width_tiles, compute_data_format);

    // Test-only debug output.
    tt::CBIndex cb_biased_dump = dump_biased ? tt::CBIndex::c_26 : cb_biased_scores;
    if (dump_biased) {
        add_cb(cb_biased_dump, compute_page_size, 2 * width_tiles, compute_data_format);
    }
    auto* biased_buffer = dump_biased ? tensor_args.biased_scores->buffer() : weights_buffer;
    TT_FATAL(biased_buffer != nullptr, "biased scores buffer must be allocated on device");
    uint32_t biased_page_size = biased_buffer->page_size();
    Buffer* biased_runtime_buffer = dump_biased ? tensor_args.biased_scores->buffer() : nullptr;

    auto cb_sorted_group_scores = tt::CBIndex::c_6;
    auto cb_sorted_expert_indices_temp = tt::CBIndex::c_7;
    auto cb_expert_index_template = tt::CBIndex::c_8;
    add_cb(cb_sorted_group_scores, compute_page_size, 2, compute_data_format);
    add_cb(cb_sorted_expert_indices_temp, uint32_page_size, 2, tt::DataFormat::UInt32);
    add_cb(cb_expert_index_template, uint32_page_size, width_tiles, tt::DataFormat::UInt32);

    uint32_t num_group_tiles = tt::div_up(operation_attributes.n_groups, 32);
    auto cb_group_index_template = tt::CBIndex::c_9;
    auto cb_group_summed_scores = tt::CBIndex::c_10;
    auto cb_top_experts_per_group = tt::CBIndex::c_11;
    auto cb_sorted_group_order = tt::CBIndex::c_12;
    add_cb(cb_group_index_template, uint32_page_size, num_group_tiles, tt::DataFormat::UInt32);
    add_cb(
        cb_top_experts_per_group,
        compute_page_size,
        operation_attributes.summed_experts_per_group,
        compute_data_format);
    add_cb(cb_group_summed_scores, compute_page_size, num_group_tiles, compute_data_format);
    add_cb(cb_sorted_group_order, indices_page_size, num_group_tiles, tt::DataFormat::UInt16);

    auto cb_winning_group_scores = tt::CBIndex::c_13;
    auto cb_winning_group_indices = tt::CBIndex::c_14;
    add_cb(cb_winning_group_scores, compute_page_size, operation_attributes.topk_groups, compute_data_format);
    add_cb(cb_winning_group_indices, uint32_page_size, operation_attributes.topk_groups, tt::DataFormat::UInt32);

    auto cb_reduce_intermediate = tt::CBIndex::c_15;
    auto cb_final_indices_transposed = tt::CBIndex::c_16;
    add_cb(cb_reduce_intermediate, compute_page_size, 2 * n_activated_expert_tiles, compute_data_format);
    add_cb(cb_final_indices_transposed, indices_page_size, 2 * n_activated_expert_tiles, tt::DataFormat::UInt16);

    auto cb_reduce_ones_scalar = tt::CBIndex::c_17;
    add_cb(cb_reduce_ones_scalar, compute_page_size, 1, compute_data_format);

    auto cb_epsilon_scalar = tt::CBIndex::c_18;
    add_cb(cb_epsilon_scalar, compute_page_size, 1, compute_data_format);

    auto cb_route_scale_scalar = tt::CBIndex::c_19;
    add_cb(cb_route_scale_scalar, compute_page_size, 1, compute_data_format);

    auto cb_normalized_scores = tt::CBIndex::c_20;
    add_cb(cb_normalized_scores, compute_page_size, 2 * n_activated_expert_tiles, compute_data_format);

    auto cb_reciprocal_sums = tt::CBIndex::c_21;
    add_cb(cb_reciprocal_sums, compute_page_size, 2 * n_activated_expert_tiles, compute_data_format);

    auto cb_gathered_sigmoid = tt::CBIndex::c_22;
    add_cb(cb_gathered_sigmoid, compute_page_size, 2 * n_activated_expert_tiles, compute_data_format);

    // Optional padding config: when absent, fall back to the output indices buffer so the writer's
    // TensorAccessor compile-time args still line up (a 0 runtime address then disables padding).
    auto cb_padding_config = tt::CBIndex::c_23;
    auto* padding_config_buffer =
        tensor_args.padding_config.has_value() ? tensor_args.padding_config->buffer() : indices_buffer;
    TT_FATAL(padding_config_buffer != nullptr, "padding config buffer must be allocated on device");
    uint32_t padding_config_page_size = static_cast<uint32_t>(padding_config_buffer->aligned_page_size());
    add_cb(cb_padding_config, padding_config_page_size, 1, tt::DataFormat::UInt32);
    Buffer* padding_runtime_buffer =
        tensor_args.padding_config.has_value() ? tensor_args.padding_config->buffer() : nullptr;

    KernelDescriptor::NamedCompileTimeArgs reader_named_compile_time_args = {
        {"cb_in_scores", cb_in_scores},
        {"cb_in_bias", cb_in_bias},
        {"cb_route_scale_scalar", cb_route_scale_scalar},
        {"width_tiles", width_tiles},
        {"scores_page_size", scores_page_size},
        {"bias_page_size", bias_page_size},
    };

    std::vector<uint32_t> reader_compile_time_args = {};
    tt::tt_metal::TensorAccessorArgs(scores_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(bias_buffer).append_to(reader_compile_time_args);

    KernelDescriptor::NamedCompileTimeArgs compute_named_compile_time_args = {
        {"cb_in_scores", cb_in_scores},
        {"cb_in_bias", cb_in_bias},
        {"cb_scores_fp32", cb_scores_fp32},
        {"cb_bias_fp32", cb_bias_fp32},
        {"cb_sigmoid_scores", cb_sigmoid_scores},
        {"cb_biased_scores", cb_biased_scores},
        {"cb_biased_dump", cb_biased_dump},
        {"dump_biased_scores", static_cast<uint32_t>(dump_biased)},
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
        {"log_group_size", floor_log2(experts / operation_attributes.n_groups)},
        {"summed_experts_per_group", operation_attributes.summed_experts_per_group},
        {"topk_groups", operation_attributes.topk_groups},
        {"n_groups", operation_attributes.n_groups},
        {"log_topk_groups", floor_log2(operation_attributes.topk_groups)},
        {"log_n_groups", floor_log2(operation_attributes.n_groups)},
        {"cb_winning_group_scores", cb_winning_group_scores},
        {"cb_winning_group_indices", cb_winning_group_indices},
        {"num_group_tiles", num_group_tiles},
        {"n_activated_experts", operation_attributes.n_activated_experts},
        {"n_activated_expert_tiles", n_activated_expert_tiles},
        {"cb_reduce_intermediate", cb_reduce_intermediate},
        {"cb_final_indices_transposed", cb_final_indices_transposed},
        {"cb_reduce_ones_scalar", cb_reduce_ones_scalar},
        {"cb_epsilon_scalar", cb_epsilon_scalar},
        {"cb_route_scale_scalar", cb_route_scale_scalar},
        {"cb_normalized_scores", cb_normalized_scores},
        {"cb_reciprocal_sums", cb_reciprocal_sums},
        {"cb_gathered_sigmoid", cb_gathered_sigmoid},
        {"stable_sort", static_cast<uint32_t>(operation_attributes.stable_sort)},
        {"score_func", static_cast<uint32_t>(operation_attributes.score_func)},
        // blocks::topk only reads log_tiles when tiles <= 2, so a ceil-log2 is safe. Used by the
        // single-group (n_groups == 1) path which runs a plain top-k over all width_tiles.
        {"log_width_tiles", ceil_log2(static_cast<uint32_t>(width_tiles))},
    };

    // Default-unpacked fp32 tiles reach DEST through SrcA as TF32, i.e. with zero low 13 mantissa bits;
    // the kernel's rank-tag stable engine keeps its tag inside those bits and is lossless only then.
    // sort_keys_tf32 certifies it from the modes actually passed for the two CBs the rank-tag sorts
    // read (an UnpackToDestFp32 mode there turns it off and the kernel keeps the comparator engine).
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    const auto default_unpack = [&](tt::CBIndex cb) {
        return unpack_to_dest_mode[static_cast<uint32_t>(cb)] == tt::tt_metal::UnpackToDestMode::Default;
    };
    const bool sort_keys_tf32 = default_unpack(cb_biased_scores) && default_unpack(cb_group_summed_scores);
    compute_named_compile_time_args.emplace_back("sort_keys_tf32", static_cast<uint32_t>(sort_keys_tf32));

    KernelDescriptor::NamedCompileTimeArgs writer_named_compile_time_args = {
        {"row_major_weights", static_cast<uint32_t>(row_major_weights)},
        {"cb_out_weights", cb_out_weights},
        {"cb_out_indices", cb_out_indices},
        {"cb_expert_index_template", cb_expert_index_template},
        {"cb_group_index_template", cb_group_index_template},
        {"cb_top_experts_per_group", cb_top_experts_per_group},
        {"cb_gathered_sigmoid", cb_gathered_sigmoid},
        {"cb_sorted_group_scores", cb_sorted_group_scores},
        {"scores_page_size", scores_page_size},
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
        {"cb_biased_dump", cb_biased_dump},
        {"dump_biased_scores", static_cast<uint32_t>(dump_biased)},
        {"biased_page_size", biased_page_size},
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
        {"cb_padding_config", cb_padding_config},
    };

    std::vector<uint32_t> writer_compile_time_args = {};
    tt::tt_metal::TensorAccessorArgs(weights_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(indices_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(padding_config_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(biased_buffer).append_to(writer_compile_time_args);

    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/kernels/dataflow/"
        "reader_moe_grouped_topk.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.named_compile_time_args = std::move(reader_named_compile_time_args);
    reader_kernel_desc.config = ReaderConfigDescriptor{};

    bool fp32_dest_acc_en = true;
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/kernels/compute/"
        "moe_grouped_topk.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = all_cores;
    compute_kernel_desc.named_compile_time_args = std::move(compute_named_compile_time_args);
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = false,
        .unpack_to_dest_mode = unpack_to_dest_mode,
        .math_approx_mode = false,
    };

    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_grouped_topk/device/kernels/dataflow/"
        "writer_moe_grouped_topk.cpp";
    writer_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_kernel_desc.core_ranges = all_cores;
    writer_kernel_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_kernel_desc.named_compile_time_args = std::move(writer_named_compile_time_args);
    writer_kernel_desc.config = WriterConfigDescriptor{};

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

        reader_kernel_desc.emplace_runtime_args(core, {scores_buffer, bias_buffer, start_height_tile, end_height_tile});
        compute_kernel_desc.emplace_runtime_args(core, {start_height_tile, end_height_tile});
        writer_kernel_desc.emplace_runtime_args(
            core,
            {weights_buffer,
             indices_buffer,
             start_height_tile,
             end_height_tile,
             padding_runtime_buffer,
             biased_runtime_buffer});
    }

    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(compute_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));
    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_grouped_topk
