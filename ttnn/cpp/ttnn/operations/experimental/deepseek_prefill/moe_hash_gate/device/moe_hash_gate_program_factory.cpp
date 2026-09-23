// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_hash_gate_device_operation.hpp"

#include <bit>

#include <tt-metalium/hal.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::moe_hash_gate {

tt::tt_metal::ProgramDescriptor MoeHashGateDeviceOperation::ProgramFactory::create_descriptor(
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
    ProgramDescriptor desc;

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

    auto* scores_buffer = scores.buffer();
    auto* input_ids_buffer = input_ids.buffer();
    auto* tid2eid_buffer = tid2eid.buffer();
    auto* weights_buffer = output_weights.buffer();
    auto* indices_buffer = output_indices.buffer();
    TT_FATAL(scores_buffer != nullptr, "scores buffer must be allocated on device");
    TT_FATAL(input_ids_buffer != nullptr, "input_ids buffer must be allocated on device");
    TT_FATAL(tid2eid_buffer != nullptr, "tid2eid buffer must be allocated on device");
    TT_FATAL(weights_buffer != nullptr, "output weights buffer must be allocated on device");
    TT_FATAL(indices_buffer != nullptr, "output indices buffer must be allocated on device");

    uint32_t scores_page_size = scores_buffer->page_size();
    uint32_t weights_page_size = weights_buffer->page_size();
    uint32_t indices_page_size = indices_buffer->page_size();
    uint32_t input_ids_page_size = input_ids_buffer->page_size();
    uint32_t tid2eid_page_size = tid2eid_buffer->page_size();
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
    add_cb(cb_out_weights, weights_page_size, 2 * n_activated_expert_tiles, weights_data_format);
    add_cb(cb_out_indices, indices_page_size, 2 * n_activated_expert_tiles, indices_data_format);
    add_cb(cb_sigmoid_scores, scores_page_size, width_tiles, scores_data_format);
    add_cb(cb_reduce_intermediate, scores_page_size, 2 * n_activated_expert_tiles, scores_data_format);
    add_cb(cb_reduce_ones_scalar, scores_page_size, 1, scores_data_format);
    add_cb(cb_epsilon_scalar, scores_page_size, 1, scores_data_format);
    add_cb(cb_route_scale_scalar, scores_page_size, 1, scores_data_format);
    add_cb(cb_normalized_scores, scores_page_size, 2 * n_activated_expert_tiles, scores_data_format);
    add_cb(cb_reciprocal_sums, scores_page_size, 2 * n_activated_expert_tiles, scores_data_format);
    add_cb(cb_gathered_sigmoid, scores_page_size, 2 * n_activated_expert_tiles, scores_data_format);

    // input_ids: one ROW_MAJOR page per height tile (tile_height uint32 token ids).
    add_cb(cb_input_ids, input_ids_page_size, 2, tt::DataFormat::UInt32);
    // tid2eid scratch: hold all tile_height looked-up rows for a tile before assembling the index tile.
    // Rows are strided by the DRAM-aligned size so each per-token read lands at an aligned destination.
    add_cb(cb_tid2eid_row, tile_height * tid2eid_row_stride, 1, tt::DataFormat::UInt16);

    // Scratch CB for the optional [num_real_tokens, pad_side] padding config row (see writer). When no
    // padding config is supplied we fall back to the output_indices buffer purely to size the CB.
    auto* padding_config_buffer =
        tensor_args.padding_config.has_value() ? tensor_args.padding_config->buffer() : indices_buffer;
    TT_FATAL(padding_config_buffer != nullptr, "padding config buffer must be allocated on device");
    uint32_t padding_config_page_size = static_cast<uint32_t>(padding_config_buffer->aligned_page_size());
    add_cb(cb_padding_config, padding_config_page_size, 1, tt::DataFormat::UInt32);
    Buffer* padding_runtime_buffer =
        tensor_args.padding_config.has_value() ? tensor_args.padding_config->buffer() : nullptr;

    // --- Reader: logits -> cb_in_scores; tid2eid[input_ids] -> cb_out_indices ---
    KernelDescriptor::NamedCompileTimeArgs reader_named_compile_time_args = {
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
    tt::tt_metal::TensorAccessorArgs(scores_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(input_ids_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(tid2eid_buffer).append_to(reader_compile_time_args);

    // --- Compute: apply_score_func -> normalize_scores -> scale ---
    KernelDescriptor::NamedCompileTimeArgs compute_named_compile_time_args = {
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

    // --- Writer: gather + sentinel patch + write ---
    KernelDescriptor::NamedCompileTimeArgs writer_named_compile_time_args = {
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
    tt::tt_metal::TensorAccessorArgs(weights_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(indices_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(padding_config_buffer).append_to(writer_compile_time_args);

    KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_hash_gate/device/kernels/dataflow/"
        "reader_moe_hash_gate.cpp";
    reader_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = all_cores;
    reader_kernel_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_kernel_desc.named_compile_time_args = std::move(reader_named_compile_time_args);
    reader_kernel_desc.config = ReaderConfigDescriptor{};

    bool fp32_dest_acc_en = true;
    KernelDescriptor compute_kernel_desc;
    compute_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_hash_gate/device/kernels/compute/"
        "moe_hash_gate.cpp";
    compute_kernel_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_kernel_desc.core_ranges = all_cores;
    compute_kernel_desc.named_compile_time_args = std::move(compute_named_compile_time_args);
    compute_kernel_desc.config = ComputeConfigDescriptor{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .dst_full_sync_en = false,
        .math_approx_mode = false,
    };

    KernelDescriptor writer_kernel_desc;
    writer_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/moe_hash_gate/device/kernels/dataflow/"
        "writer_moe_hash_gate.cpp";
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

        reader_kernel_desc.emplace_runtime_args(
            core, {scores_buffer, input_ids_buffer, tid2eid_buffer, start_height_tile, end_height_tile});
        compute_kernel_desc.emplace_runtime_args(core, {start_height_tile, end_height_tile});
        writer_kernel_desc.emplace_runtime_args(
            core, {weights_buffer, indices_buffer, start_height_tile, end_height_tile, padding_runtime_buffer});
    }

    desc.kernels.push_back(std::move(reader_kernel_desc));
    desc.kernels.push_back(std::move(compute_kernel_desc));
    desc.kernels.push_back(std::move(writer_kernel_desc));
    return desc;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_hash_gate
