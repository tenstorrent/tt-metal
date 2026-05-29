// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_llama_fused_qk_device_operation_types.hpp"
#include "rotary_embedding_llama_fused_qk_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

ProgramDescriptor RotaryEmbeddingLlamaFusedQKProgramFactory::create_descriptor(
    const RotaryEmbeddingLlamaFusedQkParams& operation_attributes,
    const RotaryEmbeddingLlamaFusedQkInputs& tensor_args,
    RotaryEmbeddingLlamaFusedQkResult& tensor_return_value) {
    ProgramDescriptor desc;

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

    constexpr uint8_t q_input_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_q_input_tiles * input_single_tile_size,
        .core_ranges = all_cores_bb,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = q_input_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
        .buffer = q_src_buffer,
    });

    constexpr uint8_t k_input_cb_index = tt::CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_k_input_tiles * input_single_tile_size,
        .core_ranges = all_cores_bb,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = k_input_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
        .buffer = k_src_buffer,
    });

    constexpr uint8_t cos_cb_index = tt::CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cos_sin_tiles * cos_single_tile_size,
        .core_ranges = all_cores_bb,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cos_cb_index,
            .data_format = cos_cb_data_format,
            .page_size = cos_single_tile_size,
        }}},
        .buffer = cos_buffer,
    });

    constexpr uint8_t sin_cb_index = tt::CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cos_sin_tiles * sin_single_tile_size,
        .core_ranges = all_cores_bb,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = sin_cb_index,
            .data_format = sin_cb_data_format,
            .page_size = sin_single_tile_size,
        }}},
        .buffer = sin_buffer,
    });

    constexpr uint8_t trans_mat_cb_index = tt::CBIndex::c_4;
    // We only take one tile of trans_mat
    uint32_t num_trans_mat_tiles = 1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_trans_mat_tiles * trans_mat_single_tile_size,
        .core_ranges = all_cores_bb,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = trans_mat_cb_index,
            .data_format = trans_mat_cb_data_format,
            .page_size = trans_mat_single_tile_size,
        }}},
        .buffer = trans_mat_buffer,
    });

    uint32_t num_interm_tiles = head_dim_t;
    constexpr uint8_t rotated_input_interm_cb_index = tt::CBIndex::c_24;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * input_single_tile_size,
        .core_ranges = all_cores_bb,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = rotated_input_interm_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
    });

    constexpr uint8_t cos_interm_cb_index = tt::CBIndex::c_25;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * input_single_tile_size,
        .core_ranges = all_cores_bb,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cos_interm_cb_index,
            .data_format = cos_cb_data_format,
            .page_size = cos_single_tile_size,
        }}},
    });

    constexpr uint8_t sin_interm_cb_index = tt::CBIndex::c_26;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * input_single_tile_size,
        .core_ranges = all_cores_bb,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = sin_interm_cb_index,
            .data_format = sin_cb_data_format,
            .page_size = sin_single_tile_size,
        }}},
    });

    constexpr uint8_t q_output_cb_index = tt::CBIndex::c_16;  // output operands start at index 16
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_q_output_tiles * output_single_tile_size,
        .core_ranges = all_cores_bb,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = q_output_cb_index,
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
        .buffer = q_dst_buffer,
    });
    constexpr uint8_t k_output_cb_index = tt::CBIndex::c_17;  // output operands start at index 17
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_k_output_tiles * output_single_tile_size,
        .core_ranges = all_cores_bb,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = k_output_cb_index,
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
        .buffer = k_dst_buffer,
    });

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

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = compute_kernel_path;
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores_bb;
    compute_desc.compile_time_args = std::move(compute_kernel_args);
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    // Runtime args to differentiate between q, k or no work groups
    // TODO: Turn off unused compute cores? (technically, it doesn't matter since only compute kernel)
    // Running into code size issues on TRISC2 with profiler turned on; need to reduce stack size by 4B
    // constexpr bool has_work = true;
    constexpr uint32_t is_q_arg = 1;  // If not q, must be k
    constexpr uint32_t is_k_arg = 0;
    const auto q_cores_vec = corerange_to_cores(q_cores, std::nullopt, /*row_wise=*/true);
    const auto k_cores_vec = corerange_to_cores(k_cores, std::nullopt, /*row_wise=*/true);
    compute_desc.runtime_args.reserve(q_cores_vec.size() + k_cores_vec.size());
    for (const auto& core : q_cores_vec) {
        compute_desc.runtime_args.emplace_back(core, std::vector<uint32_t>{is_q_arg});
    }
    for (const auto& core : k_cores_vec) {
        compute_desc.runtime_args.emplace_back(core, std::vector<uint32_t>{is_k_arg});
    }

    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
