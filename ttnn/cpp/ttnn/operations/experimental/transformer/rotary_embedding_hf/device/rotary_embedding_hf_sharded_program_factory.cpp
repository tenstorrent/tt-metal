// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "rotary_embedding_hf_sharded_program_factory.hpp"
#include <bit>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace {

ProgramDescriptor create_single_tile_decode_descriptor(
    const RotaryEmbeddingHfParams& operation_attributes, const RotaryEmbeddingHfInputs& tensor_args, Tensor& output) {
    ProgramDescriptor desc;

    const auto& input = tensor_args.input_tensor;
    const auto& cos = tensor_args.cos_cache;
    const auto& sin = tensor_args.sin_cache;

    const tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    const tt::DataFormat cos_cb_data_format = tt_metal::datatype_to_dataformat_converter(cos.dtype());
    const uint32_t cos_single_tile_size = tt::tile_size(cos_cb_data_format);

    const tt::DataFormat sin_cb_data_format = tt_metal::datatype_to_dataformat_converter(sin.dtype());
    const uint32_t sin_single_tile_size = tt::tile_size(sin_cb_data_format);

    const tt::DataFormat trans_mat_cb_data_format = input_cb_data_format == tt::DataFormat::Bfp8_b
                                                        ? tt::DataFormat::Bfp8_b
                                                    : input_cb_data_format == tt::DataFormat::Float32
                                                        ? tt::DataFormat::Float32
                                                        : tt::DataFormat::Float16_b;
    const uint32_t trans_mat_single_tile_size = tt::tile_size(trans_mat_cb_data_format);

    const tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    bool in_sharded = input.shard_spec().has_value();
    std::optional<ShardSpec> shard_spec = in_sharded ? input.shard_spec() : output.shard_spec();

    const uint32_t batch = input.padded_shape()[1];
    const uint32_t n_heads_t = shard_spec->shape[0] / constants::TILE_HEIGHT;
    const uint32_t n_heads_per_batch_t = input.padded_shape()[2] / constants::TILE_HEIGHT;
    constexpr uint32_t head_dim_t = 1;

    tt_metal::IDevice* device = input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    CoreRange all_cores = shard_spec->grid.bounding_box();
    uint32_t num_cores_x = all_cores.grid_size().x;
    uint32_t num_cores_y = all_cores.grid_size().y;

    const uint32_t num_input_tiles = n_heads_t * head_dim_t;
    const uint32_t num_output_tiles = num_input_tiles;

    const uint32_t num_cores = num_cores_x * num_cores_y;
    const uint32_t batch_parallel_factor = std::min(batch, num_cores);
    const uint32_t batch_per_core = (batch + batch_parallel_factor - 1) / batch_parallel_factor;
    const uint32_t num_cos_sin_tiles = head_dim_t * batch_per_core;

    auto* src_buffer = input.buffer();
    auto* cos_buffer = cos.buffer();
    auto* sin_buffer = sin.buffer();
    auto* dst_buffer = output.buffer();

    constexpr uint8_t input_cb_index = CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * input_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = input_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
        .buffer = src_buffer,
    });

    constexpr uint8_t cos_cb_index = CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cos_sin_tiles * cos_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cos_cb_index,
            .data_format = cos_cb_data_format,
            .page_size = cos_single_tile_size,
        }}},
        .buffer = cos_buffer,
    });

    constexpr uint8_t sin_cb_index = CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cos_sin_tiles * sin_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = sin_cb_index,
            .data_format = sin_cb_data_format,
            .page_size = sin_single_tile_size,
        }}},
        .buffer = sin_buffer,
    });

    constexpr uint8_t trans_mat_cb_index = CBIndex::c_3;
    desc.cbs.push_back(CBDescriptor{
        .total_size = trans_mat_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = trans_mat_cb_index,
            .data_format = trans_mat_cb_data_format,
            .page_size = trans_mat_single_tile_size,
        }}},
    });

    uint32_t num_interm_tiles = head_dim_t;
    constexpr uint8_t rotated_input_interm_cb_index = CBIndex::c_24;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * input_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = rotated_input_interm_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
    });

    constexpr uint8_t cos_interm_cb_index = CBIndex::c_25;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * input_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cos_interm_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
    });

    constexpr uint8_t sin_interm_cb_index = CBIndex::c_26;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * input_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = sin_interm_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
    });

    constexpr uint8_t output_cb_index = CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * output_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
        .buffer = dst_buffer,
    });

    std::vector<uint32_t> compute_kernel_args = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)cos_cb_index,
        (std::uint32_t)sin_cb_index,
        (std::uint32_t)trans_mat_cb_index,
        (std::uint32_t)rotated_input_interm_cb_index,
        (std::uint32_t)cos_interm_cb_index,
        (std::uint32_t)sin_interm_cb_index,
        (std::uint32_t)output_cb_index,
        (std::uint32_t)n_heads_per_batch_t,
        (std::uint32_t)batch_per_core,
    };

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/compute/"
        "rotary_embedding_hf_single_tile_sharded.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = CoreRangeSet(all_cores);
    compute_desc.compile_time_args = std::move(compute_kernel_args);
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)trans_mat_cb_index,
    };
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/dataflow/"
        "reader_rotary_embedding_hf_single_tile_sharded.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = CoreRangeSet(all_cores);
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

ProgramDescriptor create_multi_tile_decode_descriptor(
    const RotaryEmbeddingHfParams& operation_attributes, const RotaryEmbeddingHfInputs& tensor_args, Tensor& output) {
    ProgramDescriptor desc;

    const auto& input = tensor_args.input_tensor;
    const auto& cos = tensor_args.cos_cache;
    const auto& sin = tensor_args.sin_cache;

    const tt::DataFormat input_cb_data_format = tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    const tt::DataFormat cos_cb_data_format = tt_metal::datatype_to_dataformat_converter(cos.dtype());
    const uint32_t cos_single_tile_size = tt::tile_size(cos_cb_data_format);

    const tt::DataFormat sin_cb_data_format = tt_metal::datatype_to_dataformat_converter(sin.dtype());
    const uint32_t sin_single_tile_size = tt::tile_size(sin_cb_data_format);

    const tt::DataFormat output_cb_data_format = tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const tt::DataFormat scalar_cb_data_format = tt::DataFormat::Float16_b;
    const uint32_t scalar_single_tile_size = tt::tile_size(scalar_cb_data_format);

    bool in_sharded = input.shard_spec().has_value();
    std::optional<ShardSpec> shard_spec = in_sharded ? input.shard_spec() : output.shard_spec();

    const uint32_t batch = input.padded_shape()[1];
    const uint32_t n_heads_t = shard_spec->shape[0] / constants::TILE_HEIGHT;
    const uint32_t n_heads_per_batch_t = input.padded_shape()[2] / constants::TILE_HEIGHT;
    const uint32_t head_dim_t = shard_spec->shape[1] / constants::TILE_WIDTH;

    tt_metal::IDevice* device = input.device();

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(device->arch(), operation_attributes.compute_kernel_config);

    CoreRange all_cores = shard_spec->grid.bounding_box();
    uint32_t num_cores_x = all_cores.grid_size().x;
    uint32_t num_cores_y = all_cores.grid_size().y;

    const uint32_t num_input_tiles = n_heads_t * head_dim_t;
    const uint32_t num_output_tiles = num_input_tiles;

    const uint32_t num_cores = num_cores_x * num_cores_y;
    const uint32_t batch_parallel_factor = std::min(batch, num_cores);
    const uint32_t batch_per_core = (batch + batch_parallel_factor - 1) / batch_parallel_factor;

    const uint32_t num_sin_cos_rows_per_core = batch_per_core;
    uint32_t num_cos_sin_tiles = head_dim_t * num_sin_cos_rows_per_core;

    auto* src_buffer = input.buffer();
    auto* cos_buffer = cos.buffer();
    auto* sin_buffer = sin.buffer();
    auto* dst_buffer = output.buffer();

    constexpr uint8_t input_cb_index = CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * input_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = input_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
        .buffer = src_buffer,
    });

    constexpr uint8_t cos_cb_index = CBIndex::c_1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cos_sin_tiles * cos_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cos_cb_index,
            .data_format = cos_cb_data_format,
            .page_size = cos_single_tile_size,
        }}},
        .buffer = cos_buffer,
    });

    constexpr uint8_t sin_cb_index = CBIndex::c_2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_cos_sin_tiles * sin_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = sin_cb_index,
            .data_format = sin_cb_data_format,
            .page_size = sin_single_tile_size,
        }}},
        .buffer = sin_buffer,
    });

    constexpr uint8_t src_scalar_cb_index = CBIndex::c_3;
    uint32_t num_scalar_tiles = 1;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_scalar_tiles * scalar_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src_scalar_cb_index,
            .data_format = scalar_cb_data_format,
            .page_size = scalar_single_tile_size,
        }}},
    });

    uint32_t num_interm_tiles = head_dim_t;
    constexpr uint8_t rotated_input_interm_cb_index = CBIndex::c_24;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * input_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = rotated_input_interm_cb_index,
            .data_format = input_cb_data_format,
            .page_size = input_single_tile_size,
        }}},
    });

    constexpr uint8_t cos_interm_cb_index = CBIndex::c_25;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * cos_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cos_interm_cb_index,
            .data_format = cos_cb_data_format,
            .page_size = cos_single_tile_size,
        }}},
    });

    constexpr uint8_t sin_interm_cb_index = CBIndex::c_26;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_interm_tiles * sin_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = sin_interm_cb_index,
            .data_format = sin_cb_data_format,
            .page_size = sin_single_tile_size,
        }}},
    });

    constexpr uint8_t output_cb_index = CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * output_single_tile_size,
        .core_ranges = CoreRangeSet(all_cores),
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = output_cb_data_format,
            .page_size = output_single_tile_size,
        }}},
        .buffer = dst_buffer,
    });

    std::vector<uint32_t> compute_kernel_args = {
        (std::uint32_t)input_cb_index,
        (std::uint32_t)cos_cb_index,
        (std::uint32_t)sin_cb_index,
        (std::uint32_t)src_scalar_cb_index,
        (std::uint32_t)rotated_input_interm_cb_index,
        (std::uint32_t)cos_interm_cb_index,
        (std::uint32_t)sin_interm_cb_index,
        (std::uint32_t)output_cb_index,
        (std::uint32_t)head_dim_t,
        (std::uint32_t)n_heads_t,
        (std::uint32_t)n_heads_per_batch_t,
        (std::uint32_t)batch_per_core,
    };

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/compute/"
        "rotary_embedding_hf_sharded.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = CoreRangeSet(all_cores);
    compute_desc.compile_time_args = std::move(compute_kernel_args);
    compute_desc.config = ComputeConfigDescriptor{
        .math_fidelity = math_fidelity,
        .fp32_dest_acc_en = fp32_dest_acc_en,
    };

    const uint16_t bfloat16_neg_one = std::bit_cast<uint16_t>(bfloat16(-1.0f));
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)src_scalar_cb_index,
        (std::uint32_t)bfloat16_neg_one,
    };
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/rotary_embedding_hf/device/kernels/dataflow/"
        "reader_rotary_embedding_hf_sharded.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = CoreRangeSet(all_cores);
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace

ProgramDescriptor RotaryEmbeddingHfMultiCoreSharded::create_descriptor(
    const RotaryEmbeddingHfParams& operation_attributes, const RotaryEmbeddingHfInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input_tensor;
    if (input.padded_shape()[-1] / TILE_WIDTH == 1) {
        return create_single_tile_decode_descriptor(operation_attributes, tensor_args, output);
    }
    return create_multi_tile_decode_descriptor(operation_attributes, tensor_args, output);
}

}  // namespace ttnn::experimental::prim
