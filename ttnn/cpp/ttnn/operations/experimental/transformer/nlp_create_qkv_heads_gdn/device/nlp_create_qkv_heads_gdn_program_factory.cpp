// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/buffer.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

#include "nlp_create_qkv_heads_gdn_device_operation.hpp"

namespace ttnn::operations::experimental::transformer {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

ProgramDescriptor NlpCreateHeadsGdnDeviceOperation::Interleaved::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const Tensor& input_tensor = tensor_args.input_tensor;
    const uint32_t num_q_heads = operation_attributes.num_q_heads;
    const uint32_t num_k_heads = operation_attributes.num_k_heads;
    const uint32_t num_v_heads = operation_attributes.num_v_heads;
    const uint32_t head_dim = operation_attributes.head_dim;
    auto& output = tensor_return_value;
    CoreCoord compute_with_storage_grid_size = input_tensor.device()->compute_with_storage_grid_size();

    const auto& input_shape = input_tensor.padded_shape();
    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt_metal::Buffer* in0_buffer = input_tensor.buffer();
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);

    // TM params. Output h/w dims (S tiles, head_dim tiles) are identical for Q/K/V (shared head_dim),
    // so the per-head block HtWt is shared; only the head COUNT (and thus the batch stride) differs.
    uint32_t in0_w_tiles = input_shape[3] / TILE_WIDTH;
    uint32_t out_h_tiles = input_shape[2] / TILE_HEIGHT;
    uint32_t out_w_tiles = head_dim / TILE_WIDTH;  // tiles along head_dim
    uint32_t out_HtWt = out_h_tiles * out_w_tiles;
    uint32_t q_out_CHtWt = num_q_heads * out_HtWt;
    uint32_t k_out_CHtWt = num_k_heads * out_HtWt;
    uint32_t v_out_CHtWt = num_v_heads * out_HtWt;
    uint32_t q_num_tiles = num_q_heads * out_w_tiles;
    uint32_t k_num_tiles = num_k_heads * out_w_tiles;
    uint32_t v_num_tiles = num_v_heads * out_w_tiles;

    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block = one seq tile-row (TILE_HEIGHT tokens) = in0_w_tiles fused-width tiles.
    uint32_t num_blocks = input_shape[0] * input_shape[1] * input_shape[2] / TILE_HEIGHT;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);

    tt_metal::Tensor& q = std::get<0>(output);
    tt_metal::Tensor& k = std::get<1>(output);
    tt_metal::Tensor& v = std::get<2>(output);
    tt_metal::Buffer* q_buffer = q.buffer();
    tt_metal::Buffer* k_buffer = k.buffer();
    tt_metal::Buffer* v_buffer = v.buffer();
    TT_ASSERT(q_buffer != nullptr && k_buffer != nullptr && v_buffer != nullptr, "Output buffers must be allocated!");

    ProgramDescriptor desc;

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)q_num_tiles,
        (std::uint32_t)k_num_tiles,
        (std::uint32_t)v_num_tiles,
    };
    tt::tt_metal::TensorAccessorArgs(in0_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)out_h_tiles,
        (std::uint32_t)out_w_tiles,
        (std::uint32_t)out_HtWt,
        (std::uint32_t)num_q_heads,
        (std::uint32_t)num_k_heads,
        (std::uint32_t)num_v_heads,
    };
    tt::tt_metal::TensorAccessorArgs(q_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(k_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(v_buffer).append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_gdn/device/kernels/dataflow/"
        "reader_nlp_create_qkv_heads_gdn.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_gdn/device/kernels/dataflow/"
        "writer_nlp_create_qkv_heads_gdn.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Single circular buffer (cb1) — Q, K, V all flow through it in read order (no transpose).
    // Sized to hold a whole block (qkv_tiles = in0_w_tiles) double-buffered: the reader fills the
    // block behind one barrier and the writer drains it behind one barrier, so the two kernels
    // pipeline one block apart. The total MUST stay an integer multiple of the block size so each
    // reserve_back(qkv_tiles)/wait_front(qkv_tiles) region is contiguous in L1 (no ring wrap
    // mid-block) — the ×2 factor keeps that invariant.
    uint32_t cb_index = 1;
    constexpr uint32_t kBufferFactor = 2;                 // double-buffer the block for reader/writer overlap
    uint32_t cb_num_tiles = kBufferFactor * in0_w_tiles;  // in0_w_tiles == qkv_tiles (per block)
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_num_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_index),
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    for (uint32_t i = 0, num_blocks_written = 0; i < num_cores; i++) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        uint32_t num_blocks_per_core = 0;
        if (core_group_1.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_blocks_per_core = num_blocks_per_core_group_2;
        } else {
            TT_ASSERT(false, "Core not in specified core ranges");
        }

        uint32_t out_h_dim = num_blocks_written % out_h_tiles;
        uint32_t q_out_tensor_tile_id = (num_blocks_written / out_h_tiles * q_out_CHtWt) + (out_h_dim * out_w_tiles);
        uint32_t k_out_tensor_tile_id = (num_blocks_written / out_h_tiles * k_out_CHtWt) + (out_h_dim * out_w_tiles);
        uint32_t v_out_tensor_tile_id = (num_blocks_written / out_h_tiles * v_out_CHtWt) + (out_h_dim * out_w_tiles);

        KernelDescriptor::RTArgList reader_rt;
        reader_rt.reserve(3);
        reader_rt.push_back(in0_buffer);
        reader_rt.push_back(num_blocks_per_core);
        reader_rt.push_back(num_blocks_written * in0_w_tiles);
        reader_desc.emplace_runtime_args(core, reader_rt);

        writer_desc.emplace_runtime_args(
            core,
            {
                q_buffer,
                k_buffer,
                v_buffer,
                num_blocks_per_core,
                out_h_dim,
                q_out_tensor_tile_id,
                k_out_tensor_tile_id,
                v_out_tensor_tile_id,
            });

        num_blocks_written += num_blocks_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::operations::experimental::transformer
