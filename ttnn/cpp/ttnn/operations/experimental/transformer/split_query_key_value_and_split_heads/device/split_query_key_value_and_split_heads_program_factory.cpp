// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "split_query_key_value_and_split_heads_program_factory.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt_metal;

tt::tt_metal::ProgramDescriptor SplitFusedQKVAndSplitHeadsProgramFactory::create_descriptor(
    const SplitQueryKeyValueAndSplitHeadsParams& operation_attributes,
    const SplitQueryKeyValueAndSplitHeadsInputs& tensor_args,
    std::vector<Tensor>& output_tensors) {
    ProgramDescriptor desc;

    const auto& a = tensor_args.input_tensor;
    auto& output = output_tensors;
    auto compute_with_storage_grid_size = operation_attributes.compute_with_storage_grid_size;

    const auto& ashape = a.padded_shape();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt_metal::Buffer* in0_buffer = a.buffer();
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t per_core_tiles = ashape[3] / TILE_WIDTH;  // 96
    uint32_t num_tensors = 3;
    uint32_t num_tiles_per_tensor = per_core_tiles / num_tensors;  // 32
    uint32_t block_size = 1;  // TODO: Play around with different reader and writer block_sizes?
    bool block_size_is_one = block_size == 1;
    uint32_t num_blocks_per_tensor = num_tiles_per_tensor / block_size;

    // Per output tensor args
    // Output shape is: [B, 16, 384, 64] (Q, V heads) or [B, 16, 64, 384] (K heads)
    // For K heads, we write "w_dim" to h_dim instead, but keep nomenclature the same
    uint32_t out_h_tiles = ashape[2] / TILE_HEIGHT;
    uint32_t out_w = 64;
    uint32_t out_w_tiles = out_w / TILE_WIDTH;
    uint32_t out_c = num_tiles_per_tensor / out_w_tiles;
    uint32_t out_HtWt = out_h_tiles * out_w_tiles;
    uint32_t out_CHtWt = out_c * out_HtWt;
    // If block_size_is_one, writer kernel waits differently
    uint32_t writer_num_blocks_per_tensor = block_size_is_one ? 16 : num_blocks_per_tensor;
    uint32_t num_c_per_block = block_size_is_one ? 1 : block_size / out_w_tiles;

    // Parallelize ashape[2] (384 / 32 = 12 tiles) across columns
    // Parallelize ashape[0] (B) across rows
    uint32_t num_cores_x = ashape[2] / TILE_HEIGHT;
    uint32_t num_cores_y = ashape[0];
    TT_ASSERT(num_cores_x <= compute_with_storage_grid_size.x);
    TT_ASSERT(num_cores_y <= compute_with_storage_grid_size.y);
    CoreCoord core_range = {num_cores_x, num_cores_y};

    TT_ASSERT((output.size() == 3), "Output vector must be size 3 for split fused qkv!");
    tt_metal::Tensor& q = output[0];
    tt_metal::Tensor& k = output[1];
    tt_metal::Tensor& v = output[2];

    tt_metal::Buffer* q_buffer = q.buffer();
    TT_ASSERT(q_buffer != nullptr, "Output q buffer should be allocated on device!");
    tt_metal::Buffer* k_buffer = k.buffer();
    TT_ASSERT(k_buffer != nullptr, "Output k buffer should be allocated on device!");
    tt_metal::Buffer* v_buffer = v.buffer();
    TT_ASSERT(v_buffer != nullptr, "Output v buffer should be allocated on device!");

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    CoreRange all_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});
    CoreRangeSet all_cores_set{all_cores};

    std::vector<uint32_t> reader_compile_time_args = {
        // READER COMPILE TIME ARGS
        (std::uint32_t)block_size,             // block_size
        (std::uint32_t)num_blocks_per_tensor,  // out_num_blocks_per_tensor
    };
    tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        // WRITER COMPILE TIME ARGS
        (std::uint32_t)block_size_is_one,
        (std::uint32_t)block_size,                    // block_size
        (std::uint32_t)writer_num_blocks_per_tensor,  // out_num_blocks_per_tensor
        (std::uint32_t)num_c_per_block,               // out_num_c_per_block
        (std::uint32_t)out_w_tiles,                   // out_w_tiles
        (std::uint32_t)out_h_tiles,                   // out_h_tiles
        (std::uint32_t)out_HtWt,                      // out_HtWt
    };
    tt::tt_metal::TensorAccessorArgs(*q_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*k_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*v_buffer).append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/"
        "dataflow/reader_tm_tile_layout_create_qkv_heads.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores_set;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/"
        "dataflow/writer_tm_tile_layout_create_qkv_heads.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores_set;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Dummy compute kernel
    std::vector<uint32_t> compute_args = {num_tiles_per_tensor};
    KernelDescriptor compute_desc;
    compute_desc.kernel_source = "ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores_set;
    compute_desc.compile_time_args = std::move(compute_args);
    compute_desc.config = ComputeConfigDescriptor{};

    // Create circular buffers
    // Use cb0 and cb16 for K heads, which uses compute for transpose_wh
    constexpr uint8_t src0_cb_index = 0;
    uint32_t cb0_tiles = num_tiles_per_tensor * 2;  // double buffer
    constexpr uint8_t out_cb_index = 16;
    uint32_t out_cb_tiles = num_tiles_per_tensor;
    // Use cb1 as input and output for Q and V heads
    constexpr uint8_t src1_cb_index = 1;
    uint32_t cb1_tiles = num_tiles_per_tensor * 4;  // 2 tensors + double buffer

    desc.cbs.push_back(CBDescriptor{
        .total_size = cb0_tiles * single_tile_size,
        .core_ranges = all_cores_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = cb1_tiles * single_tile_size,
        .core_ranges = all_cores_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src1_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    desc.cbs.push_back(CBDescriptor{
        .total_size = out_cb_tiles * single_tile_size,
        .core_ranges = all_cores_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = out_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    reader_desc.runtime_args.reserve(static_cast<size_t>(num_cores_r) * num_cores_c);
    writer_desc.runtime_args.reserve(static_cast<size_t>(num_cores_r) * num_cores_c);
    for (int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for (int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};

            reader_desc.emplace_runtime_args(
                core,
                {
                    in0_buffer,  // in0_tensor_addr
                    static_cast<uint32_t>(
                        (core_idx_x + core_idx_y * num_cores_c) * per_core_tiles),  // in0_tensor_tile_id
                });
            writer_desc.emplace_runtime_args(
                core,
                {
                    q_buffer,                                                                      // q_tensor_addr
                    k_buffer,                                                                      // k_tensor_addr
                    v_buffer,                                                                      // v_tensor_addr
                    static_cast<uint32_t>((core_idx_x * out_w_tiles) + (core_idx_y * out_CHtWt)),  // out_tensor_tile_id
                    static_cast<uint32_t>(core_idx_x + (core_idx_y * out_CHtWt)),  // out_tensor_tile_id_with_transpose
                });
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
