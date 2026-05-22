// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "split_query_key_value_and_split_heads_sharded_program_factory.hpp"

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt_metal;

tt::tt_metal::ProgramDescriptor SplitFusedQKVAndSplitHeadsShardedProgramFactory::create_descriptor(
    const SplitQueryKeyValueAndSplitHeadsParams& /*operation_attributes*/,
    const SplitQueryKeyValueAndSplitHeadsInputs& tensor_args,
    std::vector<Tensor>& output_tensors) {
    ProgramDescriptor desc;

    const auto& a = tensor_args.input_tensor;
    auto& output = output_tensors;

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    auto all_cores = a.shard_spec().value().grid;
    auto bbox = all_cores.bounding_box();
    ShardOrientation shard_orientation = a.shard_spec().value().orientation;
    bool rm = shard_orientation == ShardOrientation::ROW_MAJOR;
    uint32_t num_h_cores = rm ? bbox.end_coord.y + 1 : bbox.end_coord.x + 1;
    uint32_t num_w_cores = rm ? bbox.end_coord.x + 1 : bbox.end_coord.y + 1;
    // tensor shape
    const auto& shape = a.padded_shape();
    uint32_t M = shape[2] * shape[0];  // 4608
    uint32_t K = shape[3];             // 3072
    uint32_t Mt = M / TILE_WIDTH;
    uint32_t Kt = K / TILE_WIDTH;
    uint32_t num_tensors = 3;
    uint32_t num_heads_per_tensor = 2;
    // block
    uint32_t block_w = K / num_w_cores;  // 384
    uint32_t block_h = M / num_h_cores;  // 384
    uint32_t block_wt = block_w / TILE_WIDTH;
    uint32_t block_ht = block_h / TILE_WIDTH;
    uint32_t out_block_w = block_w / num_tensors / num_heads_per_tensor;  // 64
    uint32_t out_block_wt = out_block_w / TILE_WIDTH;                     // 2
    uint32_t out_block_h = block_h * num_heads_per_tensor;                // 768
    uint32_t out_block_ht = out_block_h / TILE_WIDTH;                     // 24
    uint32_t per_core_tiles = block_ht * block_wt;
    uint32_t num_tiles_per_tensor = per_core_tiles / num_tensors;
    // check dims
    TT_ASSERT(M % TILE_WIDTH == 0 && "M must be divisible by tile width.");
    TT_ASSERT(K % TILE_WIDTH == 0 && "K must be divisible by tile width.");
    TT_ASSERT(Kt / num_w_cores == block_wt && "block_w must equal to K / num_cores_w.");
    TT_ASSERT(Mt / num_h_cores == block_ht && "block_h must equal to M / num_cores_h.");

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // block size for in0 (tensor a)
    uint32_t in0_CB_size = block_wt * block_ht * single_tile_size;
    // uint32_t im0_CB_size = 2 * single_tile_size;
    uint32_t im0_CB_size = 2 * block_ht * single_tile_size;
    uint32_t out_CB_size = out_block_wt * out_block_ht * single_tile_size;

    // reader compile arg
    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)num_heads_per_tensor,
        (std::uint32_t)block_ht,
        (std::uint32_t)block_wt,
        (std::uint32_t)out_block_wt,
        (std::uint32_t)block_wt * single_tile_size,
        (std::uint32_t)out_block_wt * single_tile_size,
        (std::uint32_t)num_tiles_per_tensor,
        (std::uint32_t)block_wt * single_tile_size / num_tensors};
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/"
        "dataflow/reader_tm_tile_layout_create_qkv_heads_sharded.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    // writer
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)num_heads_per_tensor,
        (std::uint32_t)block_ht,
        (std::uint32_t)block_wt,
        (std::uint32_t)out_block_wt,
        (std::uint32_t)block_wt * single_tile_size,
        (std::uint32_t)out_block_wt * single_tile_size,
        (std::uint32_t)num_tiles_per_tensor,
        (std::uint32_t)block_wt * single_tile_size / num_tensors};
    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/"
        "dataflow/writer_tm_tile_layout_create_qkv_heads_sharded.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // compute kernel
    std::vector<uint32_t> compute_args = {num_tiles_per_tensor};
    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/split_query_key_value_and_split_heads/device/kernels/"
        "compute/transpose_wh_sharded.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = std::move(compute_args);
    compute_desc.config = ComputeConfigDescriptor{};

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    // Create circular buffers
    // in0 sharded
    desc.cbs.push_back(CBDescriptor{
        .total_size = in0_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = CBIndex::c_0,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
        .buffer = a.buffer(),
    });
    // im
    desc.cbs.push_back(CBDescriptor{
        .total_size = im0_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = CBIndex::c_24,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });
    // q sharded
    desc.cbs.push_back(CBDescriptor{
        .total_size = out_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = CBIndex::c_16,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
        .buffer = output[0].buffer(),
    });
    // k sharded
    desc.cbs.push_back(CBDescriptor{
        .total_size = out_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = CBIndex::c_17,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
        .buffer = output[1].buffer(),
    });
    // v sharded
    desc.cbs.push_back(CBDescriptor{
        .total_size = out_CB_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = CBIndex::c_18,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
        .buffer = output[2].buffer(),
    });

    return desc;
}

}  // namespace ttnn::experimental::prim
