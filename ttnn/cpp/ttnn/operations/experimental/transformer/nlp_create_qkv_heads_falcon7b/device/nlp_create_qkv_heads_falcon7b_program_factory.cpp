// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "nlp_create_qkv_heads_falcon7b_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor NlpCreateQkvHeadsFalcon7BProgramFactory::create_descriptor(
    const NlpCreateQkvHeadsFalcon7bParams& /*operation_attributes*/,
    const Tensor& tensor_args,
    NlpCreateQkvHeadsFalcon7bResult& tensor_return_value) {
    ProgramDescriptor desc;

    const auto& a = tensor_args;
    const auto& ashape = a.padded_shape();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt_metal::Buffer* in0_buffer = a.buffer();
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t per_tensor_tiles = ashape[3] / TILE_WIDTH;  // 146
    uint32_t q_num_tiles_per_tensor = 142;
    uint32_t kv_num_tiles_per_tensor = 2;

    // Per output tensor args
    // Output shape for Q is: [B, 71, s, 64] # Needs shuffling from [B, 1, s, 4544]
    // Output shape for K/V is: [B, 1, s, 64] # Just split, no shuffling after
    uint32_t q_out_h_tiles = ashape[2] / TILE_HEIGHT;
    uint32_t q_out_w_tiles = 2;                                 // head_dim
    uint32_t q_out_c = q_num_tiles_per_tensor / q_out_w_tiles;  // num_heads
    uint32_t q_out_HtWt = q_out_h_tiles * q_out_w_tiles;
    uint32_t q_out_CHtWt = q_out_c * q_out_HtWt;

    CoreCoord compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of per_tensor_tiles per core
    uint32_t num_blocks = ashape[0] * ashape[1] * ashape[2] / TILE_HEIGHT;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);

    tt_metal::Tensor& q = tensor_return_value.q;
    tt_metal::Tensor& k = tensor_return_value.k;
    tt_metal::Tensor& v = tensor_return_value.v;

    tt_metal::Buffer* q_buffer = q.buffer();
    TT_ASSERT(q_buffer != nullptr, "Output q buffer should be allocated on device!");
    tt_metal::Buffer* k_buffer = k.buffer();
    TT_ASSERT(k_buffer != nullptr, "Output k buffer should be allocated on device!");
    tt_metal::Buffer* v_buffer = v.buffer();
    TT_ASSERT(v_buffer != nullptr, "Output v buffer should be allocated on device!");

    std::vector<uint32_t> reader_compile_time_args;
    tt_metal::TensorAccessorArgs(*in0_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)q_num_tiles_per_tensor,
        (std::uint32_t)kv_num_tiles_per_tensor,
        (std::uint32_t)q_out_h_tiles,
        (std::uint32_t)q_out_w_tiles,
        (std::uint32_t)q_out_c,
        (std::uint32_t)q_out_HtWt,
    };
    tt_metal::TensorAccessorArgs(*q_buffer).append_to(writer_compile_time_args);
    tt_metal::TensorAccessorArgs(*k_buffer).append_to(writer_compile_time_args);
    tt_metal::TensorAccessorArgs(*v_buffer).append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_falcon7b/device/kernels/dataflow/"
        "writer_tm_tile_layout_nlp_create_qkv_heads_falcon7b.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    // Create circular buffer
    constexpr uint8_t src0_cb_index = 0;
    uint32_t cb0_num_tiles = per_tensor_tiles * 2;  // double buffer
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb0_num_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    reader_desc.runtime_args.reserve(num_cores);
    writer_desc.runtime_args.reserve(num_cores);

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

        reader_desc.emplace_runtime_args(
            core,
            {
                in0_buffer,
                num_blocks_per_core * per_tensor_tiles,
                num_blocks_written * per_tensor_tiles,
            });

        uint32_t q_out_h_dim = num_blocks_written % q_out_h_tiles;
        uint32_t q_out_tensor_tile_id =
            (num_blocks_written / q_out_h_tiles * q_out_CHtWt) + (q_out_h_dim * q_out_w_tiles);

        writer_desc.emplace_runtime_args(
            core,
            {
                q_buffer,                                      // q_tensor_addr
                k_buffer,                                      // k_tensor_addr
                v_buffer,                                      // v_tensor_addr
                num_blocks_per_core,                           // num_blocks
                q_out_h_dim,                                   // q_out_h_dim
                q_out_tensor_tile_id,                          // q_out_tensor_tile_id
                num_blocks_written * kv_num_tiles_per_tensor,  // kv_out_tensor_tile_id
            });
        num_blocks_written += num_blocks_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
