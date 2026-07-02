// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "concatenate_heads_program_factory.hpp"

#include "concatenate_heads_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt;

tt::tt_metal::ProgramDescriptor ConcatenateHeadsProgramFactory::create_descriptor(
    const ConcatenateHeadsParams& operation_attributes, const ConcatenateHeadsInputs& tensor_args, Tensor& output) {
    ProgramDescriptor desc;

    const auto& a = tensor_args.input;
    const auto& ashape = a.padded_shape();
    const auto& compute_with_storage_grid_size = operation_attributes.compute_with_storage_grid_size;

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt_metal::Buffer* in0_buffer = a.buffer();
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    // Output shape is: [B, 1, 384, 1024]
    uint32_t per_core_tiles = (ashape[1] * ashape[3]) / TILE_WIDTH;
    uint32_t in0_h_tiles = ashape[2] / TILE_HEIGHT;

    // These parameters are identical to out_* in multi_core_create_qkv_heads
    uint32_t in0_w = 64;
    uint32_t in0_w_tiles = in0_w / TILE_WIDTH;
    uint32_t in0_c = per_core_tiles / in0_w_tiles;
    uint32_t in0_HtWt = in0_h_tiles * in0_w_tiles;
    uint32_t in0_CHtWt = in0_c * in0_HtWt;

    // Parallelize ashape[2] (384 / 32 = 12 tiles) across columns
    // Parallelize ashape[0] (B) across rows
    uint32_t num_cores_x = ashape[2] / TILE_HEIGHT;
    uint32_t num_cores_y = ashape[0];
    TT_ASSERT(num_cores_x <= compute_with_storage_grid_size.x);
    TT_ASSERT(num_cores_y <= compute_with_storage_grid_size.y);
    CoreCoord core_range = {num_cores_x, num_cores_y};

    tt_metal::Buffer* out_buffer = output.buffer();
    TT_ASSERT(out_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t start_core_x = 0;
    uint32_t start_core_y = 0;
    uint32_t num_cores_c = core_range.x;
    uint32_t num_cores_r = core_range.y;

    CoreRange all_cores(
        {(std::size_t)start_core_x, (std::size_t)start_core_y},
        {(std::size_t)start_core_x + num_cores_c - 1, (std::size_t)start_core_y + num_cores_r - 1});
    CoreRangeSet all_cores_set{all_cores};

    // Create circular buffer
    constexpr uint8_t src0_cb_index = 0;
    uint32_t cb0_tiles = per_core_tiles * 2;  // double buffer
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb0_tiles * single_tile_size,
        .core_ranges = all_cores_set,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    std::vector<uint32_t> reader_compile_time_args = {
        // READER COMPILE TIME ARGS
        (std::uint32_t)in0_w_tiles,  // in0_w_tiles
        (std::uint32_t)in0_c,        // in0_c
        (std::uint32_t)in0_HtWt,     // in0_HtWt
    };
    tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {
        // WRITER COMPILE TIME ARGS
        (std::uint32_t)in0_w_tiles,  // in0_w_tiles
        (std::uint32_t)in0_c,        // in0_c
    };
    tt::tt_metal::TensorAccessorArgs(*out_buffer).append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/concatenate_heads/device/kernels/dataflow/"
        "reader_tm_tile_layout_concat_heads.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores_set;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/concatenate_heads/device/kernels/dataflow/"
        "writer_tm_tile_layout_concat_heads.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores_set;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    reader_desc.runtime_args.reserve(static_cast<size_t>(num_cores_r) * num_cores_c);
    writer_desc.runtime_args.reserve(static_cast<size_t>(num_cores_r) * num_cores_c);

    for (int core_idx_y = 0; core_idx_y < num_cores_r; core_idx_y++) {
        for (int core_idx_x = 0; core_idx_x < num_cores_c; core_idx_x++) {
            CoreCoord core = {(std::size_t)start_core_x + core_idx_x, (std::size_t)start_core_y + core_idx_y};
            uint32_t in0_tensor_tile_id = (core_idx_x * in0_w_tiles) + (core_idx_y * in0_CHtWt);

            reader_desc.emplace_runtime_args(
                core,
                {
                    in0_buffer,          // in0_tensor_addr
                    in0_tensor_tile_id,  // in0_tensor_tile_id
                });
            writer_desc.emplace_runtime_args(
                core,
                {
                    out_buffer,  // out_tensor_addr
                    static_cast<uint32_t>(
                        (core_idx_x + core_idx_y * num_cores_c) * per_core_tiles),  // out_tensor_tile_id
                });
        }
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
