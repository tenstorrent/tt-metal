// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/device/nlp_create_qkv_heads_vit_program_factory.hpp"

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::experimental::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;

tt::tt_metal::ProgramDescriptor NlpCreateQkvHeadsVitProgramFactory::create_descriptor(
    const NlpCreateQkvHeadsVitParams& /*operation_attributes*/,
    const NlpCreateQkvHeadsVitInputs& tensor_args,
    NlpCreateQkvHeadsVitResult& output) {
    ProgramDescriptor desc;

    const auto& a = tensor_args.input_tensor;
    const auto& ashape = a.padded_shape();

    tt::DataFormat cb_data_format = tt_metal::datatype_to_dataformat_converter(a.dtype());

    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt_metal::Buffer* in0_buffer = a.buffer();
    TT_ASSERT(in0_buffer->size() % single_tile_size == 0);
    // Dummy
    uint32_t in1_buffer_addr = 0;

    ////////////////////////////////////////////////////////////////////////////
    //                      TM Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t per_tensor_tiles = ashape[3] / TILE_WIDTH;  // 72
    const uint32_t q_num_tiles_per_tensor = 24;
    const uint32_t num_q_heads = 12;
    const uint32_t num_kv_heads = 12;

    // Per output tensor args
    // Output shape for Q,K,V is: [B, 12, s, 64] # Needs shuffling from [B, 1, s, 2304]
    uint32_t q_out_h_tiles = ashape[2] / TILE_HEIGHT;
    uint32_t q_out_w_tiles = 2;                                 // head_dim
    uint32_t q_out_c = q_num_tiles_per_tensor / q_out_w_tiles;  // num_heads
    uint32_t q_out_HtWt = q_out_h_tiles * q_out_w_tiles;
    uint32_t q_out_CHtWt = q_out_c * q_out_HtWt;
    uint32_t kv_out_CHtWt = num_kv_heads * q_out_HtWt;
    uint32_t q_num_tiles = num_q_heads * q_out_w_tiles;
    uint32_t kv_num_tiles = num_kv_heads * q_out_w_tiles;

    CoreCoord compute_with_storage_grid_size = a.device()->compute_with_storage_grid_size();
    uint32_t num_cores_y = compute_with_storage_grid_size.y;
    // Block is a unit of work; ie. num of per_tensor_tiles per core
    uint32_t num_blocks = ashape[0] * ashape[1] * ashape[2] / TILE_HEIGHT;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_blocks_per_core_group_1, num_blocks_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_blocks);

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

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)q_num_tiles,
        (std::uint32_t)kv_num_tiles,
    };
    tt::tt_metal::TensorAccessorArgs(*in0_buffer).append_to(reader_compile_time_args);
    tt::tt_metal::TensorAccessorArgs().append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)q_out_h_tiles,
        (std::uint32_t)q_out_w_tiles,
        (std::uint32_t)q_out_HtWt,
        (std::uint32_t)num_q_heads,   // q_out_c
        (std::uint32_t)num_kv_heads,  // kv_out_c
    };
    tt::tt_metal::TensorAccessorArgs(*q_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*k_buffer).append_to(writer_compile_time_args);
    tt::tt_metal::TensorAccessorArgs(*v_buffer).append_to(writer_compile_time_args);

    ///////////// K transpose ////////////////////
    const bool transpose_k_heads = false;
    std::map<std::string, std::string> reader_defines;
    std::map<std::string, std::string> writer_defines;
    if (transpose_k_heads) {
        std::vector<uint32_t> compute_args_core_group_1 = {num_blocks_per_core_group_1 * kv_num_tiles};
        KernelDescriptor compute_desc_1;
        compute_desc_1.kernel_source = "ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp";
        compute_desc_1.source_type = KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_1.core_ranges = core_group_1;
        compute_desc_1.compile_time_args = std::move(compute_args_core_group_1);
        compute_desc_1.config = ComputeConfigDescriptor{};
        desc.kernels.push_back(std::move(compute_desc_1));

        if (core_group_2.num_cores() > 0) {
            std::vector<uint32_t> compute_args_core_group_2 = {num_blocks_per_core_group_2 * kv_num_tiles};
            KernelDescriptor compute_desc_2;
            compute_desc_2.kernel_source = "ttnn/cpp/ttnn/kernel/compute/transpose_wh.cpp";
            compute_desc_2.source_type = KernelDescriptor::SourceType::FILE_PATH;
            compute_desc_2.core_ranges = core_group_2;
            compute_desc_2.compile_time_args = std::move(compute_args_core_group_2);
            compute_desc_2.config = ComputeConfigDescriptor{};
            desc.kernels.push_back(std::move(compute_desc_2));
        }
        reader_defines["TRANSPOSE_K_HEADS"] = "1";
        writer_defines["TRANSPOSE_K_HEADS"] = "1";
    }
    //////////////////////////////////////////////

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/device/kernels/dataflow/"
        "reader_tm_tile_layout_nlp_create_qkv_heads.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.defines = KernelDescriptor::Defines{reader_defines.begin(), reader_defines.end()};
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/transformer/nlp_create_qkv_heads_vit/device/kernels/dataflow/"
        "writer_tm_tile_layout_nlp_create_qkv_heads.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = KernelDescriptor::Defines{writer_defines.begin(), writer_defines.end()};
    writer_desc.config = WriterConfigDescriptor{};

    // Create circular buffers
    constexpr uint8_t src1_cb_index = 1;
    uint32_t cb0_num_tiles = per_tensor_tiles * 2;  // double buffer
    desc.cbs.push_back(CBDescriptor{
        .total_size = cb0_num_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src1_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
        }}},
    });

    // If we transpose_k_heads:
    // - reader will write to cb0, instead of cb1
    // - compute will wait on cb0 and write to cb16
    // - writer will wait on cb 16, instead of cb1
    if (transpose_k_heads) {
        constexpr uint8_t src0_cb_index = 0;
        uint32_t cb_src0_num_tiles = per_tensor_tiles * 2;  // double buffer
        desc.cbs.push_back(CBDescriptor{
            .total_size = cb_src0_num_tiles * single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = src0_cb_index,
                .data_format = cb_data_format,
                .page_size = single_tile_size,
            }}},
        });

        constexpr uint8_t out_cb_index = 16;
        uint32_t out_cb_num_tiles = per_tensor_tiles * 2;  // double buffer
        desc.cbs.push_back(CBDescriptor{
            .total_size = out_cb_num_tiles * single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = out_cb_index,
                .data_format = cb_data_format,
                .page_size = single_tile_size,
            }}},
        });
    }

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
                static_cast<uint32_t>(in1_buffer_addr),
                num_blocks_per_core,
                num_blocks_written * per_tensor_tiles,
                static_cast<uint32_t>(0),
            });

        uint32_t q_out_h_dim = num_blocks_written % q_out_h_tiles;
        uint32_t q_out_tensor_tile_id =
            (num_blocks_written / q_out_h_tiles * q_out_CHtWt) + (q_out_h_dim * q_out_w_tiles);
        uint32_t v_out_tensor_tile_id =
            (num_blocks_written / q_out_h_tiles * kv_out_CHtWt) + (q_out_h_dim * q_out_w_tiles);
        uint32_t k_out_tensor_tile_id = transpose_k_heads
                                            ? (num_blocks_written / q_out_h_tiles * kv_out_CHtWt) + q_out_h_dim
                                            : v_out_tensor_tile_id;

        writer_desc.emplace_runtime_args(
            core,
            {
                q_buffer,             // q_tensor_addr
                k_buffer,             // k_tensor_addr
                v_buffer,             // v_tensor_addr
                num_blocks_per_core,  // num_blocks
                q_out_h_dim,
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

}  // namespace ttnn::experimental::prim
