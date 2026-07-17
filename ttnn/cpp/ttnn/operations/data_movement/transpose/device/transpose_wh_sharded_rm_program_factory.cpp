// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_wh_sharded_rm_program_factory.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-logger/tt-logger.hpp>

#include <algorithm>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor TransposeWHShardedRMProgramFactory::create_descriptor(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_wh needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_wh needs to be allocated in a buffer on device!");

    ProgramDescriptor desc;

    tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    uint32_t W = input_tensor.logical_shape()[3], H = input_tensor.logical_shape()[2];
    uint32_t stick_size_bytes = W * input_tensor.element_size();
    uint32_t ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    uint32_t output_page_size, pack_num_pages, pack_num_pages_last_col, pack_num_pages_last_row,
        pack_num_pages_last_row_col;
    if ((W % TILE_WIDTH) != 0 and (H % TILE_HEIGHT) != 0) {
        output_page_size = (W % TILE_WIDTH) * (H % TILE_HEIGHT) * output_tensor.element_size();
        pack_num_pages = dst_single_tile_size / output_page_size;
        auto output_page_size_last_col = TILE_WIDTH * (H % TILE_HEIGHT) * output_tensor.element_size();
        pack_num_pages_last_col = dst_single_tile_size / output_page_size_last_col;
        auto output_page_size_last_row = TILE_HEIGHT * (W % TILE_WIDTH) * output_tensor.element_size();
        pack_num_pages_last_row = dst_single_tile_size / output_page_size_last_row;
        pack_num_pages_last_row_col = 1;
    } else if ((W % TILE_WIDTH) != 0 and (H % TILE_HEIGHT) == 0) {
        output_page_size = (W % TILE_WIDTH) * (TILE_HEIGHT)*output_tensor.element_size();
        pack_num_pages = dst_single_tile_size / output_page_size;
        pack_num_pages_last_col = pack_num_pages;
        pack_num_pages_last_row = 1;
        pack_num_pages_last_row_col = 1;
    } else if ((W % TILE_WIDTH) == 0 and (H % TILE_HEIGHT) != 0) {
        output_page_size = (TILE_WIDTH) * (H % TILE_HEIGHT) * output_tensor.element_size();
        pack_num_pages = dst_single_tile_size / output_page_size;
        pack_num_pages_last_col = 1;
        pack_num_pages_last_row = pack_num_pages;
        pack_num_pages_last_row_col = 1;
    } else {
        output_page_size = dst_single_tile_size;
        pack_num_pages = 1;
        pack_num_pages_last_col = 1;
        pack_num_pages_last_row = 1;
        pack_num_pages_last_row_col = 1;
    }

    log_debug(tt::LogOp, "output_page_size: {}", output_page_size);
    log_debug(tt::LogOp, "pack_num_pages: {}", pack_num_pages);
    log_debug(tt::LogOp, "pack_num_pages_last_col: {}", pack_num_pages_last_col);
    log_debug(tt::LogOp, "pack_num_pages_last_row: {}", pack_num_pages_last_row);
    log_debug(tt::LogOp, "pack_num_pages_last_row_col: {}", pack_num_pages_last_row_col);

    auto shard_spec = input_tensor.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    uint32_t num_hw_blocks_per_core = shard_height / H;

    log_debug(tt::LogOp, "shard_height: {}", shard_height);
    log_debug(tt::LogOp, "dst_single_tile_size: {}", dst_single_tile_size);

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32;

    auto& all_cores = shard_spec.grid;
    [[maybe_unused]] uint32_t num_cores = shard_spec.num_cores();

    log_debug(tt::LogOp, "all_cores: {}", all_cores);
    log_debug(tt::LogOp, "num_cores: {}", num_cores);

    // sharded cb (input): .buffer triggers UpdateDynamicCircularBufferAddress on cache hit.
    uint32_t src0_cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = shard_height * stick_size_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = src0_cb_data_format,
            .page_size = stick_size_bytes,
        }}},
        .buffer = input_tensor.buffer(),
    });

    // sharded cb (output): .buffer triggers UpdateDynamicCircularBufferAddress on cache hit.
    uint32_t output_cb_index = tt::CBIndex::c_16;
    desc.cbs.push_back(CBDescriptor{
        .total_size = stick_size_bytes * shard_height,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = dst_cb_data_format,
            .page_size = output_page_size,
        }}},
        .buffer = output_tensor.buffer(),
    });

    // cb_in (double-buffered intermediate)
    uint32_t in_cb_index = tt::CBIndex::c_24;
    uint32_t num_in_tiles = wt * 2;  // double buffer
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_in_tiles * src0_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(in_cb_index),
            .data_format = src0_cb_data_format,
            .page_size = src0_single_tile_size,
        }}},
    });

    // tilize cb
    uint32_t im_cb_index = tt::CBIndex::c_25;
    uint32_t num_im_tiles = ht * wt;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_im_tiles * src0_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(im_cb_index),
            .data_format = src0_cb_data_format,
            .page_size = src0_single_tile_size,
        }}},
    });

    // untilize cb (only when ht > 8) + matching output staging CB
    if (ht > 8) {
        uint32_t im2_cb_index = tt::CBIndex::c_26;
        uint32_t num_im2_tiles = ht;
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_im2_tiles * dst_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(im2_cb_index),
                .data_format = dst_cb_data_format,
                .page_size = dst_single_tile_size,
            }}},
        });

        // compute_output_cb
        uint32_t out_cb_index = tt::CBIndex::c_27;
        uint32_t num_out_tiles = ht * 2;  // double buffer
        desc.cbs.push_back(CBDescriptor{
            .total_size = num_out_tiles * dst_single_tile_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(out_cb_index),
                .data_format = dst_cb_data_format,
                .page_size = dst_single_tile_size,
            }}},
        });
    }

    std::vector<uint32_t> reader_compile_time_args = {
        (std::uint32_t)num_hw_blocks_per_core,
        (std::uint32_t)ht,
        (std::uint32_t)H > TILE_HEIGHT ? TILE_HEIGHT : H % TILE_HEIGHT,
        (std::uint32_t)H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT,
        (std::uint32_t)wt,
        (std::uint32_t)stick_size_bytes,
        (std::uint32_t)wt * input_tensor.element_size() * TILE_WIDTH,
    };
    reader_compile_time_args.push_back(H > TILE_HEIGHT ? TILE_HEIGHT : H % TILE_HEIGHT);
    reader_compile_time_args.push_back(H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "reader_unary_transpose_wh_sharded_rm.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    std::vector<uint32_t> writer_compile_time_args = {
        (std::uint32_t)num_hw_blocks_per_core,
        (std::uint32_t)ht,
        (std::uint32_t)wt,
        (std::uint32_t)W > TILE_WIDTH ? TILE_WIDTH : W % TILE_WIDTH,
        (std::uint32_t)W % TILE_WIDTH == 0 ? TILE_WIDTH : W % TILE_WIDTH,
        (std::uint32_t)H * output_tensor.element_size(),
        (std::uint32_t)ht * output_tensor.element_size() * TILE_HEIGHT,
    };

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
        "writer_unary_transpose_wh_sharded_rm.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    std::vector<uint32_t> compute_compile_time_args = {
        (std::uint32_t)ht,
        (std::uint32_t)wt,
        (std::uint32_t)ht * wt,
        (std::uint32_t)num_hw_blocks_per_core,
        (std::uint32_t)H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT,  // last_output_row_num_datums
        (std::uint32_t)pack_num_pages,
        (std::uint32_t)pack_num_pages_last_col,
        (std::uint32_t)pack_num_pages_last_row,
        (std::uint32_t)pack_num_pages_last_row_col,
    };

    KernelDescriptor::Defines compute_defines;
    compute_defines.emplace_back("SHARDED", "1");

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (src0_cb_data_format == tt::DataFormat::Float32) {
        // Keep both the tilize input (c_24) and its output (c_25, which feeds
        // the transpose) in full Float32 on the unpack-to-dest path; otherwise
        // the unpacker falls back to tf32 and drops the low mantissa bits.
        unpack_to_dest_mode[in_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[im_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_rm.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = std::move(compute_compile_time_args);
    compute_desc.defines = std::move(compute_defines);
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
    };

    // #48928: this reader uses compile-time args only; emit one placeholder rt-arg per core (harmless,
    // re-applied by override_runtime_arguments along with the CB base addresses on cache hit).
    for (const auto& core : corerange_to_cores(all_cores)) {
        reader_desc.runtime_args.emplace_back(core, std::vector<uint32_t>{0u});
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
