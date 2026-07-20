// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "transpose_wh_program_factory.hpp"
#include "transpose_utils.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

namespace {

void emit_runtime_args_wh_tiled(
    KernelDescriptor& reader_desc,
    KernelDescriptor& compute_desc,
    KernelDescriptor& writer_desc,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    uint32_t num_cores,
    const CoreRangeSet& all_cores,
    const CoreRangeSet& core_group_1,
    uint32_t num_tiles_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_tiles_per_core_group_2) {
    auto input_shape = input_tensor.padded_shape();

    uint32_t W = input_shape[3], H = input_shape[2];

    uint32_t Wt = W / TILE_WIDTH;
    uint32_t Ht = H / TILE_HEIGHT;

    auto HtWt = Ht * Wt;

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    reader_desc.runtime_args.reserve(num_cores);
    compute_desc.runtime_args.reserve(num_cores);
    writer_desc.runtime_args.reserve(num_cores);

    for (uint32_t i = 0, num_tiles_read = 0; i < num_cores; i++) {
        const CoreCoord& core = cores[i];
        uint32_t num_tiles_per_core = 0;

        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else {
            TT_ASSERT(core_group_2.contains(core));
            num_tiles_per_core = num_tiles_per_core_group_2;
        }

        uint32_t h = num_tiles_read % Ht;
        uint32_t w = num_tiles_read / Ht % Wt;

        reader_desc.emplace_runtime_args(
            core,
            {input_tensor.buffer(),
             num_tiles_per_core,
             tt::round_down(num_tiles_read, HtWt) + (h * Wt) + w,
             h,
             w,
             Ht,
             Wt,
             HtWt});

        compute_desc.emplace_runtime_args(core, {num_tiles_per_core});

        writer_desc.emplace_runtime_args(core, {output_tensor.buffer(), num_tiles_per_core, num_tiles_read});

        num_tiles_read += num_tiles_per_core;
    }
}

void emit_runtime_args_wh_rm(
    KernelDescriptor& reader_desc,
    KernelDescriptor& compute_desc,
    KernelDescriptor& writer_desc,
    const Tensor& input_tensor,
    Tensor& output_tensor,
    uint32_t num_cores,
    const CoreRangeSet& all_cores,
    const CoreRangeSet& core_group_1,
    uint32_t num_hw_blocks_per_core_group_1,
    const CoreRangeSet& core_group_2,
    uint32_t num_hw_blocks_per_core_group_2) {
    auto input_shape = input_tensor.logical_shape();

    uint32_t W = input_shape[3], H = input_shape[2];

    auto cores = corerange_to_cores(all_cores, std::nullopt);
    reader_desc.runtime_args.reserve(num_cores);
    compute_desc.runtime_args.reserve(num_cores);
    writer_desc.runtime_args.reserve(num_cores);

    for (uint32_t i = 0, num_sticks_read = 0, num_sticks_write = 0; i < num_cores; i++) {
        const CoreCoord& core = cores[i];
        uint32_t num_hw_blocks_per_core = 0;

        if (core_group_1.contains(core)) {
            num_hw_blocks_per_core = num_hw_blocks_per_core_group_1;
        } else {
            TT_ASSERT(core_group_2.contains(core));
            num_hw_blocks_per_core = num_hw_blocks_per_core_group_2;
        }

        reader_desc.emplace_runtime_args(core, {input_tensor.buffer(), num_sticks_read, num_hw_blocks_per_core});

        compute_desc.emplace_runtime_args(core, {num_hw_blocks_per_core});

        writer_desc.emplace_runtime_args(core, {output_tensor.buffer(), num_sticks_write, num_hw_blocks_per_core});

        num_sticks_read += num_hw_blocks_per_core * H;
        num_sticks_write += num_hw_blocks_per_core * W;
    }
}

}  // namespace

tt::tt_metal::ProgramDescriptor TransposeWHProgramFactory::create_descriptor(
    const TransposeParams& /*operation_attributes*/, const TransposeInputs& tensor_args, Tensor& output_tensor) {
    const auto& input_tensor = tensor_args.input;

    TT_ASSERT(input_tensor.storage_type() == StorageType::DEVICE, "Operand to transpose_wh needs to be on device!");
    TT_ASSERT(input_tensor.buffer() != nullptr, "Operand to transpose_wh needs to be allocated in a buffer on device!");

    uint32_t num_tensor_tiles = input_tensor.physical_volume() / TILE_HW;
    uint32_t W = input_tensor.logical_shape()[3], H = input_tensor.logical_shape()[2];
    uint32_t NC = input_tensor.logical_shape()[1] * input_tensor.logical_shape()[0];
    bool row_major = input_tensor.layout() == Layout::ROW_MAJOR;

    uint32_t ht = (H + TILE_HEIGHT - 1) / TILE_HEIGHT;
    uint32_t wt = (W + TILE_WIDTH - 1) / TILE_WIDTH;

    ProgramDescriptor desc;

    tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    Buffer* src0_buffer = input_tensor.buffer();
    IDevice* device = input_tensor.device();

    bool fp32_dest_acc_en = src0_cb_data_format == tt::DataFormat::Float32 ||
                            src0_cb_data_format == tt::DataFormat::Int32 ||
                            src0_cb_data_format == tt::DataFormat::UInt32;

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores(compute_with_storage_grid_size, row_major ? NC : num_tensor_tiles);

    Buffer* dst_buffer = output_tensor.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    uint32_t src0_cb_index = 0;
    uint32_t num_input_tiles = row_major ? wt * 2 : 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_input_tiles * src0_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = src0_cb_data_format,
            .page_size = src0_single_tile_size,
        }}},
    });

    uint32_t output_cb_index = tt::CBIndex::c_16;
    uint32_t num_output_tiles = row_major ? ht * 2 : 2;
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_output_tiles * dst_single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = dst_cb_data_format,
            .page_size = dst_single_tile_size,
        }}},
    });

    if (row_major) {
        uint32_t im_cb_index = 24;
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
        // TODO REMOVE
        uint32_t im2_cb_index = 25;
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
    }

    std::vector<uint32_t> reader_compile_time_args;
    std::vector<uint32_t> reader_common_runtime_args;
    if (row_major) {
        reader_compile_time_args.push_back(ht);
        reader_compile_time_args.push_back(H > TILE_HEIGHT ? TILE_HEIGHT : H % TILE_HEIGHT);
        reader_compile_time_args.push_back(H % TILE_HEIGHT == 0 ? TILE_HEIGHT : H % TILE_HEIGHT);
        reader_compile_time_args.push_back(wt);
        reader_compile_time_args.push_back(W);
        reader_compile_time_args.push_back(ht * wt);
        reader_compile_time_args.push_back(W * input_tensor.element_size());
        reader_compile_time_args.push_back(wt * input_tensor.element_size() * TILE_WIDTH);
        reader_compile_time_args.push_back(src0_buffer->aligned_page_size());
    }
    TensorAccessorArgs(*src0_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(reader_compile_time_args, reader_common_runtime_args);

    std::vector<uint32_t> writer_compile_time_args = {output_cb_index};
    std::vector<uint32_t> writer_common_runtime_args;
    if (row_major) {
        writer_compile_time_args.push_back(ht);
        writer_compile_time_args.push_back(H);
        writer_compile_time_args.push_back(wt);
        writer_compile_time_args.push_back(W > TILE_WIDTH ? TILE_WIDTH : W % TILE_WIDTH);
        writer_compile_time_args.push_back(W % TILE_WIDTH == 0 ? TILE_WIDTH : W % TILE_WIDTH);
        writer_compile_time_args.push_back(ht * wt);
        writer_compile_time_args.push_back(H * output_tensor.element_size());
        writer_compile_time_args.push_back(ht * output_tensor.element_size() * TILE_HEIGHT);
        writer_compile_time_args.push_back(dst_buffer->aligned_page_size());
    }
    TensorAccessorArgs(*dst_buffer, tensor_accessor::ArgConfig::RuntimeTensorShape)
        .append_to(writer_compile_time_args, writer_common_runtime_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = row_major ? "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                                            "reader_unary_transpose_wh_interleaved_start_id_rm.cpp"
                                          : "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
                                            "reader_unary_transpose_wh_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};
    reader_desc.common_runtime_args = std::move(reader_common_runtime_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        row_major
            ? "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/dataflow/"
              "writer_unary_transpose_wh_interleaved_start_id_rm.cpp"
            : "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};
    writer_desc.common_runtime_args = std::move(writer_common_runtime_args);

    std::vector<uint32_t> compute_kernel_args = {};
    if (row_major) {
        compute_kernel_args.push_back(ht);
        compute_kernel_args.push_back(wt);
        compute_kernel_args.push_back(ht * wt);
    }

    KernelDescriptor::Defines compute_defines;
    if (row_major && (input_tensor.dtype() == DataType::UINT32 || input_tensor.dtype() == DataType::INT32)) {
        compute_defines.emplace_back("DST_ACCUM_MODE", "1");
    }
    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(
        NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (src0_cb_data_format == tt::DataFormat::Float32) {
        // Keep the source CB in full Float32 on the unpack-to-dest path. In
        // the row-major kernel, the tile-formatted intermediate (c_24) also
        // feeds the transpose, so it needs the same treatment.
        unpack_to_dest_mode[src0_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        if (row_major) {
            unpack_to_dest_mode[static_cast<std::size_t>(tt::CBIndex::c_24)] =
                tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
        }
    }

    KernelDescriptor compute_desc;
    compute_desc.kernel_source =
        row_major ? "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh_rm.cpp"
                  : "ttnn/cpp/ttnn/operations/data_movement/transpose/device/kernels/compute/transpose_wh.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = std::move(compute_kernel_args);
    compute_desc.defines = std::move(compute_defines);
    compute_desc.config = ComputeConfigDescriptor{
        .fp32_dest_acc_en = fp32_dest_acc_en,
        .unpack_to_dest_mode = std::move(unpack_to_dest_mode),
    };

    if (row_major) {
        emit_runtime_args_wh_rm(
            reader_desc,
            compute_desc,
            writer_desc,
            input_tensor,
            output_tensor,
            num_cores,
            all_cores,
            core_group_1,
            num_tiles_per_core_group_1,
            core_group_2,
            num_tiles_per_core_group_2);
    } else {
        emit_runtime_args_wh_tiled(
            reader_desc,
            compute_desc,
            writer_desc,
            input_tensor,
            output_tensor,
            num_cores,
            all_cores,
            core_group_1,
            num_tiles_per_core_group_1,
            core_group_2,
            num_tiles_per_core_group_2);
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
