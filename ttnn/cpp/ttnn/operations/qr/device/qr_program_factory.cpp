// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "qr_device_operation.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>

namespace ttnn::operations::qr {

using namespace tt;
using namespace tt::tt_metal;

ProgramDescriptor QrDeviceOperation::create_descriptor(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    const Tensor& input = tensor_args.input;
    const auto& [q, r] = tensor_return_value;

    const uint32_t m = input.logical_shape()[-2];
    const uint32_t n = input.logical_shape()[-1];

    // Single core: all work happens on core (0, 0).
    const CoreCoord core = {0, 0};
    const CoreRangeSet all_cores = CoreRangeSet({CoreRange(core)});

    ProgramDescriptor desc;

    constexpr uint32_t num_tiles = 1;
    const uint32_t fp32_tile_size = tile_size(tt::DataFormat::Float32);

    // Input tile.
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles * fp32_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = CBIndex::c_0,
            .data_format = tt::DataFormat::Float32,
            .page_size = fp32_tile_size,
        }}},
    });
    // R working copy (m x n, single tile).
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles * fp32_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = CBIndex::c_1,
            .data_format = tt::DataFormat::Float32,
            .page_size = fp32_tile_size,
        }}},
    });
    // Q output (m x k, single tile).
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles * fp32_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = CBIndex::c_2,
            .data_format = tt::DataFormat::Float32,
            .page_size = fp32_tile_size,
        }}},
    });
    // Reflector storage: k steps x m entries, at most 32 x 32 floats.
    desc.cbs.push_back(CBDescriptor{
        .total_size = num_tiles * fp32_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = CBIndex::c_3,
            .data_format = tt::DataFormat::Float32,
            .page_size = fp32_tile_size,
        }}},
    });

    // ---- Kernels ----

    const std::string kernels_dir_path = "ttnn/cpp/ttnn/operations/qr/device/kernels/";

    // Reader: input tile DRAM -> c_0.
    std::vector<uint32_t> reader_compile_time_args{CBIndex::c_0};
    TensorAccessorArgs(*input.buffer()).append_to(reader_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source = kernels_dir_path + "dataflow/reader_qr.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = reader_compile_time_args;
    reader_desc.config = ReaderConfigDescriptor{};

    // Writer: c_0 -> in-register Householder QR -> Q (c_2) and R (c_1) tiles
    // written straight to DRAM.
    std::vector<uint32_t> writer_compile_time_args{CBIndex::c_0, CBIndex::c_1, CBIndex::c_2, CBIndex::c_3};
    TensorAccessorArgs(*input.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(*q.buffer()).append_to(writer_compile_time_args);
    TensorAccessorArgs(*r.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = kernels_dir_path + "dataflow/writer_qr.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = writer_compile_time_args;
    writer_desc.config = WriterConfigDescriptor{};

    // ---- Runtime args ----

    reader_desc.emplace_runtime_args(core, {input.buffer()});
    writer_desc.emplace_runtime_args(core, {input.buffer(), q.buffer(), r.buffer(), m, n});

    desc.kernels = {std::move(reader_desc), std::move(writer_desc)};
    return desc;
}

}  // namespace ttnn::operations::qr
