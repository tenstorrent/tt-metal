// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
#include <algorithm>

#include "offset_cumsum_program_factory.hpp"
#include "offset_cumsum_device_operation_types.hpp"

#include "ttnn/operations/math.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include "ttnn/operation.hpp"
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::experimental::prim {

tt::tt_metal::ProgramDescriptor OffsetCumsumProgramFactory::create_descriptor(
    const OffsetCumsumParams& operation_attributes,
    const Tensor& input,
    tensor_return_value_t& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& mesh_dispatch_coordinate) {
    TT_FATAL(
        mesh_dispatch_coordinate.has_value(),
        "OffsetCumsumProgramFactory::create_descriptor requires a mesh dispatch coordinate");
    const ttnn::MeshCoordinate& coord = mesh_dispatch_coordinate.value();
    const uint32_t row_idx = coord[operation_attributes.cluster_axis];
    const uint32_t experts_per_chip = operation_attributes.experts_per_chip;

    tt::tt_metal::ProgramDescriptor desc;

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(tt::tt_metal::DataType::UINT32);

    CoreCoord core = {0, 0};
    CoreRangeSet core_set = CoreRangeSet(std::vector{CoreRange(core, core)});

    const auto& logical_shape = input.logical_shape();
    uint32_t W = logical_shape[-1];
    uint32_t H = logical_shape[-2];

    TT_FATAL(
        row_idx < H,
        "offset_cumsum: mesh row {} on cluster_axis {} is outside the {}-row gathered histogram, so this "
        "device's own offsets row does not exist. The input must be gathered along cluster_axis.",
        row_idx,
        operation_attributes.cluster_axis,
        H);

    auto* src_buffer = input.buffer();
    auto* dst_offsets_buffer = tensor_return_value.at(0).buffer();
    auto* dst_totals_buffer = tensor_return_value.at(1).buffer();
    auto* dst_expert_region_buffer = tensor_return_value.at(2).buffer();
    auto* dst_all_buffer = tensor_return_value.at(3).buffer();
    uint32_t input_page_size = src_buffer->aligned_page_size();
    uint32_t offsets_page_size = dst_offsets_buffer->aligned_page_size();
    uint32_t totals_page_size = dst_totals_buffer->aligned_page_size();
    uint32_t expert_region_page_size = dst_expert_region_buffer->aligned_page_size();
    uint32_t all_page_size = dst_all_buffer->aligned_page_size();

    uint32_t cb_in0_index = tt::CBIndex::c_0;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = input_page_size,
        .core_ranges = core_set,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_in0_index),
            .data_format = cb_data_format,
            .page_size = input_page_size,
        }}},
    });

    uint32_t cb_out0_index = tt::CBIndex::c_1;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = offsets_page_size,
        .core_ranges = core_set,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_out0_index),
            .data_format = cb_data_format,
            .page_size = offsets_page_size,
        }}},
    });

    uint32_t cb_prefix_index = tt::CBIndex::c_2;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = offsets_page_size,
        .core_ranges = core_set,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_prefix_index),
            .data_format = cb_data_format,
            .page_size = offsets_page_size,
        }}},
    });

    // Staged once per row and written to both `all_offsets` and, on this device's own row, `offsets`.
    // Both are one W-wide UINT32 row-major row, so one page covers either write.
    uint32_t cb_row_index = tt::CBIndex::c_3;
    TT_FATAL(
        all_page_size == offsets_page_size,
        "offset_cumsum: all_offsets page is {} B and offsets page is {} B; the staging buffer serves both",
        all_page_size,
        offsets_page_size);
    uint32_t cb_row_size = all_page_size;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = cb_row_size,
        .core_ranges = core_set,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_row_index),
            .data_format = cb_data_format,
            .page_size = cb_row_size,
        }}},
    });

    std::vector<uint32_t> compile_time_args = {
        cb_in0_index,
        cb_out0_index,
        cb_prefix_index,
        cb_row_index,
        input_page_size,
        offsets_page_size,
        totals_page_size,
        expert_region_page_size,
        all_page_size,
        W,
        H,
        experts_per_chip,
    };
    tt::tt_metal::TensorAccessorArgs(src_buffer).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dst_offsets_buffer).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dst_totals_buffer).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dst_expert_region_buffer).append_to(compile_time_args);
    tt::tt_metal::TensorAccessorArgs(dst_all_buffer).append_to(compile_time_args);

    tt::tt_metal::KernelDescriptor reader_kernel_desc;
    reader_kernel_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/offset_cumsum/device/kernels/"
        "reader_offset_cumsum_interleaved.cpp";
    reader_kernel_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_kernel_desc.core_ranges = core_set;
    reader_kernel_desc.compile_time_args = std::move(compile_time_args);
    reader_kernel_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    // Buffer* in the first five positions so the framework records BufferBindings
    // for the cache-hit fast path; row_idx is plain uint32_t (no override needed —
    // it is invariant for a given coord, baked into the cached program for that coord).
    tt::tt_metal::KernelDescriptor::RTArgList reader_rt_args;
    reader_rt_args.push_back(src_buffer);
    reader_rt_args.push_back(dst_offsets_buffer);
    reader_rt_args.push_back(dst_totals_buffer);
    reader_rt_args.push_back(dst_expert_region_buffer);
    reader_rt_args.push_back(dst_all_buffer);
    reader_rt_args.push_back(row_idx);
    reader_kernel_desc.emplace_runtime_args(core, reader_rt_args);

    desc.kernels.push_back(std::move(reader_kernel_desc));

    return desc;
}

}  // namespace ttnn::experimental::prim
