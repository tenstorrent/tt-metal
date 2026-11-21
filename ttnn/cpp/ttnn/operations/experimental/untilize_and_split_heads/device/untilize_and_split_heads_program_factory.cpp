// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>
#include <string>
#include <string_view>

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "untilize_and_split_heads_device_operation.hpp"

namespace ttnn::experimental::prim {

using namespace tt::tt_metal;

namespace {

constexpr std::string_view kKernelDir =
    "ttnn/cpp/ttnn/operations/experimental/untilize_and_split_heads/device/kernels/";

// Output pages (of D/32 tiles each) in flight per CB, so reading page p+1
// overlaps untilizing page p and writing page p-1.
constexpr uint32_t kPagesPerCb = 2;

}  // namespace

ProgramDescriptor UntilizeAndSplitHeadsDeviceOperation::create_descriptor(
    const operation_attributes_t& attrs, const tensor_args_t& tensor_args, Tensor& output) {
    ProgramDescriptor descriptor{};

    const auto& input = tensor_args.input;
    const auto& in_shape = input.logical_shape();
    const uint32_t batch = in_shape[0];
    const uint32_t pages_per_head = tt::div_up(in_shape[1], tt::constants::TILE_HEIGHT);
    const uint32_t head_dim = in_shape[2] / attrs.num_heads;
    const uint32_t tiles_per_page = head_dim / tt::constants::TILE_WIDTH;
    const uint32_t num_pages = batch * attrs.num_heads * pages_per_head;

    const auto data_format = datatype_to_dataformat_converter(DataType::BFLOAT16);
    const uint32_t tile_nbytes = tt::tile_size(data_format);
    const uint32_t page_nbytes = tiles_per_page * tile_nbytes;  // 32 rows of D bf16

    auto* device = input.device();
    auto [num_cores, all_cores, core_group_1, core_group_2, pages_per_core_group_1, pages_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(device->compute_with_storage_grid_size(), num_pages);

    constexpr uint8_t in_cb = tt::CBIndex::c_0;
    constexpr uint8_t out_cb = tt::CBIndex::c_16;
    for (uint8_t idx : {in_cb, out_cb}) {
        descriptor.cbs.push_back(CBDescriptor{
            .total_size = kPagesPerCb * page_nbytes,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = idx,
                .data_format = data_format,
                .page_size = tile_nbytes,
            }}},
        });
    }

    KernelDescriptor::CompileTimeArgs reader_ct{in_cb, tiles_per_page, attrs.num_heads, pages_per_head, tile_nbytes};
    TensorAccessorArgs(*input.buffer()).append_to(reader_ct);
    KernelDescriptor reader_desc;
    reader_desc.kernel_source = std::string(kKernelDir) + "reader_untilize_and_split_heads.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_ct);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor compute_desc;
    compute_desc.kernel_source = std::string(kKernelDir) + "compute_untilize_and_split_heads.cpp";
    compute_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = {in_cb, out_cb, tiles_per_page};
    compute_desc.config = ComputeConfigDescriptor{};

    KernelDescriptor::CompileTimeArgs writer_ct{out_cb, tiles_per_page, page_nbytes};
    TensorAccessorArgs(*output.buffer()).append_to(writer_ct);
    KernelDescriptor writer_desc;
    writer_desc.kernel_source = std::string(kKernelDir) + "writer_untilize_and_split_heads.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_ct);
    writer_desc.config = WriterConfigDescriptor{};

    uint32_t page_cursor = 0;
    for (const auto& core : corerange_to_cores(all_cores, num_cores, /*row_wise=*/true)) {
        uint32_t pages_here = 0;
        if (core_group_1.contains(core)) {
            pages_here = pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            pages_here = pages_per_core_group_2;
        } else {
            continue;
        }
        KernelDescriptor::RTArgList reader_args;
        reader_args.push_back(input.buffer());
        reader_args.push_back(page_cursor);
        reader_args.push_back(pages_here);
        KernelDescriptor::RTArgList writer_args;
        writer_args.push_back(output.buffer());
        writer_args.push_back(page_cursor);
        writer_args.push_back(pages_here);
        KernelDescriptor::CoreRuntimeArgs compute_args;
        compute_args.push_back(pages_here);
        reader_desc.emplace_runtime_args(core, reader_args);
        compute_desc.runtime_args.emplace_back(core, std::move(compute_args));
        writer_desc.emplace_runtime_args(core, writer_args);
        page_cursor += pages_here;
    }

    descriptor.kernels.push_back(std::move(reader_desc));
    descriptor.kernels.push_back(std::move(compute_desc));
    descriptor.kernels.push_back(std::move(writer_desc));
    return descriptor;
}

}  // namespace ttnn::experimental::prim
