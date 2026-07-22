// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "full_program_factory_common.hpp"
#include "full_program_factory_interleaved.hpp"

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::operations::full {

ProgramDescriptor FullInterleavedProgramFactory::create_descriptor(
    const operation_attributes_t& operation_attributes,
    [[maybe_unused]] const tensor_args_t&,
    tensor_return_value_t& output) {
    auto dtype = operation_attributes.dtype;
    auto fill_value = operation_attributes.fill_value;

    auto grid = operation_attributes.mesh_device->compute_with_storage_grid_size();
    auto num_pages = (uint32_t)output.buffer()->num_pages();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_pages_per_core_group_1, num_pages_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(grid, num_pages);

    uint32_t page_size = output.buffer()->page_size();
    TT_FATAL(page_size % output.element_size() == 0, "Page size must be divisible by element size");
    uint32_t elems_per_page = page_size / output.element_size();

    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(dtype);

    ProgramDescriptor desc;

    auto cb_index = tt::CBIndex::c_0;
    desc.cbs.push_back(CBDescriptor{
        .total_size = page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(cb_index),
            .data_format = data_format,
            .page_size = page_size,
        }}},
    });

    auto writer_defines_map = get_writer_defines(dtype);
    auto writer_defines = defines_from_map(writer_defines_map);
    auto u = encode_fill_value(fill_value, dtype);

    std::vector<uint32_t> writer_compile_time_args = {(uint32_t)cb_index, elems_per_page, page_size};
    tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(writer_compile_time_args);

    KernelDescriptor writer_desc;
    writer_desc.kernel_source = "ttnn/cpp/ttnn/operations/full/device/kernels/writer_full.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.defines = writer_defines;
    writer_desc.config = WriterConfigDescriptor{};

    auto cores = corerange_to_cores(all_cores, std::nullopt);

    KernelDescriptor reader_desc;
    const bool has_reader = num_pages > num_cores;
    if (has_reader) {
        auto cb_index2 = tt::CBIndex::c_1;
        desc.cbs.push_back(CBDescriptor{
            .total_size = page_size,
            .core_ranges = all_cores,
            .format_descriptors = {{CBFormatDescriptor{
                .buffer_index = static_cast<uint8_t>(cb_index2),
                .data_format = data_format,
                .page_size = page_size,
            }}},
        });

        std::vector<uint32_t> reader_compile_time_args = {(uint32_t)cb_index2, elems_per_page, page_size};
        tt::tt_metal::TensorAccessorArgs(output.buffer()).append_to(reader_compile_time_args);

        reader_desc.kernel_source = "ttnn/cpp/ttnn/operations/full/device/kernels/writer_full.cpp";
        reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
        reader_desc.core_ranges = all_cores;
        reader_desc.compile_time_args = std::move(reader_compile_time_args);
        reader_desc.defines = defines_from_map(writer_defines_map);
        reader_desc.config = ReaderConfigDescriptor{};
    }

    uint32_t page_offset = 0;

    for (const auto& core : cores) {
        uint32_t num_pages_per_core;
        if (core_group_1.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_pages_per_core = num_pages_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        if (has_reader) {
            uint32_t reader_page_start = page_offset;
            uint32_t num_pages_per_reader = num_pages_per_core / 2;
            reader_desc.emplace_runtime_args(core, {output.buffer(), u.u32, num_pages_per_reader, reader_page_start});

            uint32_t writer_page_start = reader_page_start + num_pages_per_reader;
            uint32_t num_pages_per_writer = num_pages_per_core - num_pages_per_reader;
            writer_desc.emplace_runtime_args(core, {output.buffer(), u.u32, num_pages_per_writer, writer_page_start});
        } else {
            writer_desc.emplace_runtime_args(core, {output.buffer(), u.u32, num_pages_per_core, page_offset});
        }
        page_offset += num_pages_per_core;
    }

    desc.kernels.push_back(std::move(writer_desc));
    if (has_reader) {
        desc.kernels.push_back(std::move(reader_desc));
    }

    return desc;
}

}  // namespace ttnn::operations::full
