// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/data_movement/repeat/codegen/repeat_codegen_program_factory.hpp"

#include <cstdint>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

using namespace tt::tt_metal;

ProgramDescriptor RepeatCodegenProgramFactory::create_descriptor(
    const RepeatCodegenParams& operation_attributes,
    const RepeatCodegenInputs& tensor_args,
    Tensor& tensor_return_value) {
    const auto& input = tensor_args.input;
    const auto& output = tensor_return_value;

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    const uint32_t num_repeats = operation_attributes.m_num_repeats;
    const uint32_t lower_pages = operation_attributes.m_tile_lower_pages;
    const uint32_t rep_dim_pages = operation_attributes.m_tile_rep_dim_pages;
    const uint32_t page_size = operation_attributes.m_tile_page_size_bytes;

    // DIVERGENCE 1: split total OUTPUT pages across the grid. The native factory
    // splits input higher/lower page *ranges*; the codegen reader walks output
    // pages and maps each back to a source page, so the work unit is output pages.
    const uint32_t total_src_pages = operation_attributes.m_tile_higher_pages * rep_dim_pages * lower_pages;
    const uint32_t total_out_pages = total_src_pages * num_repeats;

    IDevice* device = input.device();
    const CoreCoord grid = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, pages_per_core_1, pages_per_core_2] =
        split_work_to_cores(grid, total_out_pages);

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    constexpr uint32_t cb_id = 0;
    constexpr uint32_t READ_BATCH = 4;
    constexpr uint32_t WRITE_BATCH = 4;
    constexpr uint32_t cb_depth = 8;  // 2 * max(READ_BATCH, WRITE_BATCH)

    ProgramDescriptor desc;

    desc.cbs.push_back(CBDescriptor{
        .total_size = cb_depth * page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = cb_id,
            .data_format = cb_data_format,
            .page_size = page_size,
        }}},
    });

    // DIVERGENCE 2: two kernels (reader on NCRISC, writer on BRISC), not one.
    // DIVERGENCE 3: TensorAccessorArgs are compile-time (default ArgConfig::None)
    // and split — src onto the reader at CT index 0, dst onto the writer after
    // [cb_id, page_size]. The native factory appends BOTH src+dst with
    // RuntimeTensorShape onto its single kernel.
    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/kernels/repeat_codegen_reader.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.named_compile_time_args = {{"seq_id", 1}, {"cb_id", cb_id}, {"batch", READ_BATCH}};
    TensorAccessorArgs(*src_buffer).append_to(reader_desc.compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/repeat/codegen/kernels/repeat_codegen_writer.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = {cb_id, page_size};
    TensorAccessorArgs(*dst_buffer).append_to(writer_desc.compile_time_args);
    writer_desc.compile_time_args.push_back(WRITE_BATCH);
    writer_desc.config = WriterConfigDescriptor{};

    // DIVERGENCE 4: per-core rt-args in the codegen layout. Slot 0 is the Buffer*
    // (fast-path binding); the framework patches only that address on cache hit.
    const auto cores = corerange_to_cores(all_cores, num_cores, /*row_wise=*/false);
    uint32_t start = 0;
    for (const auto& core : cores) {
        const uint32_t n = core_group_1.contains(core) ? pages_per_core_1 : pages_per_core_2;
        reader_desc.emplace_runtime_args(core, {src_buffer, n, start, num_repeats, lower_pages, rep_dim_pages});
        writer_desc.emplace_runtime_args(core, {dst_buffer, n, start});
        start += n;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    return desc;
}

}  // namespace ttnn::prim
