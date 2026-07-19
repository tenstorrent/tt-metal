// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_row_major_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tilize_utils.hpp>

#include "ttnn/operations/data_movement/common/common.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor UntilizeWithUnpaddingRowMajorProgramFactory::create_descriptor(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    ProgramDescriptor desc;

    // output_tensor_end always slices from index 0 in every dim (unlike ttnn::slice's arbitrary
    // start offset), so there is no leading-offset misalignment to handle here.
    auto input_shape = input.padded_shape();
    auto output_shape = output.padded_shape();
    uint32_t num_dims = static_cast<uint32_t>(input_shape.rank());

    uint32_t padded_row_size_bytes = input_shape[-1] * input.element_size();
    uint32_t unpadded_row_size_bytes = output_shape[-1] * input.element_size();

    // BLOCK/WIDTH-sharded buffers split a logical row across shards - the reader/writer kernels'
    // noc_async_{read,write}_sharded helper needs the per-shard page size (shard width in bytes) to
    // do that split correctly, not the full row size. HEIGHT-sharded and interleaved buffers keep a
    // whole row per page, same as the full row size (mirrors ttnn::slice's identical helper).
    auto per_shard_page_size_bytes = [](const Tensor& t, uint32_t row_bytes) -> uint32_t {
        const auto& mc = t.memory_config();
        if (mc.is_sharded() && (mc.memory_layout() == TensorMemoryLayout::BLOCK_SHARDED ||
                                mc.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED)) {
            return mc.shard_spec().value().shape[1] * t.element_size();
        }
        return row_bytes;
    };
    uint32_t reader_page_size = per_shard_page_size_bytes(input, padded_row_size_bytes);
    uint32_t writer_page_size = per_shard_page_size_bytes(output, unpadded_row_size_bytes);

    std::vector<uint32_t> num_unpadded_sticks_per_dim(num_dims);
    std::vector<uint32_t> num_padded_sticks_per_dim(num_dims);
    std::vector<uint32_t> accumulated_total_per_dim(num_dims);
    num_unpadded_sticks_per_dim[0] = 1;
    num_padded_sticks_per_dim[0] = 0;
    accumulated_total_per_dim[0] = 1;
    for (uint32_t i = 1; i < num_dims; i++) {
        uint32_t num_unpadded_dim = output_shape[-static_cast<int32_t>(i + 1)];
        uint32_t num_total_dim = input_shape[-static_cast<int32_t>(i + 1)];
        uint32_t num_padded_dim = (num_total_dim - num_unpadded_dim) * accumulated_total_per_dim[i - 1];
        num_unpadded_sticks_per_dim[i] = num_unpadded_dim;
        num_padded_sticks_per_dim[i] = num_padded_dim;
        accumulated_total_per_dim[i] = num_total_dim * accumulated_total_per_dim[i - 1];
    }

    uint32_t alignment = std::max(
        input.buffer()->buffer_type() == BufferType::DRAM ? ::hal::get_dram_alignment() : ::hal::get_l1_alignment(),
        output.buffer()->buffer_type() == BufferType::DRAM ? ::hal::get_dram_alignment() : ::hal::get_l1_alignment());
    uint32_t unpadded_row_size_bytes_offset = tt::round_up(unpadded_row_size_bytes, alignment);

    uint32_t l1_budget = ttnn::operations::data_movement::get_max_l1_space(input);
    TT_FATAL(
        static_cast<uint64_t>(2u) * unpadded_row_size_bytes_offset <= l1_budget,
        "ttnn::untilize_with_unpadding: row-major output row size {} B exceeds per-core L1 budget {} B; "
        "wide-row chunking is not yet supported for ROW_MAJOR input",
        unpadded_row_size_bytes_offset,
        l1_budget);

    uint32_t num_unpadded_sticks = output.physical_volume() / output_shape[-1];

    IDevice* device = input.device();
    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_sticks_per_core_group_1, num_sticks_per_core_group_2] =
        operation_attributes.sub_core_grids.has_value()
            ? tt::tt_metal::split_work_to_cores(operation_attributes.sub_core_grids.value(), num_unpadded_sticks)
            : tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_unpadded_sticks);

    Buffer* src0_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();
    TT_ASSERT(dst_buffer != nullptr, "Output buffer should be allocated on device!");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    constexpr uint8_t src0_cb_index = 0;

    uint32_t num_input_pages = std::max(num_sticks_per_core_group_1, num_sticks_per_core_group_2);
    uint32_t num_sticks_per_core_pad32 = round_up_to_mul32(num_input_pages);
    uint32_t num_sticks_per_core_read =
        num_input_pages == 0
            ? 0
            : tt::tt_metal::merge_num_sticks_to_read(num_sticks_per_core_pad32, unpadded_row_size_bytes_offset, 4096);
    uint32_t num_read_per_barrier =
        num_sticks_per_core_read == 0 ? 0 : num_sticks_per_core_pad32 / num_sticks_per_core_read;

    desc.cbs.push_back(CBDescriptor{
        .total_size = num_read_per_barrier * 2 * unpadded_row_size_bytes_offset,
        .core_ranges = all_cores,
        .format_descriptors = {{CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = unpadded_row_size_bytes_offset,
        }}},
    });

    std::vector<uint32_t> reader_compile_time_args;
    TensorAccessorArgs(*src0_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {static_cast<uint32_t>(src0_cb_index)};
    TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "slice_reader_unary_unpad_dims_rm_interleaved_start_id.cpp";
    reader_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = ReaderConfigDescriptor{};

    KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/data_movement/slice/device/kernels/dataflow/"
        "slice_writer_unary_stick_layout_interleaved_start_id.cpp";
    writer_desc.source_type = KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = WriterConfigDescriptor{};

    auto all_cores_vec = corerange_to_cores(all_cores);
    reader_desc.runtime_args.reserve(all_cores_vec.size());
    writer_desc.runtime_args.reserve(all_cores_vec.size());

    std::vector<uint32_t> id_per_dim(num_dims);
    uint32_t num_sticks_written = 0;
    uint32_t start_addr = src0_buffer->address();
    // No leading-offset misalignment (see comment above), so the reader always begins at an
    // aligned address and there is no memmove-based re-alignment step.
    constexpr uint32_t misalignment = 0;
    constexpr uint32_t chunk_size = 0;
    constexpr uint32_t num_chunks_per_stick = 0;
    constexpr uint32_t last_chunk_size = 0;

    for (const auto& core : all_cores_vec) {
        uint32_t num_sticks_per_core = 0;
        if (core_group_1.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_sticks_per_core = num_sticks_per_core_group_2;
        }

        uint32_t core_num_sticks_per_core_read = 0, core_num_read_per_barrier = 0;
        if (num_sticks_per_core != 0) {
            auto pad32 = round_up_to_mul32(num_sticks_per_core);
            core_num_sticks_per_core_read =
                tt::tt_metal::merge_num_sticks_to_read(pad32, unpadded_row_size_bytes_offset, 4096);
            core_num_read_per_barrier = pad32 / core_num_sticks_per_core_read;
        }

        id_per_dim[0] = num_sticks_written % num_unpadded_sticks_per_dim[0];
        uint32_t unpadded_written = num_sticks_written / num_unpadded_sticks_per_dim[0];
        uint32_t start_id = id_per_dim[0];
        for (uint32_t j = 1; j < num_dims; j++) {
            id_per_dim[j] = unpadded_written % num_unpadded_sticks_per_dim[j];
            unpadded_written = unpadded_written / num_unpadded_sticks_per_dim[j];
            start_id += id_per_dim[j] * accumulated_total_per_dim[j - 1];
        }

        std::vector<uint32_t> reader_kernel_args = {
            start_addr,
            reader_page_size,
            unpadded_row_size_bytes,
            unpadded_row_size_bytes_offset,
            num_dims,
            misalignment,
            start_id,
            num_sticks_per_core,
            core_num_sticks_per_core_read,
            core_num_read_per_barrier,
            chunk_size,
            num_chunks_per_stick,
            last_chunk_size};
        reader_kernel_args.insert(
            reader_kernel_args.end(), num_unpadded_sticks_per_dim.begin(), num_unpadded_sticks_per_dim.end());
        reader_kernel_args.insert(
            reader_kernel_args.end(), num_padded_sticks_per_dim.begin(), num_padded_sticks_per_dim.end());
        reader_kernel_args.insert(reader_kernel_args.end(), id_per_dim.begin(), id_per_dim.end());

        std::vector<uint32_t> writer_kernel_args = {
            dst_buffer->address(),
            unpadded_row_size_bytes,
            unpadded_row_size_bytes_offset,
            num_sticks_per_core,
            core_num_sticks_per_core_read,
            core_num_read_per_barrier,
            num_sticks_written,
            writer_page_size,
            chunk_size,
            num_chunks_per_stick,
            last_chunk_size};

        num_sticks_written += num_sticks_per_core;
        reader_desc.runtime_args.emplace_back(core, std::move(reader_kernel_args));
        writer_desc.runtime_args.emplace_back(core, std::move(writer_kernel_args));
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));

    return desc;
}

}  // namespace ttnn::prim
