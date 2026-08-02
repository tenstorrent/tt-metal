// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_rm_chunked_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt_stl/span.hpp>

namespace ttnn::prim {

using namespace tt::constants;

namespace {
struct ChunkSizeConfig {
    uint32_t input_full_chunk_size_bytes;          // actual bytes read from DRAM per full chunk
    uint32_t output_full_chunk_size_bytes;         // actual bytes written to DRAM per full chunk
    uint32_t input_partial_chunk_size_bytes;       // actual bytes read from DRAM for partial chunk
    uint32_t output_partial_chunk_size_bytes;      // actual bytes written to DRAM for partial chunk
    uint32_t padded_input_full_chunk_size_bytes;   // CB page size (one full hardware tile)
    uint32_t padded_output_full_chunk_size_bytes;  // CB page size (one full hardware tile)
    uint32_t full_chunks_per_row;
    uint32_t partial_chunks_per_row;
};

ChunkSizeConfig calculate_chunk_config(
    uint32_t row_width_elements, uint32_t input_element_size, uint32_t output_element_size) {
    constexpr uint32_t max_elements_per_chunk = TILE_HW;
    const uint32_t elements_per_full_chunk = std::min(max_elements_per_chunk, row_width_elements);

    // Actual chunk sizes in bytes (for DRAM reads/writes)
    const uint32_t input_full_chunk_size_bytes = elements_per_full_chunk * input_element_size;
    const uint32_t output_full_chunk_size_bytes = elements_per_full_chunk * output_element_size;

    // copy_tile and pack_tile always access a full hardware tile. Keep each CB page at least
    // that large so an LLK operation cannot cross into the next double-buffered page.
    constexpr uint32_t padded_full_elements = TILE_HW;
    const uint32_t padded_input_full_chunk_size_bytes = padded_full_elements * input_element_size;
    const uint32_t padded_output_full_chunk_size_bytes = padded_full_elements * output_element_size;

    // Calculate how many chunks per row
    const uint32_t full_chunks_per_row = row_width_elements / elements_per_full_chunk;
    const uint32_t remainder = row_width_elements % elements_per_full_chunk;
    const uint32_t partial_chunks_per_row = (remainder > 0) ? 1 : 0;
    const uint32_t input_partial_chunk_size_bytes = remainder * input_element_size;
    const uint32_t output_partial_chunk_size_bytes = remainder * output_element_size;

    return ChunkSizeConfig{
        .input_full_chunk_size_bytes = input_full_chunk_size_bytes,
        .output_full_chunk_size_bytes = output_full_chunk_size_bytes,
        .input_partial_chunk_size_bytes = input_partial_chunk_size_bytes,
        .output_partial_chunk_size_bytes = output_partial_chunk_size_bytes,
        .padded_input_full_chunk_size_bytes = padded_input_full_chunk_size_bytes,
        .padded_output_full_chunk_size_bytes = padded_output_full_chunk_size_bytes,
        .full_chunks_per_row = full_chunks_per_row,
        .partial_chunks_per_row = partial_chunks_per_row,
    };
}

}  // anonymous namespace

tt::tt_metal::ProgramDescriptor TypecastRowMajorChunkedProgramFactory::create_descriptor(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Tensor& input = tensor_args.input;
    const DataType& input_dtype = args.input_dtype;
    const DataType& output_dtype = args.output_dtype;

    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "This factory is only for ROW_MAJOR layout");

    tt::tt_metal::ProgramDescriptor desc;

    const tt::DataFormat cb_data_format_input = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_element_size = tt::datum_size(cb_data_format_input);
    const tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t output_element_size = tt::datum_size(cb_data_format_output);

    const auto* device = input.device();

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    // Get row information
    const auto& padded_shape = input.padded_shape();
    const uint32_t row_width_elements = padded_shape[padded_shape.rank() - 1];
    const uint32_t num_rows = src_buffer->num_pages();

    // Calculate chunk configuration
    const ChunkSizeConfig chunk_config =
        calculate_chunk_config(row_width_elements, input_element_size, output_element_size);

    const uint32_t input_full_chunk_size_bytes = chunk_config.input_full_chunk_size_bytes;
    const uint32_t output_full_chunk_size_bytes = chunk_config.output_full_chunk_size_bytes;
    const uint32_t input_partial_chunk_size_bytes = chunk_config.input_partial_chunk_size_bytes;
    const uint32_t output_partial_chunk_size_bytes = chunk_config.output_partial_chunk_size_bytes;
    const uint32_t padded_input_full_chunk_size_bytes = chunk_config.padded_input_full_chunk_size_bytes;
    const uint32_t padded_output_full_chunk_size_bytes = chunk_config.padded_output_full_chunk_size_bytes;
    const uint32_t full_chunks_per_row = chunk_config.full_chunks_per_row;
    const uint32_t partial_chunks_per_row = chunk_config.partial_chunks_per_row;

    const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();

    // Split work by rows (each core handles complete rows with both full and partial chunks)
    auto [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_rows, true);
    (void)num_cores;

    constexpr uint8_t input_cb_index = tt::CBIndex::c_0;
    constexpr uint8_t output_cb_index = tt::CBIndex::c_2;
    constexpr uint32_t num_input_pages = 2;   // Always use double buffering
    constexpr uint32_t num_output_pages = 2;  // Always use double buffering

    // Additionally align CB page sizes to the source/destination buffer alignment so that the
    // double-buffered CB pages share the same residue (mod buffer alignment) as their DRAM pages.
    // This is required by the NOC: DRAM->L1 reads enforce (src_addr & alignment-1) ==
    // (dst_addr & alignment-1).  On Blackhole the DRAM alignment is 64B; without this an
    // 8-bit input with a 32-element padded chunk yields a 32B page, leaving the second
    // double-buffered page mis-aligned and causing ttsim NOC alignment crashes
    // (see test_typecast_row_major_vs_tile_layout[UINT8_TO_BFLOAT16-8x2x64x32]).
    const uint32_t input_cb_page_size_bytes = tt::align(padded_input_full_chunk_size_bytes, src_buffer->alignment());
    const uint32_t output_cb_page_size_bytes = tt::align(padded_output_full_chunk_size_bytes, dst_buffer->alignment());

    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = num_input_pages * input_cb_page_size_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = input_cb_index,
            .data_format = cb_data_format_input,
            .page_size = input_cb_page_size_bytes,
        }}},
    });

    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = num_output_pages * output_cb_page_size_bytes,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = cb_data_format_output,
            .page_size = output_cb_page_size_bytes,
        }}},
    });

    std::vector<uint32_t> reader_compile_time_args = {
        input_cb_index,                  // 0: cb_id_in
        full_chunks_per_row,             // 1: full_chunks_per_row
        partial_chunks_per_row,          // 2: partial_chunks_per_row (0 or 1)
        input_full_chunk_size_bytes,     // 3: full_chunk_size_bytes (DRAM read size)
        input_partial_chunk_size_bytes,  // 4: partial_chunk_size_bytes (DRAM read size)
    };
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);

    std::vector<uint32_t> writer_compile_time_args = {
        output_cb_index,                  // 0: cb_id_out
        full_chunks_per_row,              // 1: full_chunks_per_row
        partial_chunks_per_row,           // 2: partial_chunks_per_row (0 or 1)
        output_full_chunk_size_bytes,     // 3: full_chunk_size_bytes (DRAM write size)
        output_partial_chunk_size_bytes,  // 4: partial_chunk_size_bytes (DRAM write size)
    };
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    tt::tt_metal::KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_typecast_rm_chunked.cpp";
    reader_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/writer_typecast_rm_chunked.cpp";
    writer_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    // Create compute kernels - compute per_core_block_cnt as total chunks (full + partial) per core
    const uint32_t chunks_per_row_total = full_chunks_per_row + partial_chunks_per_row;

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        num_rows_per_core_group_1 * chunks_per_row_total,  // per_core_block_cnt (rows * total_chunks_per_row)
        1,
        input_cb_index,
        output_cb_index};

    std::vector<uint32_t> compute_kernel_args_group_2 = {
        num_rows_per_core_group_2 * chunks_per_row_total,  // per_core_block_cnt (rows * total_chunks_per_row)
        1,
        input_cb_index,
        output_cb_index};

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[input_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    constexpr bool math_approx_mode = false;

    std::map<std::string, std::string> unary_defines;
    unary_defines["TYPECAST_LLK_INIT"] = fmt::format(
        "typecast_tile_init<{0}u, {1}u>",
        static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
        static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));
    unary_defines["TYPECAST_LLK"] = fmt::format(
        "typecast_tile<{0}u, {1}u>",
        static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
        static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));

    const char* const path = "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp";

    std::optional<tt::tt_metal::KernelDescriptor> compute_desc_group_1;
    if (!core_group_1.ranges().empty()) {
        compute_desc_group_1.emplace();
        compute_desc_group_1->kernel_source = path;
        compute_desc_group_1->source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_group_1->core_ranges = core_group_1;
        compute_desc_group_1->compile_time_args = compute_kernel_args_group_1;
        for (const auto& [name, value] : unary_defines) {
            compute_desc_group_1->defines.emplace_back(name, value);
        }
        compute_desc_group_1->config = tt::tt_metal::ComputeConfigDescriptor{
            .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
            .fp32_dest_acc_en = args.fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = args.bfp8_pack_precise,
            .math_approx_mode = math_approx_mode};
    }

    std::optional<tt::tt_metal::KernelDescriptor> compute_desc_group_2;
    if (!core_group_2.ranges().empty()) {
        compute_desc_group_2.emplace();
        compute_desc_group_2->kernel_source = path;
        compute_desc_group_2->source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
        compute_desc_group_2->core_ranges = core_group_2;
        compute_desc_group_2->compile_time_args = compute_kernel_args_group_2;
        for (const auto& [name, value] : unary_defines) {
            compute_desc_group_2->defines.emplace_back(name, value);
        }
        compute_desc_group_2->config = tt::tt_metal::ComputeConfigDescriptor{
            .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
            .fp32_dest_acc_en = args.fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = args.bfp8_pack_precise,
            .math_approx_mode = math_approx_mode};
    }

    // Assign runtime args to cores (distributing rows)
    auto cores_vec = corerange_to_cores(all_cores, std::nullopt, true);
    uint32_t row_idx = 0;

    for (const auto& core : cores_vec) {
        bool is_group_1 = core_group_1.contains(core);
        uint32_t num_rows_for_core = is_group_1 ? num_rows_per_core_group_1 : num_rows_per_core_group_2;
        uint32_t start_row_id = row_idx;

        reader_desc.emplace_runtime_args(core, {src_buffer, num_rows_for_core, start_row_id});
        writer_desc.emplace_runtime_args(core, {dst_buffer, num_rows_for_core, start_row_id});

        row_idx += num_rows_for_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    if (compute_desc_group_1.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc_group_1));
    }
    if (compute_desc_group_2.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc_group_2));
    }

    return desc;
}

}  // namespace ttnn::prim
