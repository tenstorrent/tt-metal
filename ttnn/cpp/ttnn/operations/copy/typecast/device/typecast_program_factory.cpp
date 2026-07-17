// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_program_factory.hpp"

#include <optional>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/program_descriptors.hpp>

namespace ttnn::prim {

using namespace tt::constants;

tt::tt_metal::ProgramDescriptor TypecastProgramFactory::create_descriptor(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Tensor& input = tensor_args.input;
    const DataType& input_dtype = args.input_dtype;
    const DataType& output_dtype = args.output_dtype;
    const bool is_row_major = input.layout() == Layout::ROW_MAJOR;

    tt::tt_metal::ProgramDescriptor desc;

    const tt::DataFormat cb_data_format_input = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());

    std::optional<tt::tt_metal::Tile> tile = std::nullopt;
    uint32_t single_tile_size_input;
    uint32_t single_tile_size_output;
    if (is_row_major) {
        single_tile_size_input = tt::tile_size(cb_data_format_input);
        single_tile_size_output = tt::tile_size(cb_data_format_output);
    } else {
        tile = input.tensor_spec().tile();
        single_tile_size_input = tile->get_tile_size(cb_data_format_input);
        single_tile_size_output = tile->get_tile_size(cb_data_format_output);
    }

    const auto* device = input.device();

    // Get number of pages (tiles for TILE layout, rows for ROW_MAJOR layout)
    const uint32_t num_pages = input.buffer()->num_pages();

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    // Set CB page size correctly based on layout
    // - For TILE layout: page = one tile (may be tiny, e.g. 16x32)
    // - For ROW_MAJOR layout: page = one full row including padding
    const uint32_t input_page_size = is_row_major ? src_buffer->page_size() : single_tile_size_input;
    const uint32_t output_page_size = is_row_major ? dst_buffer->page_size() : single_tile_size_output;

    const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_items_per_core_group_1, num_items_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_pages, is_row_major);
    (void)num_cores;

    const std::optional<TileDescriptor> tile_descriptor =
        tile.has_value() ? std::optional<TileDescriptor>(TileDescriptor(tile.value())) : std::nullopt;

    constexpr uint8_t src0_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t num_input_pages = 2;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = num_input_pages * input_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format_input,
            .page_size = input_page_size,
            .tile = tile_descriptor,
        }}},
    });

    constexpr uint8_t output_cb_index = tt::CBIndex::c_2;
    constexpr uint32_t num_output_pages = 2;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = num_output_pages * output_page_size,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = cb_data_format_output,
            .page_size = output_page_size,
            .tile = tile_descriptor,
        }}},
    });

    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    tt::tt_metal::KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp";
    reader_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    std::vector<uint32_t> compute_kernel_args_group_1 = {
        num_items_per_core_group_1,  // per_core_block_cnt
        1,                           // per_core_block_dim (always 1, works for both tiled and row-major)
        src0_cb_index,
        output_cb_index};

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[src0_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
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

    tt::tt_metal::KernelDescriptor compute_desc_group_1;
    compute_desc_group_1.kernel_source = path;
    compute_desc_group_1.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_desc_group_1.core_ranges = core_group_1;
    compute_desc_group_1.compile_time_args = compute_kernel_args_group_1;
    for (const auto& [name, value] : unary_defines) {
        compute_desc_group_1.defines.emplace_back(name, value);
    }
    compute_desc_group_1.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .fp32_dest_acc_en = args.fp32_dest_acc_en,
        .unpack_to_dest_mode = unpack_to_dest_mode,
        .bfp8_pack_precise = args.bfp8_pack_precise,
        .math_approx_mode = math_approx_mode};

    std::optional<tt::tt_metal::KernelDescriptor> compute_desc_group_2;
    if (!core_group_2.ranges().empty()) {
        std::vector<uint32_t> compute_kernel_args_group_2 = {
            num_items_per_core_group_2,  // per_core_block_cnt
            1,                           // per_core_block_dim (always 1)
            src0_cb_index,
            output_cb_index};

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

    // Convert CoreRangeSet to vector of cores in the correct order
    // Use row_wise=true for row-major layout to match row distribution, false for tile layout
    auto cores_vec = corerange_to_cores(all_cores, std::nullopt, is_row_major);

    uint32_t num_items_written = 0;
    for (const auto& core : cores_vec) {
        uint32_t num_items_per_core = 0;
        if (core_group_1.contains(core)) {
            num_items_per_core = num_items_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_items_per_core = num_items_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        reader_desc.emplace_runtime_args(core, {src_buffer, num_items_per_core, num_items_written});
        writer_desc.emplace_runtime_args(core, {dst_buffer, num_items_per_core, num_items_written});
        num_items_written += num_items_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc_group_1));
    if (compute_desc_group_2.has_value()) {
        desc.kernels.push_back(std::move(*compute_desc_group_2));
    }

    return desc;
}

// For sub_core_grids
tt::tt_metal::ProgramDescriptor TypecastSubgridProgramFactory::create_descriptor(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& input_dtype = args.input_dtype;
    const auto& output_dtype = args.output_dtype;
    const auto& sub_core_grids = args.sub_core_grids;

    TT_FATAL(sub_core_grids.has_value(), "sub_core_grids cannot be null");

    tt::tt_metal::ProgramDescriptor desc;

    const auto& tile = input.tensor_spec().tile();
    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tile.get_tile_size(cb_data_format);
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size_output = tile.get_tile_size(cb_data_format_output);

    uint32_t ntiles = input.physical_volume() / tile.get_tile_hw();
    uint32_t ncores = sub_core_grids->num_cores();

    TT_FATAL(ncores != 0, "number of cores cannot be 0");

    for (uint32_t core_id = ncores; core_id >= 1; core_id--) {
        if (ntiles % ncores == 0) {
            break;
        }
        ncores--;
    }
    TT_FATAL(
        (ntiles % (ncores) == 0), "{} num of tiles are not split uniformly across {} num of cores", ntiles, ncores);

    auto cores = corerange_to_cores(sub_core_grids.value(), ncores, true);
    auto all_cores = num_cores_to_corerangeset_in_subcoregrids(cores[0], ncores, sub_core_grids.value(), true);
    if (ncores == 1) {
        all_cores = ttnn::CoreRangeSet(ttnn::CoreRange(cores[0]));
    }

    const TileDescriptor tile_descriptor(tile);

    constexpr uint8_t src0_cb_index = tt::CBIndex::c_0;
    constexpr uint32_t num_input_tiles = 2;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = num_input_tiles * single_tile_size,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = src0_cb_index,
            .data_format = cb_data_format,
            .page_size = single_tile_size,
            .tile = tile_descriptor,
        }}},
    });

    constexpr uint8_t output_cb_index = tt::CBIndex::c_2;
    constexpr uint32_t num_output_tiles = 2;
    desc.cbs.push_back(tt::tt_metal::CBDescriptor{
        .total_size = num_output_tiles * single_tile_size_output,
        .core_ranges = all_cores,
        .format_descriptors = {{tt::tt_metal::CBFormatDescriptor{
            .buffer_index = output_cb_index,
            .data_format = cb_data_format_output,
            .page_size = single_tile_size_output,
            .tile = tile_descriptor,
        }}},
    });

    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    std::vector<uint32_t> reader_compile_time_args;
    tt::tt_metal::TensorAccessorArgs(*src_buffer).append_to(reader_compile_time_args);
    std::vector<uint32_t> writer_compile_time_args = {(std::uint32_t)output_cb_index};
    tt::tt_metal::TensorAccessorArgs(*dst_buffer).append_to(writer_compile_time_args);

    tt::tt_metal::KernelDescriptor reader_desc;
    reader_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp";
    reader_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    reader_desc.core_ranges = all_cores;
    reader_desc.compile_time_args = std::move(reader_compile_time_args);
    reader_desc.config = tt::tt_metal::ReaderConfigDescriptor{};

    tt::tt_metal::KernelDescriptor writer_desc;
    writer_desc.kernel_source =
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
    writer_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    writer_desc.core_ranges = all_cores;
    writer_desc.compile_time_args = std::move(writer_compile_time_args);
    writer_desc.config = tt::tt_metal::WriterConfigDescriptor{};

    uint32_t ntiles_per_core = ntiles / ncores;
    std::vector<uint32_t> compute_kernel_args = {
        static_cast<uint32_t>(ntiles_per_core),  // per_core_block_cnt
        1,                                       // per_core_block_dim
        src0_cb_index,
        output_cb_index};

    std::vector<tt::tt_metal::UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, tt::tt_metal::UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[src0_cb_index] = tt::tt_metal::UnpackToDestMode::UnpackToDestFp32;
    }

    bool math_approx_mode = false;

    std::map<std::string, std::string> unary_defines;
    unary_defines["TYPECAST_LLK_INIT"] = fmt::format(
        "typecast_tile_init<{0}u, {1}u>",
        static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
        static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));
    unary_defines["TYPECAST_LLK"] = fmt::format(
        "typecast_tile<{0}u, {1}u>",
        static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
        static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));

    const auto* path = "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp";

    tt::tt_metal::KernelDescriptor compute_desc;
    compute_desc.kernel_source = path;
    compute_desc.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    compute_desc.core_ranges = all_cores;
    compute_desc.compile_time_args = compute_kernel_args;
    for (const auto& [name, value] : unary_defines) {
        compute_desc.defines.emplace_back(name, value);
    }
    compute_desc.config = tt::tt_metal::ComputeConfigDescriptor{
        .math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .fp32_dest_acc_en = args.fp32_dest_acc_en,
        .unpack_to_dest_mode = unpack_to_dest_mode,
        .bfp8_pack_precise = args.bfp8_pack_precise,
        .math_approx_mode = math_approx_mode};

    uint32_t tile_start_id = 0;

    for (auto core : cores) {
        reader_desc.emplace_runtime_args(core, {src_buffer, ntiles_per_core, tile_start_id});
        writer_desc.emplace_runtime_args(core, {dst_buffer, ntiles_per_core, tile_start_id});
        tile_start_id += ntiles_per_core;
    }

    desc.kernels.push_back(std::move(reader_desc));
    desc.kernels.push_back(std::move(writer_desc));
    desc.kernels.push_back(std::move(compute_desc));

    return desc;
}

}  // namespace ttnn::prim
