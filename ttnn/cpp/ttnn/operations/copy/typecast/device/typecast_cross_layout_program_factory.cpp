// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_cross_layout_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/operations/cb_utils.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"

namespace ttnn::prim {

using namespace tt::constants;
using namespace tt::tt_metal;

// ─── TILE → ROW_MAJOR (descriptor) ──────────────────────────────────────────

static ProgramDescriptor create_tile_to_rm_descriptor(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    const auto input_dtype = args.input_dtype;
    const auto output_dtype = args.output_dtype;

    const auto cb_data_format_input = datatype_to_dataformat_converter(input_dtype);
    const auto single_tile_size_input = tt::tile_size(cb_data_format_input);
    const auto cb_data_format_output = datatype_to_dataformat_converter(output_dtype);
    const auto single_tile_size_output = tt::tile_size(cb_data_format_output);

    const auto* device = input.device();
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    const auto padded_shape = input.padded_shape();
    const uint32_t tensor_width = padded_shape[-1];
    const uint32_t tensor_height = input.physical_volume() / tensor_width;
    const uint32_t ntiles_per_row = tensor_width / TILE_WIDTH;
    const uint32_t nblocks = tensor_height / TILE_HEIGHT;

    const auto grid_size = device->compute_with_storage_grid_size();
    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(grid_size, nblocks);

    ProgramDescriptor desc;

    // ── CBs ──────────────────────────────────────────────────────────────
    constexpr uint32_t input_cb_index = tt::CBIndex::c_0;
    {
        const uint32_t input_cb_num_tiles = ntiles_per_row * 2;
        CBDescriptor cb;
        cb.total_size = input_cb_num_tiles * single_tile_size_input;
        cb.core_ranges = all_cores;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_cb_index),
            .data_format = cb_data_format_input,
            .page_size = single_tile_size_input});
        desc.cbs.push_back(std::move(cb));
    }

    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    {
        const uint32_t output_cb_num_tiles = ntiles_per_row * 2;
        CBDescriptor cb;
        cb.total_size = output_cb_num_tiles * single_tile_size_output;
        cb.core_ranges = all_cores;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = cb_data_format_output,
            .page_size = single_tile_size_output});
        desc.cbs.push_back(std::move(cb));
    }

    // ── Reader ───────────────────────────────────────────────────────────
    std::vector<uint32_t> reader_ct_args = {(uint32_t)input_cb_index};
    TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);

    const bool has_cliff = !core_range_cliff.empty();
    const uint32_t ncores_full = ncores - has_cliff;
    uint32_t tile_start_id = 0;

    KernelDescriptor::RuntimeArgs reader_rt;
    const auto& cores = corerange_to_cores(all_cores);
    for (uint32_t i = 0; i < ncores; ++i) {
        const uint32_t cur_nblocks = (i < ncores_full) ? nblocks_per_core : nblocks_per_core_cliff;
        const uint32_t num_tiles_this_core = cur_nblocks * ntiles_per_row;
        reader_rt.emplace_back(
            cores[i], std::vector<uint32_t>{src_buffer->address(), num_tiles_this_core, tile_start_id});
        tile_start_id += num_tiles_this_core;
    }

    {
        KernelDescriptor k;
        k.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/reader_unary_start_id.cpp";
        k.core_ranges = all_cores;
        k.compile_time_args = reader_ct_args;
        k.runtime_args = std::move(reader_rt);
        k.config = ReaderConfigDescriptor{};
        desc.kernels.push_back(std::move(k));
    }

    // ── Writer ───────────────────────────────────────────────────────────
    const uint32_t output_element_size = output.element_size();
    const uint32_t output_stick_size = tensor_width * output_element_size;

    std::vector<uint32_t> writer_ct_args = {
        (uint32_t)output_cb_index,
        output_stick_size,
        (uint32_t)TILE_HEIGHT,
        ntiles_per_row,
        (uint32_t)1,
        output_element_size,
        tensor_width,
        tensor_width};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

    KernelDescriptor::RuntimeArgs writer_rt;
    uint32_t block_start_id = 0;
    for (uint32_t i = 0; i < ncores; ++i) {
        const uint32_t cur_nblocks = (i < ncores_full) ? nblocks_per_core : nblocks_per_core_cliff;
        writer_rt.emplace_back(
            cores[i], std::vector<uint32_t>{dst_buffer->address(), cur_nblocks, block_start_id, tensor_width, 0u, 0u});
        block_start_id += cur_nblocks;
    }

    {
        KernelDescriptor k;
        k.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/untilize/device/kernels/dataflow/"
            "writer_unary_stick_layout_split_rows_multi_core.cpp";
        k.core_ranges = all_cores;
        k.compile_time_args = writer_ct_args;
        k.runtime_args = std::move(writer_rt);
        k.config = WriterConfigDescriptor{};
        desc.kernels.push_back(std::move(k));
    }

    // ── Compute ──────────────────────────────────────────────────────────
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[input_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    auto compute_defines = make_typecast_compute_defines_desc(input_dtype, output_dtype);
    compute_defines.emplace_back("UNTILIZE_OUTPUT", "1");

    if (!core_range.ranges().empty()) {
        KernelDescriptor k;
        k.kernel_source = TYPECAST_COMPUTE_KERNEL_PATH;
        k.core_ranges = core_range;
        k.compile_time_args = {nblocks_per_core, ntiles_per_row, input_cb_index, output_cb_index};
        k.defines = compute_defines;
        k.config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = args.fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = args.bfp8_pack_precise};
        desc.kernels.push_back(std::move(k));
    }

    if (!core_range_cliff.empty()) {
        KernelDescriptor k;
        k.kernel_source = TYPECAST_COMPUTE_KERNEL_PATH;
        k.core_ranges = core_range_cliff;
        k.compile_time_args = {nblocks_per_core_cliff, ntiles_per_row, input_cb_index, output_cb_index};
        k.defines = compute_defines;
        k.config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = args.fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = args.bfp8_pack_precise};
        desc.kernels.push_back(std::move(k));
    }

    return desc;
}

// ─── ROW_MAJOR → TILE (descriptor) ──────────────────────────────────────────

static ProgramDescriptor create_rm_to_tile_descriptor(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    const auto input_dtype = args.input_dtype;
    const auto output_dtype = args.output_dtype;

    const auto cb_data_format_input = datatype_to_dataformat_converter(input_dtype);
    const auto single_tile_size_input = tt::tile_size(cb_data_format_input);
    const auto cb_data_format_output = datatype_to_dataformat_converter(output_dtype);
    const auto single_tile_size_output = tt::tile_size(cb_data_format_output);

    const auto* device = input.device();
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    const auto logical_shape = input.logical_shape();
    const auto logical_width = logical_shape[-1];
    const uint32_t ntiles_per_row = tt::div_up(logical_width, TILE_WIDTH);
    const uint32_t ntiles = dst_buffer->num_pages();
    const uint32_t nblocks = tt::div_up(ntiles, ntiles_per_row);

    const auto grid_size = device->compute_with_storage_grid_size();
    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(grid_size, nblocks);

    const uint32_t rm_page_size = src_buffer->page_size();

    ProgramDescriptor desc;

    // ── CBs ──────────────────────────────────────────────────────────────
    constexpr uint32_t input_cb_index = tt::CBIndex::c_0;
    {
        CBDescriptor cb;
        cb.total_size = ntiles_per_row * single_tile_size_input;
        cb.core_ranges = all_cores;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(input_cb_index),
            .data_format = cb_data_format_input,
            .page_size = single_tile_size_input});
        desc.cbs.push_back(std::move(cb));
    }

    constexpr uint32_t intermediate_cb_index = tt::CBIndex::c_1;
    {
        const uint32_t intermediate_cb_num_tiles = ntiles_per_row * 2;
        CBDescriptor cb;
        cb.total_size = intermediate_cb_num_tiles * single_tile_size_input;
        cb.core_ranges = all_cores;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(intermediate_cb_index),
            .data_format = cb_data_format_input,
            .page_size = single_tile_size_input});
        desc.cbs.push_back(std::move(cb));
    }

    constexpr uint32_t output_cb_index = tt::CBIndex::c_16;
    {
        const uint32_t output_cb_num_tiles = ntiles_per_row * 2;
        CBDescriptor cb;
        cb.total_size = output_cb_num_tiles * single_tile_size_output;
        cb.core_ranges = all_cores;
        cb.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = cb_data_format_output,
            .page_size = single_tile_size_output});
        desc.cbs.push_back(std::move(cb));
    }

    // ── Reader ───────────────────────────────────────────────────────────
    const uint32_t aligned_page_size = src_buffer->aligned_page_size();
    std::vector<uint32_t> reader_ct_args = {aligned_page_size, (uint32_t)1, rm_page_size};
    TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);

    const bool has_cliff = !core_range_cliff.empty();
    const uint32_t ncores_full = ncores - has_cliff;
    uint32_t tile_start_id = 0;
    uint32_t page_start_id = 0;

    KernelDescriptor::RuntimeArgs reader_rt;
    const auto& cores = corerange_to_cores(all_cores);
    for (uint32_t i = 0; i < ncores; ++i) {
        const uint32_t cur_nblocks = (i < ncores_full) ? nblocks_per_core : nblocks_per_core_cliff;
        reader_rt.emplace_back(
            cores[i],
            std::vector<uint32_t>{
                src_buffer->address(),
                cur_nblocks * TILE_HEIGHT,
                rm_page_size,
                ntiles_per_row,
                rm_page_size,
                1u,
                0u,
                0u,
                page_start_id});
        page_start_id += cur_nblocks * TILE_HEIGHT;
    }

    {
        KernelDescriptor k;
        k.kernel_source =
            "ttnn/cpp/ttnn/operations/data_movement/tilize/device/kernels/dataflow/"
            "reader_unary_stick_layout_split_rows_multicore.cpp";
        k.core_ranges = all_cores;
        k.compile_time_args = reader_ct_args;
        k.runtime_args = std::move(reader_rt);
        k.config = ReaderConfigDescriptor{};
        desc.kernels.push_back(std::move(k));
    }

    // ── Writer ───────────────────────────────────────────────────────────
    std::vector<uint32_t> writer_ct_args = {(uint32_t)output_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

    KernelDescriptor::RuntimeArgs writer_rt;
    tile_start_id = 0;
    for (uint32_t i = 0; i < ncores; ++i) {
        const uint32_t cur_nblocks = (i < ncores_full) ? nblocks_per_core : nblocks_per_core_cliff;
        const uint32_t num_tiles_this_core = cur_nblocks * ntiles_per_row;
        writer_rt.emplace_back(
            cores[i], std::vector<uint32_t>{dst_buffer->address(), num_tiles_this_core, tile_start_id});
        tile_start_id += num_tiles_this_core;
    }

    {
        KernelDescriptor k;
        k.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
        k.core_ranges = all_cores;
        k.compile_time_args = writer_ct_args;
        k.runtime_args = std::move(writer_rt);
        k.config = WriterConfigDescriptor{};
        desc.kernels.push_back(std::move(k));
    }

    // ── Compute ──────────────────────────────────────────────────────────
    const auto is_32bit_type = [](DataType dt) {
        return dt == DataType::FLOAT32 || dt == DataType::INT32 || dt == DataType::UINT32;
    };
    const bool rm_to_tile_fp32_dest = is_32bit_type(input_dtype) || is_32bit_type(output_dtype);

    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (rm_to_tile_fp32_dest) {
        unpack_to_dest_mode[input_cb_index] = UnpackToDestMode::UnpackToDestFp32;
        unpack_to_dest_mode[intermediate_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    auto compute_defines = make_typecast_compute_defines_desc(input_dtype, output_dtype);
    compute_defines.emplace_back("TILIZE_INPUT", "1");
    if (is_32bit_type(output_dtype)) {
        compute_defines.emplace_back("TYPECAST_OUTPUT_32BIT", "1");
    }

    if (!core_range.ranges().empty()) {
        KernelDescriptor k;
        k.kernel_source = TYPECAST_COMPUTE_KERNEL_PATH;
        k.core_ranges = core_range;
        k.compile_time_args = {
            nblocks_per_core, ntiles_per_row, input_cb_index, output_cb_index, intermediate_cb_index};
        k.defines = compute_defines;
        k.config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = rm_to_tile_fp32_dest,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = args.bfp8_pack_precise};
        desc.kernels.push_back(std::move(k));
    }

    if (!core_range_cliff.empty()) {
        KernelDescriptor k;
        k.kernel_source = TYPECAST_COMPUTE_KERNEL_PATH;
        k.core_ranges = core_range_cliff;
        k.compile_time_args = {
            nblocks_per_core_cliff, ntiles_per_row, input_cb_index, output_cb_index, intermediate_cb_index};
        k.defines = compute_defines;
        k.config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = rm_to_tile_fp32_dest,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = args.bfp8_pack_precise};
        desc.kernels.push_back(std::move(k));
    }

    return desc;
}

// ─── Public API ──────────────────────────────────────────────────────────────

ProgramDescriptor TypecastCrossLayoutProgramFactory::create_descriptor(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    const bool input_is_tile = (tensor_args.input.layout() == Layout::TILE);
    const bool output_is_tile = (output.layout() == Layout::TILE);

    TT_FATAL(
        input_is_tile != output_is_tile,
        "TypecastCrossLayoutProgramFactory requires different input and output layouts. "
        "Input: {}, Output: {}",
        input_is_tile ? "TILE" : "ROW_MAJOR",
        output_is_tile ? "TILE" : "ROW_MAJOR");

    if (input_is_tile && !output_is_tile) {
        return create_tile_to_rm_descriptor(args, tensor_args, output);
    }
    return create_rm_to_tile_descriptor(args, tensor_args, output);
}

TypecastCrossLayoutProgramFactory::cached_program_t TypecastCrossLayoutProgramFactory::create(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    auto descriptor = create_descriptor(args, tensor_args, output);
    Program program{descriptor};

    // Kernel handles assigned sequentially: reader=0, writer=1
    constexpr KernelHandle reader_kernel_id = 0;
    constexpr KernelHandle writer_kernel_id = 1;

    const auto grid_size = tensor_args.input.device()->compute_with_storage_grid_size();
    const uint32_t ncores = static_cast<uint32_t>(descriptor.kernels[0].runtime_args.size());

    return cached_program_t{std::move(program), {reader_kernel_id, writer_kernel_id, ncores, grid_size.y}};
}

void TypecastCrossLayoutProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TypecastParams& /*operation_attributes*/,
    const TypecastInputs& tensor_args,
    Tensor& output) {
    auto& reader_kernel_id = cached_program.shared_variables.reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.writer_kernel_id;
    const uint32_t num_cores = cached_program.shared_variables.num_cores;
    const uint32_t num_cores_y = cached_program.shared_variables.num_cores_y;

    auto& program = cached_program.program;
    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = output.buffer();

    for (uint32_t i = 0; i < num_cores; i++) {
        const CoreCoord core = {i / num_cores_y, i % num_cores_y};
        {
            auto& runtime_args = GetRuntimeArgs(program, reader_kernel_id, core);
            runtime_args[0] = src_buffer->address();
        }
        {
            auto& runtime_args = GetRuntimeArgs(program, writer_kernel_id, core);
            runtime_args[0] = dst_buffer->address();
        }
    }
}

}  // namespace ttnn::prim
