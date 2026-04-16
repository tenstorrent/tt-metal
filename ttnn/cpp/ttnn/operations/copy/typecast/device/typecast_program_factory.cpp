// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/cb_utils.hpp"

namespace ttnn::prim {

using namespace tt::constants;

tt::tt_metal::ProgramDescriptor TypecastProgramFactory::create_descriptor(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto input_dtype = args.input_dtype;
    const auto output_dtype = args.output_dtype;
    const bool is_row_major = input.layout() == Layout::ROW_MAJOR;

    const auto cb_data_format_input = datatype_to_dataformat_converter(input.dtype());
    const auto single_tile_size_input = tt::tile_size(cb_data_format_input);
    const auto cb_data_format_output = datatype_to_dataformat_converter(output.dtype());
    const auto single_tile_size_output = tt::tile_size(cb_data_format_output);

    const auto* device = input.device();
    auto* src_buffer = input.buffer();
    auto* dst_buffer = output.buffer();

    const uint32_t num_pages = src_buffer->num_pages();
    const uint32_t input_page_size = is_row_major ? src_buffer->page_size() : single_tile_size_input;
    const uint32_t output_page_size = is_row_major ? dst_buffer->page_size() : single_tile_size_output;

    // ── Core selection ───────────────────────────────────────────────────
    CoreRangeSet all_cores(std::vector<CoreRange>{});
    std::vector<CoreCoord> cores_vec;
    uint32_t num_cores = 0;
    std::vector<uint32_t> items_per_core;

    if (args.sub_core_grids.has_value()) {
        const auto& sub_core_grids = args.sub_core_grids.value();
        uint32_t ntiles = input.physical_volume() / TILE_HW;
        num_cores = sub_core_grids.num_cores();
        TT_FATAL(num_cores != 0, "number of cores cannot be 0");

        for (uint32_t c = num_cores; c >= 1; c--) {
            if (ntiles % c == 0) {
                num_cores = c;
                break;
            }
        }
        TT_FATAL(ntiles % num_cores == 0, "{} tiles not uniformly split across {} cores", ntiles, num_cores);

        cores_vec = corerange_to_cores(sub_core_grids, num_cores, true);
        all_cores = (num_cores == 1)
                        ? CoreRangeSet(CoreRange(cores_vec[0]))
                        : num_cores_to_corerangeset_in_subcoregrids(cores_vec[0], num_cores, sub_core_grids, true);

        const uint32_t items = ntiles / num_cores;
        items_per_core.assign(num_cores, items);
    } else {
        const auto grid_size = device->compute_with_storage_grid_size();
        auto [nc, ac, cg1, cg2, n1, n2] = tt::tt_metal::split_work_to_cores(grid_size, num_pages, is_row_major);
        num_cores = nc;
        all_cores = ac;
        cores_vec = corerange_to_cores(all_cores, std::nullopt, is_row_major);

        items_per_core.reserve(num_cores);
        for (const auto& core : cores_vec) {
            if (cg1.contains(core)) {
                items_per_core.push_back(n1);
            } else if (cg2.contains(core)) {
                items_per_core.push_back(n2);
            }
        }
    }

    ProgramDescriptor program_descriptor;

    // ── Circular Buffers ─────────────────────────────────────────────────
    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = input_page_size * 2;
        cb_desc.core_ranges = all_cores;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format_input,
            .page_size = input_page_size});
        program_descriptor.cbs.push_back(std::move(cb_desc));
    }

    constexpr uint32_t output_cb_index = tt::CBIndex::c_2;
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = output_page_size * 2;
        cb_desc.core_ranges = all_cores;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = cb_data_format_output,
            .page_size = output_page_size});
        program_descriptor.cbs.push_back(std::move(cb_desc));
    }

    // ── Reader kernel ────────────────────────────────────────────────────
    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);

    KernelDescriptor::RuntimeArgs reader_rt_args;
    uint32_t start_id = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto count = items_per_core[i];
        reader_rt_args.emplace_back(cores_vec[i], std::vector<uint32_t>{src_buffer->address(), count, start_id});
        start_id += count;
    }

    {
        KernelDescriptor kernel_desc;
        kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp";
        kernel_desc.core_ranges = all_cores;
        kernel_desc.compile_time_args = reader_ct_args;
        kernel_desc.runtime_args = std::move(reader_rt_args);
        kernel_desc.config = ReaderConfigDescriptor{};
        program_descriptor.kernels.push_back(std::move(kernel_desc));
    }

    // ── Writer kernel ────────────────────────────────────────────────────
    std::vector<uint32_t> writer_ct_args = {output_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

    KernelDescriptor::RuntimeArgs writer_rt_args;
    start_id = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto count = items_per_core[i];
        writer_rt_args.emplace_back(cores_vec[i], std::vector<uint32_t>{dst_buffer->address(), count, start_id});
        start_id += count;
    }

    {
        KernelDescriptor kernel_desc;
        kernel_desc.kernel_source =
            "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp";
        kernel_desc.core_ranges = all_cores;
        kernel_desc.compile_time_args = writer_ct_args;
        kernel_desc.runtime_args = std::move(writer_rt_args);
        kernel_desc.config = WriterConfigDescriptor{};
        program_descriptor.kernels.push_back(std::move(kernel_desc));
    }

    // ── Compute kernels ──────────────────────────────────────────────────
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    const auto unary_defines = make_typecast_compute_defines_desc(input_dtype, output_dtype);

    // Group cores by items_per_core count for efficiency
    std::map<uint32_t, CoreRangeSet> count_to_cores;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto count = items_per_core[i];
        if (count_to_cores.find(count) == count_to_cores.end()) {
            count_to_cores[count] = CoreRangeSet(CoreRange(cores_vec[i]));
        } else {
            count_to_cores[count] = count_to_cores[count].merge(CoreRangeSet(CoreRange(cores_vec[i])));
        }
    }

    for (const auto& [count, core_range] : count_to_cores) {
        KernelDescriptor kernel_desc;
        kernel_desc.kernel_source = TYPECAST_COMPUTE_KERNEL_PATH;
        kernel_desc.core_ranges = core_range;
        kernel_desc.compile_time_args = {count, 1, src0_cb_index, output_cb_index};
        kernel_desc.defines = unary_defines;
        kernel_desc.config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = args.fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = args.bfp8_pack_precise,
            .math_approx_mode = false};
        program_descriptor.kernels.push_back(std::move(kernel_desc));
    }

    return program_descriptor;
}

TypecastProgramFactory::cached_program_t TypecastProgramFactory::create(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    using namespace tt::tt_metal;

    auto descriptor = create_descriptor(args, tensor_args, output);
    Program program{descriptor};

    // Kernel handles are assigned sequentially: reader=0, writer=1
    constexpr KernelHandle reader_kernel_id = 0;
    constexpr KernelHandle writer_kernel_id = 1;

    // Reconstruct cores_vec for override_runtime_arguments
    const auto& input = tensor_args.input;
    const bool is_row_major = input.layout() == Layout::ROW_MAJOR;
    const uint32_t num_pages = input.buffer()->num_pages();
    std::vector<CoreCoord> cores_vec;

    if (args.sub_core_grids.has_value()) {
        uint32_t ntiles = input.physical_volume() / TILE_HW;
        uint32_t num_cores = args.sub_core_grids.value().num_cores();
        for (uint32_t c = num_cores; c >= 1; c--) {
            if (ntiles % c == 0) {
                num_cores = c;
                break;
            }
        }
        cores_vec = corerange_to_cores(args.sub_core_grids.value(), num_cores, true);
    } else {
        const auto grid_size = input.device()->compute_with_storage_grid_size();
        auto [nc, ac, cg1, cg2, n1, n2] = split_work_to_cores(grid_size, num_pages, is_row_major);
        cores_vec = corerange_to_cores(ac, std::nullopt, is_row_major);
    }

    return cached_program_t{std::move(program), {reader_kernel_id, writer_kernel_id, std::move(cores_vec)}};
}

void TypecastProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TypecastParams& /*operation_attributes*/,
    const TypecastInputs& tensor_args,
    Tensor& output) {
    auto& reader_kernel_id = cached_program.shared_variables.typecast_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.typecast_writer_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;

    auto& program = cached_program.program;
    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = output.buffer();

    auto& reader_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
    auto& writer_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
    for (const auto& core : cores) {
        reader_args_by_core[core.x][core.y][0] = src_buffer->address();
        writer_args_by_core[core.x][core.y][0] = dst_buffer->address();
    }
}

}  // namespace ttnn::prim
