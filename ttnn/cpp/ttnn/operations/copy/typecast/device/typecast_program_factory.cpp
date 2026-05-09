// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_program_factory.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
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
    // For RM: pad CB pages to 32-element AND DRAM-buffer alignment so (a) the unpacker does not cross
    // page boundaries and (b) double-buffered CB pages share the same residue (mod buffer alignment)
    // as their DRAM pages — required by the NOC: src_addr & (align-1) == dst_addr & (align-1). On
    // Blackhole the DRAM alignment is 64B; without (b) an 8-bit input with a 32-element row yields a
    // 32B page → mis-aligned second page → ttsim NOC alignment crash. (#41977)
    const uint32_t padded_input_page_size =
        is_row_major ? tt::align(tt::align(input_page_size, 32u * input.element_size()), src_buffer->alignment())
                     : input_page_size;
    const uint32_t padded_output_page_size =
        is_row_major ? tt::align(tt::align(output_page_size, 32u * output.element_size()), dst_buffer->alignment())
                     : output_page_size;

    // ── Core selection ───────────────────────────────────────────────────
    // Use a fixed grid (full compute grid, or sub_core_grids if provided) and let the
    // per-core work count be a runtime arg.  This makes the program structure
    // volume-independent: a single compiled program serves any tensor volume, so the
    // program-cache hash does not need to include the volume (see compute_program_hash).
    CoreRangeSet all_cores(std::vector<CoreRange>{});
    std::vector<CoreCoord> cores_vec;
    uint32_t num_cores = 0;
    std::vector<uint32_t> items_per_core;

    if (args.sub_core_grids.has_value()) {
        const auto& sub_core_grids = args.sub_core_grids.value();
        num_cores = sub_core_grids.num_cores();
        TT_FATAL(num_cores != 0, "number of cores cannot be 0");
        all_cores = sub_core_grids;
        cores_vec = corerange_to_cores(sub_core_grids, num_cores, true);
    } else {
        const auto grid_size = device->compute_with_storage_grid_size();
        all_cores = CoreRangeSet(CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1}));
        num_cores = grid_size.x * grid_size.y;
        cores_vec = corerange_to_cores(all_cores, std::nullopt, is_row_major);
    }

    // Distribute num_pages across the fixed core set. The first `extra` cores get one
    // additional page; the rest get the floor share. Cores that end up with zero pages
    // (when num_pages < num_cores) early-out in the kernel.
    items_per_core.resize(num_cores);
    {
        const uint32_t base = num_pages / num_cores;
        const uint32_t extra = num_pages % num_cores;
        for (uint32_t i = 0; i < num_cores; ++i) {
            items_per_core[i] = base + (i < extra ? 1u : 0u);
        }
    }

    ProgramDescriptor program_descriptor;

    // ── Circular Buffers ─────────────────────────────────────────────────
    // For RM: use padded page sizes so the unpacker stays within CB page boundaries.
    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = padded_input_page_size * 2;
        cb_desc.core_ranges = all_cores;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(src0_cb_index),
            .data_format = cb_data_format_input,
            .page_size = padded_input_page_size});
        program_descriptor.cbs.push_back(std::move(cb_desc));
    }

    constexpr uint32_t output_cb_index = tt::CBIndex::c_2;
    {
        CBDescriptor cb_desc;
        cb_desc.total_size = padded_output_page_size * 2;
        cb_desc.core_ranges = all_cores;
        cb_desc.format_descriptors.push_back(CBFormatDescriptor{
            .buffer_index = static_cast<uint8_t>(output_cb_index),
            .data_format = cb_data_format_output,
            .page_size = padded_output_page_size});
        program_descriptor.cbs.push_back(std::move(cb_desc));
    }

    // ── Reader kernel ────────────────────────────────────────────────────
    KernelDescriptor::RuntimeArgs reader_rt_args;
    uint32_t start_id = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto count = items_per_core[i];
        reader_rt_args.emplace_back(cores_vec[i], std::vector<uint32_t>{src_buffer->address(), count, start_id});
        start_id += count;
    }

    {
        KernelDescriptor kernel_desc;
        if (is_row_major) {
            // RM: use padded reader that zero-fills extra bytes in the CB page.
            kernel_desc.kernel_source =
                "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/"
                "reader_typecast_rm_interleaved.cpp";
            kernel_desc.compile_time_args = {src0_cb_index, input_page_size, padded_input_page_size};
            TensorAccessorArgs(*src_buffer).append_to(kernel_desc.compile_time_args);
        } else {
            std::vector<uint32_t> reader_ct_args;
            TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);
            kernel_desc.kernel_source =
                "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                "reader_unary_interleaved_start_id.cpp";
            kernel_desc.compile_time_args = std::move(reader_ct_args);
        }
        kernel_desc.core_ranges = all_cores;
        kernel_desc.runtime_args = std::move(reader_rt_args);
        kernel_desc.config = ReaderConfigDescriptor{};
        program_descriptor.kernels.push_back(std::move(kernel_desc));
    }

    // ── Writer kernel ────────────────────────────────────────────────────
    KernelDescriptor::RuntimeArgs writer_rt_args;
    start_id = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto count = items_per_core[i];
        writer_rt_args.emplace_back(cores_vec[i], std::vector<uint32_t>{dst_buffer->address(), count, start_id});
        start_id += count;
    }

    {
        KernelDescriptor kernel_desc;
        if (is_row_major) {
            // RM: use padded writer that writes only actual (non-padded) bytes to DRAM.
            kernel_desc.kernel_source =
                "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/"
                "writer_typecast_rm_interleaved.cpp";
            kernel_desc.compile_time_args = {output_cb_index, output_page_size};
            TensorAccessorArgs(*dst_buffer).append_to(kernel_desc.compile_time_args);
        } else {
            std::vector<uint32_t> writer_ct_args = {output_cb_index};
            TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);
            kernel_desc.kernel_source =
                "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/"
                "writer_unary_interleaved_start_id.cpp";
            kernel_desc.compile_time_args = std::move(writer_ct_args);
        }
        kernel_desc.core_ranges = all_cores;
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

    // Single compute kernel binary across all cores; per-core work count is a runtime
    // arg (passed alongside the kernel descriptor below).
    {
        KernelDescriptor kernel_desc;
        kernel_desc.kernel_source = TYPECAST_COMPUTE_KERNEL_PATH;
        kernel_desc.core_ranges = all_cores;
        kernel_desc.compile_time_args = {1u /*per_core_block_dim*/, src0_cb_index, output_cb_index};
        kernel_desc.defines = unary_defines;
        kernel_desc.config = ComputeConfigDescriptor{
            .math_fidelity = MathFidelity::HiFi4,
            .fp32_dest_acc_en = args.fp32_dest_acc_en,
            .unpack_to_dest_mode = unpack_to_dest_mode,
            .bfp8_pack_precise = args.bfp8_pack_precise,
            .math_approx_mode = false};
        for (uint32_t i = 0; i < num_cores; ++i) {
            kernel_desc.runtime_args.emplace_back(cores_vec[i], std::vector<uint32_t>{items_per_core[i]});
        }
        program_descriptor.kernels.push_back(std::move(kernel_desc));
    }

    return program_descriptor;
}

TypecastProgramFactory::cached_program_t TypecastProgramFactory::create(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    using namespace tt::tt_metal;

    auto descriptor = create_descriptor(args, tensor_args, output);
    Program program{descriptor};

    // Kernel handles are assigned in descriptor-push order: reader=0, writer=1, compute=2.
    constexpr KernelHandle reader_kernel_id = 0;
    constexpr KernelHandle writer_kernel_id = 1;
    constexpr KernelHandle compute_kernel_id = 2;

    // Mirror the fixed-grid core selection used in create_descriptor — same set of
    // cores as the program above, in the same order, so per-core runtime-arg updates
    // in override_runtime_arguments stay in sync with what was registered.
    const auto& input = tensor_args.input;
    const bool is_row_major = input.layout() == Layout::ROW_MAJOR;
    std::vector<CoreCoord> cores_vec;
    if (args.sub_core_grids.has_value()) {
        const auto& sub = args.sub_core_grids.value();
        cores_vec = corerange_to_cores(sub, sub.num_cores(), true);
    } else {
        const auto grid_size = input.device()->compute_with_storage_grid_size();
        const CoreRangeSet all_cores(CoreRange({0, 0}, {grid_size.x - 1, grid_size.y - 1}));
        cores_vec = corerange_to_cores(all_cores, std::nullopt, is_row_major);
    }

    return cached_program_t{
        std::move(program), {reader_kernel_id, writer_kernel_id, compute_kernel_id, std::move(cores_vec)}};
}

void TypecastProgramFactory::override_runtime_arguments(
    cached_program_t& cached_program,
    const TypecastParams& /*operation_attributes*/,
    const TypecastInputs& tensor_args,
    Tensor& output) {
    auto& reader_kernel_id = cached_program.shared_variables.typecast_reader_kernel_id;
    auto& writer_kernel_id = cached_program.shared_variables.typecast_writer_kernel_id;
    auto& compute_kernel_id = cached_program.shared_variables.typecast_compute_kernel_id;
    const auto& cores = cached_program.shared_variables.cores;

    auto& program = cached_program.program;
    auto* src_buffer = tensor_args.input.buffer();
    auto* dst_buffer = output.buffer();

    // Cache hits with a different tensor volume must rebalance the per-core work
    // (count, start_id) since hash() intentionally drops volume — see compute_program_hash.
    const uint32_t num_pages = src_buffer->num_pages();
    const uint32_t num_cores = static_cast<uint32_t>(cores.size());
    const uint32_t base = num_pages / num_cores;
    const uint32_t extra = num_pages % num_cores;

    auto& reader_args_by_core = GetRuntimeArgs(program, reader_kernel_id);
    auto& writer_args_by_core = GetRuntimeArgs(program, writer_kernel_id);
    auto& compute_args_by_core = GetRuntimeArgs(program, compute_kernel_id);

    uint32_t start_id = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores[i];
        const uint32_t count = base + (i < extra ? 1u : 0u);
        auto& reader_args = reader_args_by_core[core.x][core.y];
        reader_args[0] = src_buffer->address();
        reader_args[1] = count;
        reader_args[2] = start_id;
        auto& writer_args = writer_args_by_core[core.x][core.y];
        writer_args[0] = dst_buffer->address();
        writer_args[1] = count;
        writer_args[2] = start_id;
        compute_args_by_core[core.x][core.y][0] = count;
        start_id += count;
    }
}

}  // namespace ttnn::prim
