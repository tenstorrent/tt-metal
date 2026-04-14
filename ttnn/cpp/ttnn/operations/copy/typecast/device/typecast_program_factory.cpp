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

TypecastProgramFactory::cached_program_t TypecastProgramFactory::create(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto input_dtype = args.input_dtype;
    const auto output_dtype = args.output_dtype;
    const bool is_row_major = input.layout() == Layout::ROW_MAJOR;

    Program program{};

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

    // Per-core item counts: cores_vec[i] processes items_per_core[i] items
    std::vector<uint32_t> items_per_core;

    if (args.sub_core_grids.has_value()) {
        // Sub-core grid mode: distribute tiles uniformly across specified cores
        const auto& sub_core_grids = args.sub_core_grids.value();
        uint32_t ntiles = input.physical_volume() / TILE_HW;
        num_cores = sub_core_grids.num_cores();
        TT_FATAL(num_cores != 0, "number of cores cannot be 0");

        // Find largest divisor of ntiles that is <= num_cores
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
        // Standard mode: distribute pages across full grid
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

    // ── Circular Buffers ─────────────────────────────────────────────────
    constexpr uint32_t src0_cb_index = tt::CBIndex::c_0;
    create_cb(src0_cb_index, program, all_cores, input_page_size, 2, cb_data_format_input);

    constexpr uint32_t output_cb_index = tt::CBIndex::c_2;
    create_cb(output_cb_index, program, all_cores, output_page_size, 2, cb_data_format_output);

    // ── Reader/Writer kernels ────────────────────────────────────────────
    std::vector<uint32_t> reader_ct_args;
    TensorAccessorArgs(*src_buffer).append_to(reader_ct_args);
    std::vector<uint32_t> writer_ct_args = {(uint32_t)output_cb_index};
    TensorAccessorArgs(*dst_buffer).append_to(writer_ct_args);

    const auto reader_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/reader_unary_interleaved_start_id.cpp",
        all_cores,
        ReaderDataMovementConfig(reader_ct_args));

    const auto writer_kernel_id = CreateKernel(
        program,
        "ttnn/cpp/ttnn/operations/eltwise/unary/device/kernels/dataflow/writer_unary_interleaved_start_id.cpp",
        all_cores,
        WriterDataMovementConfig(writer_ct_args));

    // ── Compute kernel ───────────────────────────────────────────────────
    std::vector<UnpackToDestMode> unpack_to_dest_mode(NUM_CIRCULAR_BUFFERS, UnpackToDestMode::Default);
    if (args.preserve_fp32_precision) {
        unpack_to_dest_mode[src0_cb_index] = UnpackToDestMode::UnpackToDestFp32;
    }

    std::map<std::string, std::string> unary_defines;
    unary_defines["TYPECAST_LLK_INIT"] = fmt::format(
        "typecast_tile_init<{0}u, {1}u>",
        static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
        static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));
    unary_defines["TYPECAST_LLK"] = fmt::format(
        "typecast_tile<{0}u, {1}u>",
        static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
        static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype)));

    const auto compute_path = "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp";

    // Create compute kernels — group by items_per_core count for efficiency
    // For uniform distribution (sub_core_grids), one kernel covers all cores.
    // For split_work_to_cores, up to two groups with different tile counts.
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
        std::vector<uint32_t> compute_args = {count, 1, src0_cb_index, output_cb_index};
        CreateKernel(
            program,
            compute_path,
            core_range,
            ComputeConfig{
                .math_fidelity = MathFidelity::HiFi4,
                .fp32_dest_acc_en = args.fp32_dest_acc_en,
                .unpack_to_dest_mode = unpack_to_dest_mode,
                .bfp8_pack_precise = args.bfp8_pack_precise,
                .math_approx_mode = false,
                .compile_args = compute_args,
                .defines = unary_defines});
    }

    // ── Runtime args ─────────────────────────────────────────────────────
    uint32_t start_id = 0;
    for (uint32_t i = 0; i < num_cores; ++i) {
        const auto& core = cores_vec[i];
        const auto count = items_per_core[i];

        SetRuntimeArgs(program, reader_kernel_id, core, {src_buffer->address(), count, start_id});
        SetRuntimeArgs(program, writer_kernel_id, core, {dst_buffer->address(), count, start_id});
        start_id += count;
    }

    return cached_program_t{std::move(program), {reader_kernel_id, writer_kernel_id, cores_vec}};
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
