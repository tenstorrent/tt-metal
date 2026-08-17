// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_rm_chunked_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt_stl/span.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

using namespace tt::constants;
using namespace tt::tt_metal::experimental;

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

ttnn::device_operation::ProgramArtifacts TypecastRowMajorChunkedProgramFactory::create_program_artifacts(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Tensor& input = tensor_args.input;
    const DataType& input_dtype = args.input_dtype;
    const DataType& output_dtype = args.output_dtype;

    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "This factory is only for ROW_MAJOR layout");

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

    // ---- Resource names ----
    const DFBSpecName IN_DFB{"in"};    // legacy CBIndex::c_0
    const DFBSpecName OUT_DFB{"out"};  // legacy CBIndex::c_2
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_GROUP_1{"compute_group_1"};
    const KernelSpecName COMPUTE_GROUP_2{"compute_group_2"};

    constexpr uint32_t num_input_pages = 2;   // Always use double buffering
    constexpr uint32_t num_output_pages = 2;  // Always use double buffering

    // Additionally align DFB entry sizes to the source/destination buffer alignment so that the
    // double-buffered DFB entries share the same residue (mod buffer alignment) as their DRAM pages.
    // This is required by the NOC: DRAM->L1 reads enforce (src_addr & alignment-1) ==
    // (dst_addr & alignment-1).  On Blackhole the DRAM alignment is 64B; without this an
    // 8-bit input with a 32-element padded chunk yields a 32B page, leaving the second
    // double-buffered page mis-aligned and causing ttsim NOC alignment crashes
    // (see test_typecast_row_major_vs_tile_layout[UINT8_TO_BFLOAT16-8x2x64x32]).
    const uint32_t input_cb_page_size_bytes = tt::align(padded_input_full_chunk_size_bytes, src_buffer->alignment());
    const uint32_t output_cb_page_size_bytes = tt::align(padded_output_full_chunk_size_bytes, dst_buffer->alignment());

    const DataflowBufferSpec in_dfb{
        .unique_id = IN_DFB,
        .entry_size = input_cb_page_size_bytes,
        .num_entries = num_input_pages,
        .data_format_metadata = cb_data_format_input,
    };

    const DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = output_cb_page_size_bytes,
        .num_entries = num_output_pages,
        .data_format_metadata = cb_data_format_output,
    };

    const TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec()};
    const TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    const KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_typecast_rm_chunked.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {
                {"full_chunks_per_row", full_chunks_per_row},
                {"partial_chunks_per_row", partial_chunks_per_row},           // 0 or 1
                {"full_chunk_size_bytes", input_full_chunk_size_bytes},       // DRAM read size
                {"partial_chunk_size_bytes", input_partial_chunk_size_bytes}  // DRAM read size
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_rows", "start_row_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    const KernelSpec writer{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/writer_typecast_rm_chunked.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args =
            {
                {"full_chunks_per_row", full_chunks_per_row},
                {"partial_chunks_per_row", partial_chunks_per_row},            // 0 or 1
                {"full_chunk_size_bytes", output_full_chunk_size_bytes},       // DRAM write size
                {"partial_chunk_size_bytes", output_partial_chunk_size_bytes}  // DRAM write size
            },
        .runtime_arg_schema = {.runtime_arg_names = {"num_rows", "start_row_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // Create compute kernels - compute per_core_block_cnt as total chunks (full + partial) per core
    const uint32_t chunks_per_row_total = full_chunks_per_row + partial_chunks_per_row;

    // Legacy set unpack_to_dest_mode[input_cb] = UnpackToDestFp32 when preserve_fp32_precision and
    // left every other CB at Default; the named equivalent is an UnpackToDest entry for the input DFB.
    ComputeUnpackModes unpack_modes;
    if (args.preserve_fp32_precision) {
        unpack_modes.emplace(IN_DFB, tt::tt_metal::UnpackMode::UnpackToDest);
    } else if (args.fp32_dest_acc_en && cb_data_format_input == tt::DataFormat::Float32) {
        // Metal 2.0 requires an explicit entry for a consumed Float32 DFB under a 32-bit Dest, where
        // legacy silently defaulted. UnpackToSrc is that legacy default.
        unpack_modes.emplace(IN_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
    }

    KernelSpec::CompilerOptions::Defines unary_defines;
    unary_defines.emplace(
        "TYPECAST_LLK_INIT",
        fmt::format(
            "typecast_tile_init<{0}u, {1}u>",
            static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
            static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype))));
    unary_defines.emplace(
        "TYPECAST_LLK",
        fmt::format(
            "typecast_tile<{0}u, {1}u>",
            static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
            static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype))));

    const char* const path = "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp";

    // One KernelSpec per legacy compute KernelDescriptor: the per-group chunk count stays a
    // compile-time arg, so the work-split multiplicity is preserved.
    // Field values carried over from the legacy ComputeConfigDescriptor: math_fidelity=HiFi4,
    // fp32_dest_acc_en, bfp8_pack_precise, math_approx_mode=false. dst_full_sync_en was left at its
    // legacy default (false) = double_buffer_dest true, which is also the Metal 2.0 default.
    const auto make_compute = [&](const KernelSpecName& id, uint32_t per_core_block_cnt) {
        return KernelSpec{
            .unique_id = id,
            .source = path,
            .compiler_options = {.defines = unary_defines},
            .dfb_bindings =
                {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args =
                {{"per_core_block_cnt", per_core_block_cnt},  // rows * total_chunks_per_row
                 {"per_core_block_dim", 1u}},
            .hw_config = ComputeHardwareConfig{ComputeGen1Config{
                .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
                .sfpu_precision_mode = tt::tt_metal::Precision::Precise,  // legacy math_approx_mode = false
                .bfp_pack_precision_mode =
                    args.bfp8_pack_precise ? tt::tt_metal::Precision::Precise : tt::tt_metal::Precision::Approximate,
                .enable_32_bit_dest = args.fp32_dest_acc_en,
                .unpack_modes = unpack_modes,
            }},
        };
    };

    Group<KernelSpec> kernels = {reader, writer};
    Group<WorkUnitSpec> work_units;
    if (!core_group_1.ranges().empty()) {
        kernels.push_back(make_compute(COMPUTE_GROUP_1, num_rows_per_core_group_1 * chunks_per_row_total));
        work_units.push_back(WorkUnitSpec{
            .name = "typecast_rm_group_1", .kernels = {READER, WRITER, COMPUTE_GROUP_1}, .target_nodes = core_group_1});
    }
    if (!core_group_2.ranges().empty()) {
        kernels.push_back(make_compute(COMPUTE_GROUP_2, num_rows_per_core_group_2 * chunks_per_row_total));
        work_units.push_back(WorkUnitSpec{
            .name = "typecast_rm_group_2", .kernels = {READER, WRITER, COMPUTE_GROUP_2}, .target_nodes = core_group_2});
    }

    // Assign runtime args to cores (distributing rows)
    auto cores_vec = corerange_to_cores(all_cores, std::nullopt, true);
    uint32_t row_idx = 0;

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    for (const auto& core : cores_vec) {
        bool is_group_1 = core_group_1.contains(core);
        uint32_t num_rows_for_core = is_group_1 ? num_rows_per_core_group_1 : num_rows_per_core_group_2;
        uint32_t start_row_id = row_idx;

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"num_rows", num_rows_for_core}, {"start_row_id", start_row_id}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_rows", num_rows_for_core}, {"start_row_id", start_row_id}});

        row_idx += num_rows_for_core;
    }

    ProgramSpec spec{
        .name = "typecast_rm_chunked",
        .kernels = std::move(kernels),
        .dataflow_buffers = {in_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = std::move(work_units),
    };

    // The compute kernels have no runtime args, so they need no KernelRunArgs entry.
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {{INPUT, input.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
