// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_rm_chunked_program_factory.hpp"

#include <filesystem>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt_stl/span.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace {

// File-scope names, unique across the typecast device/ sibling .cpp files (unity-build safety).
constexpr const char* RM_CHUNKED_READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_typecast_rm_chunked.cpp";
constexpr const char* RM_CHUNKED_WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/writer_typecast_rm_chunked.cpp";
constexpr const char* RM_CHUNKED_COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast_m2.cpp";

struct RmChunkSizeConfig {
    uint32_t input_full_chunk_size_bytes;          // actual bytes read from DRAM per full chunk
    uint32_t output_full_chunk_size_bytes;         // actual bytes written to DRAM per full chunk
    uint32_t input_partial_chunk_size_bytes;       // actual bytes read from DRAM for partial chunk
    uint32_t output_partial_chunk_size_bytes;      // actual bytes written to DRAM for partial chunk
    uint32_t padded_input_full_chunk_size_bytes;   // CB page size (rounded up to 32-element boundary)
    uint32_t padded_output_full_chunk_size_bytes;  // CB page size (rounded up to 32-element boundary)
    uint32_t full_chunks_per_row;
    uint32_t partial_chunks_per_row;
};

RmChunkSizeConfig calculate_chunk_config(
    uint32_t row_width_elements, uint32_t input_element_size, uint32_t output_element_size) {
    constexpr uint32_t max_elements_per_chunk = 1024;
    const uint32_t elements_per_full_chunk = std::min(max_elements_per_chunk, row_width_elements);

    // Actual chunk sizes in bytes (for DRAM reads/writes)
    const uint32_t input_full_chunk_size_bytes = elements_per_full_chunk * input_element_size;
    const uint32_t output_full_chunk_size_bytes = elements_per_full_chunk * output_element_size;

    // CB page sizes: round up to next multiple of 32 elements for the unpacker.
    const uint32_t padded_full_elements = tt::align(elements_per_full_chunk, 32u);
    const uint32_t padded_input_full_chunk_size_bytes = padded_full_elements * input_element_size;
    const uint32_t padded_output_full_chunk_size_bytes = padded_full_elements * output_element_size;

    // Calculate how many chunks per row
    const uint32_t full_chunks_per_row = row_width_elements / elements_per_full_chunk;
    const uint32_t remainder = row_width_elements % elements_per_full_chunk;
    const uint32_t partial_chunks_per_row = (remainder > 0) ? 1 : 0;
    const uint32_t input_partial_chunk_size_bytes = remainder * input_element_size;
    const uint32_t output_partial_chunk_size_bytes = remainder * output_element_size;

    return RmChunkSizeConfig{
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

// TYPECAST_LLK / TYPECAST_LLK_INIT defines (was a std::map copied onto each compute KernelDescriptor).
m2::KernelSpec::CompilerOptions::Defines rm_typecast_compute_defines(DataType input_dtype, DataType output_dtype) {
    m2::KernelSpec::CompilerOptions::Defines defines;
    defines.emplace(
        "TYPECAST_LLK_INIT",
        fmt::format(
            "typecast_tile_init<{0}u, {1}u>",
            static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
            static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype))));
    defines.emplace(
        "TYPECAST_LLK",
        fmt::format(
            "typecast_tile<{0}u, {1}u>",
            static_cast<uint32_t>(datatype_to_dataformat_converter(input_dtype)),
            static_cast<uint32_t>(datatype_to_dataformat_converter(output_dtype))));
    return defines;
}

m2::ComputeHardwareConfig rm_typecast_compute_hw_config(
    const TypecastParams& args, const m2::DFBSpecName& input_dfb, tt::DataFormat input_format) {
    m2::ComputeHardwareConfig hw{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = args.fp32_dest_acc_en,
        .bfp8_pack_precise = args.bfp8_pack_precise,
        .math_approx_mode = false,
    };
    // UnpackToDestFp32 is only accepted (and only meaningful) when the input DFB carries Float32 data;
    // legacy applied it unconditionally under preserve_fp32_precision, which was a tolerated no-op for
    // non-fp32 inputs. Guard it to match the effective legacy behavior.
    if (args.preserve_fp32_precision && input_format == tt::DataFormat::Float32) {
        hw.unpack_to_dest_mode.emplace(input_dfb, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32);
    }
    return hw;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts TypecastRowMajorChunkedProgramFactory::create_program_artifacts(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    const Tensor& input = tensor_args.input;
    const DataType input_dtype = args.input_dtype;
    const DataType output_dtype = args.output_dtype;

    TT_FATAL(input.layout() == Layout::ROW_MAJOR, "This factory is only for ROW_MAJOR layout");

    const tt::DataFormat cb_data_format_input = datatype_to_dataformat_converter(input.dtype());
    const uint32_t input_element_size = tt::datum_size(cb_data_format_input);
    const tt::DataFormat cb_data_format_output = datatype_to_dataformat_converter(output.dtype());
    const uint32_t output_element_size = tt::datum_size(cb_data_format_output);

    const auto* device = input.device();

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    // Get row information
    const auto& padded_shape = input.padded_shape();
    const uint32_t row_width_elements = padded_shape[padded_shape.rank() - 1];
    const uint32_t num_rows = src_buffer->num_pages();

    // Calculate chunk configuration
    const RmChunkSizeConfig chunk_config =
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

    const m2::DFBSpecName input_dfb{"input_cb"};
    const m2::DFBSpecName output_dfb{"output_cb"};

    m2::ProgramSpec spec;
    spec.name = "typecast_rm_chunked";

    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = input_dfb,
            .entry_size = input_cb_page_size_bytes,
            .num_entries = num_input_pages,
            .data_format_metadata = cb_data_format_input,
        },
        m2::DataflowBufferSpec{
            .unique_id = output_dfb,
            .entry_size = output_cb_page_size_bytes,
            .num_entries = num_output_pages,
            .data_format_metadata = cb_data_format_output,
        }};

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = input.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()}};

    // Structural chunk scalars are NAMED compile-time args (formerly positional CTAs 1..4 + the CB id).
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{RM_CHUNKED_READER_KERNEL_PATH},
        .dfb_bindings = {m2::ProducerOf(input_dfb, "input_cb")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "src_args"}},
        .compile_time_args =
            {{"full_chunks_per_row", full_chunks_per_row},
             {"partial_chunks_per_row", partial_chunks_per_row},
             {"full_chunk_size_bytes", input_full_chunk_size_bytes},
             {"partial_chunk_size_bytes", input_partial_chunk_size_bytes}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_rows", "start_row_id"}},
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt::tt_metal::NOC::RISCV_1_default}},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{RM_CHUNKED_WRITER_KERNEL_PATH},
        .dfb_bindings = {m2::ConsumerOf(output_dfb, "output_cb")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "dst_args"}},
        .compile_time_args =
            {{"full_chunks_per_row", full_chunks_per_row},
             {"partial_chunks_per_row", partial_chunks_per_row},
             {"full_chunk_size_bytes", output_full_chunk_size_bytes},
             {"partial_chunk_size_bytes", output_partial_chunk_size_bytes}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_rows", "start_row_id"}},
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
    };

    // Compute per_core_block_cnt as total chunks (full + partial) per core; differs between core groups,
    // so it is a per-kernel compile-time arg.
    const uint32_t chunks_per_row_total = full_chunks_per_row + partial_chunks_per_row;

    auto make_compute = [&](const m2::KernelSpecName& name, uint32_t num_rows_per_core) {
        return m2::KernelSpec{
            .unique_id = name,
            .source = std::filesystem::path{RM_CHUNKED_COMPUTE_KERNEL_PATH},
            .compiler_options = {.defines = rm_typecast_compute_defines(input_dtype, output_dtype)},
            .dfb_bindings = {m2::ConsumerOf(input_dfb, "input_cb"), m2::ProducerOf(output_dfb, "output_cb")},
            .compile_time_args =
                {{"per_core_block_cnt", num_rows_per_core * chunks_per_row_total}, {"per_core_block_dim", 1u}},
            .hw_config = rm_typecast_compute_hw_config(args, input_dfb, cb_data_format_input),
        };
    };

    spec.kernels = {std::move(reader), std::move(writer)};
    spec.work_units = {};

    // Local DFBs (input_cb, output_cb) require their producer AND consumer kernels in the SAME
    // WorkUnitSpec on every node where the DFB lives. reader/writer (producer of input_cb, consumer of
    // output_cb) must therefore be co-located with the compute kernel in each core group's WorkUnitSpec.
    const bool has_group_1 = !core_group_1.ranges().empty();
    if (has_group_1) {
        spec.kernels.push_back(make_compute(m2::KernelSpecName{"compute_1"}, num_rows_per_core_group_1));
        spec.work_units.push_back(m2::WorkUnitSpec{
            .name = "typecast_rm_compute_1",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute_1"}},
            .target_nodes = core_group_1});
    }
    const bool has_group_2 = !core_group_2.ranges().empty();
    if (has_group_2) {
        spec.kernels.push_back(make_compute(m2::KernelSpecName{"compute_2"}, num_rows_per_core_group_2));
        spec.work_units.push_back(m2::WorkUnitSpec{
            .name = "typecast_rm_compute_2",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute_2"}},
            .target_nodes = core_group_2});
    }

    // Assign runtime args to cores (distributing rows)
    auto cores_vec = corerange_to_cores(all_cores, std::nullopt, true);

    m2::ProgramRunArgs run_args;
    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};

    uint32_t row_idx = 0;
    for (const auto& core : cores_vec) {
        bool is_group_1 = core_group_1.contains(core);
        uint32_t num_rows_for_core = is_group_1 ? num_rows_per_core_group_1 : num_rows_per_core_group_2;
        uint32_t start_row_id = row_idx;

        reader_args.runtime_arg_values.push_back(
            {core, {{"num_rows", num_rows_for_core}, {"start_row_id", start_row_id}}});
        writer_args.runtime_arg_values.push_back(
            {core, {{"num_rows", num_rows_for_core}, {"start_row_id", start_row_id}}});

        row_idx += num_rows_for_core;
    }
    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));

    run_args.tensor_args.emplace(
        m2::TensorParamName{"input"}, m2::ProgramRunArgs::TensorArgument{std::cref(input.mesh_tensor())});
    run_args.tensor_args.emplace(
        m2::TensorParamName{"output"}, m2::ProgramRunArgs::TensorArgument{std::cref(output.mesh_tensor())});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
