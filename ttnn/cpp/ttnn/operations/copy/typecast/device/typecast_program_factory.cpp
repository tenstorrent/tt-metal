// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_program_factory.hpp"

#include <filesystem>

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>

namespace ttnn::prim {

using namespace tt::constants;
using namespace tt;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

namespace {

// File-scope kernel paths. Names are unique across the typecast device/ sibling .cpp files to avoid
// unity-build collisions.
constexpr const char* INTERLEAVED_READER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_typecast_interleaved.cpp";
constexpr const char* INTERLEAVED_WRITER_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/writer_typecast_interleaved.cpp";
constexpr const char* INTERLEAVED_COMPUTE_KERNEL_PATH =
    "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast_m2.cpp";

// TYPECAST_LLK / TYPECAST_LLK_INIT defines, formerly assembled into a std::map and copied onto each
// compute KernelDescriptor. Shared by both core groups (and both migrated factories in this file).
m2::KernelSpec::CompilerOptions::Defines typecast_compute_defines(DataType input_dtype, DataType output_dtype) {
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

// Build the compute kernel's ComputeHardwareConfig (was ComputeConfigDescriptor). When
// preserve_fp32_precision is set, the input DFB is unpacked to Dest in full FP32 (formerly an
// UnpackToDestFp32 entry in the per-CB unpack_to_dest_mode vector).
m2::ComputeHardwareConfig typecast_compute_hw_config(
    const TypecastParams& args, const m2::DFBSpecName& input_dfb, tt::DataFormat input_format) {
    m2::ComputeHardwareConfig hw{
        .math_fidelity = MathFidelity::HiFi4,
        .fp32_dest_acc_en = args.fp32_dest_acc_en,
        .bfp8_pack_precise = args.bfp8_pack_precise,
        .math_approx_mode = false,
    };
    // UnpackToDestFp32 is only meaningful (and only accepted by ValidateProgramSpec) when the input DFB
    // actually carries Float32 data. Legacy set this mode whenever preserve_fp32_precision was true,
    // which for non-fp32 inputs (e.g. *->uint8) was a tolerated no-op; faithfully reproduce the effective
    // behavior by applying it only for Float32 input.
    if (args.preserve_fp32_precision && input_format == tt::DataFormat::Float32) {
        hw.unpack_to_dest_mode.emplace(input_dfb, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32);
    }
    return hw;
}

}  // namespace

ttnn::device_operation::ProgramArtifacts TypecastProgramFactory::create_program_artifacts(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    const Tensor& input = tensor_args.input;
    const DataType input_dtype = args.input_dtype;
    const DataType output_dtype = args.output_dtype;
    const bool is_row_major = input.layout() == Layout::ROW_MAJOR;

    const tt::DataFormat cb_data_format_input = datatype_to_dataformat_converter(input.dtype());
    const uint32_t single_tile_size_input = tt::tile_size(cb_data_format_input);
    const tt::DataFormat cb_data_format_output = datatype_to_dataformat_converter(output.dtype());
    const uint32_t single_tile_size_output = tt::tile_size(cb_data_format_output);

    const auto* device = input.device();

    // Get number of pages (tiles for TILE layout, rows for ROW_MAJOR layout)
    const uint32_t num_pages = input.buffer()->num_pages();

    Buffer* src_buffer = input.buffer();
    Buffer* dst_buffer = output.buffer();

    // Set CB page size correctly based on layout
    // - For TILE layout: page = one 32x32 tile
    // - For ROW_MAJOR layout: page = one full row including padding
    const uint32_t input_page_size = is_row_major ? src_buffer->page_size() : single_tile_size_input;
    const uint32_t output_page_size = is_row_major ? dst_buffer->page_size() : single_tile_size_output;

    const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_items_per_core_group_1, num_items_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_pages, is_row_major);

    constexpr uint32_t num_input_pages = 2;
    constexpr uint32_t num_output_pages = 2;
    const m2::DFBSpecName input_dfb{"input_cb"};
    const m2::DFBSpecName output_dfb{"output_cb"};

    m2::ProgramSpec spec;
    spec.name = "typecast";

    // DFBs: src0 is L1-allocated double-buffered; out0 is L1-allocated double-buffered. (formerly CBs
    // c_0 / c_2). The compute kernel consumes input_cb and produces output_cb.
    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = input_dfb,
            .entry_size = input_page_size,
            .num_entries = num_input_pages,
            .data_format_metadata = cb_data_format_input,
        },
        m2::DataflowBufferSpec{
            .unique_id = output_dfb,
            .entry_size = output_page_size,
            .num_entries = num_output_pages,
            .data_format_metadata = cb_data_format_output,
        }};

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = input.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()}};

    // Reader on NCRISC (RISCV_1 / NOC1), writer on BRISC — so the two DM kernels don't collide on the
    // same DM processor.
    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{INTERLEAVED_READER_KERNEL_PATH},
        .dfb_bindings = {m2::ProducerOf(input_dfb, "input_cb")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "src_args"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt::tt_metal::NOC::RISCV_1_default}},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{INTERLEAVED_WRITER_KERNEL_PATH},
        .dfb_bindings = {m2::ConsumerOf(output_dfb, "output_cb")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "dst_args"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
    };

    // Compute kernel group 1: consumes input_cb, produces output_cb. per_core_block_cnt is a per-kernel
    // compile-time arg (it differs between the two core groups); per_core_block_dim is always 1.
    auto make_compute = [&](const m2::KernelSpecName& name, uint32_t per_core_block_cnt) {
        return m2::KernelSpec{
            .unique_id = name,
            .source = std::filesystem::path{INTERLEAVED_COMPUTE_KERNEL_PATH},
            .compiler_options = {.defines = typecast_compute_defines(input_dtype, output_dtype)},
            .dfb_bindings = {m2::ConsumerOf(input_dfb, "input_cb"), m2::ProducerOf(output_dfb, "output_cb")},
            .compile_time_args = {{"per_core_block_cnt", per_core_block_cnt}, {"per_core_block_dim", 1u}},
            .hw_config = typecast_compute_hw_config(args, input_dfb, cb_data_format_input),
        };
    };

    spec.kernels = {
        std::move(reader),
        std::move(writer),
        make_compute(m2::KernelSpecName{"compute_1"}, num_items_per_core_group_1)};

    // Local DFBs (input_cb, output_cb) require their producer AND consumer kernels to be hosted by the
    // same WorkUnitSpec on every node where the DFB lives. reader produces input_cb (consumed by compute)
    // and writer consumes output_cb (produced by compute), so reader/writer must be co-located with the
    // compute kernel in each core group's WorkUnitSpec — not split into a separate DM-only WorkUnitSpec.
    spec.work_units = {m2::WorkUnitSpec{
        .name = "typecast_compute_1",
        .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute_1"}},
        .target_nodes = core_group_1}};

    const bool has_group_2 = !core_group_2.ranges().empty();
    if (has_group_2) {
        spec.kernels.push_back(make_compute(m2::KernelSpecName{"compute_2"}, num_items_per_core_group_2));
        spec.work_units.push_back(m2::WorkUnitSpec{
            .name = "typecast_compute_2",
            .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute_2"}},
            .target_nodes = core_group_2});
    }

    // Per-core runtime args. Use row_wise=true for row-major layout to match row distribution.
    auto cores_vec = corerange_to_cores(all_cores, std::nullopt, is_row_major);

    m2::ProgramRunArgs run_args;
    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};

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

        reader_args.runtime_arg_values.push_back(
            {core, {{"num_pages", num_items_per_core}, {"start_id", num_items_written}}});
        writer_args.runtime_arg_values.push_back(
            {core, {{"num_pages", num_items_per_core}, {"start_id", num_items_written}}});
        num_items_written += num_items_per_core;
    }
    run_args.kernel_run_args.push_back(std::move(reader_args));
    run_args.kernel_run_args.push_back(std::move(writer_args));

    run_args.tensor_args.emplace(
        m2::TensorParamName{"input"}, m2::ProgramRunArgs::TensorArgument{std::cref(input.mesh_tensor())});
    run_args.tensor_args.emplace(
        m2::TensorParamName{"output"}, m2::ProgramRunArgs::TensorArgument{std::cref(output.mesh_tensor())});

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

// For sub_core_grids
ttnn::device_operation::ProgramArtifacts TypecastSubgridProgramFactory::create_program_artifacts(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    const auto& input = tensor_args.input;
    const DataType input_dtype = args.input_dtype;
    const DataType output_dtype = args.output_dtype;
    const auto& sub_core_grids = args.sub_core_grids;

    TT_FATAL(sub_core_grids.has_value(), "sub_core_grids cannot be null");

    const tt::DataFormat cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const uint32_t single_tile_size = tt::tile_size(cb_data_format);
    const tt::DataFormat cb_data_format_output = datatype_to_dataformat_converter(output.dtype());
    const uint32_t single_tile_size_output = tt::tile_size(cb_data_format_output);

    uint32_t ntiles = input.physical_volume() / tt::constants::TILE_HW;
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

    constexpr uint32_t num_input_tiles = 2;
    constexpr uint32_t num_output_tiles = 2;
    const m2::DFBSpecName input_dfb{"input_cb"};
    const m2::DFBSpecName output_dfb{"output_cb"};

    m2::ProgramSpec spec;
    spec.name = "typecast_subgrid";

    spec.dataflow_buffers = {
        m2::DataflowBufferSpec{
            .unique_id = input_dfb,
            .entry_size = single_tile_size,
            .num_entries = num_input_tiles,
            .data_format_metadata = cb_data_format,
        },
        m2::DataflowBufferSpec{
            .unique_id = output_dfb,
            .entry_size = single_tile_size_output,
            .num_entries = num_output_tiles,
            .data_format_metadata = cb_data_format_output,
        }};

    spec.tensor_parameters = {
        m2::TensorParameter{.unique_id = m2::TensorParamName{"input"}, .spec = input.tensor_spec()},
        m2::TensorParameter{.unique_id = m2::TensorParamName{"output"}, .spec = output.tensor_spec()}};

    m2::KernelSpec reader{
        .unique_id = m2::KernelSpecName{"reader"},
        .source = std::filesystem::path{INTERLEAVED_READER_KERNEL_PATH},
        .dfb_bindings = {m2::ProducerOf(input_dfb, "input_cb")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"input"}, .accessor_name = "src_args"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config =
            m2::DataMovementHardwareConfig{
                .gen1_config =
                    m2::DataMovementHardwareConfig::Gen1Config{
                        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
                        .noc = tt::tt_metal::NOC::RISCV_1_default}},
    };

    m2::KernelSpec writer{
        .unique_id = m2::KernelSpecName{"writer"},
        .source = std::filesystem::path{INTERLEAVED_WRITER_KERNEL_PATH},
        .dfb_bindings = {m2::ConsumerOf(output_dfb, "output_cb")},
        .tensor_bindings = {m2::TensorBinding{
            .tensor_parameter_name = m2::TensorParamName{"output"}, .accessor_name = "dst_args"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = m2::DataMovementHardwareConfig{.gen1_config = m2::DataMovementHardwareConfig::Gen1Config{}},
    };

    const uint32_t ntiles_per_core = ntiles / ncores;
    m2::KernelSpec compute{
        .unique_id = m2::KernelSpecName{"compute"},
        .source = std::filesystem::path{INTERLEAVED_COMPUTE_KERNEL_PATH},
        .compiler_options = {.defines = typecast_compute_defines(input_dtype, output_dtype)},
        .dfb_bindings = {m2::ConsumerOf(input_dfb, "input_cb"), m2::ProducerOf(output_dfb, "output_cb")},
        .compile_time_args = {{"per_core_block_cnt", ntiles_per_core}, {"per_core_block_dim", 1u}},
        .hw_config = typecast_compute_hw_config(args, input_dfb, cb_data_format),
    };

    spec.kernels = {std::move(reader), std::move(writer), std::move(compute)};
    spec.work_units = {m2::WorkUnitSpec{
        .name = "typecast_subgrid",
        .kernels = {m2::KernelSpecName{"reader"}, m2::KernelSpecName{"writer"}, m2::KernelSpecName{"compute"}},
        .target_nodes = all_cores}};

    m2::ProgramRunArgs run_args;
    m2::ProgramRunArgs::KernelRunArgs reader_args{.kernel = m2::KernelSpecName{"reader"}};
    m2::ProgramRunArgs::KernelRunArgs writer_args{.kernel = m2::KernelSpecName{"writer"}};

    uint32_t tile_start_id = 0;
    for (const auto& core : cores) {
        reader_args.runtime_arg_values.push_back({core, {{"num_pages", ntiles_per_core}, {"start_id", tile_start_id}}});
        writer_args.runtime_arg_values.push_back({core, {{"num_pages", ntiles_per_core}, {"start_id", tile_start_id}}});
        tile_start_id += ntiles_per_core;
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
