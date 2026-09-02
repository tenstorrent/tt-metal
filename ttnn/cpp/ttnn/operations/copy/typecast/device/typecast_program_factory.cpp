// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "typecast_program_factory.hpp"

#include <tt-metalium/work_split.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/tt_align.hpp>

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::prim {

using namespace tt::constants;
using namespace tt::tt_metal::experimental;

namespace {

// Kernel sources shared by both factories in this file. The two interleaved dataflow kernels are
// Metal 2.0 forks of the eltwise/unary donors (see the fork note in their headers).
constexpr const char* kReaderSource =
    "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/reader_unary_interleaved_start_id_metal2.cpp";
constexpr const char* kWriterSource =
    "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/dataflow/writer_unary_interleaved_start_id_metal2.cpp";
constexpr const char* kComputeSource =
    "ttnn/cpp/ttnn/operations/copy/typecast/device/kernels/compute/eltwise_typecast.cpp";

// The typecast LLK is selected by input/output data format through these two defines.
KernelSpec::CompilerOptions::Defines make_typecast_defines(
    tt::tt_metal::DataType input_dtype, tt::tt_metal::DataType output_dtype) {
    KernelSpec::CompilerOptions::Defines defines;
    defines.emplace(
        "TYPECAST_LLK_INIT",
        fmt::format(
            "typecast_tile_init<{0}u, {1}u>",
            static_cast<uint32_t>(tt::tt_metal::datatype_to_dataformat_converter(input_dtype)),
            static_cast<uint32_t>(tt::tt_metal::datatype_to_dataformat_converter(output_dtype))));
    defines.emplace(
        "TYPECAST_LLK",
        fmt::format(
            "typecast_tile<{0}u, {1}u>",
            static_cast<uint32_t>(tt::tt_metal::datatype_to_dataformat_converter(input_dtype)),
            static_cast<uint32_t>(tt::tt_metal::datatype_to_dataformat_converter(output_dtype))));
    return defines;
}

// Legacy set unpack_to_dest_mode[in_cb] = UnpackToDestFp32 when preserve_fp32_precision (a no-op in
// the JIT unless the format is Float32), leaving every other CB at Default. The named equivalent is
// an UnpackToDest entry for the input DFB. Metal 2.0 additionally *requires* an explicit entry for a
// consumed Float32 DFB when enable_32_bit_dest is set, where legacy silently defaulted — so supply
// the legacy default (UnpackToSrc, which lowers to UnpackToDestMode::Default) in that case.
ComputeUnpackModes make_unpack_modes(
    const TypecastParams& args, const DFBSpecName& in_dfb, tt::DataFormat input_data_format) {
    ComputeUnpackModes unpack_modes;
    if (args.preserve_fp32_precision) {
        unpack_modes.emplace(in_dfb, tt::tt_metal::UnpackMode::UnpackToDest);
    } else if (args.fp32_dest_acc_en && input_data_format == tt::DataFormat::Float32) {
        unpack_modes.emplace(in_dfb, tt::tt_metal::UnpackMode::UnpackToSrc);
    }
    return unpack_modes;
}

// The legacy ComputeConfigDescriptor field values, carried over one-for-one:
//   math_fidelity=HiFi4 -> fpu_math_fidelity; fp32_dest_acc_en -> enable_32_bit_dest;
//   bfp8_pack_precise -> bfp_pack_precision_mode; math_approx_mode=false -> sfpu_precision_mode.
// dst_full_sync_en was left at its legacy default (false), which is double_buffer_dest = true (the
// Metal 2.0 default), so it needs no explicit setting.
ComputeGen1Config make_compute_config(const TypecastParams& args, ComputeUnpackModes unpack_modes) {
    return ComputeGen1Config{
        .fpu_math_fidelity = tt::tt_metal::MathFidelity::HiFi4,
        .sfpu_precision_mode = tt::tt_metal::Precision::Precise,  // legacy math_approx_mode = false
        .bfp_pack_precision_mode =
            args.bfp8_pack_precise ? tt::tt_metal::Precision::Precise : tt::tt_metal::Precision::Approximate,
        .enable_32_bit_dest = args.fp32_dest_acc_en,
        .unpack_modes = std::move(unpack_modes),
    };
}

}  // namespace

ttnn::device_operation::ProgramArtifacts TypecastProgramFactory::create_program_artifacts(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const Tensor& input = tensor_args.input;
    const DataType& input_dtype = args.input_dtype;
    const DataType& output_dtype = args.output_dtype;
    const bool is_row_major = input.layout() == Layout::ROW_MAJOR;

    const tt::DataFormat cb_data_format_input = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const uint32_t single_tile_size_input = tt::tile_size(cb_data_format_input);
    const tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    const uint32_t single_tile_size_output = tt::tile_size(cb_data_format_output);

    const auto* device = input.device();

    // Get number of pages (tiles for TILE layout, rows for ROW_MAJOR layout)
    const uint32_t num_pages = input.buffer()->num_pages();

    // Set DFB entry size correctly based on layout
    // - For TILE layout: entry = one 32x32 tile
    // - For ROW_MAJOR layout: entry = one full row including padding
    const uint32_t input_page_size = is_row_major ? input.buffer()->page_size() : single_tile_size_input;
    const uint32_t output_page_size = is_row_major ? output.buffer()->page_size() : single_tile_size_output;

    const CoreCoord compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto [num_cores, all_cores, core_group_1, core_group_2, num_items_per_core_group_1, num_items_per_core_group_2] =
        tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_pages, is_row_major);
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

    constexpr uint32_t num_input_pages = 2;
    constexpr uint32_t num_output_pages = 2;
    const DataflowBufferSpec in_dfb{
        .unique_id = IN_DFB,
        .entry_size = input_page_size,
        .num_entries = num_input_pages,
        .data_format_metadata = cb_data_format_input,
    };
    const DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = output_page_size,
        .num_entries = num_output_pages,
        .data_format_metadata = cb_data_format_output,
    };

    const TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec()};
    const TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    const KernelSpec reader{
        .unique_id = READER,
        .source = kReaderSource,
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    const KernelSpec writer{
        .unique_id = WRITER,
        .source = kWriterSource,
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    const auto typecast_defines = make_typecast_defines(input_dtype, output_dtype);
    const auto make_compute = [&](const KernelSpecName& id, uint32_t per_core_block_cnt) {
        return KernelSpec{
            .unique_id = id,
            .source = kComputeSource,
            .compiler_options = {.defines = typecast_defines, .opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings =
                {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args =
                {{"per_core_block_cnt", per_core_block_cnt},
                 // per_core_block_dim is always 1 (works for both tiled and row-major)
                 {"per_core_block_dim", 1u}},
            .hw_config =
                ComputeHardwareConfig{make_compute_config(args, make_unpack_modes(args, IN_DFB, cb_data_format_input))},
        };
    };

    // One KernelSpec per legacy compute KernelDescriptor: the per-group item count stays a
    // compile-time arg, so the work-split multiplicity is preserved.
    Group<KernelSpec> kernels = {reader, writer};
    Group<WorkUnitSpec> work_units;
    kernels.push_back(make_compute(COMPUTE_GROUP_1, num_items_per_core_group_1));
    work_units.push_back(WorkUnitSpec{
        .name = "typecast_group_1", .kernels = {READER, WRITER, COMPUTE_GROUP_1}, .target_nodes = core_group_1});
    const bool has_group_2 = !core_group_2.ranges().empty();
    if (has_group_2) {
        kernels.push_back(make_compute(COMPUTE_GROUP_2, num_items_per_core_group_2));
        work_units.push_back(WorkUnitSpec{
            .name = "typecast_group_2", .kernels = {READER, WRITER, COMPUTE_GROUP_2}, .target_nodes = core_group_2});
    }

    // Convert CoreRangeSet to vector of cores in the correct order
    // Use row_wise=true for row-major layout to match row distribution, false for tile layout
    auto cores_vec = corerange_to_cores(all_cores, std::nullopt, is_row_major);

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
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

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"num_pages", num_items_per_core}, {"start_id", num_items_written}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_pages", num_items_per_core}, {"start_id", num_items_written}});
        num_items_written += num_items_per_core;
    }

    ProgramSpec spec{
        .name = "typecast",
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

// For sub_core_grids
ttnn::device_operation::ProgramArtifacts TypecastSubgridProgramFactory::create_program_artifacts(
    const TypecastParams& args, const TypecastInputs& tensor_args, Tensor& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& input_dtype = args.input_dtype;
    const auto& output_dtype = args.output_dtype;
    const auto& sub_core_grids = args.sub_core_grids;

    TT_FATAL(sub_core_grids.has_value(), "sub_core_grids cannot be null");

    tt::DataFormat cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    uint32_t single_tile_size = tt::tile_size(cb_data_format);
    tt::DataFormat cb_data_format_output = tt::tt_metal::datatype_to_dataformat_converter(output.dtype());
    uint32_t single_tile_size_output = tt::tile_size(cb_data_format_output);

    const auto* device = input.device();

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

    // ---- Resource names ----
    const DFBSpecName IN_DFB{"in"};    // legacy CBIndex::c_0
    const DFBSpecName OUT_DFB{"out"};  // legacy CBIndex::c_2
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    constexpr uint32_t num_input_tiles = 2;
    constexpr uint32_t num_output_tiles = 2;
    const DataflowBufferSpec in_dfb{
        .unique_id = IN_DFB,
        .entry_size = single_tile_size,
        .num_entries = num_input_tiles,
        .data_format_metadata = cb_data_format,
    };
    const DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = single_tile_size_output,
        .num_entries = num_output_tiles,
        .data_format_metadata = cb_data_format_output,
    };

    const TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec()};
    const TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    const KernelSpec reader{
        .unique_id = READER,
        .source = kReaderSource,
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    const KernelSpec writer{
        .unique_id = WRITER,
        .source = kWriterSource,
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    uint32_t ntiles_per_core = ntiles / ncores;
    const KernelSpec compute{
        .unique_id = COMPUTE,
        .source = kComputeSource,
        .compiler_options =
            {.defines = make_typecast_defines(input_dtype, output_dtype), .opt_level = KernelBuildOptLevel::O3},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args =
            {{"per_core_block_cnt", static_cast<uint32_t>(ntiles_per_core)}, {"per_core_block_dim", 1u}},
        .hw_config = ComputeHardwareConfig{make_compute_config(args, make_unpack_modes(args, IN_DFB, cb_data_format))},
    };

    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};
    uint32_t tile_start_id = 0;
    for (auto core : cores) {
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values, core, {{"num_pages", ntiles_per_core}, {"start_id", tile_start_id}});
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values, core, {{"num_pages", ntiles_per_core}, {"start_id", tile_start_id}});
        tile_start_id += ntiles_per_core;
    }

    ProgramSpec spec{
        .name = "typecast_subgrid",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = {in_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = {WorkUnitSpec{
            .name = "typecast_subgrid", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores}},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {{INPUT, input.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
