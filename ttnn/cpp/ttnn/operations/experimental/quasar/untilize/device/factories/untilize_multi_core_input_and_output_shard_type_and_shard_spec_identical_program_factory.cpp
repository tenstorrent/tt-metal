// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.hpp"
#include "untilize_multi_core_program_factory.hpp"

#include <filesystem>
#include <optional>

#include "ttnn/common/constants.hpp"

#include <tt-metalium/constants.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

namespace {

// Lanes path: 4 reader threads copy the input shard's tiles into a ring, 4 compute threads ("lanes")
// untilize them into lane-wide row entries, and 2 writer threads copy the rows into the output shard;
// the data-movement sides use implicit sync. See create_untilize_split_rows_program for the lane
// layout. Returns std::nullopt when the shard does not fit it.
std::optional<ttnn::device_operation::ProgramArtifacts> create_shard_identical_lanes_program(
    const Tensor& a, const Tensor& output, bool fp32_dest_acc_en) {
    constexpr uint32_t num_lanes = kUntilizeNumLanes;
    const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
    const uint32_t tile_height = tile_shape[0];
    const uint32_t tile_width = tile_shape[1];
    const ShardSpec shard_spec = a.shard_spec().value();
    const uint32_t tiles_per_row = shard_spec.shape[1] / tile_width;
    const uint32_t num_blocks = shard_spec.shape[0] / tile_height;
    // Column lanes split every block's tiles four ways; otherwise each lane takes whole blocks.
    const bool column_lanes = tiles_per_row % num_lanes == 0;
    const uint32_t lane_tiles = column_lanes ? tiles_per_row / num_lanes : tiles_per_row;
    const uint32_t lane_bytes = lane_tiles * tile_width * output.element_size();
    if (lane_bytes % hal::get_l1_alignment() != 0 || (!column_lanes && num_blocks % num_lanes != 0)) {
        return std::nullopt;
    }
    const uint32_t num_lane_blocks = column_lanes ? num_blocks : num_blocks / num_lanes;

    const tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    const tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    const uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);

    const DFBSpecName IN_DFB{"in"};
    const DFBSpecName OUT_DFB{"out"};
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    const std::filesystem::path kdir("ttnn/cpp/ttnn/operations/experimental/quasar/untilize/device/kernels/");
    KernelSpec::CompilerOptions::Defines dm_defines;
    dm_defines.emplace("COLUMN_LANES", column_lanes ? "1" : "0");
    dm_defines.emplace("LOCAL_SHARD", "1");
    KernelSpec reader{
        .unique_id = READER,
        .source = kdir / "dataflow/reader_unary_lanes_metal2.cpp",
        .num_threads = num_lanes,
        .compiler_options = {.defines = dm_defines},
        .dfb_bindings = {ProducerOf(IN_DFB, "in")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {{"tiles_per_row", tiles_per_row}, {"lane_tiles", lane_tiles}, {"tile_bytes", input_single_tile_size}},
        .runtime_arg_schema = {.runtime_arg_names = {"first_block", "num_lane_blocks"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };
    KernelSpec writer{
        .unique_id = WRITER,
        .source = kdir / "dataflow/writer_unary_lanes_rows_metal2.cpp",
        .num_threads = kUntilizeNumWriterThreads,
        .compiler_options = {.defines = dm_defines},
        .dfb_bindings = {ConsumerOf(OUT_DFB, "out")},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args =
            {{"tile_height", tile_height},
             {"num_lanes", num_lanes},
             {"lane_bytes", lane_bytes},
             {"row_bytes", static_cast<uint32_t>(shard_spec.shape[1] * output.element_size())}},
        .runtime_arg_schema = {.runtime_arg_names = {"first_block", "lane_rows"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
    }
    ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(ttnn::ComputeKernelConfig{
        .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_dest_acc_en});
    if (fp32_dest_acc_en) {
        compute_hw.unpack_modes.emplace(IN_DFB, tt::tt_metal::UnpackMode::UnpackToDest);
    }
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = kdir / "compute/untilize_lanes_metal2.cpp",
        .num_threads = num_lanes,
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings = {ConsumerOf(IN_DFB, "in"), ProducerOf(OUT_DFB, "out")},
        .compile_time_args =
            {{"lane_tiles", lane_tiles},
             {"full_ct_dim", column_lanes ? tiles_per_row : num_lanes * tiles_per_row},
             {"block_rows", tile_height}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_lane_blocks", "last_block_rows"}},
        .hw_config = compute_hw,
    };

    KernelRunArgs::RuntimeArgValues reader_args;
    KernelRunArgs::RuntimeArgValues writer_args;
    KernelRunArgs::RuntimeArgValues compute_args;
    for (const auto& core : corerange_to_cores(shard_spec.grid)) {
        AddRuntimeArgsForNode(reader_args, core, {{"first_block", 0}, {"num_lane_blocks", num_lane_blocks}});
        AddRuntimeArgsForNode(writer_args, core, {{"first_block", 0}, {"lane_rows", num_lane_blocks * tile_height}});
        AddRuntimeArgsForNode(
            compute_args, core, {{"num_lane_blocks", num_lane_blocks}, {"last_block_rows", tile_height}});
    }

    ProgramSpec spec{
        .name = "untilize_input_and_output_shard_identical_lanes",
        .kernels = {reader, writer, compute},
        // One tile per entry in, two per lane; one block of lane-wide rows per lane out.
        .dataflow_buffers =
            {DataflowBufferSpec{
                 .unique_id = IN_DFB,
                 .entry_size = input_single_tile_size,
                 .num_entries = 2 * num_lanes,
                 .data_format_metadata = input_cb_data_format},
             DataflowBufferSpec{
                 .unique_id = OUT_DFB,
                 .entry_size = lane_bytes,
                 .num_entries = num_lanes * tile_height,
                 .data_format_metadata = output_cb_data_format}},
        .tensor_parameters =
            {TensorParameter{.unique_id = INPUT, .spec = a.tensor_spec()},
             TensorParameter{.unique_id = OUTPUT, .spec = output.tensor_spec()}},
        .work_units = {WorkUnitSpec{
            .name = "untilize_shard_identical_lanes",
            .kernels = {READER, WRITER, COMPUTE},
            .target_nodes = shard_spec.grid}},
    };
    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{.kernel = READER, .runtime_arg_values = std::move(reader_args)},
        KernelRunArgs{.kernel = WRITER, .runtime_arg_values = std::move(writer_args)},
        KernelRunArgs{.kernel = COMPUTE, .runtime_arg_values = std::move(compute_args)}};
    run_args.tensor_args = {{INPUT, a.mesh_tensor()}, {OUTPUT, output.mesh_tensor()}};
    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace

// Metal 2.0 port. Input and output share an identical shard spec, so BOTH DFBs are zero-copy: built on
// top of the input/output shard buffers via DataflowBufferSpec.borrowed_from. The reader just advances
// the input DFB write pointer, compute untilizes in place, and the writer is a readiness handshake on
// the output DFB. One compute KernelSpec over the shard grid; uniform per-core counts.
ttnn::device_operation::ProgramArtifacts
UntilizeMultiCoreInputAndOutputShardTypeAndShardSpecIdenticalProgramFactory::create_program_artifacts(
    const UntilizeOperationAttributes& operation_attributes,
    const UntilizeTensorArgs& tensor_args,
    UntilizeTensorReturnValue& tensor_return_value) {
    const auto& a = tensor_args.input;
    const Tensor& output = tensor_return_value;
    const auto& input_mesh_tensor = a.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();
    const auto& fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    if (auto lanes = create_shard_identical_lanes_program(a, tensor_return_value, fp32_dest_acc_en)) {
        return std::move(*lanes);
    }

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const auto& tile_shape = a.tensor_spec().tile().get_tile_shape();
    uint32_t tile_height = tile_shape[0];
    uint32_t tile_width = tile_shape[1];

    ShardSpec shard_spec = a.shard_spec().value();
    uint32_t shard_height = shard_spec.shape[0];
    uint32_t shard_width = shard_spec.shape[1];

    uint32_t num_tiles_per_block = shard_width / tile_width;
    uint32_t num_blocks_per_core = shard_height / tile_height;
    uint32_t num_tiles_per_shard = num_tiles_per_block * num_blocks_per_core;

    // ---- Resource names ----
    const DFBSpecName IN_DFB{"in"};    // legacy c_0
    const DFBSpecName OUT_DFB{"out"};  // legacy c_16
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    // ---- DataflowBuffers — both zero-copy, borrowed from the shard buffers ----
    DataflowBufferSpec in_dfb{
        .unique_id = IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_shard,
        .data_format_metadata = input_cb_data_format,
    };
    in_dfb.borrowed_from = INPUT;
    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_tiles_per_shard,
        .data_format_metadata = output_cb_data_format,
    };
    out_dfb.borrowed_from = OUTPUT;

    TensorParameter input_param{.unique_id = INPUT, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    const std::filesystem::path kdir("ttnn/cpp/ttnn/operations/experimental/quasar/untilize/device/kernels/");

    KernelSpec reader{
        .unique_id = READER,
        .source = kdir / "dataflow/reader_unary_sharded_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles_per_core"}},
        .hw_config = ttnn::create_reader_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = kdir / "dataflow/writer_unary_sharded_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_units"}},
        .hw_config = ttnn::create_writer_datamovement_config(/*disable_dfb_implicit_sync_for_all=*/true),
    };

    KernelSpec::CompilerOptions::Defines compute_defines;
    if (a.dtype() == DataType::INT32 || a.dtype() == DataType::UINT32 || a.dtype() == DataType::FLOAT32) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
    }
    ttnn::ComputeKernelConfig compute_config{
        .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_dest_acc_en};
    ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(compute_config);
    if (fp32_dest_acc_en) {
        compute_hw.unpack_modes.emplace(IN_DFB, tt::tt_metal::UnpackMode::UnpackToDest);
    }
    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = kdir / "compute/untilize_metal2.cpp",
        .compiler_options = {.defines = compute_defines},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{.dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
        .compile_time_args =
            {{"per_core_block_cnt", num_blocks_per_core}, {"per_core_block_tile_cnt", num_tiles_per_block}},
        .hw_config = compute_hw,
    };

    Group<KernelSpec> kernels = {reader, writer, compute};
    Group<WorkUnitSpec> work_units = {WorkUnitSpec{
        .name = "untilize_shard_identical", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = shard_spec.grid}};

    // Per-core runtime args (uniform across the shard grid).
    auto cores =
        corerange_to_cores(shard_spec.grid, std::nullopt, shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    KernelRunArgs::RuntimeArgValues reader_node_args;
    KernelRunArgs::RuntimeArgValues writer_node_args;
    for (const auto& core : cores) {
        reader_node_args["num_tiles_per_core"][core] = num_tiles_per_shard;
        writer_node_args["num_units"][core] = num_tiles_per_shard;
    }

    ProgramSpec spec{
        .name = "untilize_input_and_output_shard_identical",
        .kernels = std::move(kernels),
        .dataflow_buffers = {in_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{.kernel = READER, .runtime_arg_values = std::move(reader_node_args)},
        KernelRunArgs{.kernel = WRITER, .runtime_arg_values = std::move(writer_node_args)}};
    run_args.tensor_args = {{INPUT, input_mesh_tensor}, {OUTPUT, output_mesh_tensor}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
