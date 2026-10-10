// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_multi_core_input_and_output_shard_type_and_shard_spec_identical_program_factory.hpp"
#include "untilize_multi_core_program_factory.hpp"

#include <filesystem>

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

// Metal 2.0 port. Input and output share an identical shard spec, so BOTH DFBs are zero-copy: built on
// top of the input/output shard buffers via DataflowBufferSpec.borrowed_from. The reader just advances
// the input DFB write pointer, compute untilizes in place, and the writer is a readiness handshake on
// the output DFB. One compute KernelSpec over the shard grid; uniform per-core counts. The staged path
// replaces both DFBs with rings so every data-movement side moves data with implicit sync.
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

    // Staged path: 4 reader threads copy the input shard into an ALL-pattern ring by sub-block,
    // compute untilizes even and odd blocks into separate one-row-per-entry DFBs, and one
    // single-thread writer kernel per DFB copies the rows into the output shard. All sides use
    // implicit sync.
    constexpr uint32_t num_reader_threads = kUntilizeNumReaderThreads;
    const uint32_t sub_block_tiles =
        untilize_split_sub_block_tiles(num_tiles_per_block, num_blocks_per_core, fp32_dest_acc_en);
    const uint32_t row_bytes = shard_width * output.element_size();
    const bool staged = row_bytes % hal::get_l1_alignment() == 0 && sub_block_tiles != 0;

    // ---- Resource names ----
    const DFBSpecName IN_DFB{"in"};    // legacy c_0
    const DFBSpecName OUT_DFB{"out"};  // legacy c_16
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};
    const DFBSpecName OUT_ODD_DFB{"out_odd"};
    const KernelSpecName WRITER_ODD{"writer_odd"};

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
    DataflowBufferSpec out_odd_dfb{};
    if (staged) {
        // Rings replace the borrowed DFBs: two sub-blocks per reader thread in, two blocks of rows out.
        in_dfb.num_entries = 2 * num_reader_threads * sub_block_tiles;
        in_dfb.borrowed_from.reset();
        out_dfb.entry_size = row_bytes;
        out_dfb.num_entries = 2 * tile_height;
        out_dfb.borrowed_from.reset();
        out_odd_dfb = out_dfb;
        out_odd_dfb.unique_id = OUT_ODD_DFB;
    }

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

    KernelSpec writer_odd;
    if (staged) {
        reader = KernelSpec{
            .unique_id = READER,
            .source = kdir / "dataflow/reader_unary_shard_to_staged_metal2.cpp",
            .num_threads = num_reader_threads,
            .dfb_bindings = {ProducerOf(IN_DFB, "in")},
            .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
            .compile_time_args = {{"sub_block_tiles", sub_block_tiles}, {"tile_bytes", input_single_tile_size}},
            .runtime_arg_schema = {.runtime_arg_names = {"num_tiles"}},
            .hw_config = ttnn::create_reader_datamovement_config(),
        };
        auto make_row_writer = [&](const KernelSpecName& id, const DFBSpecName& dfb) {
            return KernelSpec{
                .unique_id = id,
                .source = kdir / "dataflow/writer_unary_rows_to_shard_metal2.cpp",
                .dfb_bindings = {DFBBinding{
                    .dfb_spec_name = dfb, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
                .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
                .compile_time_args = {{"tile_height", tile_height}, {"block_step", 2}, {"row_bytes", row_bytes}},
                .runtime_arg_schema = {.runtime_arg_names = {"num_blocks", "first_block"}},
                .hw_config = ttnn::create_writer_datamovement_config(),
            };
        };
        writer = make_row_writer(WRITER, OUT_DFB);
        writer_odd = make_row_writer(WRITER_ODD, OUT_ODD_DFB);
    }

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

    if (staged) {
        compute.source = kdir / "compute/untilize_split_output_metal2.cpp";
        compute.dfb_bindings = {
            AllConsumerOf(IN_DFB, "in"), ProducerOf(OUT_DFB, "out"), ProducerOf(OUT_ODD_DFB, "out_odd")};
        compute.compile_time_args = {
            {"per_core_block_tile_cnt", num_tiles_per_block},
            {"sub_block_tiles", sub_block_tiles},
            {"block_rows", tile_height}};
        compute.runtime_arg_schema = {.runtime_arg_names = {"per_core_block_cnt", "last_block_rows"}};
    }

    Group<KernelSpec> kernels = {reader, writer, compute};
    Group<KernelSpecName> work_unit_kernels = {READER, WRITER, COMPUTE};
    if (staged) {
        kernels.push_back(writer_odd);
        work_unit_kernels.push_back(WRITER_ODD);
    }
    Group<WorkUnitSpec> work_units = {WorkUnitSpec{
        .name = "untilize_shard_identical", .kernels = work_unit_kernels, .target_nodes = shard_spec.grid}};

    // Per-core runtime args (uniform across the shard grid).
    auto cores =
        corerange_to_cores(shard_spec.grid, std::nullopt, shard_spec.orientation == ShardOrientation::ROW_MAJOR);
    KernelRunArgs::RuntimeArgValues reader_node_args;
    KernelRunArgs::RuntimeArgValues writer_node_args;
    KernelRunArgs::RuntimeArgValues writer_odd_node_args;
    KernelRunArgs::RuntimeArgValues compute_node_args;
    for (const auto& core : cores) {
        if (staged) {
            // Even blocks go to WRITER, odd ones to WRITER_ODD.
            reader_node_args["num_tiles"][core] = num_tiles_per_shard;
            AddRuntimeArgsForNode(
                writer_node_args, core, {{"num_blocks", (num_blocks_per_core + 1) / 2}, {"first_block", 0}});
            AddRuntimeArgsForNode(
                writer_odd_node_args, core, {{"num_blocks", num_blocks_per_core / 2}, {"first_block", 1}});
            AddRuntimeArgsForNode(
                compute_node_args,
                core,
                {{"per_core_block_cnt", num_blocks_per_core}, {"last_block_rows", tile_height}});
        } else {
            reader_node_args["num_tiles_per_core"][core] = num_tiles_per_shard;
            writer_node_args["num_units"][core] = num_tiles_per_shard;
        }
    }

    Group<DataflowBufferSpec> dataflow_buffers = {in_dfb, out_dfb};
    if (staged) {
        dataflow_buffers.push_back(out_odd_dfb);
    }

    ProgramSpec spec{
        .name = "untilize_input_and_output_shard_identical",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dataflow_buffers),
        .tensor_parameters = {input_param, output_param},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{.kernel = READER, .runtime_arg_values = std::move(reader_node_args)},
        KernelRunArgs{.kernel = WRITER, .runtime_arg_values = std::move(writer_node_args)}};
    if (staged) {
        run_args.kernel_run_args.push_back(
            KernelRunArgs{.kernel = WRITER_ODD, .runtime_arg_values = std::move(writer_odd_node_args)});
        run_args.kernel_run_args.push_back(
            KernelRunArgs{.kernel = COMPUTE, .runtime_arg_values = std::move(compute_node_args)});
    }
    run_args.tensor_args = {{INPUT, input_mesh_tensor}, {OUTPUT, output_mesh_tensor}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
