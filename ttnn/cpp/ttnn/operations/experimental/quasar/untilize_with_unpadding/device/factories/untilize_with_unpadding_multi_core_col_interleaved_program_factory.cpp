// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "untilize_with_unpadding_multi_core_col_interleaved_program_factory.hpp"

#include <filesystem>

#include "ttnn/operations/math.hpp"
#include "ttnn/operations/core/work_split/work_split_tilize.hpp"
#include <tt-metalium/constants.hpp>
#include <tt-metalium/allocator.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operation.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::constants;
using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

ttnn::device_operation::ProgramArtifacts
UntilizeWithUnpaddingMultiCoreColInterleavedProgramFactory::create_program_artifacts(
    const UntilizeWithUnpaddingParams& operation_attributes, const Tensor& input, Tensor& output) {
    const auto& a = input;
    const auto& input_mesh_tensor = input.mesh_tensor();
    const auto& output_mesh_tensor = output.mesh_tensor();
    bool fp32_dest_acc_en = operation_attributes.fp32_dest_acc_en;
    const auto& sub_core_grids = operation_attributes.sub_core_grids;

    tt::DataFormat input_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t input_single_tile_size = tt::tile_size(input_cb_data_format);
    tt::DataFormat output_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t output_single_tile_size = tt::tile_size(output_cb_data_format);

    const auto& input_shape = a.padded_shape();
    const auto& output_shape = output.padded_shape();

    IDevice* device = a.device();
    CoreCoord grid_size = device->compute_with_storage_grid_size();
    CoreRange default_cores({0, 0}, {grid_size.x - 1, grid_size.y - 1});
    CoreRangeSet default_grid(default_cores);
    CoreRangeSet available_grid = sub_core_grids.has_value() ? sub_core_grids.value() : default_grid;

    uint32_t num_blocks = input_shape[-1] / TILE_WIDTH;
    uint32_t num_tiles_per_row = a.padded_shape()[-1] / TILE_WIDTH;
    uint32_t num_tiles_per_col = a.padded_shape()[-2] / TILE_HEIGHT;

    auto [ncores, all_cores, core_range, core_range_cliff, nblocks_per_core, nblocks_per_core_cliff] =
        ttnn::split_blocks_for_tilize(available_grid, num_blocks);

    bool has_cliff = !core_range_cliff.empty();

    uint32_t unpadded_row_size_bytes;

    uint32_t el_size;
    if (a.dtype() == DataType::BFLOAT8_B) {
        unpadded_row_size_bytes = output_shape[-1] * output.element_size();
        el_size = output.element_size();
    } else {
        unpadded_row_size_bytes = output_shape[-1] * a.element_size();
        el_size = a.element_size();
    }

    // ---- Resource names ----
    const DFBSpecName IN_DFB{"in"};    // legacy c_0
    const DFBSpecName OUT_DFB{"out"};  // legacy c_16
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_FULL{"compute_full"};
    const KernelSpecName COMPUTE_CLIFF{"compute_cliff"};

    // ---- DataflowBuffers (legacy c_0 / c_16 CBs) ----
    DataflowBufferSpec in_dfb{
        .unique_id = IN_DFB,
        .entry_size = input_single_tile_size,
        .num_entries = num_tiles_per_col,
        .data_format_metadata = input_cb_data_format,
    };
    DataflowBufferSpec out_dfb{
        .unique_id = OUT_DFB,
        .entry_size = output_single_tile_size,
        .num_entries = num_tiles_per_col,
        .data_format_metadata = output_cb_data_format,
    };

    // ---- Tensor parameters ----
    TensorParameter input_param{.unique_id = INPUT, .spec = input.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    uint32_t num_tiles_2d = a.padded_shape()[-1] * a.padded_shape()[-2] / TILE_HW;

    auto log_shape = output.logical_shape();
    uint32_t third_dim = 1;
    if (log_shape.rank() == 3) {
        third_dim = log_shape[-3];
    } else if (log_shape.rank() >= 4) {
        third_dim = log_shape[-3] * log_shape[-4];
    }

    uint32_t total_num_rows = output.logical_shape()[-2];

    // ---- Reader kernel ----
    KernelSpec reader{
        .unique_id = READER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/device/kernels/dataflow/"
            "reader_unary_interleaved_col_multicore.cpp"),
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {{"num_tiles_per_2d", num_tiles_2d},
             {"third_dim", third_dim},
             {"number_blocks_per_core", nblocks_per_core}},
        .runtime_arg_schema = {.runtime_arg_names = {"core_number", "tiles_per_row", "num_blocks"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    // ---- Writer kernel ----
    KernelSpec writer{
        .unique_id = WRITER,
        .source = std::filesystem::path(
            "ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/device/kernels/dataflow/"
            "writer_unary_stick_layout_col_multicore.cpp"),
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .compile_time_args =
            {{"total_num_rows", total_num_rows},
             {"ncores", ncores},
             {"third_dim", third_dim},
             {"tile_width", TILE_WIDTH},
             {"unpadded_X_size", unpadded_row_size_bytes}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"core_number", "size_per_row_per_block", "blocks_per_core", "width_size"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    // ---- Compute kernel (full + cliff) ----
    KernelSpec::CompilerOptions::Defines compute_defines;
    if (input_cb_data_format == tt::DataFormat::Int32 || input_cb_data_format == tt::DataFormat::UInt32 ||
        input_cb_data_format == tt::DataFormat::Float32) {
        compute_defines.emplace("DST_ACCUM_MODE", "1");
    }
    auto make_compute_hw = [&]() -> ComputeHardwareConfig {
        ttnn::ComputeKernelConfig cfg{
            .math_fidelity = MathFidelity::HiFi4, .math_approx_mode = false, .fp32_dest_acc_en = fp32_dest_acc_en};
        ComputeHardwareConfig compute_hw = ttnn::to_compute_hardware_config(device->arch(), cfg);
        if (fp32_dest_acc_en) {
            std::visit(
                [&](auto& c) {
                    c.unpack_to_dest_mode.emplace(IN_DFB, tt::tt_metal::UnpackToDestMode::UnpackToDestFp32);
                },
                compute_hw);
        }
        return compute_hw;
    };
    const std::filesystem::path compute_source(
        "ttnn/cpp/ttnn/operations/experimental/quasar/untilize_with_unpadding/device/kernels/compute/"
        "untilize_w.cpp");
    auto make_compute = [&](const KernelSpecName& id, uint32_t nblocks) {
        return KernelSpec{
            .unique_id = id,
            .source = compute_source,
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings =
                {DFBBinding{.dfb_spec_name = IN_DFB, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args =
                {{"per_core_block_cnt", nblocks},
                 {"per_core_block_tile_cnt", num_tiles_per_col},
                 {"third_dim", third_dim}},
            .hw_config = make_compute_hw(),
        };
    };

    Group<KernelSpec> kernels = {reader, writer};
    Group<WorkUnitSpec> work_units;
    if (!core_range.empty()) {
        kernels.push_back(make_compute(COMPUTE_FULL, nblocks_per_core));
        work_units.push_back(WorkUnitSpec{
            .name = "untilize_with_unpadding_col_full",
            .kernels = {READER, WRITER, COMPUTE_FULL},
            .target_nodes = core_range});
    }
    if (has_cliff) {
        kernels.push_back(make_compute(COMPUTE_CLIFF, nblocks_per_core_cliff));
        work_units.push_back(WorkUnitSpec{
            .name = "untilize_with_unpadding_col_cliff",
            .kernels = {READER, WRITER, COMPUTE_CLIFF},
            .target_nodes = core_range_cliff});
    }

    // ---- Per-core runtime args ----
    // Replicates the legacy per-core work-distribution loop verbatim; the src/dst buffer-address
    // RTAs are dropped (carried by the TensorAccessor bindings).
    Group<KernelRunArgs::NodeRuntimeArgs> reader_node_args;
    Group<KernelRunArgs::NodeRuntimeArgs> writer_node_args;
    reader_node_args.reserve(ncores);
    writer_node_args.reserve(ncores);

    const auto& cores = corerange_to_cores(available_grid);
    uint32_t number_blocks_per_core;
    for (uint32_t i = 0; i < ncores; ++i) {
        const NodeCoord node = cores[i];

        if (has_cliff && i == ncores - 1) {
            number_blocks_per_core = nblocks_per_core_cliff;
        } else {
            number_blocks_per_core = nblocks_per_core;
        }

        // Writer named RTAs. NOTE: the legacy writer kernel read its named values from positional
        // runtime-arg indices 3/4/5 while the factory only emitted 5 args (0..4) — so the legacy
        // mapping was: size_per_row_per_block(kernel) <- number_blocks_per_core,
        // blocks_per_core(kernel) <- TILE_WIDTH*el_size, width_size(kernel) <- (out-of-bounds read).
        // This factory is not reachable via select_program_factory (dormant path), so the values are
        // preserved as-observed by the legacy kernel; width_size's legacy value was undefined and is
        // set here to the intended per-block width (TILE_WIDTH * el_size). See FLAG in port notes.
        writer_node_args.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = node,
            .args = {
                {"core_number", i},
                {"size_per_row_per_block", number_blocks_per_core},
                {"blocks_per_core", TILE_WIDTH * el_size},
                {"width_size", TILE_WIDTH * el_size}}});

        // Reader named RTAs (legacy: src_addr, i, num_tiles_per_row, number_blocks_per_core).
        reader_node_args.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = node,
            .args = {
                {"core_number", i}, {"tiles_per_row", num_tiles_per_row}, {"num_blocks", number_blocks_per_core}}});
    }

    // ---- ProgramSpec ----
    ProgramSpec spec{
        .name = "untilize_with_unpadding_multi_core_col_interleaved",
        .kernels = std::move(kernels),
        .dataflow_buffers = {in_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = std::move(work_units),
    };

    // ---- ProgramRunArgs ----
    ProgramRunArgs run_args;
    KernelRunArgs reader_args{.kernel = READER, .runtime_arg_values = std::move(reader_node_args)};
    KernelRunArgs writer_args{.kernel = WRITER, .runtime_arg_values = std::move(writer_node_args)};
    run_args.kernel_run_args = {std::move(reader_args), std::move(writer_args)};
    run_args.tensor_args = {{INPUT, input_mesh_tensor}, {OUTPUT, output_mesh_tensor}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
