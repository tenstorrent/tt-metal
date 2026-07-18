// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "reduce_op_device_operation.hpp"
#include "ttnn/operations/experimental/quasar/reduction/generic/device/reduce_op.hpp"
#include <tt-metalium/allocator.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/math.hpp>
#include <tt-logger/tt-logger.hpp>  // [DIAG avgpool x1.15] remove after
#include <bit>
#include <cmath>
#include <filesystem>
#include <map>

using namespace tt::tt_metal;
using namespace tt::tt_metal::experimental;

namespace ttnn::prim::qsr {

// Metal 2.0 port of the multi-core H reduce factory — SCOPED to the resnet-reachable path.
//
// Only the interleaved, non-negate H-reduce path is on Metal 2.0 (this is what pool_sum / avg_pool2d
// reaches: Sum over H, INTERLEAVED in/out, no negate). It also covers Int32-MAX (reduce_metal2 routes
// Int32 to the SFPU path internally). The remaining branches of the legacy factory — the row-major
// dense path (rm_path, currently perf-disabled upstream), width-sharding (re-routed to INTERLEAVED
// upstream), and the negate/MIN path (dual SFPU/FPU with self-loop acc/ineg) — are NOT yet ported and
// TT_FATAL here. They stay on the legacy ProgramDescriptor path until ported (see METAL2_PORT_REPORT.md).
ttnn::device_operation::ProgramArtifacts
ReduceDeviceOperation::ReduceMultiCoreHProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& tensor_return_value) {
    using namespace tt;
    const auto& a = tensor_args.mesh_tensor();
    const auto& output = tensor_return_value.mesh_tensor();
    const auto& shape = a.padded_shape();
    uint32_t W = shape[3], H = shape[2], NC = shape[1] * shape[0];
    const uint32_t tile_height = a.tensor_spec().tile().get_height();
    const uint32_t tile_width = a.tensor_spec().tile().get_width();

    uint32_t Wt = tt::div_up(W, tile_width);
    uint32_t Ht = tt::div_up(H, tile_height);
    uint32_t HtWt = Ht * Wt;

    const bool use_width_sharding = a.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED &&
                                    output.memory_config().memory_layout() == TensorMemoryLayout::WIDTH_SHARDED;

    // Not-yet-ported branches stay on the legacy ProgramDescriptor concept.
    TT_FATAL(
        !operation_attributes.row_major_h_dense_path && !use_width_sharding && !operation_attributes.negate,
        "Reduce MultiCoreH Metal 2.0 port currently supports only the interleaved, non-negate path "
        "(rm_path={}, width_sharded={}, negate={} are not yet ported to Metal 2.0).",
        operation_attributes.row_major_h_dense_path,
        use_width_sharding,
        operation_attributes.negate);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(a.device().arch(), operation_attributes.compute_kernel_config);

    tt::DataFormat src0_cb_data_format = datatype_to_dataformat_converter(a.dtype());
    uint32_t src0_single_tile_size = tt::tile_size(src0_cb_data_format);
    tt::DataFormat scaler_cb_data_format =
        src0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    uint32_t scaler_single_tile_size = tt::tile_size(scaler_cb_data_format);
    tt::DataFormat dst_cb_data_format = datatype_to_dataformat_converter(output.dtype());
    uint32_t dst_single_tile_size = tt::tile_size(dst_cb_data_format);

    IDevice* device = &a.mutable_device();

    const bool use_post_mul = operation_attributes.post_mul_scaler != 1.0f;
    uint32_t scaler_bits = std::bit_cast<uint32_t>(operation_attributes.scaler);
    uint32_t post_mul_scaler_bits = std::bit_cast<uint32_t>(operation_attributes.post_mul_scaler);

    // [DIAG avgpool x1.15 -- remove after] What does the multi_core_H factory actually see/emit? If this
    // does NOT print during the failing run, a cached program is being reused (stale scaler baked in).
    // If it prints use_post_mul=0 / scaler=1/49, attributes lost the split. If use_post_mul=1 &
    // scaler=1.0 but output is still x1.15, the H compute kernel isn't applying REDUCE_POST_MUL.
    log_warning(
        tt::LogOp,
        "QSR_REDUCE_H_FACTORY math_op={} scaler={} post_mul_scaler={} use_post_mul={} scaler_bits=0x{:08x} "
        "post_mul_bits=0x{:08x}",
        static_cast<int>(operation_attributes.math_op),
        operation_attributes.scaler,
        operation_attributes.post_mul_scaler,
        use_post_mul,
        scaler_bits,
        post_mul_scaler_bits);

    auto compute_with_storage_grid_size = device->compute_with_storage_grid_size();
    auto num_cols = NC * Wt;
    uint32_t num_cores;
    CoreRangeSet all_cores, core_group_1, core_group_2;
    uint32_t num_cols_per_core_group_1, num_cols_per_core_group_2;
    if (operation_attributes.sub_core_grids.has_value()) {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(*operation_attributes.sub_core_grids, num_cols);
    } else {
        std::tie(
            num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2) =
            tt::tt_metal::split_work_to_cores(compute_with_storage_grid_size, num_cols);
    }
    TT_FATAL(num_cores > 0, "Reduce H requires at least one worker core");

    // ---- Resource names ----
    const DFBSpecName IN{"in"};          // legacy c_0
    const DFBSpecName SCALER{"scaler"};  // legacy c_2
    const DFBSpecName OUT{"out"};        // legacy c_3
    const TensorParamName INPUT{"input"};
    const TensorParamName OUTPUT{"output"};
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};

    DataflowBufferSpec in_dfb{
        .unique_id = IN,
        .entry_size = src0_single_tile_size,
        .num_entries = 2,
        .data_format_metadata = src0_cb_data_format};
    DataflowBufferSpec scaler_dfb{
        .unique_id = SCALER,
        .entry_size = scaler_single_tile_size,
        .num_entries = 1,
        .data_format_metadata = scaler_cb_data_format};
    DataflowBufferSpec out_dfb{
        .unique_id = OUT,
        .entry_size = dst_single_tile_size,
        .num_entries = 2,
        .data_format_metadata = dst_cb_data_format};

    TensorParameter input_param{.unique_id = INPUT, .spec = a.tensor_spec()};
    TensorParameter output_param{.unique_id = OUTPUT, .spec = output.tensor_spec()};

    // ---- Reduce defines ----
    std::map<std::string, std::string> reduce_defines =
        reduce_op_utils_qsr::get_defines(operation_attributes.math_op, tt::tt_metal::ReduceOpDim::H);
    if (use_post_mul) {
        reduce_defines["REDUCE_POST_MUL"] = "1";
    }
    KernelSpec::CompilerOptions::Defines compute_defines;
    for (const auto& [k, v] : reduce_defines) {
        compute_defines.emplace(k, v);
    }
    KernelSpec::CompilerOptions::Defines reader_defines = compute_defines;
    reader_defines.emplace("ENABLE_FP32_DEST_ACC", fp32_dest_acc_en ? "1" : "0");
    reader_defines.emplace("DST_SYNC_FULL", dst_full_sync_en ? "1" : "0");

    const std::filesystem::path kdir("ttnn/cpp/ttnn/operations/experimental/quasar/reduction/generic/device/kernels/");

    KernelSpec reader{
        .unique_id = READER,
        .source = kdir / "dataflow/reader_unary_transpose_wh_universal_input_cols_partitioned_metal2.cpp",
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = SCALER, .accessor_name = "scaler", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"}},
        .compile_time_args =
            {{"Ht", Ht}, {"Wt", Wt}, {"HtWt", HtWt}, {"scaler_bits", scaler_bits}, {"use_welford", 0u}},
        .runtime_arg_schema = {.runtime_arg_names = {"col_start_tile_id", "curr_col_in_batch", "num_cols"}},
        .hw_config =
            DataMovementHardwareConfig{
                .role = DataMovementRoleHint::READER,
                .gen2_config = DataMovementHardwareConfig::Gen2Config{.disable_dfb_implicit_sync_for_all = true}},
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = kdir / "dataflow/writer_unary_interleaved_start_id_metal2.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = OUTPUT, .accessor_name = "output"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_pages", "start_id"}},
        .hw_config =
            DataMovementHardwareConfig{
                .role = DataMovementRoleHint::WRITER,
                .gen2_config = DataMovementHardwareConfig::Gen2Config{.disable_dfb_implicit_sync_for_all = true}},
    };

    auto make_compute = [&](const KernelSpecName& id, uint32_t compute_Wt) {
        return KernelSpec{
            .unique_id = id,
            .source = kdir / "compute/reduce_metal2.cpp",
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings =
                {DFBBinding{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{
                     .dfb_spec_name = SCALER, .accessor_name = "scaler", .endpoint_type = DFBEndpointType::CONSUMER},
                 DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::PRODUCER}},
            .compile_time_args =
                {{"Ht", Ht}, {"Wt", compute_Wt}, {"NC", 1u}, {"post_mul_scaler_bits", post_mul_scaler_bits}},
            .hw_config =
                ComputeHardwareConfig{
                    .math_fidelity = math_fidelity,
                    .fp32_dest_acc_en = fp32_dest_acc_en,
                    .dst_full_sync_en = dst_full_sync_en},
        };
    };

    Group<KernelSpec> kernels = {reader, writer};
    Group<WorkUnitSpec> work_units;
    kernels.push_back(make_compute(COMPUTE_G1, num_cols_per_core_group_1));
    work_units.push_back(
        WorkUnitSpec{.name = "reduce_h_g1", .kernels = {READER, WRITER, COMPUTE_G1}, .target_nodes = core_group_1});
    const bool has_g2 = !core_group_2.ranges().empty();
    if (has_g2) {
        kernels.push_back(make_compute(COMPUTE_G2, num_cols_per_core_group_2));
        work_units.push_back(
            WorkUnitSpec{.name = "reduce_h_g2", .kernels = {READER, WRITER, COMPUTE_G2}, .target_nodes = core_group_2});
    }

    // ---- Per-core runtime args (mirror the legacy interleaved work-distribution loop) ----
    std::vector<CoreCoord> cores;
    if (operation_attributes.sub_core_grids.has_value()) {
        for (const auto& range : all_cores.ranges()) {
            for (int y = range.start_coord.y; y <= range.end_coord.y; ++y) {
                for (int x = range.start_coord.x; x <= range.end_coord.x; ++x) {
                    cores.emplace_back(x, y);
                }
            }
        }
    } else {
        cores = grid_to_cores(num_cores, compute_with_storage_grid_size.x, compute_with_storage_grid_size.y, false);
    }
    TT_FATAL(
        cores.size() == num_cores, "Resolved core list size {} must match split num_cores {}", cores.size(), num_cores);
    TT_FATAL(Wt != 0, "Width in tiles (Wt) must be non-zero (W={}, tile_width={})", W, tile_width);

    Group<KernelRunArgs::NodeRuntimeArgs> reader_node_args;
    Group<KernelRunArgs::NodeRuntimeArgs> writer_node_args;
    for (uint32_t i = 0, num_cols_read = 0; i < num_cores; i++) {
        const CoreCoord& core = cores[i];
        uint32_t num_cols_per_core = 0;
        if (core_group_1.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }
        reader_node_args.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = core,
            .args = {
                {"col_start_tile_id", (num_cols_read / Wt * HtWt) + (num_cols_read % Wt)},
                {"curr_col_in_batch", num_cols_read % Wt},
                {"num_cols", num_cols_per_core}}});
        writer_node_args.push_back(KernelRunArgs::NodeRuntimeArgs{
            .node = core, .args = {{"num_pages", num_cols_per_core}, {"start_id", num_cols_read}}});
        num_cols_read += num_cols_per_core;
        if (i == num_cores - 1) {
            TT_FATAL(
                num_cols_read == num_cols,
                "Reduce H assigned {} columns across cores, expected {}",
                num_cols_read,
                num_cols);
        }
    }

    ProgramSpec spec{
        .name = "reduce_multi_core_h",
        .kernels = std::move(kernels),
        .dataflow_buffers = {in_dfb, scaler_dfb, out_dfb},
        .tensor_parameters = {input_param, output_param},
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {
        KernelRunArgs{.kernel = READER, .runtime_arg_values = std::move(reader_node_args)},
        KernelRunArgs{.kernel = WRITER, .runtime_arg_values = std::move(writer_node_args)}};
    run_args.tensor_args = {{INPUT, a}, {OUTPUT, output}};

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim::qsr
