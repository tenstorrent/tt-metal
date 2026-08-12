// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_device_operation.hpp"

#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <ttnn/metal_v2_artifacts.hpp>

#include <string>
#include <utility>
#include <cstdint>

namespace ttnn::prim {

using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts
SoftmaxDeviceOperation::SoftmaxProgramFactoryGeneralHSmall::create_program_artifacts(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    using namespace tt::tt_metal;

    log_debug(tt::LogMetal, "SoftmaxProgramFactoryGeneralHSmall selected");

    const auto& input = tensor_args.input_tensor;
    const auto& input_mt = input.mesh_tensor();
    const auto& output_mt = output_tensor.mesh_tensor();
    const auto& compute_kernel_config = attributes.compute_kernel_config;
    auto* const device = input.device();
    const auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange core_range({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});
    const std::uint32_t tile_height = input.tensor_spec().tile().get_height();
    const std::uint32_t tile_width = input.tensor_spec().tile().get_width();
    const auto shape = input.padded_shape();
    const auto H = shape[-2];
    const auto W = shape[-1];
    const auto Ht = H / tile_height;
    const auto Wt = W / tile_width;

    // Work split
    const auto num = input.physical_volume() / H / W;
    const std::uint32_t num_cols_tiles = num * Wt;
    const std::uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;
    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        operations::split_work_to_cores_wt_core_range(core_range, num_cols_tiles);

    const auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    if (input.dtype() == DataType::FLOAT32 && !fp32_dest_acc_en) {
        TT_THROW(
            "FP32 destination accumulation must be enabled when input tensor has FLOAT32 data type. Please update the "
            "compute kernel configuration.");
    }

    const auto data_format = datatype_to_dataformat_converter(input.dtype());
    // Use Float16_b for intermediates when not accumulating in fp32, matching the AttentionOptimized path.
    // This avoids using Bfp8_b for intermediate computations where it lacks precision (issue #32934).
    const auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    // Reader generates mask/scaler with uint16_t (1024 elements = 2048 bytes). Use Float16_b for these CBs when
    // input is Bfp8_b so tile size matches; Bfp8_b tile layout is smaller and would be overflowed (issue #32934).
    const auto mask_scaler_format = (data_format == tt::DataFormat::Bfp8_b) ? tt::DataFormat::Float16_b : data_format;

    const std::uint32_t in_tile_size = tt::tile_size(data_format);
    const std::uint32_t mask_scaler_tile_size = tt::tile_size(mask_scaler_format);
    const std::uint32_t intermed_tile_size = tt::tile_size(intermed_data_format);

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};

    const TensorParamName SRC{"src"};
    const TensorParamName DST{"dst"};

    const DFBSpecName IN{"in"};
    const DFBSpecName MASK{"mask"};
    const DFBSpecName MAX_SCALER{"max_scaler"};
    const DFBSpecName SUM_SCALER{"sum_scaler"};
    const DFBSpecName OUT{"out"};
    const DFBSpecName EXPS{"exps"};
    const DFBSpecName RECIP{"recip_sum_exps"};
    const DFBSpecName MAX{"max"};
    const DFBSpecName X_MINUS_MAX{"x_minus_max"};
    const DFBSpecName TMP{"tmp"};

    // Circular buffers
    Group<DataflowBufferSpec> dfbs = {
        DataflowBufferSpec{
            .unique_id = IN, .entry_size = in_tile_size, .num_entries = Ht, .data_format_metadata = data_format},
        DataflowBufferSpec{
            .unique_id = MASK,
            .entry_size = mask_scaler_tile_size,
            .num_entries = 1,
            .data_format_metadata = mask_scaler_format},
        DataflowBufferSpec{
            .unique_id = MAX_SCALER,
            .entry_size = mask_scaler_tile_size,
            .num_entries = 1,
            .data_format_metadata = mask_scaler_format},
        DataflowBufferSpec{
            .unique_id = SUM_SCALER,
            .entry_size = mask_scaler_tile_size,
            .num_entries = 1,
            .data_format_metadata = mask_scaler_format},
        DataflowBufferSpec{
            .unique_id = OUT, .entry_size = in_tile_size, .num_entries = Ht, .data_format_metadata = data_format},
        DataflowBufferSpec{
            .unique_id = EXPS,
            .entry_size = intermed_tile_size,
            .num_entries = Ht,
            .data_format_metadata = intermed_data_format},
        // reduce
        DataflowBufferSpec{
            .unique_id = RECIP,
            .entry_size = intermed_tile_size,
            .num_entries = 1,
            .data_format_metadata = intermed_data_format},
        DataflowBufferSpec{
            .unique_id = MAX,
            .entry_size = intermed_tile_size,
            .num_entries = 1,
            .data_format_metadata = intermed_data_format},
        DataflowBufferSpec{
            .unique_id = X_MINUS_MAX,
            .entry_size = intermed_tile_size,
            .num_entries = Ht,
            .data_format_metadata = intermed_data_format},
        DataflowBufferSpec{
            .unique_id = TMP,
            .entry_size = intermed_tile_size,
            .num_entries = 1,
            .data_format_metadata = intermed_data_format},
    };

    // Data movement kernel
    KernelSpec reader{
        .unique_id = READER,
        .source = std::string(SOFTMAX_KERNEL_PATH_GENERAL) + "/reader_moreh_softmax_h.cpp",
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{.dfb_spec_name = MASK, .accessor_name = "mask", .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = MAX_SCALER,
                 .accessor_name = "max_scaler",
                 .endpoint_type = DFBEndpointType::PRODUCER},
             DFBBinding{
                 .dfb_spec_name = SUM_SCALER,
                 .accessor_name = "sum_scaler",
                 .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = SRC, .accessor_name = "src"}},
        .compile_time_args = {{"is_fp32", static_cast<std::uint32_t>(input.dtype() == DataType::FLOAT32)}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_rows", "tile_offset", "Ht", "Wt", "mask_h"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = std::string(SOFTMAX_KERNEL_PATH_GENERAL) + "/writer_moreh_softmax_h.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = DST, .accessor_name = "dst"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_rows", "tile_offset", "Ht", "Wt"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    // Compute kernel
    KernelSpec::CompilerOptions::Defines compute_defines;
    compute_defines["SOFTMAX"] = "1";
    // Enable FP32_DEST_ACC_EN for format reconfiguration in moreh compute helpers when using mixed
    // data formats (Bfp8_b input with Float16_b intermediates/mask/scaler) (issue #32934).
    if (fp32_dest_acc_en || data_format == tt::DataFormat::Bfp8_b) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    auto make_compute_hw = [&]() {
        auto hw = ttnn::to_compute_hardware_config(arch, compute_kernel_config);
        if (fp32_dest_acc_en) {
            std::get<ComputeGen1Config>(hw).unpack_modes = {
                {IN, tt::tt_metal::UnpackMode::UnpackToSrc},
                {MASK, tt::tt_metal::UnpackMode::UnpackToSrc},
                {MAX_SCALER, tt::tt_metal::UnpackMode::UnpackToSrc},
                {SUM_SCALER, tt::tt_metal::UnpackMode::UnpackToSrc},
                {EXPS, tt::tt_metal::UnpackMode::UnpackToSrc},
                {RECIP, tt::tt_metal::UnpackMode::UnpackToSrc},
                {MAX, tt::tt_metal::UnpackMode::UnpackToSrc},
                {X_MINUS_MAX, tt::tt_metal::UnpackMode::UnpackToSrc},
                {TMP, tt::tt_metal::UnpackMode::UnpackToSrc},
            };
        }
        return hw;
    };

    auto compute_dfb_bindings = [&]() {
        return Group<DFBBinding>{
            DFBBinding{.dfb_spec_name = IN, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = MASK, .accessor_name = "mask", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = MAX_SCALER, .accessor_name = "max_scaler", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = SUM_SCALER, .accessor_name = "sum_scaler", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out0", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{.dfb_spec_name = EXPS, .accessor_name = "exps", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{.dfb_spec_name = EXPS, .accessor_name = "exps", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = RECIP, .accessor_name = "recip_sum_exps", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{
                .dfb_spec_name = RECIP, .accessor_name = "recip_sum_exps", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = MAX, .accessor_name = "max", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{.dfb_spec_name = MAX, .accessor_name = "max", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = X_MINUS_MAX,
                .accessor_name = "x_minus_max",
                .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{
                .dfb_spec_name = X_MINUS_MAX,
                .accessor_name = "x_minus_max",
                .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = TMP, .accessor_name = "tmp", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{.dfb_spec_name = TMP, .accessor_name = "tmp", .endpoint_type = DFBEndpointType::CONSUMER},
        };
    };

    auto make_compute = [&](const KernelSpecName& id, std::uint32_t N) {
        return KernelSpec{
            .unique_id = id,
            .source = std::string(SOFTMAX_KERNEL_PATH_GENERAL) + "/moreh_softmax_h.cpp",
            .compiler_options = {.defines = compute_defines, .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
            .dfb_bindings = compute_dfb_bindings(),
            .compile_time_args = {{"N", N}, {"Ht", Ht}},
            .hw_config = make_compute_hw(),
        };
    };

    bool has_core_group_2 = num_tiles_per_core_group_2 > 0;

    Group<KernelSpec> kernels = {reader, writer, make_compute(COMPUTE_G1, num_tiles_per_core_group_1)};
    if (has_core_group_2) {
        kernels.push_back(make_compute(COMPUTE_G2, num_tiles_per_core_group_2));
    }

    Group<WorkUnitSpec> work_units;
    work_units.push_back(
        WorkUnitSpec{.name = "wu_g1", .kernels = {READER, WRITER, COMPUTE_G1}, .target_nodes = core_group_1});
    if (has_core_group_2) {
        work_units.push_back(
            WorkUnitSpec{.name = "wu_g2", .kernels = {READER, WRITER, COMPUTE_G2}, .target_nodes = core_group_2});
    }

    ProgramSpec spec{
        .name = "softmax_general_h_small",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {TensorParameter{.unique_id = SRC, .spec = input.tensor_spec()},
             TensorParameter{.unique_id = DST, .spec = output_tensor.tensor_spec()}},
        .work_units = std::move(work_units),
    };

    // Runtime Args
    ProgramRunArgs run_args;
    KernelRunArgs reader_ra{.kernel = READER};
    KernelRunArgs writer_ra{.kernel = WRITER};

    const auto core_x_offset = core_range.start_coord.x;
    const auto core_y_offset = core_range.start_coord.y;

    std::uint32_t mask_h = input.logical_shape()[-2] % tile_height;
    if (mask_h == 0) {
        mask_h = tile_height;
    }

    for (std::uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        const CoreCoord core = {(i / core_h) + core_x_offset, (i % core_h) + core_y_offset};
        std::uint32_t num_tiles_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        // Reader computes the reduce scaler in-kernel; only shape-derived args are passed.
        AddRuntimeArgsForNode(
            reader_ra.runtime_arg_values,
            core,
            {{"num_rows", num_tiles_per_core},
             {"tile_offset", tile_offset},
             {"Ht", Ht},
             {"Wt", Wt},
             {"mask_h", mask_h}});
        AddRuntimeArgsForNode(
            writer_ra.runtime_arg_values,
            core,
            {{"num_rows", num_tiles_per_core}, {"tile_offset", tile_offset}, {"Ht", Ht}, {"Wt", Wt}});

        tile_offset += num_tiles_per_core;
    }

    run_args.kernel_run_args = {std::move(reader_ra), std::move(writer_ra)};
    run_args.tensor_args.emplace(SRC, input_mt);
    run_args.tensor_args.emplace(DST, output_mt);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::prim
