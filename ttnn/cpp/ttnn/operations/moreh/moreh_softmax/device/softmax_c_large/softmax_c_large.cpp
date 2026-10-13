// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>

#include "ttnn/operations/moreh/moreh_softmax/device/moreh_softmax_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <ttnn/metal_v2_artifacts.hpp>
#include <cstdint>

namespace ttnn::operations::moreh::moreh_softmax {

using namespace tt::tt_metal::experimental;

ttnn::device_operation::ProgramArtifacts MorehSoftmaxOperation::MorehSoftmaxCLargeFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output) {
    using namespace tt;
    using namespace tt::tt_metal;

    log_info(tt::LogTest, "Large tensor algorithm selected");
    const auto& input = tensor_args.input;
    const auto& input_mt = input.mesh_tensor();
    const auto& output_mt = output.mesh_tensor();
    const auto dim = operation_attributes.dim;
    const auto op = operation_attributes.op;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    auto* device = input.device();
    auto grid_coord = device->compute_with_storage_grid_size();
    const CoreRange core_range({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});
    // split work
    auto shape = input.padded_shape();
    auto H = shape[-2];
    auto W = shape[-1];
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;

    std::uint32_t num_tiles = input.physical_volume() / shape[dim] / H / W * Ht * Wt;

    std::uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores_wt_core_range(core_range, num_tiles);

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    if (input.dtype() == DataType::FLOAT32 && !fp32_dest_acc_en) {
        TT_THROW(
            "FP32 destination accumulation must be enabled when input tensor has FLOAT32 data type. Please update the "
            "compute kernel configuration.");
    }

    // create circular buffers
    auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    auto intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;
    const std::uint32_t tile_size_data = tile_size(data_format);
    const std::uint32_t tile_size_intermed = tile_size(intermed_data_format);

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};

    const TensorParamName SRC{"src"};
    const TensorParamName DST{"dst"};

    const DFBSpecName IN{"in"};
    const DFBSpecName OUT{"out"};
    const DFBSpecName EXPS{"exps"};
    const DFBSpecName RECIP{"recip_sum_exps"};
    const DFBSpecName ADD{"add"};
    const DFBSpecName MAX{"max"};
    const DFBSpecName TMP{"tmp"};

    Group<DataflowBufferSpec> dfbs = {
        DataflowBufferSpec{
            .unique_id = IN, .entry_size = tile_size_data, .num_entries = 2, .data_format_metadata = data_format},
        DataflowBufferSpec{
            .unique_id = OUT, .entry_size = tile_size_data, .num_entries = 2, .data_format_metadata = data_format},
        DataflowBufferSpec{
            .unique_id = EXPS,
            .entry_size = tile_size_intermed,
            .num_entries = 1,
            .data_format_metadata = intermed_data_format},
        DataflowBufferSpec{
            .unique_id = RECIP,
            .entry_size = tile_size_intermed,
            .num_entries = 1,
            .data_format_metadata = intermed_data_format},
        DataflowBufferSpec{
            .unique_id = ADD,
            .entry_size = tile_size_intermed,
            .num_entries = 2,
            .data_format_metadata = intermed_data_format},
        // max uses the input data_format (matches legacy CBIndex::c_27 in this factory)
        DataflowBufferSpec{
            .unique_id = MAX, .entry_size = tile_size_data, .num_entries = 1, .data_format_metadata = data_format},
        DataflowBufferSpec{
            .unique_id = TMP,
            .entry_size = tile_size_intermed,
            .num_entries = 1,
            .data_format_metadata = intermed_data_format},
    };

    // create read/write kernel
    KernelSpec reader{
        .unique_id = READER,
        .source = "ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/reader_moreh_softmax_c_large.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = IN, .accessor_name = "in", .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = SRC, .accessor_name = "src"}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"num_tiles", "tile_offset", "outer_stride", "inner_size", "dim_size"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = "ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/writer_moreh_softmax_c_large.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = OUT, .accessor_name = "out", .endpoint_type = DFBEndpointType::CONSUMER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = DST, .accessor_name = "dst"}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"num_tiles", "tile_offset", "outer_stride", "inner_size", "dim_size"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    auto outer_stride = Ht * Wt;
    for (int i = dim; i < shape.rank() - 2; i++) {
        outer_stride *= shape[i];
    }
    auto dim_size = shape[dim];
    auto inner_size = outer_stride / dim_size;

    KernelSpec::CompilerOptions::Defines compute_defines;
    if (op == MorehSoftmaxOp::SOFTMAX || op == MorehSoftmaxOp::LOGSOFTMAX) {
        compute_defines["SOFTMAX"] = "1";
    } else {
        compute_defines["SOFTMIN"] = "1";
    }
    if (op == MorehSoftmaxOp::LOGSOFTMAX) {
        compute_defines["LOG"] = "1";
    }
    if (fp32_dest_acc_en) {
        compute_defines["FP32_DEST_ACC_EN"] = "1";
    }

    // create compute kernel
    auto make_compute_hw = [&]() {
        auto hw = ttnn::to_compute_hardware_config(compute_kernel_config);
        if (fp32_dest_acc_en) {
            hw.unpack_modes = {
                {IN, tt::tt_metal::UnpackMode::UnpackToSrc},
                {EXPS, tt::tt_metal::UnpackMode::UnpackToSrc},
                {RECIP, tt::tt_metal::UnpackMode::UnpackToSrc},
                {ADD, tt::tt_metal::UnpackMode::UnpackToSrc},
                {MAX, tt::tt_metal::UnpackMode::UnpackToSrc},
                {TMP, tt::tt_metal::UnpackMode::UnpackToSrc},
            };
        }
        return hw;
    };

    auto compute_dfb_bindings = [&]() {
        return Group<DFBBinding>{
            DFBBinding{.dfb_spec_name = IN, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = OUT, .accessor_name = "out0", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{.dfb_spec_name = EXPS, .accessor_name = "exps", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{.dfb_spec_name = EXPS, .accessor_name = "exps", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{
                .dfb_spec_name = RECIP, .accessor_name = "recip_sum_exps", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{
                .dfb_spec_name = RECIP, .accessor_name = "recip_sum_exps", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = ADD, .accessor_name = "add", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{.dfb_spec_name = ADD, .accessor_name = "add", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = MAX, .accessor_name = "max", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{.dfb_spec_name = MAX, .accessor_name = "max", .endpoint_type = DFBEndpointType::CONSUMER},
            DFBBinding{.dfb_spec_name = TMP, .accessor_name = "tmp", .endpoint_type = DFBEndpointType::PRODUCER},
            DFBBinding{.dfb_spec_name = TMP, .accessor_name = "tmp", .endpoint_type = DFBEndpointType::CONSUMER},
        };
    };

    auto make_compute = [&](const KernelSpecName& id, std::uint32_t N) {
        return KernelSpec{
            .unique_id = id,
            .source = "ttnn/cpp/ttnn/operations/moreh/moreh_softmax/device/kernels/moreh_softmax_c_large.cpp",
            .compiler_options = {.defines = compute_defines, .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
            .dfb_bindings = compute_dfb_bindings(),
            .compile_time_args = {{"N", N}, {"dim_size", static_cast<std::uint32_t>(dim_size)}},
            .hw_config = make_compute_hw(),
        };
    };

    bool has_core_group_2 = !core_group_2.ranges().empty();

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
        .name = "moreh_softmax_c_large",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters =
            {TensorParameter{.unique_id = SRC, .spec = input.tensor_spec()},
             TensorParameter{.unique_id = DST, .spec = output.tensor_spec()}},
        .work_units = std::move(work_units),
    };

    // Set Runtime Args
    ProgramRunArgs run_args;
    KernelRunArgs reader_ra{.kernel = READER};
    KernelRunArgs writer_ra{.kernel = WRITER};

    auto core_x_offset = core_range.start_coord.x;
    auto core_y_offset = core_range.start_coord.y;

    for (std::uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {(i / core_h) + core_x_offset, (i % core_h) + core_y_offset};
        std::uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        AddRuntimeArgsForNode(
            reader_ra.runtime_arg_values,
            core,
            {{"num_tiles", num_tiles_per_core},
             {"tile_offset", tile_offset},
             {"outer_stride", outer_stride},
             {"inner_size", inner_size},
             {"dim_size", static_cast<std::uint32_t>(dim_size)}});
        AddRuntimeArgsForNode(
            writer_ra.runtime_arg_values,
            core,
            {{"num_tiles", num_tiles_per_core},
             {"tile_offset", tile_offset},
             {"outer_stride", outer_stride},
             {"inner_size", inner_size},
             {"dim_size", static_cast<std::uint32_t>(dim_size)}});

        tile_offset += num_tiles_per_core;
    }

    run_args.kernel_run_args = {std::move(reader_ra), std::move(writer_ra)};
    run_args.tensor_args.emplace(SRC, input_mt);
    run_args.tensor_args.emplace(DST, output_mt);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}
}  // namespace ttnn::operations::moreh::moreh_softmax
