// SPDX-FileCopyrightText: © 2023 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <utility>

#include "ttnn/operations/moreh/moreh_softmax_backward/device/moreh_softmax_backward_device_operation.hpp"
#include "ttnn/operations/moreh/moreh_softmax_backward/device/moreh_softmax_backward_metal2_common.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>

namespace ttnn::operations::moreh::moreh_softmax_backward {

ttnn::device_operation::ProgramArtifacts
MorehSoftmaxBackwardOperation::MorehSoftmaxBackwardCLargeFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& input_grad_tensor) {
    using namespace tt;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;
    using namespace metal2;

    log_info(tt::LogTest, "Large tensor algorithm selected");
    const auto& output = tensor_args.output_tensor.mesh_tensor();
    const auto& output_grad = tensor_args.output_grad_tensor.mesh_tensor();
    const auto& input_grad = input_grad_tensor.mesh_tensor();
    const auto dim = operation_attributes.dim;
    const auto op = operation_attributes.op;
    const auto& compute_kernel_config = operation_attributes.compute_kernel_config;

    const auto& device = output_grad.device();
    auto grid_coord = device.compute_with_storage_grid_size();
    const CoreRange core_range({0, 0}, {grid_coord.x - 1, grid_coord.y - 1});
    // split work
    auto shape = input_grad.padded_shape();
    auto H = shape[-2];
    auto W = shape[-1];
    auto Ht = H / tt::constants::TILE_HEIGHT;
    auto Wt = W / tt::constants::TILE_WIDTH;

    uint32_t num_tiles = input_grad.physical_volume() / shape[dim] / H / W * Ht * Wt;

    uint32_t core_h = core_range.end_coord.y - core_range.start_coord.y + 1;

    auto [num_cores, all_cores, core_group_1, core_group_2, num_tiles_per_core_group_1, num_tiles_per_core_group_2] =
        split_work_to_cores_wt_core_range(core_range, num_tiles);

    const bool fp32_dest_acc_en = compute_kernel_config.fp32_dest_acc_en;

    // dataflow buffer formats
    tt::DataFormat data_format = tt::tt_metal::datatype_to_dataformat_converter(input_grad.dtype());
    tt::DataFormat intermed_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : data_format;
    const uint32_t tile_size_data = tile_size(data_format);
    const uint32_t tile_size_intermed = tile_size(intermed_data_format);

    // create read/write kernel
    KernelSpec::CompilerOptions::Defines reader_defines;
    if (op == MorehSoftmaxBackwardOp::LOGSOFTMAX) {
        reader_defines.emplace("LOG", "1");
    }

    KernelSpec reader_spec{
        .unique_id = READER_KERNEL,
        .source =
            "ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/"
            "reader_moreh_softmax_backward_c.cpp",
        .compiler_options = {.defines = std::move(reader_defines)},
        .dfb_bindings =
            {DFBBinding{
                 .dfb_spec_name = Y_DFB,
                 .accessor_name = "y",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             },
             DFBBinding{
                 .dfb_spec_name = DY_DFB,
                 .accessor_name = "dy",
                 .endpoint_type = DFBEndpointType::PRODUCER,
             }},
        .tensor_bindings =
            {TensorBinding{.tensor_parameter_name = OUTPUT_TENSOR, .accessor_name = "y"},
             TensorBinding{.tensor_parameter_name = OUTPUT_GRAD_TENSOR, .accessor_name = "dy"}},
        .runtime_arg_schema =
            {.runtime_arg_names = {"num_tiles", "tile_offset", "outer_stride", "inner_size", "dim_size"}},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer_spec{
        .unique_id = WRITER_KERNEL,
        .source =
            "ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/"
            "writer_moreh_softmax_backward_c.cpp",
        .dfb_bindings = {DFBBinding{
            .dfb_spec_name = DX_DFB,
            .accessor_name = "out",
            .endpoint_type = DFBEndpointType::CONSUMER,
        }},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = INPUT_GRAD_TENSOR, .accessor_name = "dx"}},
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

    // create compute kernel
    auto compute_hw_config = ttnn::to_compute_hardware_config(compute_kernel_config);
    compute_hw_config.unpack_modes = MakeUnpackModes(
        fp32_dest_acc_en,
        {{Y_DFB, data_format},
         {DY_DFB, data_format},
         {YDY_DFB, intermed_data_format},
         {SUM_DFB, intermed_data_format},
         {DY_M_SUM_DFB, intermed_data_format}});
    const auto compute_defines = MakeComputeDefines(op, fp32_dest_acc_en);

    // One KernelSpec per core group, each baking in that group's per-core tile count. The groups
    // cover disjoint nodes, so every node still runs exactly one compute instance.
    auto make_compute_spec = [&](const KernelSpecName& unique_id, uint32_t num_tiles_per_core) {
        return KernelSpec{
            .unique_id = unique_id,
            .source = "ttnn/cpp/ttnn/operations/moreh/moreh_softmax_backward/device/kernels/"
                      "moreh_softmax_backward_c_large.cpp",
            // Legacy ComputeConfigDescriptor left opt_level unset, which resolves to O3 for a
            // compute kernel; Metal 2.0's type-agnostic default is O2, so state O3 here.
            .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings =
                {DFBBinding{
                     .dfb_spec_name = Y_DFB,
                     .accessor_name = "y",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = DY_DFB,
                     .accessor_name = "dy",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = DX_DFB,
                     .accessor_name = "dx",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 },
                 // The three intermediates never leave the compute kernel: it is their only
                 // toucher, so each is bound at both ends of its own FIFO (a self-loop).
                 DFBBinding{
                     .dfb_spec_name = YDY_DFB,
                     .accessor_name = "ydy",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 },
                 DFBBinding{
                     .dfb_spec_name = YDY_DFB,
                     .accessor_name = "ydy",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = SUM_DFB,
                     .accessor_name = "sum",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 },
                 DFBBinding{
                     .dfb_spec_name = SUM_DFB,
                     .accessor_name = "sum",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 },
                 DFBBinding{
                     .dfb_spec_name = DY_M_SUM_DFB,
                     .accessor_name = "dy_m_sum",
                     .endpoint_type = DFBEndpointType::PRODUCER,
                 },
                 DFBBinding{
                     .dfb_spec_name = DY_M_SUM_DFB,
                     .accessor_name = "dy_m_sum",
                     .endpoint_type = DFBEndpointType::CONSUMER,
                 }},
            .compile_time_args = {{"N", num_tiles_per_core}, {"dim_size", dim_size}},
            .hw_config = compute_hw_config,
        };
    };

    // Set Runtime Args
    KernelRunArgs reader_run_args{.kernel = READER_KERNEL};
    KernelRunArgs writer_run_args{.kernel = WRITER_KERNEL};

    auto core_x_offset = core_range.start_coord.x;
    auto core_y_offset = core_range.start_coord.y;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; i++) {
        CoreCoord core = {(i / core_h) + core_x_offset, (i % core_h) + core_y_offset};
        uint32_t num_tiles_per_core;
        if (core_group_1.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tiles_per_core = num_tiles_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"num_tiles", num_tiles_per_core},
             {"tile_offset", tile_offset},
             {"outer_stride", outer_stride},
             {"inner_size", inner_size},
             {"dim_size", dim_size}});

        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"num_tiles", num_tiles_per_core},
             {"tile_offset", tile_offset},
             {"outer_stride", outer_stride},
             {"inner_size", inner_size},
             {"dim_size", dim_size}});

        tile_offset += num_tiles_per_core;
    }

    ProgramSpec spec{
        .name = "moreh_softmax_backward_c_large",
        .kernels = {reader_spec, writer_spec, make_compute_spec(COMPUTE_KERNEL_G1, num_tiles_per_core_group_1)},
        .dataflow_buffers =
            {MakeDFB(Y_DFB, 2, tile_size_data, data_format),
             MakeDFB(DY_DFB, 2, tile_size_data, data_format),
             MakeDFB(DX_DFB, 2, tile_size_data, data_format),
             MakeDFB(YDY_DFB, 1, tile_size_intermed, intermed_data_format),
             MakeDFB(SUM_DFB, 2, tile_size_intermed, intermed_data_format),
             MakeDFB(DY_M_SUM_DFB, 1, tile_size_intermed, intermed_data_format)},
        .tensor_parameters =
            {TensorParameter{.unique_id = OUTPUT_TENSOR, .spec = output.tensor_spec()},
             TensorParameter{.unique_id = OUTPUT_GRAD_TENSOR, .spec = output_grad.tensor_spec()},
             TensorParameter{.unique_id = INPUT_GRAD_TENSOR, .spec = input_grad.tensor_spec()}},
        .work_units = {WorkUnitSpec{
            .name = "core_group_1",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL_G1},
            .target_nodes = core_group_1,
        }},
    };

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {OUTPUT_TENSOR, TensorArgument{output}},
        {OUTPUT_GRAD_TENSOR, TensorArgument{output_grad}},
        {INPUT_GRAD_TENSOR, TensorArgument{input_grad}},
    };

    if (!core_group_2.ranges().empty()) {
        spec.kernels.push_back(make_compute_spec(COMPUTE_KERNEL_G2, num_tiles_per_core_group_2));
        spec.work_units.push_back(WorkUnitSpec{
            .name = "core_group_2",
            .kernels = {READER_KERNEL, WRITER_KERNEL, COMPUTE_KERNEL_G2},
            .target_nodes = core_group_2,
        });
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_softmax_backward
