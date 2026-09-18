// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_backward_gamma_beta_grad_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_group_norm_backward {

ttnn::device_operation::ProgramArtifacts
MorehGroupNormBackwardGammaBetaGradOperation::MorehGroupNormBackwardGammaBetaGradFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    using namespace tt;
    using namespace tt::constants;
    using namespace tt::tt_metal;
    using namespace tt::tt_metal::experimental;

    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& mean = tensor_args.mean;
    const auto& rstd = tensor_args.rstd;

    // Bind straight to the vector elements: the framework resolves each TensorArgument to its tensor
    // by MeshTensor identity, so a copy would fail the match at dispatch.
    std::optional<Tensor>& gamma_grad = outputs[0];
    std::optional<Tensor>& beta_grad = outputs[1];
    auto num_groups = operation_attributes.num_groups;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = output_grad.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto output_grad_shape = output_grad.padded_shape();

    const auto n = output_grad_shape[0];
    const auto c = output_grad_shape[1];
    const auto h = output_grad_shape[2];
    const auto w = output_grad_shape[3];

    const auto origin_output_grad_shape = output_grad.logical_shape();

    const auto origin_h = origin_output_grad_shape[2];
    const auto origin_w = origin_output_grad_shape[3];

    const bool is_groupnorm = true;
    const bool is_lastdim_layernorm = false;

    const bool do_mask_h = (origin_h % TILE_HEIGHT) != 0;
    const bool do_mask_w = (origin_w % TILE_WIDTH) != 0;

    const auto Ht = h / TILE_HEIGHT;
    const auto Wt = w / TILE_WIDTH;

    const auto num_channels = c;

    const auto batch = n;
    const auto HtWt = Ht * Wt;
    const auto num_inner_tiles = batch * HtWt;  // inner_size

    const bool gamma_grad_has_value = gamma_grad.has_value();
    const bool beta_grad_has_value = beta_grad.has_value();

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const auto
        [num_cores_to_be_used,
         all_cores,
         core_group_1,
         core_group_2,
         num_channels_per_core_group_1,
         num_channels_per_core_group_2] = tt_metal::split_work_to_cores(grid, num_channels);

    log_debug(LogTest, "num_cores_to_be_used: {}", num_cores_to_be_used);
    log_debug(LogTest, "num_channels_per_core_group_1: {}", num_channels_per_core_group_1);
    log_debug(LogTest, "num_channels_per_core_group_2: {}", num_channels_per_core_group_2);

    ////////////////////////////////////////////////////////////////////////////
    //                         Resource names
    ////////////////////////////////////////////////////////////////////////////
    // Declared function-local rather than in an anonymous namespace: ttnn_op_moreh is a unity-build
    // target and this op's two factories share one namespace, so anonymous-namespace constants of the
    // same name would collide outright.
    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE_G1{"compute_g1"};
    const KernelSpecName COMPUTE_G2{"compute_g2"};

    const DFBSpecName DY{"dy"};          // output_grad(==dy)
    const DFBSpecName X{"x"};            // input(==x)
    const DFBSpecName MEAN{"mean"};      // mean
    const DFBSpecName RSTD{"rstd"};      // rstd
    const DFBSpecName SCALER{"scaler"};  // this op fills it with 1.0; the compute kernel reduces with it
    const DFBSpecName MASK_H{"mask_h"};  // mask_h
    const DFBSpecName MASK_W{"mask_w"};  // mask_w
    const DFBSpecName DGAMMA{"dgamma"};  // gamma_grad(==dgamma)
    const DFBSpecName DBETA{"dbeta"};    // beta_grad(==dbeta)
    const DFBSpecName Y{"y"};            // output(==y)
    const DFBSpecName YDY{"ydy"};        // y * dy
    const DFBSpecName DYADD{"dyadd"};    // Add[dy]
    const DFBSpecName YDYADD{"ydyadd"};  // Add[y * dy]
    const DFBSpecName XMM{"xmm"};        // x - mean
    const DFBSpecName DYCOPY{"dycopy"};  // dycopy

    const TensorParamName OUTPUT_GRAD{"output_grad"};
    const TensorParamName INPUT{"input"};
    const TensorParamName MEAN_T{"mean"};
    const TensorParamName RSTD_T{"rstd"};
    const TensorParamName GAMMA_GRAD{"gamma_grad"};
    const TensorParamName BETA_GRAD{"beta_grad"};

    ////////////////////////////////////////////////////////////////////////////
    //                         DataflowBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 1;                  // output_grad(==dy)
    const uint32_t in1_t = 1;                  // input(==x)
    const uint32_t in2_t = 1;                  // mean
    const uint32_t in3_t = 1;                  // rstd
    const uint32_t in4_t = 1;                  // one
    const uint32_t in5_t = do_mask_h ? 1 : 0;  // mask_h
    const uint32_t in6_t = do_mask_w ? 1 : 0;  // mask_w

    const uint32_t out0_t = gamma_grad_has_value ? 1 : 0;  // gamma_grad(==dgamma)
    const uint32_t out1_t = beta_grad_has_value ? 1 : 0;   // beta_grad(==dbeta)

    const uint32_t im0_t = 1;  // output(==y)
    const uint32_t im1_t = 1;  // y * dy
    const uint32_t im2_t = 1;  // Add[dy]
    const uint32_t im3_t = 1;  // Add[y * dy]
    const uint32_t im4_t = 1;  // x - mean
    const uint32_t im5_t = 1;  // dycopy

    const auto data_format = tt_metal::datatype_to_dataformat_converter(output_grad.dtype());
    const auto single_tile_size = tt::tile_size(data_format);

    Group<DataflowBufferSpec> dfbs;

    auto add_dfb = [&](const DFBSpecName& name, uint32_t num_tiles) {
        if (num_tiles == 0) {
            return;
        }
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = name,
            .entry_size = single_tile_size,
            .num_entries = num_tiles,
            .data_format_metadata = data_format,
        });
    };

    add_dfb(DY, in0_t);       // output_grad(==dy)
    add_dfb(X, in1_t);        // input(==x)
    add_dfb(MEAN, in2_t);     // mean
    add_dfb(RSTD, in3_t);     // rstd
    add_dfb(SCALER, in4_t);   // one
    add_dfb(MASK_H, in5_t);   // mask_h
    add_dfb(MASK_W, in6_t);   // mask_w
    add_dfb(DGAMMA, out0_t);  // gamma_grad(==dgamma)
    add_dfb(DBETA, out1_t);   // beta_grad(==dbeta)
    add_dfb(Y, im0_t);        // output(==y)
    add_dfb(YDY, im1_t);      // y * dy
    add_dfb(DYADD, im2_t);    // Add[dy]
    add_dfb(YDYADD, im3_t);   // Add[y * dy]
    add_dfb(XMM, im4_t);      // x - mean
    add_dfb(DYCOPY, im5_t);   // dycopy

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::string reader_kernel_file(
        "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/gamma_beta_grad/kernels/dataflow/"
        "reader_moreh_group_norm_backward_gamma_beta_grad.cpp");
    const std::string writer_kernel_file(
        "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/gamma_beta_grad/kernels/dataflow/"
        "writer_moreh_group_norm_backward_gamma_beta_grad.cpp");

    // gamma_grad_has_value / beta_grad_has_value / do_mask_h / do_mask_w each select whether a
    // resource is bound, so they reach the kernels as preprocessor defines: an unbound name does not
    // exist in the generated header, and a C++-level `if constexpr` would still name-look-up the
    // discarded branch. The borrowed compute kernel needs the same flags as the reader and writer.
    KernelSpec::CompilerOptions::Defines reader_defines{};
    KernelSpec::CompilerOptions::Defines writer_defines{};
    KernelSpec::CompilerOptions::Defines compute_defines{
        {"REDUCE_OP", "PoolType::SUM"},
        {"REDUCE_DIM", "ReduceDim::REDUCE_SCALAR"},
    };
    if (gamma_grad_has_value) {
        reader_defines["GAMMA_GRAD_HAS_VALUE"] = "1";
        writer_defines["GAMMA_GRAD_HAS_VALUE"] = "1";
        compute_defines["GAMMA_GRAD_HAS_VALUE"] = "1";
    }
    if (beta_grad_has_value) {
        writer_defines["BETA_GRAD_HAS_VALUE"] = "1";
        compute_defines["BETA_GRAD_HAS_VALUE"] = "1";
    }
    if (do_mask_h) {
        reader_defines["DO_MASK_H"] = "1";
        compute_defines["DO_MASK_H"] = "1";
    }
    if (do_mask_w) {
        reader_defines["DO_MASK_W"] = "1";
        compute_defines["DO_MASK_W"] = "1";
    }

    Group<DFBBinding> reader_dfb_bindings{
        DFBBinding{.dfb_spec_name = DY, .accessor_name = "output_grad", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = X, .accessor_name = "input", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = MEAN, .accessor_name = "mean", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = RSTD, .accessor_name = "rstd", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = SCALER, .accessor_name = "scaler", .endpoint_type = DFBEndpointType::PRODUCER},
    };
    if (do_mask_h) {
        reader_dfb_bindings.push_back(
            DFBBinding{.dfb_spec_name = MASK_H, .accessor_name = "mask_h", .endpoint_type = DFBEndpointType::PRODUCER});
    }
    if (do_mask_w) {
        reader_dfb_bindings.push_back(
            DFBBinding{.dfb_spec_name = MASK_W, .accessor_name = "mask_w", .endpoint_type = DFBEndpointType::PRODUCER});
    }

    KernelSpec reader{
        .unique_id = READER,
        .source = reader_kernel_file,
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings = reader_dfb_bindings,
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = OUTPUT_GRAD, .accessor_name = "output_grad"},
                TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
                TensorBinding{.tensor_parameter_name = MEAN_T, .accessor_name = "mean"},
                TensorBinding{.tensor_parameter_name = RSTD_T, .accessor_name = "rstd"},
            },
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"tile_offset",
                     "num_channels_per_core",
                     "num_inner_tiles",
                     "num_channels",
                     "num_groups",
                     "origin_h",
                     "origin_w"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    Group<DFBBinding> writer_dfb_bindings{};
    Group<TensorBinding> writer_tensor_bindings{};
    if (gamma_grad_has_value) {
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DGAMMA, .accessor_name = "gamma_grad", .endpoint_type = DFBEndpointType::CONSUMER});
        writer_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = GAMMA_GRAD, .accessor_name = "gamma_grad"});
    }
    if (beta_grad_has_value) {
        writer_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = DBETA, .accessor_name = "beta_grad", .endpoint_type = DFBEndpointType::CONSUMER});
        writer_tensor_bindings.push_back(
            TensorBinding{.tensor_parameter_name = BETA_GRAD, .accessor_name = "beta_grad"});
    }

    KernelSpec writer{
        .unique_id = WRITER,
        .source = writer_kernel_file,
        .compiler_options = {.defines = writer_defines},
        .dfb_bindings = writer_dfb_bindings,
        .tensor_bindings = writer_tensor_bindings,
        .runtime_arg_schema =
            {
                .runtime_arg_names = {"tile_offset", "num_channels_per_core", "num_inner_tiles", "batch"},
            },
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const std::string compute_kernel_file(
        "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
        "moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp");

    Group<DFBBinding> compute_dfb_bindings{
        DFBBinding{.dfb_spec_name = DY, .accessor_name = "dy", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = X, .accessor_name = "x", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = MEAN, .accessor_name = "mean", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = RSTD, .accessor_name = "rstd", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = SCALER, .accessor_name = "scaler", .endpoint_type = DFBEndpointType::CONSUMER},
        // Compute-only intermediates: one toucher, so the compute kernel is bound as both endpoints.
        DFBBinding{.dfb_spec_name = Y, .accessor_name = "y", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = Y, .accessor_name = "y", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = YDY, .accessor_name = "ydy", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = YDY, .accessor_name = "ydy", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = DYADD, .accessor_name = "dyadd", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = DYADD, .accessor_name = "dyadd", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = YDYADD, .accessor_name = "ydyadd", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = YDYADD, .accessor_name = "ydyadd", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = XMM, .accessor_name = "xmm", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = XMM, .accessor_name = "xmm", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = DYCOPY, .accessor_name = "dycopy", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = DYCOPY, .accessor_name = "dycopy", .endpoint_type = DFBEndpointType::CONSUMER},
    };
    if (do_mask_h) {
        compute_dfb_bindings.push_back(
            DFBBinding{.dfb_spec_name = MASK_H, .accessor_name = "mask_h", .endpoint_type = DFBEndpointType::CONSUMER});
    }
    if (do_mask_w) {
        compute_dfb_bindings.push_back(
            DFBBinding{.dfb_spec_name = MASK_W, .accessor_name = "mask_w", .endpoint_type = DFBEndpointType::CONSUMER});
    }
    if (gamma_grad_has_value) {
        compute_dfb_bindings.push_back(
            DFBBinding{.dfb_spec_name = DGAMMA, .accessor_name = "dgamma", .endpoint_type = DFBEndpointType::PRODUCER});
    }
    if (beta_grad_has_value) {
        compute_dfb_bindings.push_back(
            DFBBinding{.dfb_spec_name = DBETA, .accessor_name = "dbeta", .endpoint_type = DFBEndpointType::PRODUCER});
    }

    // The descriptor factory set an all-default ComputeConfigDescriptor{}, so the Gen1 config is built
    // directly and left at its own defaults. Routing this through the TTNN ComputeKernelConfig helper
    // would silently flip every field, because that helper's defaults favor performance while the
    // Metal struct's favor precision. With enable_32_bit_dest at its default false, the Float32
    // unpack_modes requirement does not apply, so the table stays empty.
    const ComputeHardwareConfig compute_hw = ComputeGen1Config{};

    auto make_compute = [&](const KernelSpecName& unique_id, uint32_t num_channels_per_core) {
        return KernelSpec{
            .unique_id = unique_id,
            .source = compute_kernel_file,
            // Legacy ComputeConfig defaults opt_level to O3, Metal 2.0's type-agnostic
            // CompilerOptions to O2; state it so the compute kernels keep their level.
            .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings = compute_dfb_bindings,
            .compile_time_args =
                {
                    {"num_cols_per_core", num_channels_per_core},
                    {"origin_H", origin_h},
                    {"origin_W", origin_w},
                    {"NCHt", num_inner_tiles},
                    {"Wt", Wt},
                    {"is_lastdim_layernorm", static_cast<uint32_t>(is_lastdim_layernorm)},
                    {"is_groupnorm", static_cast<uint32_t>(is_groupnorm)},
                },
            .hw_config = compute_hw,
        };
    };

    const bool has_core_group_2 = !core_group_2.ranges().empty();

    Group<KernelSpec> kernels{reader, writer, make_compute(COMPUTE_G1, num_channels_per_core_group_1)};
    if (has_core_group_2) {
        kernels.push_back(make_compute(COMPUTE_G2, num_channels_per_core_group_2));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Tensor parameters
    ////////////////////////////////////////////////////////////////////////////
    Group<TensorParameter> tensor_parameters{
        TensorParameter{.unique_id = OUTPUT_GRAD, .spec = output_grad.tensor_spec()},
        TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = MEAN_T, .spec = mean.tensor_spec()},
        TensorParameter{.unique_id = RSTD_T, .spec = rstd.tensor_spec()},
    };
    if (gamma_grad_has_value) {
        tensor_parameters.push_back(TensorParameter{.unique_id = GAMMA_GRAD, .spec = gamma_grad->tensor_spec()});
    }
    if (beta_grad_has_value) {
        tensor_parameters.push_back(TensorParameter{.unique_id = BETA_GRAD, .spec = beta_grad->tensor_spec()});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Work units
    ////////////////////////////////////////////////////////////////////////////
    Group<WorkUnitSpec> work_units{
        WorkUnitSpec{.name = "wu_g1", .kernels = {READER, WRITER, COMPUTE_G1}, .target_nodes = core_group_1},
    };
    if (has_core_group_2) {
        work_units.push_back(
            WorkUnitSpec{.name = "wu_g2", .kernels = {READER, WRITER, COMPUTE_G2}, .target_nodes = core_group_2});
    }

    ProgramSpec spec{
        .name = "moreh_group_norm_backward_gamma_beta_grad",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    KernelRunArgs reader_run_args{.kernel = READER};
    KernelRunArgs writer_run_args{.kernel = WRITER};

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        NodeCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_channels_per_core;
        if (core_group_1.contains(core)) {
            num_channels_per_core = num_channels_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_channels_per_core = num_channels_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // reader
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"tile_offset", tile_offset},
             {"num_channels_per_core", num_channels_per_core},
             {"num_inner_tiles", num_inner_tiles},
             {"num_channels", num_channels},
             {"num_groups", num_groups},
             {"origin_h", origin_h},
             {"origin_w", origin_w}});

        // writer
        AddRuntimeArgsForNode(
            writer_run_args.runtime_arg_values,
            core,
            {{"tile_offset", tile_offset},
             {"num_channels_per_core", num_channels_per_core},
             {"num_inner_tiles", num_inner_tiles},
             {"batch", batch}});

        tile_offset += num_channels_per_core * HtWt;
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {OUTPUT_GRAD, output_grad.mesh_tensor()},
        {INPUT, input.mesh_tensor()},
        {MEAN_T, mean.mesh_tensor()},
        {RSTD_T, rstd.mesh_tensor()},
    };
    if (gamma_grad_has_value) {
        run_args.tensor_args.emplace(GAMMA_GRAD, gamma_grad->mesh_tensor());
    }
    if (beta_grad_has_value) {
        run_args.tensor_args.emplace(BETA_GRAD, beta_grad->mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_group_norm_backward
