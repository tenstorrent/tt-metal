// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moreh_group_norm_backward_input_grad_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_group_norm_backward {

ttnn::device_operation::ProgramArtifacts
MorehGroupNormBackwardInputGradOperation::MorehGroupNormBackwardInputGradFactory::create_program_artifacts(
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

    const auto& input_grad = outputs;
    const auto& gamma = tensor_args.gamma;
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

    const bool is_lastdim_layernorm = false;
    const bool is_groupnorm = true;

    const bool do_mask_h = (origin_h % TILE_HEIGHT) != 0;
    const bool do_mask_w = (origin_w % TILE_WIDTH) != 0;

    const auto Ht = h / TILE_HEIGHT;
    const auto Wt = w / TILE_WIDTH;

    const auto num_channels = c;
    const auto num_rows = n * num_groups;
    const auto num_inner_tiles = (num_channels / num_groups) * Ht * Wt;

    const bool gamma_has_value = gamma.has_value();

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
         num_rows_per_core_group_1,
         num_rows_per_core_group_2] = tt_metal::split_work_to_cores(grid, num_rows);

    log_debug(LogTest, "num_cores_to_be_used: {}", num_cores_to_be_used);
    log_debug(LogTest, "num_rows_per_core_group_1: {}", num_rows_per_core_group_1);
    log_debug(LogTest, "num_rows_per_core_group_2: {}", num_rows_per_core_group_2);

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

    const DFBSpecName DY{"dy"};                    // output_grad(==dy)
    const DFBSpecName X{"x"};                      // input(==x)
    const DFBSpecName MEAN{"mean"};                // mean
    const DFBSpecName RSTD{"rstd"};                // rstd
    const DFBSpecName SCALER{"scaler"};            // this op fills it with 1.0; the compute kernel reduces with it
    const DFBSpecName N_RECIP_N{"n_recip_n"};      // inner_size(==n) and its reciprocal
    const DFBSpecName GAMMA{"gamma"};              // gamma
    const DFBSpecName MASK_H_W{"mask_h_w"};        // mask_h_w
    const DFBSpecName DX{"dx"};                    // input_grad(==dx)
    const DFBSpecName DYCOPY{"dycopy"};            // copy output_grad(==dycopy)
    const DFBSpecName Y{"y"};                      // output(==y)
    const DFBSpecName DYSUM{"dysum"};              // Sum[dy]
    const DFBSpecName YDYSUM{"ydysum"};            // Sum[y * dy]
    const DFBSpecName RECIP_NRSTD{"recip_nrstd"};  // rstd / n — a buffer of its own on the small path
    const DFBSpecName TMP1{"tmp1"};                // scratch; the compute kernel reaches it under
    const DFBSpecName TMP2{"tmp2"};                // several working names, all one buffer each
    const DFBSpecName TMP3{"tmp3"};

    const TensorParamName OUTPUT_GRAD{"output_grad"};
    const TensorParamName INPUT{"input"};
    const TensorParamName MEAN_T{"mean"};
    const TensorParamName RSTD_T{"rstd"};
    const TensorParamName GAMMA_T{"gamma"};
    const TensorParamName INPUT_GRAD{"input_grad"};

    ////////////////////////////////////////////////////////////////////////////
    //                         DataflowBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t{1};
    const uint32_t in1_t{1};
    const uint32_t in2_t{1};
    const uint32_t in3_t{1};
    const uint32_t in4_t{1};
    const uint32_t in5_t{2};
    const uint32_t in6_t = gamma_has_value ? 1 : 0;
    const uint32_t in7_t = (do_mask_h || do_mask_w) ? 2 : 0;

    const uint32_t out0_t{1};

    uint32_t im0_t{num_inner_tiles};
    uint32_t im1_t{num_inner_tiles};
    const uint32_t im2_t{1};
    const uint32_t im3_t{1};
    const uint32_t im4_t{1};
    const uint32_t im5_t{1};
    const uint32_t im6_t{1};
    uint32_t im7_t{1};

    const auto data_format = tt_metal::datatype_to_dataformat_converter(output_grad.dtype());
    const auto single_tile_size = tt::tile_size(data_format);

    const auto dfb_usage = (in0_t + in1_t + in2_t + in3_t + in4_t + in5_t + in6_t + in7_t + out0_t + im0_t + im1_t +
                            im2_t + im3_t + im4_t + im5_t + im6_t + im7_t) *
                           single_tile_size;
    const auto available_L1 = device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const bool use_large_algorithm = dfb_usage >= available_L1;

    if (use_large_algorithm) {
        log_info(LogTest, "Large moreh_group_norm_backward_input_grad algorithm is selected.");
        im0_t = 1;
        im1_t = 1;
        im7_t = 0;
    } else {
        log_info(LogTest, "Small moreh_group_norm_backward_input_grad algorithm is selected.");
    }

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

    add_dfb(DY, in0_t);         // output_grad
    add_dfb(X, in1_t);          // input
    add_dfb(MEAN, in2_t);       // mean
    add_dfb(RSTD, in3_t);       // rstd
    add_dfb(SCALER, in4_t);     // one
    add_dfb(N_RECIP_N, in5_t);  // inner_size(==n)
    add_dfb(GAMMA, in6_t);
    add_dfb(MASK_H_W, in7_t);
    add_dfb(DX, out0_t);  // input_grad
    add_dfb(DYCOPY, im0_t);
    add_dfb(Y, im1_t);
    add_dfb(DYSUM, im2_t);
    add_dfb(YDYSUM, im3_t);
    // The last four buffers mean different things to the two borrowed compute kernels, so they are
    // named for the source that will actually be selected. The large kernel folds rstd/n into tmp3 and
    // needs no fourth scratch buffer, which is why im7_t is zeroed above.
    if (use_large_algorithm) {
        add_dfb(TMP1, im4_t);
        add_dfb(TMP2, im5_t);
        add_dfb(TMP3, im6_t);
    } else {
        add_dfb(RECIP_NRSTD, im4_t);
        add_dfb(TMP1, im5_t);
        add_dfb(TMP2, im6_t);
        add_dfb(TMP3, im7_t);
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto* const reader_kernel_file =
        use_large_algorithm
            ? "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/input_grad/kernels/dataflow/"
              "reader_moreh_group_norm_backward_input_grad_large.cpp"
            : "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/input_grad/kernels/dataflow/"
              "reader_moreh_group_norm_backward_input_grad_small.cpp";

    const std::string writer_kernel_file(
        "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm_backward/device/input_grad/kernels/dataflow/"
        "writer_moreh_group_norm_backward_input_grad.cpp");

    // gamma_has_value / do_mask_h / do_mask_w each select whether a resource is bound, so they reach
    // the kernels as preprocessor defines: an unbound name does not exist in the generated header, and
    // a C++-level `if constexpr` would still name-look-up the discarded branch. The borrowed compute
    // kernel needs the same flags as the reader.
    KernelSpec::CompilerOptions::Defines reader_defines{};
    KernelSpec::CompilerOptions::Defines compute_defines{
        {"REDUCE_OP", "PoolType::SUM"},
        {"REDUCE_DIM", "ReduceDim::REDUCE_SCALAR"},
    };
    if (gamma_has_value) {
        reader_defines["GAMMA_HAS_VALUE"] = "1";
        compute_defines["GAMMA_HAS_VALUE"] = "1";
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
        DFBBinding{
            .dfb_spec_name = N_RECIP_N, .accessor_name = "n_recip_n", .endpoint_type = DFBEndpointType::PRODUCER},
    };
    Group<TensorBinding> reader_tensor_bindings{
        TensorBinding{.tensor_parameter_name = OUTPUT_GRAD, .accessor_name = "output_grad"},
        TensorBinding{.tensor_parameter_name = INPUT, .accessor_name = "input"},
        TensorBinding{.tensor_parameter_name = MEAN_T, .accessor_name = "mean"},
        TensorBinding{.tensor_parameter_name = RSTD_T, .accessor_name = "rstd"},
    };
    if (gamma_has_value) {
        reader_dfb_bindings.push_back(
            DFBBinding{.dfb_spec_name = GAMMA, .accessor_name = "gamma", .endpoint_type = DFBEndpointType::PRODUCER});
        reader_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = GAMMA_T, .accessor_name = "gamma"});
    }
    if (do_mask_h || do_mask_w) {
        reader_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = MASK_H_W, .accessor_name = "mask_h_w", .endpoint_type = DFBEndpointType::PRODUCER});
    }

    KernelSpec reader{
        .unique_id = READER,
        .source = reader_kernel_file,
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings = reader_dfb_bindings,
        .tensor_bindings = reader_tensor_bindings,
        .runtime_arg_schema =
            {
                .runtime_arg_names =
                    {"tile_offset",
                     "num_rows_per_core",
                     "num_inner_tiles",
                     "num_channels",
                     "num_groups",
                     "origin_h",
                     "origin_w"},
            },
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source = writer_kernel_file,
        .dfb_bindings =
            {
                DFBBinding{
                    .dfb_spec_name = DX, .accessor_name = "input_grad", .endpoint_type = DFBEndpointType::CONSUMER},
            },
        .tensor_bindings =
            {
                TensorBinding{.tensor_parameter_name = INPUT_GRAD, .accessor_name = "input_grad"},
            },
        .runtime_arg_schema = {.runtime_arg_names = {"tile_offset", "num_rows_per_core", "num_inner_tiles"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                      ComputeKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    const auto* const compute_kernel_file =
        use_large_algorithm ? "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
                              "moreh_layer_norm_backward_input_grad_large_kernel.cpp"
                            : "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
                              "moreh_layer_norm_backward_input_grad_small_kernel.cpp";

    Group<DFBBinding> compute_dfb_bindings{
        DFBBinding{.dfb_spec_name = DY, .accessor_name = "dy", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = X, .accessor_name = "x", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = MEAN, .accessor_name = "mean", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = RSTD, .accessor_name = "rstd", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = SCALER, .accessor_name = "scaler", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{
            .dfb_spec_name = N_RECIP_N, .accessor_name = "n_recip_n", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = DX, .accessor_name = "dx", .endpoint_type = DFBEndpointType::PRODUCER},
        // Compute-only intermediates: one toucher, so the compute kernel is bound as both endpoints.
        DFBBinding{.dfb_spec_name = DYCOPY, .accessor_name = "dycopy", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = DYCOPY, .accessor_name = "dycopy", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = Y, .accessor_name = "y", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = Y, .accessor_name = "y", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = DYSUM, .accessor_name = "dysum", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = DYSUM, .accessor_name = "dysum", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = YDYSUM, .accessor_name = "ydysum", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = YDYSUM, .accessor_name = "ydysum", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = TMP1, .accessor_name = "tmp1", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = TMP1, .accessor_name = "tmp1", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = TMP2, .accessor_name = "tmp2", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = TMP2, .accessor_name = "tmp2", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = TMP3, .accessor_name = "tmp3", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = TMP3, .accessor_name = "tmp3", .endpoint_type = DFBEndpointType::CONSUMER},
    };
    if (!use_large_algorithm) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RECIP_NRSTD, .accessor_name = "recip_nrstd", .endpoint_type = DFBEndpointType::PRODUCER});
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = RECIP_NRSTD, .accessor_name = "recip_nrstd", .endpoint_type = DFBEndpointType::CONSUMER});
    }
    if (gamma_has_value) {
        compute_dfb_bindings.push_back(
            DFBBinding{.dfb_spec_name = GAMMA, .accessor_name = "gamma", .endpoint_type = DFBEndpointType::CONSUMER});
    }
    if (do_mask_h || do_mask_w) {
        compute_dfb_bindings.push_back(DFBBinding{
            .dfb_spec_name = MASK_H_W, .accessor_name = "mask_h_w", .endpoint_type = DFBEndpointType::CONSUMER});
    }

    // The descriptor factory set an all-default ComputeConfigDescriptor{}, so the Gen1 config is built
    // directly and left at its own defaults. Routing this through the TTNN ComputeKernelConfig helper
    // would silently flip every field, because that helper's defaults favor performance while the
    // Metal struct's favor precision. With enable_32_bit_dest at its default false, the Float32
    // unpack_modes requirement does not apply, so the table stays empty.
    const ComputeHardwareConfig compute_hw{};

    auto make_compute = [&](const KernelSpecName& unique_id, uint32_t num_rows_per_core) {
        return KernelSpec{
            .unique_id = unique_id,
            .source = compute_kernel_file,
            // Legacy ComputeConfig defaults opt_level to O3, Metal 2.0's type-agnostic
            // CompilerOptions to O2; state it so the compute kernels keep their level.
            .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
            .dfb_bindings = compute_dfb_bindings,
            .compile_time_args =
                {
                    {"num_rows_per_core", num_rows_per_core},
                    {"origin_H", origin_h},
                    {"origin_W", origin_w},
                    // Carried over as-is: the kernel calls this argument Wt, but the value is the
                    // inner-dimension tile count.
                    {"Wt", num_inner_tiles},
                    {"is_lastdim_layernorm", static_cast<uint32_t>(is_lastdim_layernorm)},
                    {"is_groupnorm", static_cast<uint32_t>(is_groupnorm)},
                },
            .hw_config = compute_hw,
        };
    };

    const bool has_core_group_2 = !core_group_2.ranges().empty();

    Group<KernelSpec> kernels{reader, writer, make_compute(COMPUTE_G1, num_rows_per_core_group_1)};
    if (has_core_group_2) {
        kernels.push_back(make_compute(COMPUTE_G2, num_rows_per_core_group_2));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Tensor parameters
    ////////////////////////////////////////////////////////////////////////////
    Group<TensorParameter> tensor_parameters{
        TensorParameter{.unique_id = OUTPUT_GRAD, .spec = output_grad.tensor_spec()},
        TensorParameter{.unique_id = INPUT, .spec = input.tensor_spec()},
        TensorParameter{.unique_id = MEAN_T, .spec = mean.tensor_spec()},
        TensorParameter{.unique_id = RSTD_T, .spec = rstd.tensor_spec()},
        TensorParameter{.unique_id = INPUT_GRAD, .spec = input_grad.tensor_spec()},
    };
    if (gamma_has_value) {
        tensor_parameters.push_back(TensorParameter{.unique_id = GAMMA_T, .spec = gamma->tensor_spec()});
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
        .name = "moreh_group_norm_backward_input_grad",
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

        uint32_t num_rows_per_core;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        // reader
        AddRuntimeArgsForNode(
            reader_run_args.runtime_arg_values,
            core,
            {{"tile_offset", tile_offset},
             {"num_rows_per_core", num_rows_per_core},
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
             {"num_rows_per_core", num_rows_per_core},
             {"num_inner_tiles", num_inner_tiles}});

        tile_offset += num_rows_per_core * num_inner_tiles;
    }

    ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run_args), std::move(writer_run_args)};
    run_args.tensor_args = {
        {OUTPUT_GRAD, output_grad.mesh_tensor()},
        {INPUT, input.mesh_tensor()},
        {MEAN_T, mean.mesh_tensor()},
        {RSTD_T, rstd.mesh_tensor()},
        {INPUT_GRAD, input_grad.mesh_tensor()},
    };
    if (gamma_has_value) {
        run_args.tensor_args.emplace(GAMMA_T, gamma->mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_group_norm_backward
