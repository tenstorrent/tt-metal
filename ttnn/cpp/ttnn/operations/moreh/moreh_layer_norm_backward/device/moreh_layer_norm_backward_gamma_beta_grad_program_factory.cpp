// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <string>
#include <vector>

#include "moreh_layer_norm_backward_gamma_beta_grad_device_operation.hpp"
#include "moreh_layer_norm_backward_metal2_helpers.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm_backward_gamma_beta_grad {

namespace {
namespace m2 = tt::tt_metal::experimental;
using namespace ttnn::operations::moreh::moreh_layer_norm_backward_metal2;

const m2::KernelSpecName READER{"reader"};
const m2::KernelSpecName WRITER{"writer"};
const m2::KernelSpecName COMPUTE_G1{"compute_g1"};
const m2::KernelSpecName COMPUTE_G2{"compute_g2"};

// Reader -> compute inputs.
const m2::DFBSpecName DY{"dy"};          // output_grad(==dy)
const m2::DFBSpecName X{"x"};            // input(==x)
const m2::DFBSpecName MEAN{"mean"};      // mean
const m2::DFBSpecName RSTD{"rstd"};      // rstd
const m2::DFBSpecName SCALER{"scaler"};  // scaler
const m2::DFBSpecName MASK_H{"mask_h"};  // mask_h; allocated only when the last row tile is partial

// Compute -> writer outputs.
const m2::DFBSpecName DGAMMA{"dgamma"};  // gamma_grad(==dgamma), holds Sum[y * dy]
const m2::DFBSpecName DBETA{"dbeta"};    // beta_grad(==dbeta), holds Sum[dy]

// Compute-private intermediates: the compute kernel packs into each and unpacks it back, so it is
// the only endpoint on both sides.
const m2::DFBSpecName Y{"y"};            // output(==y), y = (x - mean) * rstd
const m2::DFBSpecName YDY{"ydy"};        // y * dy
const m2::DFBSpecName DYADD{"dyadd"};    // Add[dy]
const m2::DFBSpecName YDYADD{"ydyadd"};  // Add[y * dy]
const m2::DFBSpecName XMM{"xmm"};        // x - mean
const m2::DFBSpecName DYCOPY{"dycopy"};  // dycopy

const m2::TensorParamName OUTPUT_GRAD_T{"output_grad"};
const m2::TensorParamName INPUT_T{"input"};
const m2::TensorParamName MEAN_T{"mean"};
const m2::TensorParamName RSTD_T{"rstd"};
const m2::TensorParamName GAMMA_GRAD_T{"gamma_grad"};
const m2::TensorParamName BETA_GRAD_T{"beta_grad"};

constexpr const char* READER_KERNEL =
    "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
    "reader_moreh_layer_norm_backward_gamma_beta_grad.cpp";
constexpr const char* WRITER_KERNEL =
    "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
    "writer_moreh_layer_norm_backward_gamma_beta_grad.cpp";
constexpr const char* COMPUTE_KERNEL =
    "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
    "moreh_layer_norm_backward_gamma_beta_grad_kernel.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts
MorehLayerNormBackwardGammaBetaGradOperation::MorehLayerNormBackwardGammaBetaGradFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& mean = tensor_args.mean;
    const auto& rstd = tensor_args.rstd;

    const std::optional<const Tensor>& gamma_grad = output_tensor.at(0);
    const std::optional<const Tensor>& beta_grad = output_tensor.at(1);

    auto normalized_dims = operation_attributes.normalized_dims;

    auto compute_kernel_config =
        init_device_compute_kernel_config(input.device()->arch(), operation_attributes.compute_kernel_config);

    using namespace tt::constants;
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    IDevice* device = output_grad.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto output_grad_shape = output_grad.padded_shape();
    const auto output_grad_shape_without_padding = output_grad.logical_shape();

    const bool is_lastdim_layer_norm = normalized_dims == 1;
    const bool is_groupnorm = false;

    const auto origin_H = output_grad_shape_without_padding[-2];
    const auto origin_W = output_grad_shape_without_padding[-1];

    const bool do_mask_h = (origin_H % TILE_HEIGHT) != 0 && is_lastdim_layer_norm;
    const uint32_t mask_h = do_mask_h ? origin_H % TILE_HEIGHT : TILE_HEIGHT;

    const auto mean_rstd_shape = mean.padded_shape();
    const auto mean_rstd_shape_without_padding = mean.logical_shape();
    auto mean_rstd_height = mean_rstd_shape_without_padding[-2];
    auto mean_rstd_width = mean_rstd_shape_without_padding[-1];

    auto num_inner = compute_inner(output_grad_shape, normalized_dims);
    auto num_outer = compute_outer(output_grad_shape, normalized_dims);

    const bool gamma_grad_has_value = gamma_grad.has_value();
    const bool beta_grad_has_value = beta_grad.has_value();
    TT_FATAL(gamma_grad_has_value || beta_grad_has_value, "gamma_grad and beta_grad must have values");

    const auto& output_grad_mesh = output_grad.mesh_tensor();
    const auto& input_mesh = input.mesh_tensor();
    const auto& mean_mesh = mean.mesh_tensor();
    const auto& rstd_mesh = rstd.mesh_tensor();

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_cols_per_core_group_1, num_cols_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, num_inner);
    const bool has_core_group_2 = !core_group_2.ranges().empty();

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    ////////////////////////////////////////////////////////////////////////////
    //                         DataflowBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 1;                  // output_grad(==dy)
    const uint32_t in1_t = 1;                  // input(==x)
    const uint32_t in2_t = 1;                  // mean
    const uint32_t in3_t = 1;                  // rstd
    const uint32_t in4_t = 1;                  // scaler
    const uint32_t in5_t = do_mask_h ? 1 : 0;  // mask_h

    const uint32_t out0_t = 1;  // gamma_grad(==dgamma)
    const uint32_t out1_t = 1;  // beta_grad(==dbeta)

    const uint32_t im0_t = 1;  // output(==y)
    const uint32_t im1_t = 1;  // y * dy
    const uint32_t im2_t = 1;  // Add[dy]
    const uint32_t im3_t = 1;  // Add[y * dy]
    const uint32_t im4_t = 1;  // x - mean
    const uint32_t im5_t = 1;  // dycopy

    const auto dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_grad.dtype());
    auto intermed_dfb_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : dfb_data_format;

    m2::Group<m2::DataflowBufferSpec> dfbs;
    dfbs.push_back(make_dfb(DY, in0_t, dfb_data_format));
    dfbs.push_back(make_dfb(X, in1_t, dfb_data_format));
    dfbs.push_back(make_dfb(MEAN, in2_t, dfb_data_format));
    dfbs.push_back(make_dfb(RSTD, in3_t, dfb_data_format));
    dfbs.push_back(make_dfb(SCALER, in4_t, dfb_data_format));
    if (do_mask_h) {
        dfbs.push_back(make_dfb(MASK_H, in5_t, dfb_data_format));
    }
    dfbs.push_back(make_dfb(DGAMMA, out0_t, dfb_data_format));
    dfbs.push_back(make_dfb(DBETA, out1_t, dfb_data_format));
    dfbs.push_back(make_dfb(Y, im0_t, intermed_dfb_format));
    dfbs.push_back(make_dfb(YDY, im1_t, intermed_dfb_format));
    dfbs.push_back(make_dfb(DYADD, im2_t, intermed_dfb_format));
    dfbs.push_back(make_dfb(YDYADD, im3_t, intermed_dfb_format));
    dfbs.push_back(make_dfb(XMM, im4_t, intermed_dfb_format));
    dfbs.push_back(make_dfb(DYCOPY, im5_t, intermed_dfb_format));

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // The kernels name a buffer's binding token only where the host actually binds it, so a condition
    // that decides a binding has to reach the kernel as a preprocessor define rather than as a
    // compile-time argument: `if constexpr (false)` still looks the name up. mask_h gates on the
    // factory's do_mask_h — the condition that decides whether the buffer exists at all. (The compute
    // kernel derives its own do_mask_h from a slightly wider expression that also admits groupnorm;
    // that one still guards the work, but it cannot be what decides the binding.)
    m2::KernelSpec::CompilerOptions::Defines reader_defines;
    m2::KernelSpec::CompilerOptions::Defines writer_defines;
    m2::KernelSpec::CompilerOptions::Defines compute_defines;
    compute_defines.emplace("REDUCE_OP", "PoolType::SUM");
    compute_defines.emplace("REDUCE_DIM", "ReduceDim::REDUCE_COL");
    if (fp32_dest_acc_en) {
        reader_defines.emplace("FP32_DEST_ACC_EN", "1");
        compute_defines.emplace("FP32_DEST_ACC_EN", "1");
    }
    if (do_mask_h) {
        reader_defines.emplace("DO_MASK_H", "1");
        compute_defines.emplace("DO_MASK_H", "1");
    }
    if (gamma_grad_has_value) {
        writer_defines.emplace("GAMMA_GRAD_HAS_VALUE", "1");
    }
    if (beta_grad_has_value) {
        writer_defines.emplace("BETA_GRAD_HAS_VALUE", "1");
    }

    m2::KernelSpec reader{
        .unique_id = READER,
        .source = READER_KERNEL,
        .compiler_options = {.defines = reader_defines},
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = OUTPUT_GRAD_T, .accessor_name = "output_grad"},
                m2::TensorBinding{.tensor_parameter_name = INPUT_T, .accessor_name = "input"},
                m2::TensorBinding{.tensor_parameter_name = MEAN_T, .accessor_name = "mean"},
                m2::TensorBinding{.tensor_parameter_name = RSTD_T, .accessor_name = "rstd"},
            },
        .compile_time_args = {{"gamma_grad_has_value", static_cast<uint32_t>(gamma_grad_has_value)}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_cols_per_core",
                  "num_outer",
                  "num_inner",
                  "tile_offset",
                  "mask_h",
                  "normalized_dims",
                  "mean_rstd_height",
                  "mean_rstd_width"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };
    // Accessor names follow each kernel's own vocabulary: the reader talks about the tensors it moves,
    // the compute kernel about the algebra it evaluates.
    bind_dfb(reader, DY, "output_grad", m2::DFBEndpointType::PRODUCER);
    bind_dfb(reader, X, "input", m2::DFBEndpointType::PRODUCER);
    bind_dfb(reader, MEAN, "mean", m2::DFBEndpointType::PRODUCER);
    bind_dfb(reader, RSTD, "rstd", m2::DFBEndpointType::PRODUCER);
    bind_dfb(reader, SCALER, "scaler", m2::DFBEndpointType::PRODUCER);
    if (do_mask_h) {
        bind_dfb(reader, MASK_H, "mask_h", m2::DFBEndpointType::PRODUCER);
    }

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = WRITER_KERNEL,
        .compiler_options = {.defines = writer_defines},
        .runtime_arg_schema = {.runtime_arg_names = {"num_cols_per_core", "tile_offset"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };
    bind_dfb(writer, DGAMMA, "gamma_grad", m2::DFBEndpointType::CONSUMER);
    bind_dfb(writer, DBETA, "beta_grad", m2::DFBEndpointType::CONSUMER);
    // Only the output tensors are optional here; both output buffers are always allocated, so the
    // kernel's pack-target selection needs no gate.
    if (gamma_grad_has_value) {
        writer.tensor_bindings.push_back(
            m2::TensorBinding{.tensor_parameter_name = GAMMA_GRAD_T, .accessor_name = "gamma_grad"});
    }
    if (beta_grad_has_value) {
        writer.tensor_bindings.push_back(
            m2::TensorBinding{.tensor_parameter_name = BETA_GRAD_T, .accessor_name = "beta_grad"});
    }

    // One compute KernelSpec per work-split core group, differing only in the per-group column count.
    // Keeping that count a compile-time argument is what lets the kernel unroll its outer loop.
    auto make_compute = [&](const m2::KernelSpecName& unique_id, uint32_t num_cols_per_core_group) {
        auto compute_hw = ttnn::to_compute_hardware_config(device->arch(), compute_kernel_config);
        auto& compute_gen1 = gen1_compute_config(compute_hw);
        if (compute_gen1.enable_32_bit_dest) {
            // With the 32-bit Dest register enabled the intermediates are Float32, and a Float32
            // buffer a compute kernel consumes has to state its unpack mode. All of these feed FPU
            // ops (add/sub/mul tiles, the column reduce), which read their operands out of SrcA/SrcB,
            // so SrcA/B is the mode for every one — the same mode the pre-Metal-2.0 kernel got by
            // leaving unpack_to_dest_mode unset. The inputs and outputs stay at the io dtype, which
            // this op validates as bfloat16, so only the intermediates can qualify.
            unpack_via_src(compute_gen1, Y);
            unpack_via_src(compute_gen1, YDY);
            unpack_via_src(compute_gen1, DYADD);
            unpack_via_src(compute_gen1, YDYADD);
            unpack_via_src(compute_gen1, XMM);
            unpack_via_src(compute_gen1, DYCOPY);
        }

        m2::KernelSpec compute{
            .unique_id = unique_id,
            .source = COMPUTE_KERNEL,
            // Legacy ComputeConfig defaults opt_level to O3; Metal 2.0's CompilerOptions defaults to
            // O2 for every kernel kind, so a compute kernel has to ask for O3 explicitly.
            .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
            .compile_time_args =
                {{"num_cols_per_core", num_cols_per_core_group},
                 {"origin_H", origin_H},
                 {"origin_W", origin_W},
                 {"NCHt", num_outer},
                 {"Wt", num_inner},
                 {"gamma_grad_has_value", static_cast<uint32_t>(gamma_grad_has_value)},
                 {"beta_grad_has_value", static_cast<uint32_t>(beta_grad_has_value)},
                 {"is_lastdim_layernorm", static_cast<uint32_t>(is_lastdim_layer_norm)},
                 {"is_groupnorm", static_cast<uint32_t>(is_groupnorm)}},
            .hw_config = compute_hw,
        };
        bind_dfb(compute, DY, "dy", m2::DFBEndpointType::CONSUMER);
        bind_dfb(compute, X, "x", m2::DFBEndpointType::CONSUMER);
        bind_dfb(compute, MEAN, "mean", m2::DFBEndpointType::CONSUMER);
        bind_dfb(compute, RSTD, "rstd", m2::DFBEndpointType::CONSUMER);
        bind_dfb(compute, SCALER, "scaler", m2::DFBEndpointType::CONSUMER);
        if (do_mask_h) {
            bind_dfb(compute, MASK_H, "mask_h", m2::DFBEndpointType::CONSUMER);
        }
        bind_dfb(compute, DGAMMA, "dgamma", m2::DFBEndpointType::PRODUCER);
        bind_dfb(compute, DBETA, "dbeta", m2::DFBEndpointType::PRODUCER);
        bind_self_loop(compute, Y, "y");
        bind_self_loop(compute, YDY, "ydy");
        bind_self_loop(compute, DYADD, "dyadd");
        bind_self_loop(compute, YDYADD, "ydyadd");
        bind_self_loop(compute, XMM, "xmm");
        bind_self_loop(compute, DYCOPY, "dycopy");
        return compute;
    };

    m2::Group<m2::KernelSpec> kernels;
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));
    kernels.push_back(make_compute(COMPUTE_G1, num_cols_per_core_group_1));
    if (has_core_group_2) {
        kernels.push_back(make_compute(COMPUTE_G2, num_cols_per_core_group_2));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Tensor parameters
    ////////////////////////////////////////////////////////////////////////////
    m2::Group<m2::TensorParameter> tensor_parameters;
    tensor_parameters.push_back(
        m2::TensorParameter{.unique_id = OUTPUT_GRAD_T, .spec = output_grad_mesh.tensor_spec()});
    tensor_parameters.push_back(m2::TensorParameter{.unique_id = INPUT_T, .spec = input_mesh.tensor_spec()});
    tensor_parameters.push_back(m2::TensorParameter{.unique_id = MEAN_T, .spec = mean_mesh.tensor_spec()});
    tensor_parameters.push_back(m2::TensorParameter{.unique_id = RSTD_T, .spec = rstd_mesh.tensor_spec()});
    if (gamma_grad_has_value) {
        tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = GAMMA_GRAD_T, .spec = gamma_grad->mesh_tensor().tensor_spec()});
    }
    if (beta_grad_has_value) {
        tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = BETA_GRAD_T, .spec = beta_grad->mesh_tensor().tensor_spec()});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelRunArgs reader_run{.kernel = READER};
    m2::KernelRunArgs writer_run{.kernel = WRITER};

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_cols_per_core;
        if (core_group_1.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_cols_per_core = num_cols_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        m2::AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"num_cols_per_core", num_cols_per_core},
             {"num_outer", num_outer},
             {"num_inner", num_inner},
             {"tile_offset", tile_offset},
             {"mask_h", mask_h},
             {"normalized_dims", normalized_dims},
             {"mean_rstd_height", mean_rstd_height},
             {"mean_rstd_width", mean_rstd_width}});

        m2::AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"num_cols_per_core", num_cols_per_core}, {"tile_offset", tile_offset}});

        tile_offset += num_cols_per_core;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      Assemble
    ////////////////////////////////////////////////////////////////////////////
    // Reader and writer belong to both work units, so their node set is the union of the two core
    // groups — the all_cores placement they had before.
    m2::Group<m2::WorkUnitSpec> work_units;
    work_units.push_back(
        m2::WorkUnitSpec{.name = "wu_g1", .kernels = {READER, WRITER, COMPUTE_G1}, .target_nodes = core_group_1});
    if (has_core_group_2) {
        work_units.push_back(
            m2::WorkUnitSpec{.name = "wu_g2", .kernels = {READER, WRITER, COMPUTE_G2}, .target_nodes = core_group_2});
    }

    m2::ProgramSpec spec{
        .name = "moreh_layer_norm_backward_gamma_beta_grad",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    m2::ProgramRunArgs run_args;
    run_args.kernel_run_args = {std::move(reader_run), std::move(writer_run)};
    run_args.tensor_args.emplace(OUTPUT_GRAD_T, output_grad_mesh);
    run_args.tensor_args.emplace(INPUT_T, input_mesh);
    run_args.tensor_args.emplace(MEAN_T, mean_mesh);
    run_args.tensor_args.emplace(RSTD_T, rstd_mesh);
    if (gamma_grad_has_value) {
        run_args.tensor_args.emplace(GAMMA_GRAD_T, gamma_grad->mesh_tensor());
    }
    if (beta_grad_has_value) {
        run_args.tensor_args.emplace(BETA_GRAD_T, beta_grad->mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_layer_norm_backward_gamma_beta_grad
