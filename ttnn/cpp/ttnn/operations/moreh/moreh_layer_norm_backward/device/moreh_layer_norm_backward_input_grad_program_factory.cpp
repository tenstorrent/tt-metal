// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <string>

#include "moreh_layer_norm_backward_input_grad_device_operation.hpp"
#include "moreh_layer_norm_backward_metal2_helpers.hpp"
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

namespace ttnn::operations::moreh::moreh_layer_norm_backward_input_grad {

namespace {
namespace m2 = tt::tt_metal::experimental;
using namespace ttnn::operations::moreh::moreh_layer_norm_backward_metal2;

const m2::KernelSpecName READER{"reader"};
const m2::KernelSpecName WRITER{"writer"};
const m2::KernelSpecName COMPUTE_G1{"compute_g1"};
const m2::KernelSpecName COMPUTE_G2{"compute_g2"};

// Reader -> compute inputs.
const m2::DFBSpecName DY{"dy"};                // output_grad(==dy)
const m2::DFBSpecName X{"x"};                  // input(==x)
const m2::DFBSpecName MEAN{"mean"};            // mean
const m2::DFBSpecName RSTD{"rstd"};            // rstd
const m2::DFBSpecName SCALER{"scaler"};        // scaler
const m2::DFBSpecName N_RECIP_N{"n_recip_n"};  // n and 1/n, one tile each
const m2::DFBSpecName GAMMA{"gamma"};          // gamma; allocated only when gamma is supplied
const m2::DFBSpecName MASK_H_W{"mask_h_w"};    // mask_h and mask_w; allocated only when a tile is partial

// Compute -> writer output.
// dx = ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
const m2::DFBSpecName DX{"dx"};  // input_grad(==dx)

// Compute-private intermediates: the compute kernel packs into each and unpacks it back, so it is
// the only endpoint on both sides.
const m2::DFBSpecName DYCOPY{"dycopy"};  // copy output_grad(==dy or dy * gamma)
const m2::DFBSpecName Y{"y"};            // output(==y), y = (x - mean) * rstd
const m2::DFBSpecName DYSUM{"dysum"};    // Sum[dy]
const m2::DFBSpecName YDYSUM{"ydysum"};  // Sum[y * dy]
// rstd / n. The small algorithm gives it a buffer of its own; the large algorithm is tighter on L1
// and stages it in tmp3 instead, so this buffer exists on the small path only.
const m2::DFBSpecName RECIP_NRSTD{"recip_nrstd"};
// Three scratch buffers the compute kernel reuses across phases, under a different name in each.
const m2::DFBSpecName TMP1{"tmp1"};
const m2::DFBSpecName TMP2{"tmp2"};
const m2::DFBSpecName TMP3{"tmp3"};

const m2::TensorParamName OUTPUT_GRAD_T{"output_grad"};
const m2::TensorParamName INPUT_T{"input"};
const m2::TensorParamName MEAN_T{"mean"};
const m2::TensorParamName RSTD_T{"rstd"};
const m2::TensorParamName GAMMA_T{"gamma"};
const m2::TensorParamName INPUT_GRAD_T{"input_grad"};

constexpr const char* READER_SMALL_KERNEL =
    "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
    "reader_moreh_layer_norm_backward_input_grad_small.cpp";
constexpr const char* READER_LARGE_KERNEL =
    "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
    "reader_moreh_layer_norm_backward_input_grad_large.cpp";
constexpr const char* WRITER_KERNEL =
    "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
    "writer_moreh_layer_norm_backward_input_grad.cpp";
constexpr const char* COMPUTE_SMALL_KERNEL =
    "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
    "moreh_layer_norm_backward_input_grad_small_kernel.cpp";
constexpr const char* COMPUTE_LARGE_KERNEL =
    "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm_backward/device/kernels/"
    "moreh_layer_norm_backward_input_grad_large_kernel.cpp";

}  // namespace

ttnn::device_operation::ProgramArtifacts
MorehLayerNormBackwardInputGradOperation::MorehLayerNormBackwardInputGradFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& input_grad) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& output_grad = tensor_args.output_grad;
    const auto& input = tensor_args.input;
    const auto& mean = tensor_args.mean;
    const auto& rstd = tensor_args.rstd;

    auto normalized_dims = operation_attributes.normalized_dims;

    const std::optional<const Tensor>& gamma = tensor_args.gamma;

    auto compute_kernel_config =
        init_device_compute_kernel_config(output_grad.device()->arch(), operation_attributes.compute_kernel_config);

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
    const auto output_grad_rank = output_grad_shape.rank();

    const bool is_lastdim_layer_norm = normalized_dims == 1;
    const bool is_groupnorm = false;

    const auto origin_H = output_grad_shape_without_padding[-2];
    const auto origin_W = output_grad_shape_without_padding[-1];

    const bool do_mask_h = (origin_H % TILE_HEIGHT) != 0 && !is_lastdim_layer_norm;
    const uint32_t mask_h = do_mask_h ? origin_H % TILE_HEIGHT : TILE_HEIGHT;

    const bool do_mask_w = (origin_W % TILE_WIDTH) != 0;
    const uint32_t mask_w = do_mask_w ? origin_W % TILE_WIDTH : TILE_WIDTH;

    const auto mean_rstd_shape = mean.padded_shape();
    const auto mean_rstd_shape_without_padding = mean.logical_shape();
    auto mean_rstd_height = mean_rstd_shape_without_padding[-2];
    auto mean_rstd_width = mean_rstd_shape_without_padding[-1];

    auto normalized_numel = 1.0f;
    for (uint32_t i = output_grad_rank - normalized_dims; i < output_grad_rank; i++) {
        auto size = output_grad_shape_without_padding[i];
        normalized_numel *= size;
    }

    auto n = static_cast<float>(normalized_numel);
    auto recip_n = 1.0f / n;

    auto num_inner = compute_inner(output_grad_shape, normalized_dims);
    auto num_outer = compute_outer(output_grad_shape, normalized_dims);

    const bool gamma_has_value = gamma.has_value();

    const auto& output_grad_mesh = output_grad.mesh_tensor();
    const auto& input_mesh = input.mesh_tensor();
    const auto& mean_mesh = mean.mesh_tensor();
    const auto& rstd_mesh = rstd.mesh_tensor();
    const auto& input_grad_mesh = input_grad.mesh_tensor();

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, num_outer);
    const bool has_core_group_2 = !core_group_2.ranges().empty();

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);
    ////////////////////////////////////////////////////////////////////////////
    //                         DataflowBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t in0_t = 1;                                 // output_grad(==dy)
    const uint32_t in1_t = 1;                                 // input(==x)
    const uint32_t in2_t = 1;                                 // mean
    const uint32_t in3_t = 1;                                 // rstd
    const uint32_t in4_t = 1;                                 // scaler
    const uint32_t in5_t = 2;                                 // n_recip_n
    const uint32_t in6_t = gamma_has_value ? 1 : 0;           // gamma
    const uint32_t in7_t = (do_mask_h || do_mask_w) ? 2 : 0;  // mask_h_w

    // dx = ((n * dy - Sum[dy]) - (y * Sum[y * dy])) * (rstd / n)
    const uint32_t out0_t = 1;  // input_grad(==dx)

    uint32_t im0_t = num_inner;  // copy output_grad(==dycopy)
    uint32_t im1_t = num_inner;  // output(==y)
    const uint32_t im2_t = 1;    // Sum[dy]
    const uint32_t im3_t = 1;    // Sum[y * dy]
    const uint32_t im4_t = 1;    // rstd / n

    const uint32_t im5_t = 1;
    const uint32_t im6_t = 1;
    uint32_t im7_t = 1;

    const auto dfb_data_format = tt::tt_metal::datatype_to_dataformat_converter(output_grad.dtype());
    const auto single_tile_size = tt::tile_size(dfb_data_format);
    auto intermed_dfb_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : dfb_data_format;
    const auto intermed_single_tile_size = tt::tile_size(intermed_dfb_format);

    const uint32_t dfb_usage =
        ((in0_t + in1_t + in2_t + in3_t + in4_t + in5_t + in6_t + in7_t + out0_t) * single_tile_size) +
        ((im0_t + im1_t + im2_t + im3_t + im4_t + im5_t + im6_t + im7_t) * intermed_single_tile_size);
    const uint32_t available_L1 =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const bool use_large_algorithm = dfb_usage >= available_L1;

    if (use_large_algorithm) {
        log_info(tt::LogTest, "Large moreh_layer_norm_backward_input_grad algorithm is selected.");
        im0_t = 1;
        im1_t = 1;
        im7_t = 0;
    } else {
        log_info(tt::LogTest, "Small moreh_layer_norm_backward_input_grad algorithm is selected.");
    }

    // The two algorithms differ in more than their kernel sources: the large one keeps a single tile
    // of dycopy and y instead of a whole row, and folds rstd / n into tmp3 rather than giving it a
    // buffer of its own. So the buffer list branches here, alongside the sources.
    m2::Group<m2::DataflowBufferSpec> dfbs;
    dfbs.push_back(make_dfb(DY, in0_t, dfb_data_format));
    dfbs.push_back(make_dfb(X, in1_t, dfb_data_format));
    dfbs.push_back(make_dfb(MEAN, in2_t, dfb_data_format));
    dfbs.push_back(make_dfb(RSTD, in3_t, dfb_data_format));
    dfbs.push_back(make_dfb(SCALER, in4_t, dfb_data_format));
    dfbs.push_back(make_dfb(N_RECIP_N, in5_t, dfb_data_format));
    if (gamma_has_value) {
        dfbs.push_back(make_dfb(GAMMA, in6_t, dfb_data_format));
    }
    if (do_mask_h || do_mask_w) {
        dfbs.push_back(make_dfb(MASK_H_W, in7_t, dfb_data_format));
    }
    dfbs.push_back(make_dfb(DX, out0_t, dfb_data_format));
    dfbs.push_back(make_dfb(DYCOPY, im0_t, intermed_dfb_format));
    dfbs.push_back(make_dfb(Y, im1_t, intermed_dfb_format));
    dfbs.push_back(make_dfb(DYSUM, im2_t, intermed_dfb_format));
    dfbs.push_back(make_dfb(YDYSUM, im3_t, intermed_dfb_format));
    if (use_large_algorithm) {
        // The large algorithm has no buffer of its own for rstd / n, so its three scratch buffers
        // occupy the slots the small algorithm spends on rstd / n plus its first two scratch tiles.
        dfbs.push_back(make_dfb(TMP1, im4_t, intermed_dfb_format));
        dfbs.push_back(make_dfb(TMP2, im5_t, intermed_dfb_format));
        dfbs.push_back(make_dfb(TMP3, im6_t, intermed_dfb_format));
    } else {
        dfbs.push_back(make_dfb(RECIP_NRSTD, im4_t, intermed_dfb_format));
        dfbs.push_back(make_dfb(TMP1, im5_t, intermed_dfb_format));
        dfbs.push_back(make_dfb(TMP2, im6_t, intermed_dfb_format));
        dfbs.push_back(make_dfb(TMP3, im7_t, intermed_dfb_format));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                      DataMovementKernel SetUp
    ////////////////////////////////////////////////////////////////////////////
    // A kernel may only name a buffer's (or tensor's) binding token where the host actually binds it,
    // and `if constexpr (false)` still looks the name up — so a condition that decides a binding has
    // to reach the kernel as a preprocessor define rather than as a compile-time argument.
    m2::KernelSpec::CompilerOptions::Defines reader_defines;
    m2::KernelSpec::CompilerOptions::Defines compute_defines;
    compute_defines.emplace("REDUCE_OP", "PoolType::AVG");
    if (is_lastdim_layer_norm) {
        compute_defines.emplace("REDUCE_DIM", "ReduceDim::REDUCE_ROW");
    } else {
        compute_defines.emplace("REDUCE_DIM", "ReduceDim::REDUCE_SCALAR");
    }
    if (fp32_dest_acc_en) {
        reader_defines.emplace("FP32_DEST_ACC_EN", "1");
        compute_defines.emplace("FP32_DEST_ACC_EN", "1");
    }
    if (gamma_has_value) {
        reader_defines.emplace("GAMMA_HAS_VALUE", "1");
        compute_defines.emplace("GAMMA_HAS_VALUE", "1");
    }
    if (do_mask_h || do_mask_w) {
        reader_defines.emplace("DO_MASK_H_W", "1");
        compute_defines.emplace("DO_MASK_H_W", "1");
    }

    const auto* const reader_kernel_file = use_large_algorithm ? READER_LARGE_KERNEL : READER_SMALL_KERNEL;
    const auto* const compute_kernel_file = use_large_algorithm ? COMPUTE_LARGE_KERNEL : COMPUTE_SMALL_KERNEL;

    m2::KernelSpec reader{
        .unique_id = READER,
        .source = reader_kernel_file,
        .compiler_options = {.defines = reader_defines},
        .tensor_bindings =
            {
                m2::TensorBinding{.tensor_parameter_name = OUTPUT_GRAD_T, .accessor_name = "output_grad"},
                m2::TensorBinding{.tensor_parameter_name = INPUT_T, .accessor_name = "input"},
                m2::TensorBinding{.tensor_parameter_name = MEAN_T, .accessor_name = "mean"},
                m2::TensorBinding{.tensor_parameter_name = RSTD_T, .accessor_name = "rstd"},
            },
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_rows_per_core",
                  "num_inner",
                  "tile_offset",
                  "n",
                  "recip_n",
                  "mask_h",
                  "mask_w",
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
    bind_dfb(reader, N_RECIP_N, "n_recip_n", m2::DFBEndpointType::PRODUCER);
    if (gamma_has_value) {
        bind_dfb(reader, GAMMA, "gamma", m2::DFBEndpointType::PRODUCER);
        reader.tensor_bindings.push_back(m2::TensorBinding{.tensor_parameter_name = GAMMA_T, .accessor_name = "gamma"});
    }
    if (do_mask_h || do_mask_w) {
        bind_dfb(reader, MASK_H_W, "mask_h_w", m2::DFBEndpointType::PRODUCER);
    }

    m2::KernelSpec writer{
        .unique_id = WRITER,
        .source = WRITER_KERNEL,
        .tensor_bindings = {m2::TensorBinding{.tensor_parameter_name = INPUT_GRAD_T, .accessor_name = "input_grad"}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_rows_per_core", "Wt", "tile_offset"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };
    bind_dfb(writer, DX, "input_grad", m2::DFBEndpointType::CONSUMER);

    // One compute KernelSpec per work-split core group, differing only in the per-group row count.
    // Keeping that count a compile-time argument is what lets the kernel unroll its outer loop.
    auto make_compute = [&](const m2::KernelSpecName& unique_id, uint32_t num_rows_per_core_group) {
        auto compute_hw = ttnn::to_compute_hardware_config(device->arch(), compute_kernel_config);
        auto& compute_gen1 = gen1_compute_config(compute_hw);
        if (compute_gen1.enable_32_bit_dest) {
            // With the 32-bit Dest register enabled the intermediates are Float32, and a Float32
            // buffer a compute kernel consumes has to state its unpack mode. All of these feed FPU
            // ops (add/sub/mul tiles, the reduces), which read their operands out of SrcA/SrcB, so
            // SrcA/B is the mode for every one — the same mode the pre-Metal-2.0 kernel got by
            // leaving unpack_to_dest_mode unset. The inputs and the output stay at the io dtype,
            // which this op validates as bfloat16, so only the intermediates can qualify.
            unpack_via_src(compute_gen1, DYCOPY);
            unpack_via_src(compute_gen1, Y);
            unpack_via_src(compute_gen1, DYSUM);
            unpack_via_src(compute_gen1, YDYSUM);
            if (!use_large_algorithm) {
                unpack_via_src(compute_gen1, RECIP_NRSTD);
            }
            unpack_via_src(compute_gen1, TMP1);
            unpack_via_src(compute_gen1, TMP2);
            unpack_via_src(compute_gen1, TMP3);
        }

        m2::KernelSpec compute{
            .unique_id = unique_id,
            .source = compute_kernel_file,
            // Legacy ComputeConfig defaults opt_level to O3; Metal 2.0's CompilerOptions defaults to
            // O2 for every kernel kind, so a compute kernel has to ask for O3 explicitly.
            .compiler_options = {.defines = compute_defines, .opt_level = KernelBuildOptLevel::O3},
            .compile_time_args =
                {{"num_rows_per_core", num_rows_per_core_group},
                 {"origin_H", origin_H},
                 {"origin_W", origin_W},
                 {"Wt", num_inner},
                 {"is_lastdim_layernorm", static_cast<uint32_t>(is_lastdim_layer_norm)},
                 {"is_groupnorm", static_cast<uint32_t>(is_groupnorm)}},
            .hw_config = compute_hw,
        };
        bind_dfb(compute, DY, "dy", m2::DFBEndpointType::CONSUMER);
        bind_dfb(compute, X, "x", m2::DFBEndpointType::CONSUMER);
        bind_dfb(compute, MEAN, "mean", m2::DFBEndpointType::CONSUMER);
        bind_dfb(compute, RSTD, "rstd", m2::DFBEndpointType::CONSUMER);
        bind_dfb(compute, SCALER, "scaler", m2::DFBEndpointType::CONSUMER);
        bind_dfb(compute, N_RECIP_N, "n_recip_n", m2::DFBEndpointType::CONSUMER);
        if (gamma_has_value) {
            bind_dfb(compute, GAMMA, "gamma", m2::DFBEndpointType::CONSUMER);
        }
        if (do_mask_h || do_mask_w) {
            bind_dfb(compute, MASK_H_W, "mask_h_w", m2::DFBEndpointType::CONSUMER);
        }
        bind_dfb(compute, DX, "dx", m2::DFBEndpointType::PRODUCER);
        bind_self_loop(compute, DYCOPY, "dycopy");
        bind_self_loop(compute, Y, "y");
        bind_self_loop(compute, DYSUM, "dysum");
        bind_self_loop(compute, YDYSUM, "ydysum");
        if (!use_large_algorithm) {
            bind_self_loop(compute, RECIP_NRSTD, "recip_nrstd");
        }
        bind_self_loop(compute, TMP1, "tmp1");
        bind_self_loop(compute, TMP2, "tmp2");
        bind_self_loop(compute, TMP3, "tmp3");
        return compute;
    };

    m2::Group<m2::KernelSpec> kernels;
    kernels.push_back(std::move(reader));
    kernels.push_back(std::move(writer));
    kernels.push_back(make_compute(COMPUTE_G1, num_rows_per_core_group_1));
    if (has_core_group_2) {
        kernels.push_back(make_compute(COMPUTE_G2, num_rows_per_core_group_2));
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
    if (gamma_has_value) {
        tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = GAMMA_T, .spec = gamma->mesh_tensor().tensor_spec()});
    }
    tensor_parameters.push_back(m2::TensorParameter{.unique_id = INPUT_GRAD_T, .spec = input_grad_mesh.tensor_spec()});

    ////////////////////////////////////////////////////////////////////////////
    //                      RuntimeArgs SetUp
    ////////////////////////////////////////////////////////////////////////////
    const uint32_t n_u = std::bit_cast<uint32_t>(n);
    const uint32_t recip_n_u = std::bit_cast<uint32_t>(recip_n);

    m2::KernelRunArgs reader_run{.kernel = READER};
    m2::KernelRunArgs writer_run{.kernel = WRITER};

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};

        uint32_t num_rows_per_core;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        m2::AddRuntimeArgsForNode(
            reader_run.runtime_arg_values,
            core,
            {{"num_rows_per_core", num_rows_per_core},
             {"num_inner", num_inner},
             {"tile_offset", tile_offset},
             {"n", n_u},
             {"recip_n", recip_n_u},
             {"mask_h", mask_h},
             {"mask_w", mask_w},
             {"normalized_dims", normalized_dims},
             {"mean_rstd_height", mean_rstd_height},
             {"mean_rstd_width", mean_rstd_width}});

        m2::AddRuntimeArgsForNode(
            writer_run.runtime_arg_values,
            core,
            {{"num_rows_per_core", num_rows_per_core}, {"Wt", num_inner}, {"tile_offset", tile_offset}});

        tile_offset += num_rows_per_core * num_inner;
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
        .name = "moreh_layer_norm_backward_input_grad",
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
    if (gamma_has_value) {
        run_args.tensor_args.emplace(GAMMA_T, gamma->mesh_tensor());
    }
    run_args.tensor_args.emplace(INPUT_GRAD_T, input_grad_mesh);

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}  // namespace ttnn::operations::moreh::moreh_layer_norm_backward_input_grad
