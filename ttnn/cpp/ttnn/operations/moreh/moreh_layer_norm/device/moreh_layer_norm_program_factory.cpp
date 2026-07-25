// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <bit>
#include <cmath>
#include <filesystem>
#include <string>
#include <vector>

#include "moreh_layer_norm_device_operation.hpp"
#include <tt-metalium/work_split.hpp>
#include "ttnn/operations/moreh/moreh_helper_functions.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"

#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <tt-metalium/experimental/metal2_host_api/kernel_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/dataflow_buffer_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/tensor_parameter.hpp>
#include <tt-metalium/experimental/metal2_host_api/compute_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/data_movement_hardware_config.hpp>
#include <tt-metalium/experimental/metal2_host_api/node_coord.hpp>

namespace ttnn::operations::moreh::moreh_layer_norm {

namespace m2 = tt::tt_metal::experimental;

inline uint32_t find_divisor_with_max_block_size(uint32_t val, uint32_t max_block_size) {
    uint32_t divisor{1};
    for (uint32_t current_divisor = max_block_size; current_divisor >= 1; current_divisor--) {
        if (val % current_divisor == 0) {
            divisor = current_divisor;
            break;
        }
    }
    return divisor;
}

ttnn::device_operation::ProgramArtifacts MorehLayerNormOperation::ProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& output_tensor) {
    using namespace tt;
    using namespace tt::tt_metal;

    const auto& input = tensor_args.input;
    const auto& gamma = tensor_args.gamma;
    const auto& beta = tensor_args.beta;

    const auto& mean_inp = tensor_args.mean;
    const auto& rstd_inp = tensor_args.rstd;

    std::optional<Tensor> mean = std::nullopt;
    if (mean_inp.has_value()) {
        mean = output_tensor.at(1);
    }

    std::optional<Tensor> rstd = std::nullopt;
    if (rstd_inp.has_value()) {
        rstd = output_tensor.at(2);
    }

    auto normalized_dims = operation_attributes.normalized_dims;
    auto eps = operation_attributes.eps;

    auto compute_kernel_config =
        init_device_compute_kernel_config(input.device()->arch(), operation_attributes.compute_kernel_config);

    using namespace tt::constants;
    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    IDevice* device = input.device();

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.padded_shape();
    const auto input_shape_without_padding = input.logical_shape();
    const auto input_rank = input_shape.rank();

    const bool is_lastdim_layer_norm = normalized_dims == 1;
    const bool is_groupnorm = false;

    auto num_inner = compute_inner(input_shape, normalized_dims);
    auto num_outer = compute_outer(input_shape, normalized_dims);

    const auto gamma_has_value = gamma.has_value();
    const auto beta_has_value = beta.has_value();
    const auto mean_has_value = mean.has_value();
    const auto rstd_has_value = rstd.has_value();

    const auto origin_H = input_shape_without_padding[-2];
    const auto origin_W = input_shape_without_padding[-1];

    uint32_t mean_rstd_height = 0;
    uint32_t mean_rstd_width = 0;

    if (mean_has_value || rstd_has_value) {
        const auto mean_rstd_shape_without_padding = mean_has_value ? mean->logical_shape() : rstd->logical_shape();
        mean_rstd_height = mean_rstd_shape_without_padding[-2];
        mean_rstd_width = mean_rstd_shape_without_padding[-1];
    }

    const bool do_mask_h = (origin_H % TILE_HEIGHT) != 0 && !is_lastdim_layer_norm;
    const auto mask_h = do_mask_h ? origin_H % TILE_HEIGHT : TILE_HEIGHT;

    const bool do_mask_w = (origin_W % TILE_WIDTH) != 0;
    const auto mask_w = do_mask_w ? origin_W % TILE_WIDTH : TILE_WIDTH;

    ////////////////////////////////////////////////////////////////////////////
    //                         Core Setup
    ////////////////////////////////////////////////////////////////////////////
    auto grid = device->compute_with_storage_grid_size();
    const auto num_cores_y = grid.y;

    // core_group_2 works more.
    // If number of working cores is 108 and num_outer is 110,
    // core_group_2[(x=0, y=0), (x=0, y=1)] works for 2 rows. Others work for 1 row.
    const auto
        [num_cores, all_cores, core_group_1, core_group_2, num_rows_per_core_group_1, num_rows_per_core_group_2] =
            tt::tt_metal::split_work_to_cores(grid, num_outer);

    auto arch = input.device()->arch();
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, compute_kernel_config);

    // This could be inefficient.
    // If Wt is 65, the block_size will be 5. Then, the number of iteration is 13.
    // It can be 8 * 8 + 1, so the number of iterations is 9. It's more efficient.
    uint32_t MAX_BLOCK_SIZE = 4;
    if (fp32_dest_acc_en) {
        MAX_BLOCK_SIZE = 2;
    }
    const uint32_t block_size = find_divisor_with_max_block_size(num_inner, MAX_BLOCK_SIZE);

    ////////////////////////////////////////////////////////////////////////////
    //                         DataflowBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_t = num_inner;                                   // input
    const uint32_t in1_t = 1;                                     // scaler
    const uint32_t in2_t = 1;                                     // epsilon
    const uint32_t in3_t = gamma_has_value ? 2 * block_size : 0;  // gamma
    const uint32_t in4_t = beta_has_value ? 2 * block_size : 0;   // beta
    const uint32_t in5_t = do_mask_h ? 1 : 0;                     // mask_h
    const uint32_t in6_t = do_mask_w ? 1 : 0;                     // mask_w

    const uint32_t out0_t = 2 * block_size;          // output
    const uint32_t out1_t = mean_has_value ? 1 : 0;  // mean
    const uint32_t out2_t = rstd_has_value ? 1 : 0;  // rstd

    const uint32_t im0_t = 1;                                                         // E[x]
    uint32_t im1_t = num_inner;                                                       // x - E[x]
    uint32_t im2_t = 1;                                                               // (x - E[x])^2
    const uint32_t im3_t = 1;                                                         // Sum[(x - E[x])^2]
    const uint32_t im4_t = 1;                                                         // E[(x - E[x])^2] = Var[x]
    const uint32_t im5_t = 1;                                                         // 1.0/(sqrt(Var[x] + eps))
    const uint32_t im6_t = (gamma_has_value || beta_has_value) ? 2 * block_size : 0;  // x * gamm + beta
    const uint32_t im7_t = 2;                                                         // Sum[x]

    const auto cb_data_format = tt::tt_metal::datatype_to_dataformat_converter(input.dtype());
    const auto single_tile_size = tt::tile_size(cb_data_format);
    auto intermed_cb_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : cb_data_format;
    const auto intermed_single_tile_size = tt::tile_size(intermed_cb_format);

    const uint32_t cb_usage =
        ((in0_t + in1_t + in2_t + in3_t + in4_t + in5_t + in6_t + out0_t + out1_t + out2_t) * single_tile_size) +
        ((im0_t + im1_t + im2_t + im3_t + im4_t + im5_t + im6_t + im7_t) * intermed_single_tile_size);
    const uint32_t available_L1 =
        device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const bool use_large_algorithm = cb_usage >= available_L1;

    if (use_large_algorithm) {
        log_info(tt::LogTest, "Large moreh_layer_norm algorithm is selected.");
        in0_t = 2 * block_size;
        im1_t = 2 * block_size;
        im2_t = 2 * block_size;
    } else {
        log_info(tt::LogTest, "Small moreh_layer_norm algorithm is selected.");
    }

    ////////////////////////////////////////////////////////////////////////////
    //                    Metal 2.0 resource names
    ////////////////////////////////////////////////////////////////////////////
    // DFB specs (one per legacy CB index). Optional ones are only pushed when present, matching the
    // legacy push_cb skip-zero-size behavior.
    const m2::DFBSpecName INPUT_DFB{"moreh_ln_input"};           // c_0
    const m2::DFBSpecName SCALER_DFB{"moreh_ln_scaler"};         // c_1
    const m2::DFBSpecName EPS_DFB{"moreh_ln_eps"};               // c_2
    const m2::DFBSpecName GAMMA_DFB{"moreh_ln_gamma"};           // c_3
    const m2::DFBSpecName BETA_DFB{"moreh_ln_beta"};             // c_4
    const m2::DFBSpecName MASK_H_DFB{"moreh_ln_mask_h"};         // c_5
    const m2::DFBSpecName MASK_W_DFB{"moreh_ln_mask_w"};         // c_6
    const m2::DFBSpecName OUT_DFB{"moreh_ln_out"};               // c_16
    const m2::DFBSpecName MEAN_DFB{"moreh_ln_mean"};             // c_17
    const m2::DFBSpecName RSTD_DFB{"moreh_ln_rstd"};             // c_18
    const m2::DFBSpecName EX_DFB{"moreh_ln_ex"};                 // c_24
    const m2::DFBSpecName XMM_DFB{"moreh_ln_xmm"};               // c_25
    const m2::DFBSpecName XMM2_DFB{"moreh_ln_xmm2"};             // c_26
    const m2::DFBSpecName XMM2SUM_DFB{"moreh_ln_xmm2sum"};       // c_27
    const m2::DFBSpecName VAR_DFB{"moreh_ln_var"};               // c_28
    const m2::DFBSpecName RECIP_STD_DFB{"moreh_ln_recipstd"};    // c_29
    const m2::DFBSpecName GAMMA_BETA_DFB{"moreh_ln_gammabeta"};  // c_30
    const m2::DFBSpecName XSUM_DFB{"moreh_ln_xsum"};             // c_31

    const m2::TensorParamName INPUT_T{"moreh_ln_input_t"};
    const m2::TensorParamName GAMMA_T{"moreh_ln_gamma_t"};
    const m2::TensorParamName BETA_T{"moreh_ln_beta_t"};
    const m2::TensorParamName OUTPUT_T{"moreh_ln_output_t"};
    const m2::TensorParamName MEAN_T{"moreh_ln_mean_t"};
    const m2::TensorParamName RSTD_T{"moreh_ln_rstd_t"};

    const m2::KernelSpecName READER{"moreh_ln_reader"};
    const m2::KernelSpecName WRITER{"moreh_ln_writer"};
    const m2::KernelSpecName COMPUTE_G1{"moreh_ln_compute_g1"};
    const m2::KernelSpecName COMPUTE_G2{"moreh_ln_compute_g2"};

    ////////////////////////////////////////////////////////////////////////////
    //                    DataflowBufferSpecs
    ////////////////////////////////////////////////////////////////////////////
    std::vector<m2::DataflowBufferSpec> dfbs;
    auto push_dfb = [&](const m2::DFBSpecName& name, uint32_t num_tiles, tt::DataFormat fmt) {
        if (num_tiles == 0) {
            // Preserve original behavior: skip zero-size (absent optional) buffers.
            return;
        }
        dfbs.push_back(m2::DataflowBufferSpec{
            .unique_id = name,
            .entry_size = static_cast<uint32_t>(tile_size(fmt)),
            .num_entries = num_tiles,
            .data_format_metadata = fmt,
        });
    };

    push_dfb(INPUT_DFB, in0_t, cb_data_format);
    push_dfb(SCALER_DFB, in1_t, cb_data_format);
    push_dfb(EPS_DFB, in2_t, cb_data_format);
    push_dfb(GAMMA_DFB, in3_t, cb_data_format);
    push_dfb(BETA_DFB, in4_t, cb_data_format);
    push_dfb(MASK_H_DFB, in5_t, cb_data_format);
    push_dfb(MASK_W_DFB, in6_t, cb_data_format);
    push_dfb(OUT_DFB, out0_t, cb_data_format);
    push_dfb(MEAN_DFB, out1_t, cb_data_format);
    push_dfb(RSTD_DFB, out2_t, cb_data_format);
    push_dfb(EX_DFB, im0_t, intermed_cb_format);
    push_dfb(XMM_DFB, im1_t, intermed_cb_format);
    push_dfb(XMM2_DFB, im2_t, intermed_cb_format);
    push_dfb(XMM2SUM_DFB, im3_t, intermed_cb_format);
    push_dfb(VAR_DFB, im4_t, intermed_cb_format);
    push_dfb(RECIP_STD_DFB, im5_t, intermed_cb_format);
    push_dfb(GAMMA_BETA_DFB, im6_t, intermed_cb_format);
    push_dfb(XSUM_DFB, im7_t, intermed_cb_format);

    ////////////////////////////////////////////////////////////////////////////
    //                    Kernel sources (small / large selection)
    ////////////////////////////////////////////////////////////////////////////
    const char* reader_kernel_file =
        use_large_algorithm
            ? "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/reader_moreh_layer_norm_large.cpp"
            : "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/reader_moreh_layer_norm_small.cpp";
    const char* writer_kernel_file =
        "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/writer_moreh_layer_norm.cpp";
    // Metal 2.0 forks of the compute kernels. The legacy compute sources are still borrowed by file
    // path by moreh_group_norm (not yet ported), so they are left untouched; these _metal2 copies
    // carry the DFB / named-arg / #ifdef conversion.
    const char* compute_kernel_file = use_large_algorithm
                                          ? "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/"
                                            "moreh_layer_norm_large_kernel_metal2.cpp"
                                          : "ttnn/cpp/ttnn/operations/moreh/moreh_layer_norm/device/kernels/"
                                            "moreh_layer_norm_small_kernel_metal2.cpp";

    ////////////////////////////////////////////////////////////////////////////
    //                    Defines
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelSpec::CompilerOptions::Defines reader_defines;
    m2::KernelSpec::CompilerOptions::Defines compute_defines;
    m2::KernelSpec::CompilerOptions::Defines writer_defines;
    if (gamma_has_value) {
        reader_defines.emplace("GAMMA_HAS_VALUE", "1");
        compute_defines.emplace("GAMMA_HAS_VALUE", "1");
    }
    if (beta_has_value) {
        reader_defines.emplace("BETA_HAS_VALUE", "1");
        compute_defines.emplace("BETA_HAS_VALUE", "1");
    }
    if (gamma_has_value || beta_has_value) {
        // Gate for the c_30 (gamma*+beta) intermediate DFB and the cb_gamma_beta_or_out selection.
        compute_defines.emplace("GAMMA_OR_BETA", "1");
    }
    if (do_mask_h) {
        reader_defines.emplace("DO_MASK_H", "1");
        compute_defines.emplace("DO_MASK_H", "1");
    }
    if (do_mask_w) {
        reader_defines.emplace("DO_MASK_W", "1");
        compute_defines.emplace("DO_MASK_W", "1");
    }
    if (mean_has_value) {
        compute_defines.emplace("MEAN_HAS_VALUE", "1");
        writer_defines.emplace("MEAN_HAS_VALUE", "1");
    }
    if (rstd_has_value) {
        compute_defines.emplace("RSTD_HAS_VALUE", "1");
        writer_defines.emplace("RSTD_HAS_VALUE", "1");
    }
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

    ////////////////////////////////////////////////////////////////////////////
    //                    Reader / Writer KernelSpecs
    ////////////////////////////////////////////////////////////////////////////
    m2::Group<m2::DFBBinding> reader_dfb_bindings = {
        m2::DFBBinding{
            .dfb_spec_name = INPUT_DFB, .accessor_name = "input", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = SCALER_DFB, .accessor_name = "scaler", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        m2::DFBBinding{
            .dfb_spec_name = EPS_DFB, .accessor_name = "eps", .endpoint_type = m2::DFBEndpointType::PRODUCER},
    };
    m2::Group<m2::TensorBinding> reader_tensor_bindings = {
        m2::TensorBinding{.tensor_parameter_name = INPUT_T, .accessor_name = "input"},
    };
    if (gamma_has_value) {
        reader_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = GAMMA_DFB, .accessor_name = "gamma", .endpoint_type = m2::DFBEndpointType::PRODUCER});
        reader_tensor_bindings.push_back(m2::TensorBinding{.tensor_parameter_name = GAMMA_T, .accessor_name = "gamma"});
    }
    if (beta_has_value) {
        reader_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = BETA_DFB, .accessor_name = "beta", .endpoint_type = m2::DFBEndpointType::PRODUCER});
        reader_tensor_bindings.push_back(m2::TensorBinding{.tensor_parameter_name = BETA_T, .accessor_name = "beta"});
    }
    if (do_mask_h) {
        reader_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = MASK_H_DFB, .accessor_name = "mask_h", .endpoint_type = m2::DFBEndpointType::PRODUCER});
    }
    if (do_mask_w) {
        reader_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = MASK_W_DFB, .accessor_name = "mask_w", .endpoint_type = m2::DFBEndpointType::PRODUCER});
    }

    m2::KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path(reader_kernel_file),
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings = reader_dfb_bindings,
        .tensor_bindings = reader_tensor_bindings,
        .compile_time_args = {{"block_size", block_size}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_rows_per_core", "num_inner", "tile_offset", "scaler", "eps", "mask_h", "mask_w"}},
        .hw_config = ttnn::create_reader_datamovement_config(arch),
    };

    m2::Group<m2::DFBBinding> writer_dfb_bindings = {
        m2::DFBBinding{
            .dfb_spec_name = OUT_DFB, .accessor_name = "output", .endpoint_type = m2::DFBEndpointType::CONSUMER},
    };
    m2::Group<m2::TensorBinding> writer_tensor_bindings = {
        m2::TensorBinding{.tensor_parameter_name = OUTPUT_T, .accessor_name = "output"},
    };
    if (mean_has_value) {
        writer_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = MEAN_DFB, .accessor_name = "mean", .endpoint_type = m2::DFBEndpointType::CONSUMER});
        writer_tensor_bindings.push_back(m2::TensorBinding{.tensor_parameter_name = MEAN_T, .accessor_name = "mean"});
    }
    if (rstd_has_value) {
        writer_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = RSTD_DFB, .accessor_name = "rstd", .endpoint_type = m2::DFBEndpointType::CONSUMER});
        writer_tensor_bindings.push_back(m2::TensorBinding{.tensor_parameter_name = RSTD_T, .accessor_name = "rstd"});
    }

    m2::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path(writer_kernel_file),
        .compiler_options = {.defines = writer_defines},
        .dfb_bindings = writer_dfb_bindings,
        .tensor_bindings = writer_tensor_bindings,
        .compile_time_args = {{"block_size", block_size}},
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"num_rows_per_core",
                  "num_inner",
                  "tile_offset",
                  "mean_rstd_height",
                  "mean_rstd_width",
                  "normalized_dims"}},
        .hw_config = ttnn::create_writer_datamovement_config(arch),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                    Compute KernelSpecs (one per work-split group)
    ////////////////////////////////////////////////////////////////////////////
    // Compute config: Style A — the op resolves a TTNN ComputeKernelConfig, so translate it with
    // to_compute_hardware_config (fpu_math_fidelity / sfpu_precision_mode / enable_32_bit_dest /
    // double_buffer_dest). This reproduces the legacy ComputeConfigDescriptor's four knobs exactly.
    auto compute_hw = ttnn::to_compute_hardware_config(
        arch,
        ttnn::ComputeKernelConfig{
            .math_fidelity = math_fidelity,
            .math_approx_mode = math_approx_mode,
            .fp32_dest_acc_en = fp32_dest_acc_en,
            .dst_full_sync_en = dst_full_sync_en,
        });

    // unpack_modes: the validator requires an explicit entry for every Float32 DFB the compute kernel
    // consumes when enable_32_bit_dest (fp32_dest_acc_en) is true. The legacy op set no
    // unpack_to_dest_mode (all Default) → UnpackToSrc. Only the intermediate DFBs are Float32 (and only
    // when fp32_dest_acc_en); the bfloat16 input/scaler/etc DFBs need no entry.
    if (fp32_dest_acc_en) {
        m2::ComputeUnpackModes unpack_modes;
        unpack_modes.emplace(EX_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        unpack_modes.emplace(XMM_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        unpack_modes.emplace(XMM2_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        unpack_modes.emplace(XMM2SUM_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        unpack_modes.emplace(VAR_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        unpack_modes.emplace(RECIP_STD_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        unpack_modes.emplace(XSUM_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        if (gamma_has_value || beta_has_value) {
            unpack_modes.emplace(GAMMA_BETA_DFB, tt::tt_metal::UnpackMode::UnpackToSrc);
        }
        std::visit([&](auto& cfg) { cfg.unpack_modes = unpack_modes; }, compute_hw);
    }

    auto make_compute_dfb_bindings = [&]() {
        m2::Group<m2::DFBBinding> b = {
            m2::DFBBinding{
                .dfb_spec_name = INPUT_DFB, .accessor_name = "x", .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = SCALER_DFB, .accessor_name = "scaler", .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = EPS_DFB, .accessor_name = "eps", .endpoint_type = m2::DFBEndpointType::CONSUMER},
            m2::DFBBinding{
                .dfb_spec_name = OUT_DFB, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::PRODUCER},
        };
        if (gamma_has_value) {
            b.push_back(m2::DFBBinding{
                .dfb_spec_name = GAMMA_DFB, .accessor_name = "gamma", .endpoint_type = m2::DFBEndpointType::CONSUMER});
        }
        if (beta_has_value) {
            b.push_back(m2::DFBBinding{
                .dfb_spec_name = BETA_DFB, .accessor_name = "beta", .endpoint_type = m2::DFBEndpointType::CONSUMER});
        }
        if (do_mask_h) {
            b.push_back(m2::DFBBinding{
                .dfb_spec_name = MASK_H_DFB,
                .accessor_name = "mask_h",
                .endpoint_type = m2::DFBEndpointType::CONSUMER});
        }
        if (do_mask_w) {
            b.push_back(m2::DFBBinding{
                .dfb_spec_name = MASK_W_DFB,
                .accessor_name = "mask_w",
                .endpoint_type = m2::DFBEndpointType::CONSUMER});
        }
        if (mean_has_value) {
            b.push_back(m2::DFBBinding{
                .dfb_spec_name = MEAN_DFB, .accessor_name = "mean", .endpoint_type = m2::DFBEndpointType::PRODUCER});
        }
        if (rstd_has_value) {
            b.push_back(m2::DFBBinding{
                .dfb_spec_name = RSTD_DFB, .accessor_name = "rstd", .endpoint_type = m2::DFBEndpointType::PRODUCER});
        }
        // Self-loop intermediates: each compute KernelSpec both produces and consumes them.
        auto self_loop = [&](const m2::DFBSpecName& name, const char* acc) {
            b.push_back(m2::DFBBinding{
                .dfb_spec_name = name, .accessor_name = acc, .endpoint_type = m2::DFBEndpointType::PRODUCER});
            b.push_back(m2::DFBBinding{
                .dfb_spec_name = name, .accessor_name = acc, .endpoint_type = m2::DFBEndpointType::CONSUMER});
        };
        self_loop(EX_DFB, "ex");
        self_loop(XMM_DFB, "xmm");
        self_loop(XMM2_DFB, "xmm2");
        self_loop(XMM2SUM_DFB, "xmm2sum");
        self_loop(VAR_DFB, "var");
        self_loop(RECIP_STD_DFB, "recip_std");
        if (gamma_has_value || beta_has_value) {
            self_loop(GAMMA_BETA_DFB, "gamma_beta");
        }
        self_loop(XSUM_DFB, "xsum");
        return b;
    };

    auto make_compute = [&](const m2::KernelSpecName& id, uint32_t num_rows_per_core_group) {
        return m2::KernelSpec{
            .unique_id = id,
            .source = std::filesystem::path(compute_kernel_file),
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings = make_compute_dfb_bindings(),
            .compile_time_args =
                {{"num_rows_per_core", num_rows_per_core_group},
                 {"origin_H", static_cast<uint32_t>(origin_H)},
                 {"origin_W", static_cast<uint32_t>(origin_W)},
                 {"num_inner", num_inner},
                 {"block_size", block_size},
                 {"is_lastdim_layernorm", static_cast<uint32_t>(is_lastdim_layer_norm)},
                 {"is_groupnorm", static_cast<uint32_t>(is_groupnorm)}},
            .hw_config = compute_hw,
        };
    };

    ////////////////////////////////////////////////////////////////////////////
    //                    Assemble kernels + work units
    ////////////////////////////////////////////////////////////////////////////
    m2::Group<m2::KernelSpec> kernels = {reader_spec, writer_spec};
    m2::Group<m2::WorkUnitSpec> work_units;

    kernels.push_back(make_compute(COMPUTE_G1, num_rows_per_core_group_1));
    work_units.push_back(
        m2::WorkUnitSpec{.name = "moreh_ln_g1", .kernels = {READER, WRITER, COMPUTE_G1}, .target_nodes = core_group_1});

    const bool has_core_group_2 = !core_group_2.ranges().empty();
    if (has_core_group_2) {
        kernels.push_back(make_compute(COMPUTE_G2, num_rows_per_core_group_2));
        work_units.push_back(m2::WorkUnitSpec{
            .name = "moreh_ln_g2", .kernels = {READER, WRITER, COMPUTE_G2}, .target_nodes = core_group_2});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                    TensorParameters
    ////////////////////////////////////////////////////////////////////////////
    // Bind against the real io tensors (from tensor_args / output_tensor) — the framework matches a
    // TensorArgument back to its parameter by MeshTensor identity, so a copy (e.g. the local
    // optional<const Tensor> views above) must not be used here.
    const Tensor& output_ref = output_tensor.at(0).value();
    m2::Group<m2::TensorParameter> tensor_parameters = {
        m2::TensorParameter{.unique_id = INPUT_T, .spec = input.tensor_spec()},
        m2::TensorParameter{.unique_id = OUTPUT_T, .spec = output_ref.tensor_spec()},
    };
    if (gamma_has_value) {
        tensor_parameters.push_back(m2::TensorParameter{.unique_id = GAMMA_T, .spec = gamma->tensor_spec()});
    }
    if (beta_has_value) {
        tensor_parameters.push_back(m2::TensorParameter{.unique_id = BETA_T, .spec = beta->tensor_spec()});
    }
    if (mean_has_value) {
        tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = MEAN_T, .spec = output_tensor.at(1).value().tensor_spec()});
    }
    if (rstd_has_value) {
        tensor_parameters.push_back(
            m2::TensorParameter{.unique_id = RSTD_T, .spec = output_tensor.at(2).value().tensor_spec()});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                    RuntimeArgs (name-first, per node)
    ////////////////////////////////////////////////////////////////////////////
    float scaler_f = 0.0f;
    if (normalized_dims == 1) {
        scaler_f = 1.0f / static_cast<float>(origin_W);
    } else {
        uint32_t reduce_size = 1;
        for (uint32_t i = input_rank - normalized_dims; i < input_rank; i++) {
            auto size = input_shape_without_padding[i];
            reduce_size *= size;
        }

        scaler_f = 1.0f / std::sqrt(static_cast<float>(reduce_size));
    }
    const uint32_t scaler_u = std::bit_cast<uint32_t>(scaler_f);
    const uint32_t e_u = std::bit_cast<uint32_t>(eps);  // epsilon

    m2::KernelRunArgs::RuntimeArgValues reader_args, writer_args;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const m2::NodeCoord node{static_cast<uint32_t>(core.x), static_cast<uint32_t>(core.y)};

        uint32_t num_rows_per_core;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        reader_args["num_rows_per_core"][node] = num_rows_per_core;
        reader_args["num_inner"][node] = num_inner;
        reader_args["tile_offset"][node] = tile_offset;
        reader_args["scaler"][node] = scaler_u;
        reader_args["eps"][node] = e_u;
        reader_args["mask_h"][node] = mask_h;
        reader_args["mask_w"][node] = mask_w;

        writer_args["num_rows_per_core"][node] = num_rows_per_core;
        writer_args["num_inner"][node] = num_inner;
        writer_args["tile_offset"][node] = tile_offset;
        writer_args["mean_rstd_height"][node] = mean_rstd_height;
        writer_args["mean_rstd_width"][node] = mean_rstd_width;
        writer_args["normalized_dims"][node] = normalized_dims;

        tile_offset += num_rows_per_core * num_inner;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                    Assemble ProgramSpec + ProgramRunArgs
    ////////////////////////////////////////////////////////////////////////////
    m2::ProgramSpec spec{
        .name = "moreh_layer_norm",
        .kernels = std::move(kernels),
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    m2::ProgramRunArgs run_params;
    run_params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{.kernel = READER, .runtime_arg_values = std::move(reader_args)},
        m2::ProgramRunArgs::KernelRunArgs{.kernel = WRITER, .runtime_arg_values = std::move(writer_args)},
    };

    run_params.tensor_args.emplace(INPUT_T, m2::ProgramRunArgs::TensorArgument{input.mesh_tensor()});
    run_params.tensor_args.emplace(OUTPUT_T, m2::ProgramRunArgs::TensorArgument{output_ref.mesh_tensor()});
    if (gamma_has_value) {
        run_params.tensor_args.emplace(GAMMA_T, m2::ProgramRunArgs::TensorArgument{gamma->mesh_tensor()});
    }
    if (beta_has_value) {
        run_params.tensor_args.emplace(BETA_T, m2::ProgramRunArgs::TensorArgument{beta->mesh_tensor()});
    }
    if (mean_has_value) {
        run_params.tensor_args.emplace(
            MEAN_T, m2::ProgramRunArgs::TensorArgument{output_tensor.at(1).value().mesh_tensor()});
    }
    if (rstd_has_value) {
        run_params.tensor_args.emplace(
            RSTD_T, m2::ProgramRunArgs::TensorArgument{output_tensor.at(2).value().mesh_tensor()});
    }

    return ttnn::device_operation::ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::operations::moreh::moreh_layer_norm
