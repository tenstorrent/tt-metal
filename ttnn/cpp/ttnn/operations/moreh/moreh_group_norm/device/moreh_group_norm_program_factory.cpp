// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Metal 2.0 / DataflowBuffer (DFB) program factory for moreh_group_norm.
//
// Faithful port of the legacy ProgramDescriptor factory (create_descriptor). It preserves the op's
// logic — parameter derivation, the use_large_algorithm small/large kernel selection, the
// per-config CB push (push_cb_if_nonzero), the per-core work split and runtime args — and differs
// only in the CB->DFB / named-binding translation:
//   - CBDescriptor -> DataflowBufferSpec (one per present CB index).
//   - Buffer* runtime args (input/gamma/beta, output/mean/rstd) -> typed TensorParameter /
//     TensorBinding; kernels build TensorAccessor(tensor::name).
//   - TensorAccessorArgs(...).append_to(cta) plumbing -> the binding mechanism.
//   - Positional CTAs -> named CTAs (compute) or #define presence flags (conditional bindings).
//   - Magic CB ids -> dfb::name handles.
// The compute kernels are FORKED into this op's own directory (kernels/compute/
// moreh_group_norm_{small,large}_kernel.cpp) from the legacy moreh_layer_norm kernels, because the
// donor op is being ported in parallel; this port is fully self-contained.

#include "moreh_group_norm_device_operation.hpp"
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

#include <bit>
#include <cmath>
#include <filesystem>
#include <optional>
#include <vector>

inline uint32_t get_block_size(uint32_t num_tiles, uint32_t max_block_size) {
    uint32_t block_size{1};
    for (uint32_t current_block_size = max_block_size; current_block_size >= 1; current_block_size >>= 1) {
        if (num_tiles % current_block_size == 0) {
            block_size = current_block_size;
            break;
        }
    }
    return block_size;
}

namespace ttnn::operations::moreh::moreh_group_norm {

using namespace tt;
using namespace tt::constants;
using namespace tt::tt_metal;
namespace m2 = tt::tt_metal::experimental;

using ttnn::device_operation::ProgramArtifacts;

ttnn::device_operation::ProgramArtifacts MorehGroupNormOperation::ProgramFactory::create_program_artifacts(
    const operation_attributes_t& operation_attributes,
    const tensor_args_t& tensor_args,
    tensor_return_value_t& outputs) {
    const auto& input = tensor_args.input;
    auto gamma = tensor_args.gamma;
    auto beta = tensor_args.beta;
    auto mean = outputs[1];
    auto rstd = outputs[2];

    auto& output = outputs[0].value();

    auto num_groups = operation_attributes.num_groups;
    auto eps = operation_attributes.eps;

    ////////////////////////////////////////////////////////////////////////////
    //                      Device Setup
    ////////////////////////////////////////////////////////////////////////////
    auto* device = input.device();
    // Compute config is translated wholesale via to_compute_hardware_config below (Style A). Only
    // fp32_dest_acc_en is needed separately here, to decide whether the validator-required Float32
    // unpack_modes entries must be emitted. The resolved value lives on the (already-resolved)
    // operation_attributes.compute_kernel_config.
    const bool fp32_dest_acc_en = operation_attributes.compute_kernel_config.fp32_dest_acc_en;

    ////////////////////////////////////////////////////////////////////////////
    //                         Parameters Setup
    ////////////////////////////////////////////////////////////////////////////
    const auto input_shape = input.padded_shape();

    const auto n = input_shape[0];
    const auto c = input_shape[1];
    const auto h = input_shape[2];
    const auto w = input_shape[3];

    const auto origin_input_shape = input.logical_shape();

    const auto origin_h = origin_input_shape[2];
    const auto origin_w = origin_input_shape[3];

    const bool is_lastdim_layernorm = false;
    const bool is_group_norm = true;

    const bool do_mask_h = (origin_h % TILE_HEIGHT) != 0;
    const bool do_mask_w = (origin_w % TILE_WIDTH) != 0;

    const auto Ht = h / TILE_HEIGHT;
    const auto Wt = w / TILE_WIDTH;

    const auto num_channels = c;
    const auto num_rows = n * num_groups;
    TT_FATAL(
        num_channels % num_groups == 0,
        "Group norm requires num_channels ({}) to be divisible by num_groups ({})",
        num_channels,
        num_groups);
    const auto num_inner_tiles = (num_channels / num_groups) * Ht * Wt;

    const auto f_c = static_cast<float>(num_channels) / static_cast<float>(num_groups);
    const auto f_ht = static_cast<float>(origin_h) / static_cast<float>(TILE_HEIGHT);
    const auto f_wt = static_cast<float>(origin_w) / static_cast<float>(TILE_WIDTH);
    float scaler = 1.0f / (static_cast<float>(TILE_WIDTH) * std::sqrt(f_c * f_ht * f_wt));

    const bool gamma_has_value = gamma.has_value();
    const bool beta_has_value = beta.has_value();
    const bool mean_has_value = mean.has_value();
    const bool rstd_has_value = rstd.has_value();

    constexpr uint32_t MAX_BLOCK_SIZE = 8;
    const uint32_t block_size = get_block_size(num_inner_tiles, MAX_BLOCK_SIZE);

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
         num_rows_per_core_group_2] = split_work_to_cores(grid, num_rows);

    log_debug(LogTest, "num_cores_to_be_used: {}", num_cores_to_be_used);
    log_debug(LogTest, "num_rows_per_core_group_1: {}", num_rows_per_core_group_1);
    log_debug(LogTest, "num_rows_per_core_group_2: {}", num_rows_per_core_group_2);
    log_debug(LogTest, "block_size: {}", block_size);

    ////////////////////////////////////////////////////////////////////////////
    //                         DataflowBuffer Setup
    ////////////////////////////////////////////////////////////////////////////
    uint32_t in0_t = num_inner_tiles;                         // input
    const uint32_t in1_t = 1;                                 // scaler
    const uint32_t in2_t = 1;                                 // epsilon
    const uint32_t in3_t = gamma_has_value ? block_size : 0;  // gamma
    const uint32_t in4_t = beta_has_value ? block_size : 0;   // beta
    const uint32_t in5_t = do_mask_h ? 1 : 0;                 // mask_h
    const uint32_t in6_t = do_mask_w ? 1 : 0;                 // mask_w

    const uint32_t out0_t = block_size;              // output
    const uint32_t out1_t = mean_has_value ? 1 : 0;  // mean
    const uint32_t out2_t = rstd_has_value ? 1 : 0;  // rstd

    const uint32_t im0_t = 1;                                                         // E[x]
    uint32_t im1_t = num_inner_tiles;                                                 // x - E[x]
    uint32_t im2_t = 1;                                                               // (x - E[x])^2
    const uint32_t im3_t = 1;                                                         // Sum[(x - E[x])^2]
    const uint32_t im4_t = 1;                                                         // E[(x - E[x])^2] = Var[x]
    const uint32_t im5_t = 1;                                                         // 1.0/(sqrt(Var[x] + eps))
    const uint32_t im6_t = (gamma_has_value || beta_has_value) ? 2 * block_size : 0;  // x * gamm + beta
    const uint32_t im7_t = 2;                                                         // Sum[x]

    const auto cb_data_format = datatype_to_dataformat_converter(input.dtype());
    const auto single_tile_size = tt::tile_size(cb_data_format);
    const auto tile = input.tensor_spec().tile();

    const auto cb_usage = (in0_t + in1_t + in2_t + in3_t + in4_t + in5_t + in6_t + out0_t + out1_t + out2_t + im0_t +
                           im1_t + im2_t + im3_t + im4_t + im5_t + im6_t + im7_t) *
                          single_tile_size;
    const auto available_L1 = device->l1_size_per_core() - device->allocator()->get_base_allocator_addr(HalMemType::L1);
    const bool use_large_algorithm = cb_usage >= available_L1;

    if (use_large_algorithm) {
        log_info(LogTest, "Large moreh_group_norm algorithm is selected.");
        in0_t = block_size;
        im1_t = 2 * block_size;
        im2_t = 2 * block_size;
    } else {
        log_info(LogTest, "Small moreh_group_norm algorithm is selected.");
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         DFB / tensor / kernel names
    ////////////////////////////////////////////////////////////////////////////
    // DFB spec names (mirror the legacy CBIndex assignments).
    const m2::DFBSpecName D_INPUT{"gn_input"};      // c_0
    const m2::DFBSpecName D_SCALER{"gn_scaler"};    // c_1
    const m2::DFBSpecName D_EPS{"gn_eps"};          // c_2
    const m2::DFBSpecName D_GAMMA{"gn_gamma"};      // c_3
    const m2::DFBSpecName D_BETA{"gn_beta"};        // c_4
    const m2::DFBSpecName D_MASK_H{"gn_mask_h"};    // c_5
    const m2::DFBSpecName D_MASK_W{"gn_mask_w"};    // c_6
    const m2::DFBSpecName D_OUTPUT{"gn_output"};    // c_16
    const m2::DFBSpecName D_MEAN{"gn_mean"};        // c_17
    const m2::DFBSpecName D_RSTD{"gn_rstd"};        // c_18
    const m2::DFBSpecName D_EX{"gn_ex"};            // c_24
    const m2::DFBSpecName D_XMM{"gn_xmm"};          // c_25
    const m2::DFBSpecName D_XMM2{"gn_xmm2"};        // c_26
    const m2::DFBSpecName D_XMM2SUM{"gn_xmm2sum"};  // c_27
    const m2::DFBSpecName D_VAR{"gn_var"};          // c_28
    const m2::DFBSpecName D_RECIP{"gn_recip_std"};  // c_29
    const m2::DFBSpecName D_GAMMA_BETA{"gn_gb"};    // c_30
    const m2::DFBSpecName D_XSUM{"gn_xsum"};        // c_31

    const m2::TensorParamName T_INPUT{"gn_t_input"};
    const m2::TensorParamName T_GAMMA{"gn_t_gamma"};
    const m2::TensorParamName T_BETA{"gn_t_beta"};
    const m2::TensorParamName T_OUTPUT{"gn_t_output"};
    const m2::TensorParamName T_MEAN{"gn_t_mean"};
    const m2::TensorParamName T_RSTD{"gn_t_rstd"};

    const m2::KernelSpecName READER{"gn_reader"};
    const m2::KernelSpecName WRITER{"gn_writer"};
    const m2::KernelSpecName COMPUTE_G1{"gn_compute_g1"};
    const m2::KernelSpecName COMPUTE_G2{"gn_compute_g2"};

    ////////////////////////////////////////////////////////////////////////////
    //                         DataflowBufferSpecs (mirror push_cb_if_nonzero)
    ////////////////////////////////////////////////////////////////////////////
    std::vector<m2::DataflowBufferSpec> dfbs;
    auto push_dfb_if_nonzero = [&](const m2::DFBSpecName& name, uint32_t num_tiles) {
        if (num_tiles > 0) {
            dfbs.push_back(m2::DataflowBufferSpec{
                .unique_id = name,
                .entry_size = single_tile_size,
                .num_entries = num_tiles,
                .data_format_metadata = cb_data_format,
                .tile_format_metadata = tile,
            });
        }
    };
    push_dfb_if_nonzero(D_INPUT, in0_t);
    push_dfb_if_nonzero(D_SCALER, in1_t);
    push_dfb_if_nonzero(D_EPS, in2_t);
    push_dfb_if_nonzero(D_GAMMA, in3_t);
    push_dfb_if_nonzero(D_BETA, in4_t);
    push_dfb_if_nonzero(D_MASK_H, in5_t);
    push_dfb_if_nonzero(D_MASK_W, in6_t);
    push_dfb_if_nonzero(D_OUTPUT, out0_t);
    push_dfb_if_nonzero(D_MEAN, out1_t);
    push_dfb_if_nonzero(D_RSTD, out2_t);
    push_dfb_if_nonzero(D_EX, im0_t);
    push_dfb_if_nonzero(D_XMM, im1_t);
    push_dfb_if_nonzero(D_XMM2, im2_t);
    push_dfb_if_nonzero(D_XMM2SUM, im3_t);
    push_dfb_if_nonzero(D_VAR, im4_t);
    push_dfb_if_nonzero(D_RECIP, im5_t);
    push_dfb_if_nonzero(D_GAMMA_BETA, im6_t);
    push_dfb_if_nonzero(D_XSUM, im7_t);

    ////////////////////////////////////////////////////////////////////////////
    //                         Tensor parameters
    ////////////////////////////////////////////////////////////////////////////
    m2::Group<m2::TensorParameter> tensor_parameters;
    tensor_parameters.push_back(m2::TensorParameter{.unique_id = T_INPUT, .spec = input.tensor_spec()});
    tensor_parameters.push_back(m2::TensorParameter{.unique_id = T_OUTPUT, .spec = output.tensor_spec()});
    if (gamma_has_value) {
        tensor_parameters.push_back(m2::TensorParameter{.unique_id = T_GAMMA, .spec = gamma->tensor_spec()});
    }
    if (beta_has_value) {
        tensor_parameters.push_back(m2::TensorParameter{.unique_id = T_BETA, .spec = beta->tensor_spec()});
    }
    if (mean_has_value) {
        tensor_parameters.push_back(m2::TensorParameter{.unique_id = T_MEAN, .spec = mean->tensor_spec()});
    }
    if (rstd_has_value) {
        tensor_parameters.push_back(m2::TensorParameter{.unique_id = T_RSTD, .spec = rstd->tensor_spec()});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Kernel defines (conditional-binding gates)
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelSpec::CompilerOptions::Defines reader_defines;
    m2::KernelSpec::CompilerOptions::Defines writer_defines;
    m2::KernelSpec::CompilerOptions::Defines compute_defines;
    compute_defines.emplace("REDUCE_OP", "PoolType::AVG");
    compute_defines.emplace("REDUCE_DIM", "ReduceDim::REDUCE_SCALAR");
    if (gamma_has_value) {
        reader_defines.emplace("GAMMA_HAS_VALUE", "1");
        compute_defines.emplace("GAMMA_HAS_VALUE", "1");
    }
    if (beta_has_value) {
        reader_defines.emplace("BETA_HAS_VALUE", "1");
        compute_defines.emplace("BETA_HAS_VALUE", "1");
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
        writer_defines.emplace("MEAN_HAS_VALUE", "1");
        compute_defines.emplace("MEAN_HAS_VALUE", "1");
    }
    if (rstd_has_value) {
        writer_defines.emplace("RSTD_HAS_VALUE", "1");
        compute_defines.emplace("RSTD_HAS_VALUE", "1");
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Reader kernel
    ////////////////////////////////////////////////////////////////////////////
    const char* reader_kernel_file = use_large_algorithm
                                         ? "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/kernels/dataflow/"
                                           "reader_moreh_group_norm_large.cpp"
                                         : "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/kernels/dataflow/"
                                           "reader_moreh_group_norm_small.cpp";

    m2::Group<m2::DFBBinding> reader_dfb_bindings;
    reader_dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = D_INPUT, .accessor_name = "input", .endpoint_type = m2::DFBEndpointType::PRODUCER});
    reader_dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = D_SCALER, .accessor_name = "scaler", .endpoint_type = m2::DFBEndpointType::PRODUCER});
    reader_dfb_bindings.push_back(
        m2::DFBBinding{.dfb_spec_name = D_EPS, .accessor_name = "eps", .endpoint_type = m2::DFBEndpointType::PRODUCER});
    if (gamma_has_value) {
        reader_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = D_GAMMA, .accessor_name = "gamma", .endpoint_type = m2::DFBEndpointType::PRODUCER});
    }
    if (beta_has_value) {
        reader_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = D_BETA, .accessor_name = "beta", .endpoint_type = m2::DFBEndpointType::PRODUCER});
    }
    if (do_mask_h) {
        reader_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = D_MASK_H, .accessor_name = "mask_h", .endpoint_type = m2::DFBEndpointType::PRODUCER});
    }
    if (do_mask_w) {
        reader_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = D_MASK_W, .accessor_name = "mask_w", .endpoint_type = m2::DFBEndpointType::PRODUCER});
    }

    m2::Group<m2::TensorBinding> reader_tensor_bindings;
    reader_tensor_bindings.push_back(m2::TensorBinding{T_INPUT, "input"});
    if (gamma_has_value) {
        reader_tensor_bindings.push_back(m2::TensorBinding{T_GAMMA, "gamma"});
    }
    if (beta_has_value) {
        reader_tensor_bindings.push_back(m2::TensorBinding{T_BETA, "beta"});
    }

    m2::KernelSpec reader_spec{
        .unique_id = READER,
        .source = std::filesystem::path(reader_kernel_file),
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings = reader_dfb_bindings,
        .tensor_bindings = reader_tensor_bindings,
        .runtime_arg_schema =
            {.runtime_arg_names =
                 {"scaler",
                  "eps",
                  "tile_offset",
                  "num_rows_per_core",
                  "num_inner_tiles",
                  "num_channels",
                  "origin_h",
                  "origin_w",
                  "block_size"}},
        .hw_config = ttnn::create_reader_datamovement_config(device->arch()),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                         Writer kernel
    ////////////////////////////////////////////////////////////////////////////
    static constexpr const char* WRITER_KERNEL_PATH =
        "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/kernels/dataflow/writer_moreh_group_norm.cpp";

    m2::Group<m2::DFBBinding> writer_dfb_bindings;
    writer_dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = D_OUTPUT, .accessor_name = "output", .endpoint_type = m2::DFBEndpointType::CONSUMER});
    if (mean_has_value) {
        writer_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = D_MEAN, .accessor_name = "mean", .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }
    if (rstd_has_value) {
        writer_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = D_RSTD, .accessor_name = "rstd", .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }

    m2::Group<m2::TensorBinding> writer_tensor_bindings;
    writer_tensor_bindings.push_back(m2::TensorBinding{T_OUTPUT, "output"});
    if (mean_has_value) {
        writer_tensor_bindings.push_back(m2::TensorBinding{T_MEAN, "mean"});
    }
    if (rstd_has_value) {
        writer_tensor_bindings.push_back(m2::TensorBinding{T_RSTD, "rstd"});
    }

    m2::KernelSpec writer_spec{
        .unique_id = WRITER,
        .source = std::filesystem::path(WRITER_KERNEL_PATH),
        .compiler_options = {.defines = writer_defines},
        .dfb_bindings = writer_dfb_bindings,
        .tensor_bindings = writer_tensor_bindings,
        .runtime_arg_schema =
            {.runtime_arg_names = {"tile_offset", "num_rows_per_core", "num_inner_tiles", "num_groups", "block_size"}},
        .hw_config = ttnn::create_writer_datamovement_config(device->arch()),
    };

    ////////////////////////////////////////////////////////////////////////////
    //                         Compute kernel(s)
    ////////////////////////////////////////////////////////////////////////////
    const char* compute_kernel_file =
        use_large_algorithm
            ? "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/kernels/compute/moreh_group_norm_large_kernel.cpp"
            : "ttnn/cpp/ttnn/operations/moreh/moreh_group_norm/device/kernels/compute/"
              "moreh_group_norm_small_kernel.cpp";

    m2::Group<m2::DFBBinding> compute_dfb_bindings;
    // Consumed inputs.
    compute_dfb_bindings.push_back(
        m2::DFBBinding{.dfb_spec_name = D_INPUT, .accessor_name = "x", .endpoint_type = m2::DFBEndpointType::CONSUMER});
    compute_dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = D_SCALER, .accessor_name = "scaler", .endpoint_type = m2::DFBEndpointType::CONSUMER});
    compute_dfb_bindings.push_back(
        m2::DFBBinding{.dfb_spec_name = D_EPS, .accessor_name = "eps", .endpoint_type = m2::DFBEndpointType::CONSUMER});
    if (gamma_has_value) {
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = D_GAMMA, .accessor_name = "gamma", .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }
    if (beta_has_value) {
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = D_BETA, .accessor_name = "beta", .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }
    if (do_mask_h) {
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = D_MASK_H, .accessor_name = "mask_h", .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }
    if (do_mask_w) {
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = D_MASK_W, .accessor_name = "mask_w", .endpoint_type = m2::DFBEndpointType::CONSUMER});
    }
    // Produced outputs.
    compute_dfb_bindings.push_back(m2::DFBBinding{
        .dfb_spec_name = D_OUTPUT, .accessor_name = "out", .endpoint_type = m2::DFBEndpointType::PRODUCER});
    if (mean_has_value) {
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = D_MEAN, .accessor_name = "mean", .endpoint_type = m2::DFBEndpointType::PRODUCER});
    }
    if (rstd_has_value) {
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = D_RSTD, .accessor_name = "rstd", .endpoint_type = m2::DFBEndpointType::PRODUCER});
    }
    // Self-loop intermediates (compute produces and consumes each).
    auto self_loop = [&](const m2::DFBSpecName& name, const std::string& acc) {
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = name, .accessor_name = acc, .endpoint_type = m2::DFBEndpointType::PRODUCER});
        compute_dfb_bindings.push_back(m2::DFBBinding{
            .dfb_spec_name = name, .accessor_name = acc, .endpoint_type = m2::DFBEndpointType::CONSUMER});
    };
    self_loop(D_EX, "ex");
    self_loop(D_XMM, "xmm");
    self_loop(D_XMM2, "xmm2");
    self_loop(D_XMM2SUM, "xmm2sum");
    self_loop(D_VAR, "var");
    self_loop(D_RECIP, "recip_std");
    if (gamma_has_value || beta_has_value) {
        self_loop(D_GAMMA_BETA, "gamma_beta");
    }
    self_loop(D_XSUM, "xsum");

    // Compute hardware config (Style A: translate the resolved TTNN ComputeKernelConfig). The four
    // knobs (math_fidelity / math_approx_mode / fp32_dest_acc_en / dst_full_sync_en) carry over
    // exactly as the legacy ComputeConfigDescriptor set them.
    auto compute_hw = ttnn::to_compute_hardware_config(device->arch(), operation_attributes.compute_kernel_config);

    // unpack_modes: the legacy kernel left unpack_to_dest_mode at its default (== UnpackToSrc). The
    // Metal 2.0 validator additionally REQUIRES an explicit entry for any compute-consumed Float32 DFB
    // when enable_32_bit_dest (fp32_dest_acc_en) is true. In that case only, add the legacy-equivalent
    // UnpackToSrc entry for each present compute-consumed DFB; otherwise leave default (omit).
    if (fp32_dest_acc_en && cb_data_format == tt::DataFormat::Float32) {
        m2::ComputeUnpackModes unpack_modes;
        auto add_src = [&](const m2::DFBSpecName& name) { unpack_modes.emplace(name, UnpackMode::UnpackToSrc); };
        add_src(D_INPUT);
        add_src(D_SCALER);
        add_src(D_EPS);
        if (gamma_has_value) {
            add_src(D_GAMMA);
        }
        if (beta_has_value) {
            add_src(D_BETA);
        }
        if (do_mask_h) {
            add_src(D_MASK_H);
        }
        if (do_mask_w) {
            add_src(D_MASK_W);
        }
        add_src(D_EX);
        add_src(D_XMM);
        add_src(D_XMM2);
        add_src(D_XMM2SUM);
        add_src(D_VAR);
        add_src(D_RECIP);
        if (gamma_has_value || beta_has_value) {
            add_src(D_GAMMA_BETA);
        }
        add_src(D_XSUM);
        std::visit([&](auto& cfg) { cfg.unpack_modes = unpack_modes; }, compute_hw);
    }

    auto make_compute_spec = [&](const m2::KernelSpecName& id, uint32_t num_rows_per_core) {
        return m2::KernelSpec{
            .unique_id = id,
            .source = std::filesystem::path(compute_kernel_file),
            .compiler_options = {.defines = compute_defines},
            .dfb_bindings = compute_dfb_bindings,
            .compile_time_args =
                {{"num_rows_per_core", num_rows_per_core},
                 {"origin_H", static_cast<uint32_t>(origin_h)},
                 {"origin_W", static_cast<uint32_t>(origin_w)},
                 {"num_inner", static_cast<uint32_t>(num_inner_tiles)},
                 {"block_size", static_cast<uint32_t>(block_size)},
                 {"is_lastdim_layernorm", static_cast<uint32_t>(is_lastdim_layernorm)},
                 {"is_groupnorm", static_cast<uint32_t>(is_group_norm)}},
            .hw_config = compute_hw,
        };
    };

    const bool has_core_group_2 = !core_group_2.ranges().empty();

    m2::KernelSpec compute_spec_1 = make_compute_spec(COMPUTE_G1, num_rows_per_core_group_1);
    m2::Group<m2::KernelSpec> kernels = {reader_spec, writer_spec, compute_spec_1};
    if (has_core_group_2) {
        kernels.push_back(make_compute_spec(COMPUTE_G2, num_rows_per_core_group_2));
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Runtime args (per node)
    ////////////////////////////////////////////////////////////////////////////
    m2::KernelRunArgs::RuntimeArgValues reader_args;
    m2::KernelRunArgs::RuntimeArgValues writer_args;

    for (uint32_t i = 0, tile_offset = 0; i < num_cores_to_be_used; ++i) {
        CoreCoord core = {i / num_cores_y, i % num_cores_y};
        const m2::NodeCoord node{core.x, core.y};

        uint32_t num_rows_per_core;
        if (core_group_1.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_rows_per_core = num_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges.");
        }

        reader_args["scaler"][node] = std::bit_cast<uint32_t>(scaler);
        reader_args["eps"][node] = std::bit_cast<uint32_t>(eps);
        reader_args["tile_offset"][node] = tile_offset;
        reader_args["num_rows_per_core"][node] = num_rows_per_core;
        reader_args["num_inner_tiles"][node] = num_inner_tiles;
        reader_args["num_channels"][node] = num_channels;
        reader_args["origin_h"][node] = origin_h;
        reader_args["origin_w"][node] = origin_w;
        reader_args["block_size"][node] = block_size;

        writer_args["tile_offset"][node] = tile_offset;
        writer_args["num_rows_per_core"][node] = num_rows_per_core;
        writer_args["num_inner_tiles"][node] = num_inner_tiles;
        writer_args["num_groups"][node] = num_groups;
        writer_args["block_size"][node] = block_size;

        tile_offset += num_rows_per_core * num_inner_tiles;
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Work units
    ////////////////////////////////////////////////////////////////////////////
    m2::Group<m2::WorkUnitSpec> work_units;
    work_units.push_back(m2::WorkUnitSpec{
        .name = "gn_wu_group_1", .kernels = {READER, WRITER, COMPUTE_G1}, .target_nodes = core_group_1});
    if (has_core_group_2) {
        work_units.push_back(m2::WorkUnitSpec{
            .name = "gn_wu_group_2", .kernels = {READER, WRITER, COMPUTE_G2}, .target_nodes = core_group_2});
    }

    ////////////////////////////////////////////////////////////////////////////
    //                         Assemble spec + run args
    ////////////////////////////////////////////////////////////////////////////
    m2::ProgramSpec spec{
        .name = "moreh_group_norm",
        .kernels = kernels,
        .dataflow_buffers = dfbs,
        .tensor_parameters = tensor_parameters,
        .work_units = work_units,
    };

    m2::ProgramRunArgs run_params;
    run_params.kernel_run_args = {
        m2::ProgramRunArgs::KernelRunArgs{.kernel = READER, .runtime_arg_values = std::move(reader_args)},
        m2::ProgramRunArgs::KernelRunArgs{.kernel = WRITER, .runtime_arg_values = std::move(writer_args)},
    };

    // Bind each TensorParameter to the MeshTensor of the SOURCE io tensor (in tensor_args /
    // tensor_return_value); the framework matches TensorArguments by MeshTensor identity.
    run_params.tensor_args.emplace(T_INPUT, m2::ProgramRunArgs::TensorArgument{tensor_args.input.mesh_tensor()});
    run_params.tensor_args.emplace(T_OUTPUT, m2::ProgramRunArgs::TensorArgument{outputs[0]->mesh_tensor()});
    if (gamma_has_value) {
        run_params.tensor_args.emplace(T_GAMMA, m2::ProgramRunArgs::TensorArgument{tensor_args.gamma->mesh_tensor()});
    }
    if (beta_has_value) {
        run_params.tensor_args.emplace(T_BETA, m2::ProgramRunArgs::TensorArgument{tensor_args.beta->mesh_tensor()});
    }
    if (mean_has_value) {
        run_params.tensor_args.emplace(T_MEAN, m2::ProgramRunArgs::TensorArgument{outputs[1]->mesh_tensor()});
    }
    if (rstd_has_value) {
        run_params.tensor_args.emplace(T_RSTD, m2::ProgramRunArgs::TensorArgument{outputs[2]->mesh_tensor()});
    }

    return ProgramArtifacts{
        .spec = std::move(spec),
        .run_params = std::move(run_params),
    };
}

}  // namespace ttnn::operations::moreh::moreh_group_norm
