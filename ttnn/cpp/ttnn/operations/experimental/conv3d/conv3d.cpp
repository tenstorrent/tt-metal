// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d.hpp"
#include "device/conv3d_device_operation.hpp"
#include "ttnn/operations/experimental/conv3d/prepare_conv3d_weights.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/hal.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental {

// Estimate the total circular buffer bytes that the program factory will allocate for
// a given C_in_block / C_out_block combination.  The formula mirrors the CB creation
// sequence in conv3d_program_factory.cpp so that auto-blocking can pick values that
// are guaranteed to fit in L1.
static uint64_t estimate_conv3d_cb_bytes(
    uint32_t C_in_block,
    uint32_t C_out_block,
    uint32_t C_in,
    const std::array<uint32_t, 3>& kernel_size,
    uint32_t num_patches,
    uint32_t tile_size,
    uint32_t fp32_tile_size,
    uint32_t dtype_bytes,
    bool use_bias,
    bool fp32_dest_acc_en) {
    uint32_t patch_size = kernel_size[0] * kernel_size[1] * kernel_size[2] * C_in_block;
    uint32_t padded_patch_size = tt::round_up(patch_size, tt::constants::TILE_WIDTH);
    uint32_t padded_patch_size_bytes = padded_patch_size * dtype_bytes;

    uint32_t matmul_M_t = tt::div_up(num_patches, tt::constants::TILE_HEIGHT);
    uint32_t matmul_K_t = tt::div_up(patch_size, tt::constants::TILE_WIDTH);
    uint32_t matmul_N_t = tt::div_up(C_out_block, tt::constants::TILE_WIDTH);

    uint32_t C_in_num_blocks = tt::div_up(C_in, C_in_block);
    bool use_fp32_partials = fp32_dest_acc_en && C_in_num_blocks > 1;
    uint32_t partial_tile_size = use_fp32_partials ? fp32_tile_size : tile_size;

    uint32_t vol2col_rm_pages = std::min(num_patches, 2 * tt::constants::TILE_HEIGHT);

    uint64_t total = 0;
    total += (uint64_t)padded_patch_size_bytes * vol2col_rm_pages;   // vol2col_rm
    total += (uint64_t)tile_size * matmul_M_t * matmul_K_t;          // vol2col_tiled
    total += (uint64_t)tile_size * matmul_K_t * matmul_N_t;          // weight_tiled
    total += (uint64_t)partial_tile_size * matmul_M_t * matmul_N_t;  // matmul_interm
    total += (uint64_t)tile_size * matmul_M_t * matmul_N_t;          // matmul_result_rm
    if (use_fp32_partials) {
        total += tile_size;  // zero tile for FPU accumulate
    }
    if (C_in_num_blocks > 1) {
        total += (uint64_t)partial_tile_size * matmul_M_t * matmul_N_t;  // reduction
        total += tile_size;                                              // worker_ack
    }
    if (use_bias) {
        total += (uint64_t)tile_size * matmul_N_t;  // bias
    }
    return total;
}

static Tensor prepare_and_check_weight_tensor(
    const Tensor& weight_tensor,
    uint32_t groups_,
    const ttnn::experimental::prim::Conv3dConfig& config,
    std::optional<ttnn::MeshDevice*> device) {
    Tensor prepared_weight_tensor = weight_tensor;
    switch (prepared_weight_tensor.logical_shape().rank()) {
        case 5:
            TT_FATAL(prepared_weight_tensor.device() == nullptr, "Unprepared weight tensor must be on host");
            TT_FATAL(device.has_value(), "Device must be provided when weight tensor is unprepared (rank 5)");
            prepared_weight_tensor = ttnn::operations::experimental::conv3d::prepare_conv3d_weights(
                prepared_weight_tensor, groups_, config.C_in_block, config.alignment, device.value());
            break;
        case 2: break;
        default: TT_THROW("Unsupported weight tensor rank: {}", prepared_weight_tensor.logical_shape().rank());
    }

    if (prepared_weight_tensor.layout() != Layout::TILE) {
        prepared_weight_tensor = ttnn::to_layout(prepared_weight_tensor, ttnn::Layout::TILE);
    }

    return prepared_weight_tensor;
}

ttnn::Tensor conv3d(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    std::optional<ttnn::MeshDevice*> device,
    const std::optional<ttnn::Tensor>& bias_tensor,
    const std::optional<ttnn::experimental::prim::Conv3dConfig>& config_opt,
    tt::tt_metal::DataType dtype_,
    uint32_t output_channels_,
    const std::array<uint32_t, 3>& kernel_size_,
    const std::array<uint32_t, 3>& stride_,
    const std::array<uint32_t, 3>& padding_,
    const std::array<uint32_t, 3>& dilation_,
    const std::string& padding_mode_,
    uint32_t groups_,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    ttnn::experimental::prim::Conv3dConfig config = [&]() {
        if (config_opt.has_value()) {
            return config_opt.value();
        }
        // Auto-blocking: pick the largest C_in_block that keeps total CB allocation within L1.
        // The dominant CB consumers are vol2col_tiled and weight_tiled, both proportional to
        // matmul_K_t = ceil(kD*kH*kW*C_in_block / TILE_WIDTH).  For large kernels (e.g. 2×14×14
        // in Qwen2.5-VL), a fixed C_in_block=TILE_WIDTH can overflow L1.
        auto grid_size = input_tensor.device()->compute_with_storage_grid_size();
        uint32_t C_in = input_tensor.logical_shape()[-1];  // last dim is channels (NDHWC layout)
        uint32_t C_out_block = tt::constants::TILE_WIDTH;
        uint32_t num_patches = 1;  // T_out_block=1, H_out_block=1, W_out_block=1

        // Get tile sizes for the input data format
        auto data_format = tt::tt_metal::datatype_to_dataformat_converter(dtype_);
        uint32_t tile_size = tt::tile_size(data_format);
        uint32_t fp32_tile_size = tt::tile_size(tt::DataFormat::Float32);
        uint32_t dtype_bytes = tt::datum_size(data_format);

        auto resolved_compute_config =
            ttnn::init_device_compute_kernel_config(input_tensor.device()->arch(), compute_kernel_config);
        auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
            get_compute_kernel_config_args(tt::tt_metal::hal::get_arch(), resolved_compute_config);

        bool use_bias = bias_tensor.has_value();

        // L1 budget for CBs (same reserve as program factory)
        constexpr uint32_t L1_KERNEL_CODE_RESERVE = 200 * 1024;
        uint32_t l1_cb_budget = tt::tt_metal::hal::get_max_worker_l1_unreserved_size() - L1_KERNEL_CODE_RESERVE;

        // Try C_in_block values from largest (full C_in) down to 1, halving each time.
        // Pick the largest that fits in L1.
        uint32_t best_c_in_block = 0;
        for (uint32_t candidate = C_in; candidate >= 1; candidate /= 2) {
            if (C_in % candidate != 0) {
                continue;  // C_in_block must evenly divide C_in
            }
            uint64_t estimated_bytes = estimate_conv3d_cb_bytes(
                candidate,
                C_out_block,
                C_in,
                kernel_size_,
                num_patches,
                tile_size,
                fp32_tile_size,
                dtype_bytes,
                use_bias,
                fp32_dest_acc_en);
            log_debug(
                tt::LogOp,
                "Conv3d auto-blocking: candidate C_in_block={}, estimated CB bytes={}, budget={}",
                candidate,
                estimated_bytes,
                l1_cb_budget);
            if (estimated_bytes <= l1_cb_budget) {
                best_c_in_block = candidate;
                break;
            }
        }

        TT_FATAL(
            best_c_in_block > 0,
            "Conv3d auto-blocking: cannot find a C_in_block that fits in L1 ({} bytes). "
            "kernel_size={}x{}x{}, C_in={}, C_out={}. Provide an explicit Conv3dConfig.",
            l1_cb_budget,
            kernel_size_[0],
            kernel_size_[1],
            kernel_size_[2],
            C_in,
            output_channels_);

        log_debug(tt::LogOp, "Conv3d auto-blocking: selected C_in_block={}", best_c_in_block);

        return ttnn::experimental::prim::Conv3dConfig(
            tt::tt_metal::DataType::BFLOAT16,  // weights_dtype
            tt::tt_metal::Layout::ROW_MAJOR,   // output_layout
            1,                                 // T_out_block
            1,                                 // W_out_block
            1,                                 // H_out_block
            C_out_block,                       // C_out_block
            best_c_in_block,                   // C_in_block (adaptively chosen)
            dilation_,                         // dilation
            32,                                // alignment
            grid_size                          // use full device grid
        );
    }();

    Tensor prepared_weight_tensor = prepare_and_check_weight_tensor(weight_tensor, groups_, config, device);
    return ttnn::prim::conv3d(
        input_tensor,
        prepared_weight_tensor,
        bias_tensor,
        config,
        dtype_,
        output_channels_,
        kernel_size_,
        stride_,
        padding_,
        dilation_,
        padding_mode_,
        groups_,
        memory_config,
        compute_kernel_config);
}

}  // namespace ttnn::experimental
