// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "conv3d.hpp"
#include "device/conv3d_device_operation.hpp"
#include "ttnn/operations/experimental/conv3d/prepare_conv3d_weights.hpp"
#include <tt-metalium/math.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/tt_metal.hpp>
#include "ttnn/common/constants.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

using namespace tt::tt_metal;

namespace ttnn::experimental {

namespace {

// Largest divisor of n that is <= cap. Mirrors the helper in
// `device/conv3d_program_factory.cpp` (kept in sync).
uint32_t largest_divisor_up_to(uint32_t n, uint32_t cap) {
    for (uint32_t d = std::min(n, cap); d >= 1; --d) {
        if (n % d == 0) {
            return d;
        }
    }
    return 1;
}

// Project the per-core circular-buffer byte allocation for the Conv3D op
// given a candidate (C_in_block, C_out_block). Mirrors the `create_cb`
// allocations in `device/conv3d_program_factory.cpp` — keep this in sync
// when CB sizing in the program factory changes.
uint64_t project_conv3d_cb_bytes(
    uint32_t C_in,
    uint32_t cib,
    uint32_t cob,
    uint32_t num_patches,
    uint32_t kT,
    uint32_t kH,
    uint32_t kW,
    uint32_t dtype_bytes,
    uint32_t tile_size,
    tt::DataFormat data_format,
    bool fp32_dest_acc_en,
    bool use_bias,
    tt::ARCH arch) {
    uint32_t patch_size_p = kT * kH * kW * cib;
    uint32_t padded_patch_size_p = tt::round_up(patch_size_p, tt::constants::TILE_WIDTH);
    uint32_t C_in_num_blocks_p = tt::div_up(C_in, cib);
    uint32_t matmul_K_t_p = tt::div_up(patch_size_p, tt::constants::TILE_WIDTH);
    uint32_t matmul_M_t_p = tt::div_up(num_patches, tt::constants::TILE_HEIGHT);
    uint32_t matmul_N_t_p = tt::div_up(cob, tt::constants::TILE_WIDTH);
    uint32_t padded_patch_size_bytes_p = padded_patch_size_p * dtype_bytes;
    uint32_t vol2col_rm_pages_p = (num_patches % tt::constants::TILE_HEIGHT == 0)
                                      ? std::min(num_patches, (uint32_t)tt::constants::TILE_HEIGHT)
                                      : std::min(num_patches, 2 * tt::constants::TILE_HEIGHT);
    const uint32_t dst_size_p = fp32_dest_acc_en ? 4 : 8;
    const uint32_t out_subblock_w_p = std::min(matmul_N_t_p, dst_size_p);
    const bool scale_subblock_h_p = arch == tt::ARCH::WORMHOLE_B0 && out_subblock_w_p == matmul_N_t_p;
    const uint32_t out_subblock_h_p =
        scale_subblock_h_p ? largest_divisor_up_to(matmul_M_t_p, dst_size_p / out_subblock_w_p) : 1;
    const bool use_fp32_partials_p = fp32_dest_acc_en && C_in_num_blocks_p > 1;
    const auto partial_data_format_p = use_fp32_partials_p ? tt::DataFormat::Float32 : data_format;
    const uint32_t partial_tile_size_p = tt::tile_size(partial_data_format_p);
    uint64_t bytes =
        (uint64_t)padded_patch_size_bytes_p * vol2col_rm_pages_p +
        (uint64_t)tile_size * out_subblock_h_p * matmul_K_t_p + (uint64_t)tile_size * matmul_K_t_p * matmul_N_t_p +
        (uint64_t)partial_tile_size_p * matmul_M_t_p * matmul_N_t_p + (uint64_t)tile_size * matmul_M_t_p * matmul_N_t_p;
    if (C_in_num_blocks_p > 1) {
        bytes += (uint64_t)partial_tile_size_p * matmul_M_t_p * matmul_N_t_p;
        bytes += tile_size;
    }
    if (use_fp32_partials_p) {
        bytes += tile_size;
    }
    if (use_bias) {
        bytes += (uint64_t)tile_size * matmul_N_t_p;
    }
    return bytes;
}

// L1-aware block-size resolution. The defaults — or the values supplied
// by frontends like tt-mlir — can overflow per-core L1 for ViT-style
// patch-embed shapes (large spatial kernel × small C_in × big C_out),
// where matmul K = kT*kH*kW*C_in_block is huge and the weight + vol2col
// CBs blow the L1 budget. This helper picks the largest valid
// (C_in_block, C_out_block) pair ≤ initial values that fits L1.
//
// Returns (resolved_C_in_block, resolved_C_out_block). TT_FATALs if no
// valid pair fits.
//
// Must be called BEFORE `prepare_conv3d_weights` because the weight
// layout depends on `C_in_block`. If `freeze_C_in_block` is true (e.g.
// the caller passed a pre-prepared rank-2 weight whose layout is fixed
// to `initial_C_in_block`), the resolver only shrinks `C_out_block`.
// `C_out_block` does not affect the weight layout, so it can always
// be auto-shrunk.
std::pair<uint32_t, uint32_t> resolve_block_sizes_for_l1(
    uint32_t initial_C_in_block,
    uint32_t initial_C_out_block,
    uint32_t C_in,
    uint32_t padded_C_out,
    uint32_t num_patches,
    uint32_t kT,
    uint32_t kH,
    uint32_t kW,
    uint32_t dtype_bytes,
    uint32_t tile_size,
    tt::DataFormat data_format,
    bool fp32_dest_acc_en,
    bool use_bias,
    bool freeze_C_in_block) {
    const tt::ARCH arch = tt::tt_metal::hal::get_arch();
    // Match the kernel-code reserve used in `conv3d_program_factory.cpp` for
    // prefetch-buffer sizing so the resolver and the runtime CB validator
    // agree on the L1 budget.
    constexpr uint32_t L1_KERNEL_CODE_RESERVE = 200 * 1024;
    const uint64_t l1_budget = tt::tt_metal::hal::get_max_worker_l1_unreserved_size() - L1_KERNEL_CODE_RESERVE;

    auto fits = [&](uint32_t cib, uint32_t cob) -> bool {
        return project_conv3d_cb_bytes(
                   C_in,
                   cib,
                   cob,
                   num_patches,
                   kT,
                   kH,
                   kW,
                   dtype_bytes,
                   tile_size,
                   data_format,
                   fp32_dest_acc_en,
                   use_bias,
                   arch) <= l1_budget;
    };

    uint32_t C_in_block = initial_C_in_block;
    uint32_t C_out_block = initial_C_out_block;
    if (fits(C_in_block, C_out_block)) {
        return {C_in_block, C_out_block};
    }

    constexpr uint32_t C_IN_BLOCK_ALIGNMENT = 16;
    const uint32_t C_OUT_BLOCK_ALIGNMENT = tt::constants::TILE_WIDTH;

    // Prefer larger C_in_block (fewer reduction passes) over larger C_out_block.
    // Outer loop: candidate C_in_block from initial down to alignment floor.
    // Inner loop: largest C_out_block (≤ initial) that fits, given the C_in_block.
    // When the caller has a pre-prepared weight, only shrink C_out_block.
    bool found = false;
    uint32_t resolved_cib = 0;
    uint32_t resolved_cob = 0;
    const uint32_t cib_min = freeze_C_in_block ? initial_C_in_block : C_IN_BLOCK_ALIGNMENT;
    for (uint32_t cib = initial_C_in_block; cib >= cib_min && !found; cib -= C_IN_BLOCK_ALIGNMENT) {
        if (C_in % cib != 0) {
            continue;
        }
        for (uint32_t cob = initial_C_out_block; cob >= C_OUT_BLOCK_ALIGNMENT && !found; cob -= C_OUT_BLOCK_ALIGNMENT) {
            if (padded_C_out % cob != 0) {
                continue;
            }
            if (fits(cib, cob)) {
                resolved_cib = cib;
                resolved_cob = cob;
                found = true;
            }
        }
    }

    TT_FATAL(
        found,
        "Conv3D circular buffers exceed L1 budget ({} B) for all valid block "
        "sizes (initial C_in_block={}, C_out_block={}; C_in={}, padded_C_out={}). "
        "{}"
        "Reduce T_out_block / H_out_block / W_out_block, or use a different op.",
        l1_budget,
        initial_C_in_block,
        initial_C_out_block,
        C_in,
        padded_C_out,
        freeze_C_in_block ? "C_in_block was frozen at the user-supplied value because a pre-prepared "
                            "(rank-2) weight tensor was passed; pass a rank-5 weight to allow auto-shrink. "
                          : "");

    log_info(
        tt::LogOp,
        "Conv3D auto-shrink: (C_in_block, C_out_block) {}/{} -> {}/{} to fit L1.",
        initial_C_in_block,
        initial_C_out_block,
        resolved_cib,
        resolved_cob);

    return {resolved_cib, resolved_cob};
}

}  // namespace

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
    // If no config provided, use conservative default blocking:
    // minimal spatial blocks (1,1,1), smallest valid channel blocks (TILE_WIDTH) to minimize L1 pressure
    auto config = config_opt.value_or(ttnn::experimental::prim::Conv3dConfig(
        tt::tt_metal::DataType::BFLOAT16,                        // weights_dtype
        tt::tt_metal::Layout::ROW_MAJOR,                         // output_layout
        1,                                                       // T_out_block
        1,                                                       // W_out_block
        1,                                                       // H_out_block
        tt::constants::TILE_WIDTH,                               // C_out_block (one tile width)
        tt::constants::TILE_WIDTH,                               // C_in_block (one tile width, min L1)
        dilation_,                                               // dilation (match the op's dilation)
        32,                                                      // alignment
        input_tensor.device()->compute_with_storage_grid_size()  // use full device grid
        ));

    // Resolve (C_in_block, C_out_block) up-front so the weight preparation
    // and the program factory agree on a layout that fits L1. Without this,
    // ViT-style patch-embed shapes (e.g. Qwen 2.5-VL: kernel=[2,14,14],
    // C_out=1280) produce a per-core CB allocation that exceeds Wormhole's
    // L1, and the failure surfaces as a cryptic
    //   `Bad StatusOr access: INTERNAL: Error code: 13`
    // at program-launch time. The resolver shrinks the blocks if needed and
    // fails early with an actionable message if no valid pair fits.
    {
        const auto& input_shape = input_tensor.logical_shape();
        const uint32_t C_in = input_shape[input_shape.rank() - 1];
        const uint32_t padded_C_out = tt::round_up(output_channels_, tt::constants::TILE_WIDTH);
        const uint32_t initial_C_out_block = config.C_out_block > 0 ? config.C_out_block : padded_C_out;
        const uint32_t initial_C_in_block = config.C_in_block > 0 ? config.C_in_block : C_in;

        const auto data_format = tt::tt_metal::datatype_to_dataformat_converter(input_tensor.dtype());
        const uint32_t dtype_bytes = input_tensor.element_size();
        const uint32_t tile_size = tt::tile_size(data_format);
        const uint32_t num_patches = config.T_out_block * config.H_out_block * config.W_out_block;

        // Match the defaults used by `Conv3dDeviceOperation::validate_on_program_cache_miss`
        // (see device/conv3d_device_operation.cpp call to init_device_compute_kernel_config)
        // so the resolver and the program factory agree on `fp32_dest_acc_en`.
        const auto compute_kernel_config_resolved = ttnn::init_device_compute_kernel_config(
            input_tensor.device()->arch(),
            compute_kernel_config,
            tt::tt_metal::MathFidelity::HiFi2,  // default_fidelity
            true,                               // default_approx_mode
            false,                              // default_fp32_acc
            false,                              // default_l1_acc
            false                               // default_dst_full_sync_en
        );
        [[maybe_unused]] auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
            get_compute_kernel_config_args(input_tensor.device()->arch(), compute_kernel_config_resolved);

        // If the weight is already prepared (rank 2), its physical layout is
        // baked-in to the user-supplied C_in_block — auto-shrinking C_in_block
        // here would silently produce garbage output. Freeze it.
        const bool freeze_C_in_block = (weight_tensor.logical_shape().rank() == 2);

        auto [resolved_C_in_block, resolved_C_out_block] = resolve_block_sizes_for_l1(
            initial_C_in_block,
            initial_C_out_block,
            C_in,
            padded_C_out,
            num_patches,
            kernel_size_[0],
            kernel_size_[1],
            kernel_size_[2],
            dtype_bytes,
            tile_size,
            data_format,
            fp32_dest_acc_en,
            bias_tensor.has_value(),
            freeze_C_in_block);

        config.C_in_block = resolved_C_in_block;
        config.C_out_block = resolved_C_out_block;
    }

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
