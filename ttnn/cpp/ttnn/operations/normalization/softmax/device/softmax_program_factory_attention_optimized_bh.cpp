// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "softmax_device_operation.hpp"

#include "ttnn/operations/core/data_movement_kernel/datamovement_kernel_config.hpp"
#include "ttnn/operations/core/program_cache_l1.hpp"

#include <tt-logger/tt-logger.hpp>
#include <tt-metalium/math.hpp>
#include <tt-metalium/work_split.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_spec.hpp>
#include <tt-metalium/experimental/metal2_host_api/program_run_args.hpp>
#include <ttnn/metal_v2_artifacts.hpp>

#include <algorithm>
#include <bit>
#include <string>
#include <utility>
#include <cstdint>
#include <cstdlib>
#include <vector>

namespace ttnn::prim {

using namespace tt::tt_metal::experimental;

namespace {
std::uint32_t g_bh_rows = 1;
template <typename D>
std::uint32_t bh_split_w(D* dev, std::uint32_t gx, std::uint32_t gy) {
    const auto v0 = dev->worker_core_from_logical_core(CoreCoord{0, 0});
    std::uint32_t w = 1;
    for (; w < gx; ++w) {
        bool ok = true;
        for (std::uint32_t y = 0; y < gy && ok; ++y) {
            const auto v = dev->worker_core_from_logical_core(CoreCoord{w, y});
            const auto vy = dev->worker_core_from_logical_core(CoreCoord{0, y});
            ok = v.x == v0.x + w && v.y == vy.y && vy.y == v0.y + y;
        }
        if (!ok) {
            break;
        }
    }
    const std::uint32_t cap = 0;
    return cap != 0 ? std::min(w, cap) : w;
}
constexpr std::uint32_t bh_deep_out_depth = 8;
constexpr std::uint64_t bh_l1_reserve_bytes = 16384;
constexpr double bh_l1_cb_fraction = 0.9;
constexpr std::uint32_t bh_small_kernel_single_tile_cbs = 5;
constexpr std::uint32_t bh_row_split_im_single_tiles = 4;
constexpr std::uint32_t bh_row_split_acc_single_tiles = 5;
constexpr std::uint32_t bh_row_chain_im_single_tiles = 3;
constexpr std::uint32_t bh_row_chain_acc_single_tiles = 6;
constexpr const char* bh_read_batch_tiles = "8";
constexpr const char* bh_large_kernel_read_batch_tiles = "16";
constexpr std::uint32_t bh_row_parallel_min_tiles = 8;
constexpr std::uint32_t bh_short_row_max_tiles = 16;
constexpr std::uint32_t bh_few_rows_per_core = 4;
constexpr std::uint32_t bh_row_split_max_tile_rows = 16;
constexpr std::uint32_t bh_row_chain_max_cores = 11;
constexpr std::uint32_t bh_row_chain_min_tiles_per_core = 12;
constexpr std::uint32_t bh_row_chain_min_tiles_per_core_fp32 = 4;
const char* bh_kdef() {
    return g_bh_rows >= 4 ? ",BH_ACC,BH_READ_ROW_ONCE,BH_RECIP_FACES," : ",BH_ACC,BH_RECIP_MCAST,";
}
}

SoftmaxInterleavedPlan SoftmaxDeviceOperation::SoftmaxProgramFactoryAttentionOptimizedBH::select_plan(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args) {
    using namespace tt::tt_metal;
    SoftmaxInterleavedPlan plan;
    const auto& input_tensor = tensor_args.input_tensor;
    auto* device = input_tensor.device();
    const auto arch = device->arch();
    const auto out_spec = SoftmaxDeviceOperation::compute_output_specs(attributes, tensor_args);
    const auto plain_dram = [](BufferType buffer_type, DataType dtype) {
        return buffer_type == BufferType::DRAM && (dtype == DataType::BFLOAT16 || dtype == DataType::FLOAT32);
    };
    const auto& shape = input_tensor.padded_shape();
    const std::uint32_t NC = shape.rank() >= 4 ? shape[0] : 1, W = shape[-1], H = (input_tensor.physical_volume() / (NC * shape[-1]));
    const std::uint32_t Wt = W / input_tensor.tensor_spec().tile().get_width();
    const std::uint32_t Ht = H / input_tensor.tensor_spec().tile().get_height();
    if ((arch != tt::ARCH::WORMHOLE_B0 && !(arch == tt::ARCH::BLACKHOLE && 1 != 0)) || tensor_args.mask.has_value() || Wt < bh_row_parallel_min_tiles ||
        !plain_dram(input_tensor.memory_config().buffer_type(), input_tensor.dtype()) ||
        !plain_dram(out_spec.memory_config().buffer_type(), out_spec.data_type())) {
        return plan;
    }
    const bool mask_padded_data = W > input_tensor.logical_shape()[-1];
    const std::uint64_t l1_budget_bytes = ttnn::operations::core::usable_program_l1_capacity(device);
    const std::uint64_t l1_bank = device->allocator()->get_bank_size(BufferType::L1);

    const tt::DataFormat in0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const auto grid = device->compute_with_storage_grid_size();
    const bool fp32_in = in0_cb_data_format == tt::DataFormat::Float32;
    const bool short_work = Wt <= bh_short_row_max_tiles || tt::div_up(NC * Ht, grid.x * grid.y) <= bh_few_rows_per_core;
    const std::uint32_t in0_tile_size = tt::tile_size(in0_cb_data_format);
    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, attributes.compute_kernel_config);
    const std::uint32_t max_scaler_tile_size = tt::tile_size(
        in0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b);
    const std::uint32_t bf16_tile_size = tt::tile_size(tt::DataFormat::Float16_b);
    const std::uint32_t out0_tile_size = tt::tile_size(datatype_to_dataformat_converter(out_spec.data_type()));
    const bool bulk_fp32 = fp32_dest_acc_en && (in0_cb_data_format == tt::DataFormat::Float32 ||
                                                attributes.has_explicit_compute_kernel_config);
    const std::uint32_t im_tile_size = tt::tile_size(bulk_fp32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b);
    const std::uint32_t acc_tile_size =
        tt::tile_size(fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b);

    std::uint32_t block_size = fp32_dest_acc_en ? 4 : 8;
    if (Wt % block_size != 0) {
        const std::uint32_t divisor = static_cast<std::uint32_t>(tt::tt_metal::find_max_divisor(Wt, block_size));
        if (tt::div_up(Wt, block_size) == Wt / divisor) {
            block_size = divisor;
        }
    }
    const std::uint32_t wt_blocks = tt::div_up(Wt, block_size) * block_size;
    const std::uint32_t in0_t = attributes.numeric_stable && !mask_padded_data ? wt_blocks : block_size * 2;
    const std::uint32_t out0_t = block_size * 2;
    std::uint64_t cb_size_sum_bytes =
        (std::uint64_t{in0_t} * in0_tile_size) + (std::uint64_t{wt_blocks} * im_tile_size) +
        (std::uint64_t{out0_t} * out0_tile_size) + (std::uint64_t{bh_small_kernel_single_tile_cbs} * im_tile_size);
    if (attributes.numeric_stable && mask_padded_data) {
        cb_size_sum_bytes += std::uint64_t{wt_blocks} * im_tile_size;
    }
    const bool bh_fs = arch == tt::ARCH::BLACKHOLE && 0 != 0 && attributes.numeric_stable &&
                       !mask_padded_data && !tensor_args.mask.has_value() && Wt >= 0 &&
                       NC * Ht <= bh_row_split_max_tile_rows && Wt % block_size == 0;
    const bool use_large_kernel = (device->l1_size_per_core() * bh_l1_cb_fraction) < cb_size_sum_bytes || bh_fs;
    const bool batched_io = (attributes.numeric_stable || (Wt > bh_row_parallel_min_tiles && !fp32_in && short_work)) &&
                            (!use_large_kernel || fp32_in);
    plan.flush_writes = batched_io;
    plan.read_batch = batched_io && (use_large_kernel || Wt >= bh_short_row_max_tiles);
    const std::uint64_t deep_need = cb_size_sum_bytes + (std::uint64_t{bh_deep_out_depth - 1} * out0_t * out0_tile_size);
    plan.deep_out = batched_io && short_work && !use_large_kernel && Wt % block_size == 0 &&
                    (device->l1_size_per_core() * bh_l1_cb_fraction) >= deep_need &&
                    l1_budget_bytes >= deep_need + bh_l1_reserve_bytes;

    if (use_large_kernel && !mask_padded_data && attributes.numeric_stable && NC * Ht <= bh_row_split_max_tile_rows &&
        Wt % block_size == 0) {
        const std::uint64_t streamed_entry_bytes = in0_tile_size + (2ull * im_tile_size);
        const std::uint64_t single_entry_bytes = (std::uint64_t{out0_t} * out0_tile_size) + (3ull * acc_tile_size) +
                                                 (2ull * im_tile_size) + max_scaler_tile_size + acc_tile_size +
                                                 (2ull * bf16_tile_size);
        const std::uint64_t entries_that_fit =
            l1_bank > single_entry_bytes ? (l1_bank - single_entry_bytes) / streamed_entry_bytes : 0;
        constexpr std::uint32_t max_large_kernel_cb_entries = 80;
        const std::uint32_t L =
            (std::min<std::uint32_t>(max_large_kernel_cb_entries, static_cast<std::uint32_t>(entries_that_fit)) /
             block_size) *
            block_size;
        if (L >= block_size) {
            const std::uint32_t nchunk = tt::div_up(Wt, L);
            const std::uint32_t last_len = Wt - (nchunk - 1) * L;
            const std::uint32_t bh_sx = 2 == 1 ? bh_split_w(device, grid.x, grid.y) : grid.x;
            const std::uint32_t grid_n = std::min<std::uint32_t>(bh_sx * grid.y, 1000000);
            const std::uint32_t req = grid_n / std::max<std::uint32_t>(1u, std::min<std::uint32_t>(NC * Ht, grid.y));
            for (std::uint32_t active = std::min<std::uint32_t>(std::min<std::uint32_t>(req, nchunk), grid_n);
                 active <= std::min<std::uint32_t>(nchunk, grid_n);
                 ++active) {
                std::uint32_t t_max = 0, nck_max = 0;
                for (std::uint32_t c = 0; c < active; ++c) {
                    const std::uint32_t ck0 = c * nchunk / active;
                    const std::uint32_t nck = (c + 1) * nchunk / active - ck0;
                    std::uint32_t tcount = 0;
                    for (std::uint32_t k = ck0; k < ck0 + nck; ++k) {
                        tcount += (k == nchunk - 1) ? last_len : L;
                    }
                    t_max = std::max(t_max, tcount);
                    nck_max = std::max(nck_max, nck);
                }
                const std::uint64_t need =
                    std::uint64_t{t_max} * in0_tile_size + std::uint64_t{L} * im_tile_size +
                    2ull * block_size * out0_tile_size +
                    (std::uint64_t{bh_row_split_im_single_tiles} + active) * im_tile_size +
                    (std::uint64_t{bh_row_split_acc_single_tiles} + nck_max + nchunk) * acc_tile_size +
                    max_scaler_tile_size + acc_tile_size + bh_l1_reserve_bytes;
                if (need <= std::min(l1_bank, l1_budget_bytes)) {
                    plan.split_active = active;
                    break;
                }
            }
        }
    }

    const bool bh_bh_chain = arch == tt::ARCH::BLACKHOLE && 1 != 0;
    const bool bh_ns_chain = !attributes.numeric_stable &&
                             ((2 == 1 && !fp32_in) || 2 == 2);
    const bool bh_chain_large = 0 != 0;
    if ((!use_large_kernel || bh_chain_large) && !mask_padded_data && (attributes.numeric_stable || bh_ns_chain)) {
        const std::uint32_t lim =
            std::min<std::uint32_t>(bh_row_chain_max_cores, (grid.x * grid.y) / std::max<std::uint32_t>(1u, NC * Ht));
        const bool bh_any = bh_bh_chain && 1 != 0;
        const bool bh_col = bh_bh_chain && 0 != 0;
        const std::uint32_t min_tiles = in0_cb_data_format == tt::DataFormat::Float32
                                            ? bh_row_chain_min_tiles_per_core_fp32
                                            : bh_row_chain_min_tiles_per_core;
        std::uint32_t C = 1;
        if (bh_any || bh_col) {
            auto bh_cap = [&](std::uint32_t c) -> std::uint32_t {
                if (bh_col) {
                    return grid.x * (grid.y / c);
                }
                if (c <= grid.x) {
                    return (grid.x / c) * grid.y;
                }
                return c % grid.x == 0 ? grid.y / (c / grid.x) : (c % 8 == 0 ? grid.y / (c / 8) : 0);
            };
            C = bh_col ? std::min<std::uint32_t>(lim, grid.y) : lim;
            while (C >= 2 && (Wt % block_size != 0 || Wt / block_size < C || Wt / C < min_tiles ||
                              bh_cap(C) == 0 || (Wt % (C * block_size) != 0 && NC * Ht > bh_cap(C)))) {
                --C;
            }
        } else {
            while (C * 2 <= lim) {
                C *= 2;
            }
            while (C >= 2 && (Wt % (C * block_size) != 0 || Wt / C < min_tiles)) {
                C /= 2;
            }
        }
        const bool shape_ok =
            bh_col ? C <= grid.y
            : bh_bh_chain ? (C <= grid.x || (C % grid.x == 0 && C / grid.x <= grid.y) || (C % 8 == 0 && C / 8 <= grid.y))
                          : ((C <= grid.x && grid.x % C == 0) || (C % grid.x == 0 && C / grid.x <= grid.y));
        const std::uint32_t bh_tmax =
            C >= 2 ? (0 != 0 ? 2 * tt::div_up(Wt / block_size, C)
                      : 0 != 0 ? tt::div_up(Wt / block_size - 1, C - 1)
                                                     : tt::div_up(Wt / block_size, C)) *
                         block_size
                   : Wt;
        const std::uint64_t need = std::uint64_t{bh_tmax} * (in0_tile_size + im_tile_size) +
                                   2ull * block_size * out0_tile_size +
                                   (std::uint64_t{bh_row_chain_im_single_tiles} + C) * im_tile_size +
                                   std::uint64_t{bh_row_chain_acc_single_tiles} * acc_tile_size + max_scaler_tile_size +
                                   acc_tile_size + bh_l1_reserve_bytes;
        if (C >= 2 && shape_ok && need <= std::min(l1_bank, l1_budget_bytes)) {
            plan.chain_c = C;
            if (use_large_kernel) {
                plan.split_active = 0;
            }
        }
    }
    if (fp32_in && 1 == 0) { plan.deep_out = false; }
    if (fp32_in && 0 != 0 && Wt > 0) { plan.deep_out = false; }
    if (fp32_in && !use_large_kernel && 0 != 0 && Wt > 0) { plan.read_batch = false; }
    if (0) { plan.deep_out = false; }
    if (0) { plan.read_batch = false; }
    if (0) { plan.flush_writes = false; }
    if (0) { plan.split_active = 0; }
    if (0) { plan.chain_c = 0; }
    return plan;
}

#undef bh_row_parallel_min_tiles
#undef bh_short_row_max_tiles
#undef bh_few_rows_per_core
#undef bh_row_split_max_tile_rows
#undef bh_row_chain_max_cores
#undef bh_row_chain_min_tiles_per_core
#undef bh_row_chain_min_tiles_per_core_fp32
ttnn::device_operation::ProgramArtifacts
SoftmaxDeviceOperation::SoftmaxProgramFactoryAttentionOptimizedBH::create_program_artifacts(
    const operation_attributes_t& attributes, const tensor_args_t& tensor_args, tensor_return_value_t& output_tensor) {
    using namespace tt::tt_metal;

    log_debug(tt::LogMetal, "SoftmaxProgramFactoryAttentionOptimizedBH selected");

    const auto& input_tensor = tensor_args.input_tensor;
    const auto& shape = input_tensor.padded_shape();
    const std::uint32_t NC = shape.rank() >= 4 ? shape[0] : 1, W = shape[-1], H = (input_tensor.physical_volume() / (NC * shape[-1]));
    const std::uint32_t tile_width = input_tensor.tensor_spec().tile().get_width();
    const std::uint32_t tile_height = input_tensor.tensor_spec().tile().get_height();
    const std::uint32_t Wt = W / tile_width;
    const std::uint32_t Ht = H / tile_height;
    g_bh_rows = NC * Ht;
    const auto& shape_unpadded = input_tensor.logical_shape();

    const bool has_mask = tensor_args.mask.has_value();
    TT_FATAL(
        has_mask || !attributes.is_causal_mask,
        "Causal-mask softmax requires an attention mask (is_causal_mask=true with no mask is unsupported).");

    bool mask_padded_data = false;
    std::uint32_t num_datum_padded = 0;
    std::uint32_t W_unpadded = shape_unpadded[-1];
    if (W > W_unpadded) {
        mask_padded_data = true;
        num_datum_padded = W - W_unpadded;
    }

    std::uint32_t mask_H = H;
    if (has_mask) {
        mask_H = tensor_args.mask.value().padded_shape()[2];
    }
    const std::uint32_t mask_Ht = mask_H / tile_height;

    auto* device = input_tensor.device();
    const auto arch = device->arch();
    const auto& plan = attributes.interleaved_plan;
    const auto check_rect = [device](const CoreRangeSet& nodes, const CoreCoord& lo, const CoreCoord& hi) {
        TT_FATAL(
            nodes.contains(CoreRange(lo, hi)),
            "Softmax remote-write targets {} are outside the program's buffer cores",
            CoreRange(lo, hi).str());
        const auto v0 = device->worker_core_from_logical_core(lo);
        for (std::size_t y = lo.y; y <= hi.y; ++y) {
            for (std::size_t x = lo.x; x <= hi.x; ++x) {
                const auto v = device->worker_core_from_logical_core(CoreCoord{x, y});
                TT_FATAL(
                    v.x == v0.x + (x - lo.x) && v.y == v0.y + (y - lo.y),
                    "Softmax cores {} do not map to a contiguous worker rectangle",
                    CoreRange(lo, hi).str());
            }
        }
    };

    const tt::DataFormat in0_cb_data_format = datatype_to_dataformat_converter(input_tensor.dtype());
    const std::uint32_t in0_tile_size = tt::tile_size(in0_cb_data_format);

    auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(arch, attributes.compute_kernel_config);

    const tt::DataFormat max_scaler_cb_data_format =
        in0_cb_data_format == tt::DataFormat::Float32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const std::uint32_t max_scaler_tile_size = tt::tile_size(max_scaler_cb_data_format);
    const std::uint32_t fused_attention_scale_tile_size = tt::tile_size(tt::DataFormat::Float16_b);

    const tt::DataFormat out0_cb_data_format = datatype_to_dataformat_converter(output_tensor.dtype());
    const std::uint32_t out0_tile_size = tt::tile_size(out0_cb_data_format);

    const tt::DataFormat mask_cb_data_format =
        has_mask ? datatype_to_dataformat_converter(tensor_args.mask.value().dtype()) : tt::DataFormat::Float16_b;
    const std::uint32_t mask_tile_size = tt::tile_size(mask_cb_data_format);
    const std::uint32_t mask_padded_tile_size = tt::tile_size(tt::DataFormat::Float16_b);

    const bool bulk_fp32 = fp32_dest_acc_en && (in0_cb_data_format == tt::DataFormat::Float32 ||
                                                attributes.has_explicit_compute_kernel_config);
    const tt::DataFormat im_cb_data_format = bulk_fp32 ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const std::uint32_t im_tile_size = tt::tile_size(im_cb_data_format);

    const tt::DataFormat acc_cb_data_format = fp32_dest_acc_en ? tt::DataFormat::Float32 : tt::DataFormat::Float16_b;
    const std::uint32_t acc_tile_size = tt::tile_size(acc_cb_data_format);

    const tt::DataFormat sum_scaler_cb_data_format = acc_cb_data_format;
    const std::uint32_t sum_scaler_tile_size = acc_tile_size;

    std::uint32_t block_size = fp32_dest_acc_en ? 4 : 8;

    if (Wt % block_size != 0) {
        const std::uint32_t divisor = static_cast<std::uint32_t>(tt::tt_metal::find_max_divisor(Wt, block_size));
        if (tt::div_up(Wt, block_size) == Wt / divisor) {
            block_size = divisor;
        }
    }

    const bool small_kernel_numeric_stable_uses_cb_in0_at_wt =
        attributes.numeric_stable && !has_mask && !mask_padded_data;
    const bool small_kernel_numeric_stable_uses_cb_x_at_wt =
        attributes.numeric_stable && (has_mask || mask_padded_data);

    std::uint32_t in0_t =
        small_kernel_numeric_stable_uses_cb_in0_at_wt ? tt::div_up(Wt, block_size) * block_size : block_size * 2;
    std::uint32_t out0_t = block_size * 2;
    std::uint32_t im1_t = 1;
    std::uint32_t in2_t = 1;
    std::uint32_t in3_t = 1;
    std::uint32_t in4_t =
        tt::div_up(Wt, block_size) * block_size;
    std::uint32_t in5_t = 1;
    std::uint32_t im2_t = 1;
    std::uint32_t im4_t = tt::div_up(Wt, block_size) * block_size;

    std::uint32_t im0_t = block_size * tt::div_up(Wt, block_size);

    std::uint32_t im3_t = block_size * (tt::div_up(Wt, block_size) + 1);

    std::uint32_t dfb_length = in0_t;
    bool use_large_kernel = false;
    constexpr std::uint32_t single_tile_cb_count = 5;
    std::uint32_t cb_size_sum_bytes = (in0_t * in0_tile_size) + (im0_t * im_tile_size) + (out0_t * out0_tile_size) +
                                      (single_tile_cb_count * im_tile_size);
    if (small_kernel_numeric_stable_uses_cb_x_at_wt) {
        cb_size_sum_bytes += im4_t * im_tile_size;
    }
    if (has_mask) {
        cb_size_sum_bytes +=
            (im3_t * im_tile_size) + (in3_t * fused_attention_scale_tile_size) + (in4_t * mask_tile_size);
    }

    if ((device->l1_size_per_core() * 0.9) < cb_size_sum_bytes || plan.split_active != 0) {
        use_large_kernel = true;
        const std::uint32_t streamed_entry_bytes =
            in0_tile_size + (2 * im_tile_size) + (has_mask ? im_tile_size + mask_tile_size : 0);
        const std::uint32_t single_entry_bytes = (out0_t * out0_tile_size) + (3 * acc_tile_size) +
                                                 (2 * im_tile_size) + max_scaler_tile_size + sum_scaler_tile_size +
                                                 mask_padded_tile_size + fused_attention_scale_tile_size;
        const std::uint32_t l1_free = device->allocator()->get_bank_size(BufferType::L1);
        const std::uint32_t entries_that_fit =
            l1_free > single_entry_bytes ? (l1_free - single_entry_bytes) / streamed_entry_bytes : 0;
        constexpr std::uint32_t max_large_kernel_cb_entries = 80;
        const std::uint32_t large_kernel_cb_size =
            (std::min(max_large_kernel_cb_entries, entries_that_fit) / block_size) * block_size;
        TT_FATAL(
            large_kernel_cb_size >= block_size,
            "Softmax buffers of {} bytes per streamed entry do not fit L1 ({} bytes free)",
            streamed_entry_bytes,
            l1_free);
        dfb_length = large_kernel_cb_size;
        in0_t = large_kernel_cb_size;
        im4_t = large_kernel_cb_size;
        im0_t = large_kernel_cb_size;
        im3_t = large_kernel_cb_size;
        in4_t = large_kernel_cb_size;
    }
    if (!use_large_kernel) {
        TT_FATAL(
            im3_t >= Wt + block_size,
            "im3_t {} must be >= Width in tiles {} + num_dest_regs to use {}",
            im3_t,
            Wt,
            block_size);
    }
    TT_FATAL(
        im0_t % block_size == 0,
        "Size of cb: {} must be divisible by the size of block used by the reader and compute kernel.",
        im0_t);
    TT_FATAL(
        out0_t % block_size == 0,
        "Size of cb: {} must be divisible by the size of block used by the reader and compute kernel.",
        out0_t);
    TT_FATAL(
        in4_t % block_size == 0,
        "Size of cb: {} must be divisible by the size of block used by the reader and compute kernel.",
        in4_t);
    const bool deep_out = plan.deep_out;

    if (plan.split_active != 0) {
        TT_FATAL(
            use_large_kernel && dfb_length % block_size == 0, "Softmax row split needs the large-kernel buffer layout");
        const auto ws_grid = device->compute_with_storage_grid_size();
        const std::uint32_t L = dfb_length;
        const std::uint32_t nchunk = tt::div_up(Wt, L);
        const std::uint32_t last_len = Wt - (nchunk - 1) * L;
        const std::uint32_t active = plan.split_active;
        const std::uint32_t bh_mode = 2;
        const std::uint32_t rect_w = std::min<std::uint32_t>(
            active, bh_mode == 1 ? bh_split_w(device, ws_grid.x, ws_grid.y) : ws_grid.x);
        const std::uint32_t rect_h = tt::div_up(active, rect_w);
        const std::uint32_t rect_n = rect_w * rect_h;
        std::vector<std::uint32_t> ck0(rect_n, 0), nck(rect_n, 0), tcount(rect_n, 0);
        std::uint32_t t_max = 0, nck_max = 0;
        for (std::uint32_t c = 0; c < active; ++c) {
            ck0[c] = c * nchunk / active;
            nck[c] = (c + 1) * nchunk / active - ck0[c];
            for (std::uint32_t k = ck0[c]; k < ck0[c] + nck[c]; ++k) {
                tcount[c] += (k == nchunk - 1) ? last_len : L;
            }
            t_max = std::max(t_max, tcount[c]);
            nck_max = std::max(nck_max, nck[c]);
        }
        {
            const std::uint32_t n_rows = NC * Ht;
            const std::uint32_t n_groups = std::max<std::uint32_t>(
                1u, std::min<std::uint32_t>(n_rows, static_cast<std::uint32_t>(ws_grid.y) / rect_h));
            const CoreRange ws_rect({0, 0}, {rect_w - 1, n_groups * rect_h - 1});
            const CoreRangeSet ws_nodes(ws_rect);
            std::uint32_t bh_gx0 = 0, bh_gx1 = 0;
            if (bh_mode == 2) {
                for (std::uint32_t x = 1; x < rect_w; ++x) {
                    const auto a = device->worker_core_from_logical_core(CoreCoord{x - 1, 0});
                    const auto b = device->worker_core_from_logical_core(CoreCoord{x, 0});
                    if (b.x != a.x + 1) {
                        TT_FATAL(bh_gx1 == 0 && b.x > a.x, "softmax split: more than one column gap");
                        bh_gx0 = static_cast<std::uint32_t>(a.x + 1);
                        bh_gx1 = static_cast<std::uint32_t>(b.x);
                    }
                    for (std::uint32_t y = 0; y < n_groups * rect_h; ++y) {
                        TT_FATAL(device->worker_core_from_logical_core(CoreCoord{x, y}).x == b.x &&
                                     device->worker_core_from_logical_core(CoreCoord{x, y}).y ==
                                         device->worker_core_from_logical_core(CoreCoord{0, 0}).y + y,
                                 "softmax split: rows not contiguous");
                    }
                }
            } else {
                for (std::uint32_t grp = 0; grp < n_groups; ++grp) {
                    check_rect(ws_nodes, CoreCoord{0, grp * rect_h}, CoreCoord{rect_w - 1, grp * rect_h + rect_h - 1});
                }
            }

            const KernelSpecName W_READER{"reader"};
            const KernelSpecName W_WRITER{"writer"};
            const KernelSpecName W_COMPUTE{"compute"};
            const TensorParamName W_SRC{"src"};
            const TensorParamName W_DST{"dst"};
            const DFBSpecName W_IN0{"in0"}, W_OUT0{"out0"}, W_MAXSC{"max_scaler"}, W_SUMSC{"sum_scaler"},
                W_EXPS{"exps"}, W_PMAX{"pmax"}, W_RMAX{"rmax"}, W_MOUT{"mout"}, W_MCB{"mcb"}, W_PSUM{"psum"},
                W_RSUM{"rsum"}, W_PREV{"prev"}, W_ROUT{"rout"}, W_RCB{"rcb"};
            const SemaphoreSpecName S_RMAX{"rmax_sem"}, S_RSUM{"rsum_sem"}, S_M{"m_sem"}, S_R{"r_sem"};

            Group<DataflowBufferSpec> wdfbs;
            auto add_dfb = [&](const DFBSpecName& n, std::uint32_t sz, std::uint32_t cnt, tt::DataFormat f) {
                wdfbs.push_back(DataflowBufferSpec{
                    .unique_id = n, .entry_size = sz, .num_entries = cnt, .data_format_metadata = f});
            };
            add_dfb(W_IN0, in0_tile_size, t_max, in0_cb_data_format);
            add_dfb(W_OUT0, out0_tile_size, block_size * 2, out0_cb_data_format);
            add_dfb(W_MAXSC, max_scaler_tile_size, 1, max_scaler_cb_data_format);
            add_dfb(W_SUMSC, sum_scaler_tile_size, 1, sum_scaler_cb_data_format);
            const std::uint64_t bh_keep_need =
                std::uint64_t{t_max} * (in0_tile_size + im_tile_size) + 2ull * block_size * out0_tile_size +
                (std::uint64_t{bh_row_split_im_single_tiles} + active) * im_tile_size +
                (std::uint64_t{bh_row_split_acc_single_tiles} + nck_max + nchunk) * acc_tile_size + max_scaler_tile_size +
                acc_tile_size + bh_l1_reserve_bytes;
            bool bh_even = true;
            for (std::uint32_t c = 0; c < active; ++c) {
                bh_even = bh_even && tcount[c] == t_max;
            }
            const bool bh_keep = 1 != 0 && (n_rows <= n_groups || bh_even) &&
                                 bh_keep_need <= static_cast<std::uint64_t>(device->l1_size_per_core() * bh_l1_cb_fraction);
            add_dfb(W_EXPS, im_tile_size, bh_keep ? t_max : L, im_cb_data_format);
            add_dfb(W_PMAX, im_tile_size, 1, im_cb_data_format);
            add_dfb(W_RMAX, im_tile_size, active, im_cb_data_format);
            add_dfb(W_MOUT, im_tile_size, 1, im_cb_data_format);
            add_dfb(W_MCB, im_tile_size, 1, im_cb_data_format);
            add_dfb(W_PSUM, acc_tile_size, nck_max, acc_cb_data_format);
            add_dfb(W_RSUM, acc_tile_size, nchunk, acc_cb_data_format);
            add_dfb(W_PREV, acc_tile_size, 2, acc_cb_data_format);
            add_dfb(W_ROUT, acc_tile_size, 1, acc_cb_data_format);
            add_dfb(W_RCB, acc_tile_size, 1, acc_cb_data_format);

            KernelSpec::CompilerOptions::Defines wbase;
            wbase["NUMERIC_STABLE"] = "1";
            KernelSpec::CompilerOptions::Defines wcomp = wbase;
            wcomp["EXP_APPROX"] = math_approx_mode ? "1" : "0";
            if (bh_keep) {
                wcomp["BH_KEEP_EXPS"] = "1";
            }
            {
                std::string kd = bh_kdef();
                std::size_t p0 = 0;
                while (p0 < kd.size()) {
                    std::size_t p1 = kd.find(',', p0);
                    if (p1 == std::string::npos) {
                        p1 = kd.size();
                    }
                    if (p1 > p0) {
                        wcomp[kd.substr(p0, p1 - p0)] = "1";
                    }
                    p0 = p1 + 1;
                }
            }
            wcomp["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";

            auto w_hw = ttnn::to_compute_hardware_config(attributes.compute_kernel_config);
            if (fp32_dest_acc_en) {
                auto um = [&](const DFBSpecName& n, tt::DataFormat f, bool to_dest) {
                    if (f == tt::DataFormat::Float32) {
                        w_hw.unpack_modes.insert(
                            {n,
                             to_dest ? tt::tt_metal::UnpackMode::UnpackToDest : tt::tt_metal::UnpackMode::UnpackToSrc});
                    }
                };
                um(W_IN0, in0_cb_data_format, false);
                um(W_MAXSC, max_scaler_cb_data_format, false);
                um(W_SUMSC, sum_scaler_cb_data_format, false);
                um(W_EXPS, im_cb_data_format, false);
                um(W_RMAX, im_cb_data_format, false);
                um(W_MCB, im_cb_data_format, false);
                um(W_RSUM, acc_cb_data_format, true);
                um(W_PREV, acc_cb_data_format, false);
                um(W_RCB, acc_cb_data_format, false);
            }

            KernelSpec w_reader{
                .unique_id = W_READER,
                .source =
                    std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/dataflow/reader_unary_interleaved_sm_row_split_bh.cpp",
                .compiler_options = {.defines = wbase},
                .dfb_bindings =
                    {DFBBinding{
                         .dfb_spec_name = W_IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
                     DFBBinding{
                         .dfb_spec_name = W_MAXSC,
                         .accessor_name = "max_scaler",
                         .endpoint_type = DFBEndpointType::PRODUCER},
                     DFBBinding{
                         .dfb_spec_name = W_SUMSC,
                         .accessor_name = "sum_scaler",
                         .endpoint_type = DFBEndpointType::PRODUCER}},
                .tensor_bindings = {TensorBinding{.tensor_parameter_name = W_SRC, .accessor_name = "src"}},
                .runtime_arg_schema =
                    {.runtime_arg_names = {"T", "t0", "num_rows", "Wt", "blk", "pad", "row0", "rstride"}},
                .hw_config = ttnn::create_reader_datamovement_config(),
            };
            KernelSpec w_writer{
                .unique_id = W_WRITER,
                .source =
                    std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/dataflow/writer_unary_interleaved_sm_row_split_bh.cpp",
                .compiler_options = {.defines = wbase},
                .dfb_bindings =
                    {DFBBinding{
                         .dfb_spec_name = W_OUT0, .accessor_name = "out0", .endpoint_type = DFBEndpointType::CONSUMER},
                     DFBBinding{
                         .dfb_spec_name = W_PMAX, .accessor_name = "pmax", .endpoint_type = DFBEndpointType::CONSUMER},
                     DFBBinding{
                         .dfb_spec_name = W_RMAX, .accessor_name = "rmax", .endpoint_type = DFBEndpointType::PRODUCER},
                     DFBBinding{
                         .dfb_spec_name = W_MOUT, .accessor_name = "mout", .endpoint_type = DFBEndpointType::CONSUMER},
                     DFBBinding{
                         .dfb_spec_name = W_MCB, .accessor_name = "mcb", .endpoint_type = DFBEndpointType::PRODUCER},
                     DFBBinding{
                         .dfb_spec_name = W_PSUM, .accessor_name = "psum", .endpoint_type = DFBEndpointType::CONSUMER},
                     DFBBinding{
                         .dfb_spec_name = W_RSUM, .accessor_name = "rsum", .endpoint_type = DFBEndpointType::PRODUCER},
                     DFBBinding{
                         .dfb_spec_name = W_ROUT, .accessor_name = "rout", .endpoint_type = DFBEndpointType::CONSUMER},
                     DFBBinding{
                         .dfb_spec_name = W_RCB, .accessor_name = "rcb", .endpoint_type = DFBEndpointType::PRODUCER}},
                .semaphore_bindings =
                    {SemaphoreBinding{.semaphore_spec_name = S_RMAX, .accessor_name = "rmax_sem"},
                     SemaphoreBinding{.semaphore_spec_name = S_RSUM, .accessor_name = "rsum_sem"},
                     SemaphoreBinding{.semaphore_spec_name = S_M, .accessor_name = "m_sem"},
                     SemaphoreBinding{.semaphore_spec_name = S_R, .accessor_name = "r_sem"}},
                .tensor_bindings = {TensorBinding{.tensor_parameter_name = W_DST, .accessor_name = "dst"}},
                .runtime_arg_schema = {.runtime_arg_names = {"c",   "active",   "is_comb", "comb_x", "comb_y",
                                                             "x0",  "y0",       "x1",      "y1",     "rect_n",
                                                             "ck0", "nck",      "nchunk",  "T",      "t0",
                                                             "Wt",  "num_rows", "blk",     "row0",   "rstride", "gx0", "gx1"}},
                .hw_config = ttnn::create_writer_datamovement_config(),
            };
            Group<DFBBinding> w_cb = {
                DFBBinding{.dfb_spec_name = W_IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = W_MAXSC,
                    .accessor_name = "max_scaler",
                    .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = W_SUMSC,
                    .accessor_name = "sum_scaler",
                    .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = W_PMAX, .accessor_name = "pmax", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = W_RMAX, .accessor_name = "rmax", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = W_MOUT, .accessor_name = "mout", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = W_MCB, .accessor_name = "mcb", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = W_PSUM, .accessor_name = "psum", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{
                    .dfb_spec_name = W_RSUM, .accessor_name = "rsum", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = W_ROUT, .accessor_name = "rout", .endpoint_type = DFBEndpointType::PRODUCER},
                DFBBinding{.dfb_spec_name = W_RCB, .accessor_name = "rcb", .endpoint_type = DFBEndpointType::CONSUMER},
                DFBBinding{
                    .dfb_spec_name = W_OUT0, .accessor_name = "out0", .endpoint_type = DFBEndpointType::PRODUCER},
            };
            for (const auto* n : {&W_EXPS, &W_PREV}) {
                const char* acc = (n == &W_EXPS) ? "exps" : "prev";
                w_cb.push_back(
                    DFBBinding{.dfb_spec_name = *n, .accessor_name = acc, .endpoint_type = DFBEndpointType::PRODUCER});
                w_cb.push_back(
                    DFBBinding{.dfb_spec_name = *n, .accessor_name = acc, .endpoint_type = DFBEndpointType::CONSUMER});
            }
            KernelSpec w_compute{
                .unique_id = W_COMPUTE,
                .source = std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/compute/softmax_row_split_bh.cpp",
                .compiler_options = {.defines = wcomp, .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
                .dfb_bindings = w_cb,
                .runtime_arg_schema =
                    {.runtime_arg_names =
                         {"T", "nck", "ck0", "nchunk", "L", "last_len", "is_comb", "active", "num_rows", "blk", "pad"}},
                .hw_config = w_hw,
            };

            Group<SemaphoreSpec> wsems;
            for (const auto* s : {&S_RMAX, &S_RSUM, &S_M, &S_R}) {
                wsems.push_back(SemaphoreSpec{.unique_id = *s, .target_nodes = ws_nodes});
            }
            Group<WorkUnitSpec> wwus;
            wwus.push_back(
                WorkUnitSpec{.name = "wu", .kernels = {W_READER, W_WRITER, W_COMPUTE}, .target_nodes = ws_nodes});
            ProgramSpec wspec{
                .name = "softmax_row_split",
                .kernels = {w_reader, w_writer, w_compute},
                .dataflow_buffers = std::move(wdfbs),
                .semaphores = std::move(wsems),
                .tensor_parameters =
                    {TensorParameter{.unique_id = W_SRC, .spec = input_tensor.tensor_spec()},
                     TensorParameter{.unique_id = W_DST, .spec = output_tensor.tensor_spec()}},
                .work_units = std::move(wwus),
            };

            ProgramRunArgs wrun;
            KernelRunArgs wr_ra{.kernel = W_READER};
            KernelRunArgs ww_ra{.kernel = W_WRITER};
            KernelRunArgs wc_ra{.kernel = W_COMPUTE};
            for (std::uint32_t gi = 0; gi < n_groups * rect_n; ++gi) {
                const std::uint32_t grp = gi / rect_n;
                const std::uint32_t c = gi % rect_n;
                const CoreCoord core = {c % rect_w, grp * rect_h + c / rect_w};
                const auto p0 = device->worker_core_from_logical_core(CoreCoord{0, grp * rect_h});
                const auto p1 = device->worker_core_from_logical_core(CoreCoord{rect_w - 1, grp * rect_h + rect_h - 1});
                const std::uint32_t g_rows = (n_rows - grp + n_groups - 1) / n_groups;
                const std::uint32_t t0 = ck0[c] * L;
                AddRuntimeArgsForNode(
                    wr_ra.runtime_arg_values,
                    core,
                    {{"T", tcount[c]},
                     {"t0", t0},
                     {"num_rows", g_rows},
                     {"Wt", Wt},
                     {"blk", block_size},
                     {"pad", t_max - tcount[c]},
                     {"row0", grp},
                     {"rstride", n_groups}});
                AddRuntimeArgsForNode(
                    ww_ra.runtime_arg_values,
                    core,
                    {{"c", c},
                     {"active", active},
                     {"is_comb", c == 0 ? 1u : 0u},
                     {"comb_x", static_cast<std::uint32_t>(p0.x)},
                     {"comb_y", static_cast<std::uint32_t>(p0.y)},
                     {"x0", static_cast<std::uint32_t>(p0.x)},
                     {"y0", static_cast<std::uint32_t>(p0.y)},
                     {"x1", static_cast<std::uint32_t>(p1.x)},
                     {"y1", static_cast<std::uint32_t>(p1.y)},
                     {"rect_n", rect_n},
                     {"ck0", ck0[c]},
                     {"nck", nck[c]},
                     {"nchunk", nchunk},
                     {"T", tcount[c]},
                     {"t0", t0},
                     {"Wt", Wt},
                     {"num_rows", g_rows},
                     {"blk", block_size},
                     {"row0", grp},
                     {"rstride", n_groups},
                     {"gx0", bh_gx0},
                     {"gx1", bh_gx1}});
                AddRuntimeArgsForNode(
                    wc_ra.runtime_arg_values,
                    core,
                    {{"T", tcount[c]},
                     {"nck", nck[c]},
                     {"ck0", ck0[c]},
                     {"nchunk", nchunk},
                     {"L", L},
                     {"last_len", last_len},
                     {"is_comb", c == 0 ? 1u : 0u},
                     {"active", active},
                     {"num_rows", g_rows},
                     {"blk", block_size},
                     {"pad", t_max - tcount[c]}});
            }
            wrun.kernel_run_args = {std::move(wr_ra), std::move(ww_ra), std::move(wc_ra)};
            wrun.tensor_args.emplace(W_SRC, input_tensor.mesh_tensor());
            wrun.tensor_args.emplace(W_DST, output_tensor.mesh_tensor());
            return ttnn::device_operation::ProgramArtifacts{.spec = std::move(wspec), .run_params = std::move(wrun)};
        }
    }

    if (plan.chain_c >= 2) {
        const auto wc_grid = device->compute_with_storage_grid_size();
        const std::uint32_t C = plan.chain_c;
        const std::uint32_t gx = wc_grid.x, gy = wc_grid.y;
        const std::uint32_t bh_nb = Wt / block_size;
        std::uint32_t T = tt::div_up(bh_nb, C) * block_size;
        std::uint32_t bh_f0 = std::min<std::uint32_t>(0, bh_nb >= C ? bh_nb - (C - 1) : 0u);
        std::uint32_t bh_st = 0;
        while (bh_st > 0 && bh_st * C * (C - 1) / 2 + C > bh_nb) {
            --bh_st;
        }
        std::vector<std::uint32_t> bh_pref(C + 1, 0);
        if (bh_st > 0) {
            const std::uint32_t base = (bh_nb - bh_st * C * (C - 1) / 2) / C;
            for (std::uint32_t c = 0; c < C; ++c) {
                bh_pref[c + 1] = bh_pref[c] + base + bh_st * c;
            }
            bh_pref[C] = bh_nb;
        }
        auto bh_b0 = [&](std::uint32_t c) -> std::uint32_t {
            if (bh_st > 0) {
                return bh_pref[c];
            }
            if (bh_f0 == 0) {
                return c * bh_nb / C;
            }
            return c == 0 ? 0u : bh_f0 + (c - 1) * (bh_nb - bh_f0) / (C - 1);
        };
        auto bh_t0 = [&](std::uint32_t c) -> std::uint32_t { return bh_b0(c) * block_size; };
        auto bh_tc = [&](std::uint32_t c) -> std::uint32_t { return (bh_b0(c + 1) - bh_b0(c)) * block_size; };
        {
            const bool bh_bhc = arch == tt::ARCH::BLACKHOLE && 1 != 0;
            const bool bh_colm = bh_bhc && 0 != 0;
            const std::uint32_t cw = (bh_bhc && C > gx) ? (C % gx == 0 ? gx : 8u) : gx;
            const std::uint32_t n_rows = NC * Ht;
            const std::uint32_t per_row = C <= gx ? gx / C : 0;
            const std::uint32_t per_col = gy / C;
            const std::uint32_t cap = bh_colm ? gx * per_col : (C <= gx ? per_row * gy : gy / (C / cw));
            const std::uint32_t G = std::max<std::uint32_t>(1u, std::min<std::uint32_t>(n_rows, cap));
            if (n_rows > G) {
                bh_f0 = 0;
                bh_st = 0;
            }
            if (bh_f0 != 0 || bh_st != 0) {
                T = 0;
                for (std::uint32_t c = 0; c < C; ++c) {
                    T = std::max(T, bh_tc(c));
                }
            }
            auto logical = [&](std::uint32_t g, std::uint32_t c) -> CoreCoord {
                if (bh_colm) {
                    return CoreCoord{g % gx, (g / gx) * C + c};
                }
                if (C <= gx) {
                    return CoreCoord{(g % per_row) * C + c, g / per_row};
                }
                return CoreCoord{c % cw, g * (C / cw) + c / cw};
            };
            CoreRangeSet wc_nodes;
            for (std::uint32_t g = 0; g < G; ++g) {
                const CoreCoord a = logical(g, 0), b = logical(g, C - 1);
                const CoreCoord hi{(bh_colm || C <= gx) ? std::max(a.x, b.x) : cw - 1, b.y};
                wc_nodes = wc_nodes.merge(CoreRangeSet(CoreRange(a, hi)));
            }
            std::uint32_t bh_cx0 = 0, bh_cx1 = 0;
            if (bh_bhc) {
                for (std::uint32_t x = 1; x < gx; ++x) {
                    const auto pa0 = device->worker_core_from_logical_core(CoreCoord{x - 1, 0});
                    const auto pb0 = device->worker_core_from_logical_core(CoreCoord{x, 0});
                    if (pb0.x != pa0.x + 1) {
                        TT_FATAL(bh_cx1 == 0 && pb0.x > pa0.x, "softmax chain: more than one column gap");
                        bh_cx0 = static_cast<std::uint32_t>(pa0.x + 1);
                        bh_cx1 = static_cast<std::uint32_t>(pb0.x);
                    }
                }
                const auto v00 = device->worker_core_from_logical_core(CoreCoord{0, 0});
                for (std::uint32_t y = 0; y < gy; ++y) {
                    for (std::uint32_t x = 0; x < gx; ++x) {
                        const auto v = device->worker_core_from_logical_core(CoreCoord{x, y});
                        const auto vr = device->worker_core_from_logical_core(CoreCoord{x, 0});
                        TT_FATAL(v.x == vr.x && v.y == v00.y + y, "softmax chain: rows not contiguous");
                    }
                }
            } else {
                for (std::uint32_t g = 0; g < G; ++g) {
                    const CoreCoord a = logical(g, 0), b = logical(g, C - 1);
                    check_rect(wc_nodes, a, CoreCoord{std::max(a.x, b.x), b.y});
                }
            }

            const KernelSpecName C_READER{"reader"};
            const KernelSpecName C_WRITER{"writer"};
            const KernelSpecName C_COMPUTE{"compute"};
            const TensorParamName C_SRC{"src"};
            const TensorParamName C_DST{"dst"};
            const DFBSpecName C_IN0{"in0"}, C_OUT0{"out0"}, C_MAXSC{"max_scaler"}, C_SUMSC{"sum_scaler"},
                C_EXPS{"exps"}, C_PMAX{"pmax"}, C_RMAX{"rmax"}, C_MOUT{"mout"}, C_MCB{"mcb"}, C_PACC{"pacc"},
                C_RACC{"racc"}, C_ROUT{"rout"}, C_RCB{"rcb"};
            const SemaphoreSpecName CS_RMAX{"rmax_sem"}, CS_M{"m_sem"}, CS_ACC{"acc_sem"}, CS_R{"r_sem"};

            Group<DataflowBufferSpec> cdfbs;
            auto add_dfb = [&](const DFBSpecName& n, std::uint32_t sz, std::uint32_t cnt, tt::DataFormat f) {
                cdfbs.push_back(DataflowBufferSpec{
                    .unique_id = n, .entry_size = sz, .num_entries = cnt, .data_format_metadata = f});
            };
            add_dfb(C_IN0, in0_tile_size, T, in0_cb_data_format);
            add_dfb(C_OUT0, out0_tile_size, block_size * 2, out0_cb_data_format);
            add_dfb(C_MAXSC, max_scaler_tile_size, 1, max_scaler_cb_data_format);
            add_dfb(C_SUMSC, sum_scaler_tile_size, 1, sum_scaler_cb_data_format);
            add_dfb(C_EXPS, im_tile_size, T, im_cb_data_format);
            add_dfb(C_PMAX, im_tile_size, 1, im_cb_data_format);
            add_dfb(C_RMAX, im_tile_size, C, im_cb_data_format);
            add_dfb(C_MOUT, im_tile_size, 1, im_cb_data_format);
            add_dfb(C_MCB, im_tile_size, 1, im_cb_data_format);
            const std::uint32_t bh_hn = std::string(bh_kdef()).find("BH_HALF") != std::string::npos ? 2u : 1u;
            add_dfb(C_PACC, acc_tile_size, bh_hn, acc_cb_data_format);
            add_dfb(C_RACC, acc_tile_size, bh_hn, acc_cb_data_format);
            add_dfb(C_ROUT, acc_tile_size, bh_hn, acc_cb_data_format);
            add_dfb(C_RCB, acc_tile_size, 1, acc_cb_data_format);
            const bool bh_localrecip = std::string(bh_kdef()).find("BH_LOCALRECIP") != std::string::npos;
            const DFBSpecName C_RCP{"rcp"};
            add_dfb(C_RCP, acc_tile_size, 1, acc_cb_data_format);
            const std::string bh_kdt = std::string(",") + std::string(bh_kdef()) + ",";
            const bool bh_acc = bh_kdt.find(",BH_ACC,") != std::string::npos && attributes.numeric_stable &&
                                std::uint64_t{Wt} * im_tile_size <= 640ull * 1024ull;
            const DFBSpecName C_EXR{"exr"}, C_EXS{"exs"}, C_FLG{"flg"};
            if (bh_acc) {
                add_dfb(C_EXR, im_tile_size, Wt, im_cb_data_format);
                add_dfb(C_EXS, im_tile_size, 2 * block_size, im_cb_data_format);
                add_dfb(C_FLG, im_tile_size, tt::div_up(Wt * 16, im_tile_size), im_cb_data_format);
            }

            KernelSpec::CompilerOptions::Defines cbase;
            if (attributes.numeric_stable) {
                cbase["NUMERIC_STABLE"] = "1";
            }
            {
                const std::uint32_t abl = 0;
                const char* abl_names[] = {"BH_ABL_NOSUM", "BH_ABL_NOMAXX", "BH_ABL_NOEXP", "BH_ABL_NORD", "BH_ABL_NOWR", "BH_ABL_NORB", "BH_ABL_SUMIND"};
                for (std::uint32_t i = 0; i < 7; ++i) {
                    if (abl & (1u << i)) {
                        cbase[abl_names[i]] = "1";
                    }
                }
            }
            if (const char* v = "3"; v != nullptr && *v != 0) {
                cbase["BH_SUM_FIDELITY"] = v;
            }
            if (const char* v = static_cast<const char*>(nullptr); v != nullptr && *v != 0) {
                cbase["BH_MAXFID"] = v;
            }
            {
                std::string kd = bh_kdef();
                std::size_t p0 = 0;
                while (p0 < kd.size()) {
                    std::size_t p1 = kd.find(',', p0);
                    if (p1 == std::string::npos) {
                        p1 = kd.size();
                    }
                    if (p1 > p0) {
                        cbase[kd.substr(p0, p1 - p0)] = "1";
                    }
                    p0 = p1 + 1;
                }
            }
            if (!bh_acc) {
                cbase.erase("BH_ACC");
                cbase.erase("BH_ACCNOC2");
                cbase.erase("BH_ACCNORED");
                cbase.erase("BH_AH2");
                cbase.erase("BH_AH2NOC");
            }
            KernelSpec::CompilerOptions::Defines ccomp = cbase;
            KernelSpec::CompilerOptions::Defines cread = cbase;
            cread["READ_BATCH"] = bh_read_batch_tiles;
            ccomp["EXP_APPROX"] = math_approx_mode ? "1" : "0";
            ccomp["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
            auto c_hw = ttnn::to_compute_hardware_config(attributes.compute_kernel_config);
            if (fp32_dest_acc_en) {
                auto um = [&](const DFBSpecName& n, tt::DataFormat f, bool to_dest) {
                    if (f == tt::DataFormat::Float32) {
                        c_hw.unpack_modes.insert(
                            {n,
                             to_dest ? tt::tt_metal::UnpackMode::UnpackToDest : tt::tt_metal::UnpackMode::UnpackToSrc});
                    }
                };
                um(C_IN0, in0_cb_data_format, false);
                um(C_MAXSC, max_scaler_cb_data_format, false);
                um(C_SUMSC, sum_scaler_cb_data_format, false);
                um(C_EXPS, im_cb_data_format, false);
                um(C_RMAX, im_cb_data_format, false);
                um(C_MCB, im_cb_data_format, false);
                um(C_RACC, acc_cb_data_format, true);
                um(C_RCB, acc_cb_data_format, bh_localrecip);
                um(C_RCP, acc_cb_data_format, false);
                if (bh_acc) {
                    um(C_EXR, im_cb_data_format, false);
                }
            }
            auto P = [](const DFBSpecName& n, const char* a) {
                return DFBBinding{.dfb_spec_name = n, .accessor_name = a, .endpoint_type = DFBEndpointType::PRODUCER};
            };
            auto Q = [](const DFBSpecName& n, const char* a) {
                return DFBBinding{.dfb_spec_name = n, .accessor_name = a, .endpoint_type = DFBEndpointType::CONSUMER};
            };
            KernelSpec c_reader{
                .unique_id = C_READER,
                .source =
                    std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/dataflow/reader_unary_interleaved_sm_row_split_bh.cpp",
                .compiler_options = {.defines = cread},
                .dfb_bindings = {P(C_IN0, "in0"), P(C_MAXSC, "max_scaler"), P(C_SUMSC, "sum_scaler")},
                .tensor_bindings = {TensorBinding{.tensor_parameter_name = C_SRC, .accessor_name = "src"}},
                .runtime_arg_schema =
                    {.runtime_arg_names = {"T", "t0", "num_rows", "Wt", "blk", "pad", "row0", "rstride"}},
                .hw_config = ttnn::create_reader_datamovement_config(),
            };
            Group<DFBBinding> bh_wb = {
                Q(C_OUT0, "out0"),
                Q(C_PMAX, "pmax"),
                P(C_RMAX, "rmax"),
                Q(C_MOUT, "mout"),
                P(C_MCB, "mcb"),
                Q(C_PACC, "pacc"),
                P(C_RACC, "racc"),
                Q(C_ROUT, "rout"),
                P(C_RCB, "rcb")};
            if (bh_acc) {
                bh_wb.push_back(P(C_EXR, "exr"));
                bh_wb.push_back(Q(C_EXS, "exs"));
                bh_wb.push_back(P(C_FLG, "flg"));
                bh_wb.push_back(Q(C_FLG, "flg"));
            }
            KernelSpec c_writer{
                .unique_id = C_WRITER,
                .source =
                    std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/dataflow/writer_unary_interleaved_sm_row_chain_bh.cpp",
                .compiler_options = {.defines = cbase},
                .dfb_bindings = bh_wb,
                .semaphore_bindings =
                    {SemaphoreBinding{.semaphore_spec_name = CS_RMAX, .accessor_name = "rmax_sem"},
                     SemaphoreBinding{.semaphore_spec_name = CS_M, .accessor_name = "m_sem"},
                     SemaphoreBinding{.semaphore_spec_name = CS_ACC, .accessor_name = "acc_sem"},
                     SemaphoreBinding{.semaphore_spec_name = CS_R, .accessor_name = "r_sem"}},
                .tensor_bindings = {TensorBinding{.tensor_parameter_name = C_DST, .accessor_name = "dst"}},
                .runtime_arg_schema =
                    {.runtime_arg_names =
                         {"c",
                          "C",
                          "comb_x",
                          "comb_y",
                          "next_x",
                          "next_y",
                          "x0",
                          "y0",
                          "x1",
                          "y1",
                          "T",
                          "t0",
                          "Wt",
                          "num_rows",
                          "blk",
                          "row0",
                          "rstride",
                          "gx0",
                          "gx1"}},
                .hw_config = ttnn::create_writer_datamovement_config(),
            };
            Group<DFBBinding> c_cb = {
                Q(C_IN0, "in0"),
                Q(C_MAXSC, "max_scaler"),
                Q(C_SUMSC, "sum_scaler"),
                P(C_PMAX, "pmax"),
                Q(C_RMAX, "rmax"),
                P(C_MOUT, "mout"),
                Q(C_MCB, "mcb"),
                P(C_PACC, "pacc"),
                Q(C_RACC, "racc"),
                P(C_ROUT, "rout"),
                Q(C_RCB, "rcb"),
                P(C_RCP, "rcp"),
                Q(C_RCP, "rcp"),
                P(C_OUT0, "out0"),
                P(C_EXPS, "exps"),
                Q(C_EXPS, "exps")};
            if (bh_acc) {
                c_cb.push_back(Q(C_EXR, "exr"));
                c_cb.push_back(P(C_EXS, "exs"));
            }
            KernelSpec c_compute{
                .unique_id = C_COMPUTE,
                .source = std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/compute/softmax_row_chain_bh.cpp",
                .compiler_options = {.defines = ccomp, .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
                .dfb_bindings = c_cb,
                .runtime_arg_schema = {.runtime_arg_names = {"T", "c", "C", "num_rows", "blk", "Wt", "t0"}},
                .hw_config = c_hw,
            };
            Group<SemaphoreSpec> csems;
            for (const auto* s : {&CS_RMAX, &CS_M, &CS_ACC, &CS_R}) {
                csems.push_back(SemaphoreSpec{.unique_id = *s, .target_nodes = wc_nodes});
            }
            Group<WorkUnitSpec> cwus;
            cwus.push_back(
                WorkUnitSpec{.name = "wu", .kernels = {C_READER, C_WRITER, C_COMPUTE}, .target_nodes = wc_nodes});
            ProgramSpec cspec{
                .name = "softmax_row_chain",
                .kernels = {c_reader, c_writer, c_compute},
                .dataflow_buffers = std::move(cdfbs),
                .semaphores = std::move(csems),
                .tensor_parameters =
                    {TensorParameter{.unique_id = C_SRC, .spec = input_tensor.tensor_spec()},
                     TensorParameter{.unique_id = C_DST, .spec = output_tensor.tensor_spec()}},
                .work_units = std::move(cwus),
            };
            ProgramRunArgs crun;
            KernelRunArgs cr_ra{.kernel = C_READER};
            KernelRunArgs cw_ra{.kernel = C_WRITER};
            KernelRunArgs cc_ra{.kernel = C_COMPUTE};
            for (std::uint32_t g = 0; g < G; ++g) {
                const std::uint32_t g_rows = (n_rows - g + G - 1) / G;
                const auto pa = device->worker_core_from_logical_core(logical(g, 0));
                const auto pb = device->worker_core_from_logical_core(logical(g, C - 1));
                for (std::uint32_t c = 0; c < C; ++c) {
                    const CoreCoord core = logical(g, c);
                    const auto pn = device->worker_core_from_logical_core(logical(g, c + 1 < C ? c + 1 : c));
                    const std::uint32_t t0 = bh_t0(c);
                    const std::uint32_t Tc = bh_tc(c);
                    AddRuntimeArgsForNode(
                        cr_ra.runtime_arg_values,
                        core,
                        {{"T", Tc},
                         {"t0", t0},
                         {"num_rows", g_rows},
                         {"Wt", Wt},
                         {"blk", block_size},
                         {"pad", 0u},
                         {"row0", g},
                         {"rstride", G}});
                    AddRuntimeArgsForNode(
                        cw_ra.runtime_arg_values,
                        core,
                        {{"c", c},
                         {"C", C},
                         {"comb_x", static_cast<std::uint32_t>(pa.x)},
                         {"comb_y", static_cast<std::uint32_t>(pa.y)},
                         {"next_x", static_cast<std::uint32_t>(pn.x)},
                         {"next_y", static_cast<std::uint32_t>(pn.y)},
                         {"x0", static_cast<std::uint32_t>(std::min(pa.x, pb.x))},
                         {"y0", static_cast<std::uint32_t>(pa.y)},
                         {"x1", static_cast<std::uint32_t>(std::max(pa.x, pb.x))},
                         {"y1", static_cast<std::uint32_t>(pb.y)},
                         {"T", Tc},
                         {"t0", t0},
                         {"Wt", Wt},
                         {"num_rows", g_rows},
                         {"blk", block_size},
                         {"row0", g},
                         {"rstride", G},
                         {"gx0", bh_cx0},
                         {"gx1", bh_cx1}});
                    AddRuntimeArgsForNode(
                        cc_ra.runtime_arg_values,
                        core,
                        {{"T", Tc}, {"c", c}, {"C", C}, {"num_rows", g_rows}, {"blk", block_size}, {"Wt", Wt}, {"t0", t0}});
                }
            }
            crun.kernel_run_args = {std::move(cr_ra), std::move(cw_ra), std::move(cc_ra)};
            crun.tensor_args.emplace(C_SRC, input_tensor.mesh_tensor());
            crun.tensor_args.emplace(C_DST, output_tensor.mesh_tensor());
            return ttnn::device_operation::ProgramArtifacts{.spec = std::move(cspec), .run_params = std::move(crun)};
        }
    }

    const std::uint32_t num_tile_rows = NC * Ht;
    const auto grid_size = device->compute_with_storage_grid_size();
    auto
        [num_cores,
         all_cores,
         core_group_1,
         core_group_2,
         num_tile_rows_per_core_group_1,
         num_tile_rows_per_core_group_2] = tt::tt_metal::split_work_to_cores(grid_size, num_tile_rows, true);

    const KernelSpecName READER{"reader"};
    const KernelSpecName WRITER{"writer"};
    const KernelSpecName COMPUTE{"compute"};

    const TensorParamName SRC{"src"};
    const TensorParamName DST{"dst"};
    const TensorParamName MASK{"mask"};

    const DFBSpecName IN0{"in0"};
    const DFBSpecName OUT0{"out0"};
    const DFBSpecName MAX_SCALER{"max_scaler"};
    const DFBSpecName SUM_SCALER{"sum_scaler"};
    const DFBSpecName FUSED_SCALE{"fused_scale"};
    const DFBSpecName FUSED_ATTN{"fused_attn"};
    const DFBSpecName MASK_PADDED{"mask_padded"};
    const DFBSpecName EXPS{"exps"};
    const DFBSpecName SCALE_MASK{"scale_mask"};
    const DFBSpecName RECIP_SUM_EXPS{"recip_sum_exps"};
    const DFBSpecName MAX{"max"};
    const DFBSpecName X{"x"};
    const DFBSpecName PREV_REDUCE{"prev_reduce"};
    const DFBSpecName PREV_MAX{"prev_max"};
    const DFBSpecName RECIP{"recip"};

    Group<DataflowBufferSpec> dfbs;
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = IN0,
        .entry_size = in0_tile_size,
        .num_entries = in0_t,
        .data_format_metadata = in0_cb_data_format});
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = OUT0,
        .entry_size = out0_tile_size,
        .num_entries = out0_t * (deep_out ? bh_deep_out_depth : 1u),
        .data_format_metadata = out0_cb_data_format});
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = RECIP_SUM_EXPS,
        .entry_size = acc_tile_size,
        .num_entries = im1_t,
        .data_format_metadata = acc_cb_data_format});
    if (use_large_kernel) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = PREV_REDUCE,
            .entry_size = acc_tile_size,
            .num_entries = im1_t,
            .data_format_metadata = acc_cb_data_format});
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = PREV_MAX,
            .entry_size = im_tile_size,
            .num_entries = im1_t,
            .data_format_metadata = im_cb_data_format});
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = RECIP,
            .entry_size = acc_tile_size,
            .num_entries = im1_t,
            .data_format_metadata = acc_cb_data_format});
    }
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = MAX_SCALER,
        .entry_size = max_scaler_tile_size,
        .num_entries = in2_t,
        .data_format_metadata = max_scaler_cb_data_format});
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = SUM_SCALER,
        .entry_size = sum_scaler_tile_size,
        .num_entries = in2_t,
        .data_format_metadata = sum_scaler_cb_data_format});
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = EXPS,
        .entry_size = im_tile_size,
        .num_entries = im0_t,
        .data_format_metadata = im_cb_data_format});
    if (has_mask) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = SCALE_MASK,
            .entry_size = im_tile_size,
            .num_entries = im3_t,
            .data_format_metadata = im_cb_data_format});
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = FUSED_SCALE,
            .entry_size = fused_attention_scale_tile_size,
            .num_entries = in3_t,
            .data_format_metadata = tt::DataFormat::Float16_b});
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = FUSED_ATTN,
            .entry_size = mask_tile_size,
            .num_entries = in4_t,
            .data_format_metadata = mask_cb_data_format});
    }
    dfbs.push_back(DataflowBufferSpec{
        .unique_id = MASK_PADDED,
        .entry_size = mask_padded_tile_size,
        .num_entries = in5_t,
        .data_format_metadata = tt::DataFormat::Float16_b});
    if (attributes.numeric_stable) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = MAX,
            .entry_size = im_tile_size,
            .num_entries = im2_t,
            .data_format_metadata = im_cb_data_format});
    }
    const bool alloc_cb_x = small_kernel_numeric_stable_uses_cb_x_at_wt || use_large_kernel;
    if (alloc_cb_x) {
        dfbs.push_back(DataflowBufferSpec{
            .unique_id = X,
            .entry_size = im_tile_size,
            .num_entries = im4_t,
            .data_format_metadata = im_cb_data_format});
    }

    KernelSpec::CompilerOptions::Defines base_defines;
    if (has_mask) {
        base_defines["FUSED_SCALE_MASK"] = "1";
    }
    if (attributes.is_causal_mask) {
        base_defines["CAUSAL_MASK"] = "1";
    }
    if (attributes.numeric_stable) {
        base_defines["NUMERIC_STABLE"] = "1";
    }

    KernelSpec::CompilerOptions::Defines compute_defines = base_defines;
    KernelSpec::CompilerOptions::Defines reader_defines = base_defines;
    KernelSpec::CompilerOptions::Defines writer_defines;
    if (plan.read_batch) {
        reader_defines["READ_BATCH"] = use_large_kernel ? bh_large_kernel_read_batch_tiles : bh_read_batch_tiles;
    }
    if (plan.flush_writes) {
        writer_defines["FLUSH_WRITES"] = "1";
    }
    compute_defines["EXP_APPROX"] = math_approx_mode ? "1" : "0";
    compute_defines["ENABLE_FP32_DEST_ACC"] = fp32_dest_acc_en ? "1" : "0";
    if (!use_large_kernel && mask_padded_data) {
        compute_defines["MASK_PADDED_DATA"] = "1";
    }

    Group<DFBBinding> reader_bindings = {
        DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{
            .dfb_spec_name = MAX_SCALER, .accessor_name = "max_scaler", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{
            .dfb_spec_name = SUM_SCALER, .accessor_name = "sum_scaler", .endpoint_type = DFBEndpointType::PRODUCER},
    };
    Group<TensorBinding> reader_tensor_bindings = {TensorBinding{.tensor_parameter_name = SRC, .accessor_name = "src"}};
    std::vector<std::string> reader_rta_names = {"blk", "num_rows", "tile_offset", "Wt"};
    if (has_mask) {
        reader_bindings.push_back(DFBBinding{
            .dfb_spec_name = FUSED_ATTN, .accessor_name = "fused_attn", .endpoint_type = DFBEndpointType::PRODUCER});
        reader_bindings.push_back(DFBBinding{
            .dfb_spec_name = FUSED_SCALE, .accessor_name = "fused_scale", .endpoint_type = DFBEndpointType::PRODUCER});
        reader_tensor_bindings.push_back(TensorBinding{.tensor_parameter_name = MASK, .accessor_name = "mask"});
        reader_rta_names.push_back("pre_scale");
        reader_rta_names.push_back("Ht");
        reader_rta_names.push_back("start_ht");
        reader_rta_names.push_back("start_mask_id");
        if (attributes.is_causal_mask) {
            reader_rta_names.push_back("mask_start_ht");
            reader_rta_names.push_back("mask_offset");
        }
    }
    KernelSpec::CompileTimeArgs reader_cta;
    if (use_large_kernel) {
        reader_cta.insert({"dfb_length", dfb_length});
    } else {
        reader_cta.insert({"in0_t", in0_t});
    }
    if (has_mask && attributes.is_causal_mask) {
        const std::uint32_t num_tiles_causal_mask = tensor_args.mask.value().padded_shape()[-1] *
                                                    tensor_args.mask.value().padded_shape()[-2] / tile_width /
                                                    tile_height;
        reader_cta.insert({"num_tiles_causal_mask", num_tiles_causal_mask});
    }
    KernelSpec reader{
        .unique_id = READER,
        .source = std::string(SOFTMAX_KERNEL_PATH_ATTENTION) +
                  (use_large_kernel ? "/dataflow/reader_unary_interleaved_sm_large_tensor.cpp"
                                    : "/dataflow/reader_unary_interleaved_sm.cpp"),
        .compiler_options = {.defines = reader_defines},
        .dfb_bindings = reader_bindings,
        .tensor_bindings = reader_tensor_bindings,
        .compile_time_args = reader_cta,
        .runtime_arg_schema = {.runtime_arg_names = reader_rta_names},
        .hw_config = ttnn::create_reader_datamovement_config(),
    };

    KernelSpec writer{
        .unique_id = WRITER,
        .source =
            std::string(SOFTMAX_KERNEL_PATH_ATTENTION) + "/dataflow/writer_unary_interleaved_start_id_blocked_sm.cpp",
        .compiler_options = {.defines = writer_defines},
        .dfb_bindings =
            {DFBBinding{.dfb_spec_name = OUT0, .accessor_name = "out0", .endpoint_type = DFBEndpointType::CONSUMER},
             DFBBinding{
                 .dfb_spec_name = MASK_PADDED,
                 .accessor_name = "mask_padded",
                 .endpoint_type = DFBEndpointType::PRODUCER}},
        .tensor_bindings = {TensorBinding{.tensor_parameter_name = DST, .accessor_name = "dst"}},
        .compile_time_args = {{"num_datum_padded", num_datum_padded}, {"tile_hw", tile_height * tile_width}},
        .runtime_arg_schema = {.runtime_arg_names = {"num_tiles", "tile_offset", "blk", "mask_padded_data", "Wt"}},
        .hw_config = ttnn::create_writer_datamovement_config(),
    };

    Group<DFBBinding> compute_bindings = {
        DFBBinding{.dfb_spec_name = IN0, .accessor_name = "in0", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{
            .dfb_spec_name = MAX_SCALER, .accessor_name = "max_scaler", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{
            .dfb_spec_name = SUM_SCALER, .accessor_name = "sum_scaler", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{
            .dfb_spec_name = MASK_PADDED, .accessor_name = "mask_padded", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{.dfb_spec_name = OUT0, .accessor_name = "out0", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = EXPS, .accessor_name = "exps", .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{.dfb_spec_name = EXPS, .accessor_name = "exps", .endpoint_type = DFBEndpointType::CONSUMER},
        DFBBinding{
            .dfb_spec_name = RECIP_SUM_EXPS,
            .accessor_name = "recip_sum_exps",
            .endpoint_type = DFBEndpointType::PRODUCER},
        DFBBinding{
            .dfb_spec_name = RECIP_SUM_EXPS,
            .accessor_name = "recip_sum_exps",
            .endpoint_type = DFBEndpointType::CONSUMER},
    };
    auto add_self_loop = [&](const DFBSpecName& name, const char* accessor) {
        compute_bindings.push_back(
            DFBBinding{.dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = DFBEndpointType::PRODUCER});
        compute_bindings.push_back(
            DFBBinding{.dfb_spec_name = name, .accessor_name = accessor, .endpoint_type = DFBEndpointType::CONSUMER});
    };
    if (has_mask) {
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = FUSED_SCALE, .accessor_name = "fused_scale", .endpoint_type = DFBEndpointType::CONSUMER});
        compute_bindings.push_back(DFBBinding{
            .dfb_spec_name = FUSED_ATTN, .accessor_name = "fused_attn", .endpoint_type = DFBEndpointType::CONSUMER});
        add_self_loop(SCALE_MASK, "scale_mask");
    }
    if (attributes.numeric_stable) {
        add_self_loop(MAX, "max");
    }
    if (alloc_cb_x) {
        add_self_loop(X, "x");
    }
    if (use_large_kernel) {
        add_self_loop(PREV_REDUCE, "prev_reduce");
        add_self_loop(PREV_MAX, "prev_max");
        add_self_loop(RECIP, "recip");
    }

    std::vector<std::string> compute_rta_names;
    if (use_large_kernel) {
        compute_rta_names = {"num_rows", "Wt", "blk", "mask_padded_data", "dfb_length"};
    } else {
        compute_rta_names = {"num_rows", "Ht", "Wt", "blk", "start_ht"};
    }

    auto compute_hw = ttnn::to_compute_hardware_config(attributes.compute_kernel_config);
    if (fp32_dest_acc_en) {
        auto& gen1 = compute_hw;
        auto add_unpack = [&](const DFBSpecName& name, tt::DataFormat fmt) {
            if (fmt == tt::DataFormat::Float32) {
                gen1.unpack_modes.insert({name, tt::tt_metal::UnpackMode::UnpackToSrc});
            }
        };
        add_unpack(IN0, in0_cb_data_format);
        add_unpack(MAX_SCALER, max_scaler_cb_data_format);
        add_unpack(SUM_SCALER, sum_scaler_cb_data_format);
        add_unpack(MASK_PADDED, tt::DataFormat::Float16_b);
        add_unpack(EXPS, im_cb_data_format);
        add_unpack(RECIP_SUM_EXPS, acc_cb_data_format);
        if (has_mask) {
            add_unpack(FUSED_SCALE, tt::DataFormat::Float16_b);
            add_unpack(FUSED_ATTN, mask_cb_data_format);
            add_unpack(SCALE_MASK, im_cb_data_format);
        }
        if (attributes.numeric_stable) {
            add_unpack(MAX, im_cb_data_format);
        }
        if (alloc_cb_x) {
            add_unpack(X, im_cb_data_format);
        }
        if (use_large_kernel) {
            add_unpack(PREV_REDUCE, acc_cb_data_format);
            add_unpack(PREV_MAX, im_cb_data_format);
            add_unpack(RECIP, acc_cb_data_format);
        }
    }

    KernelSpec::CompileTimeArgs compute_cta;
    if (!use_large_kernel) {
        compute_cta.insert({"in0_t", in0_t});
    }

    KernelSpec compute{
        .unique_id = COMPUTE,
        .source = std::string(SOFTMAX_KERNEL_PATH_ATTENTION) +
                  (use_large_kernel ? "/compute/softmax_large_tensor.cpp" : "/compute/softmax.cpp"),
        .compiler_options = {.defines = compute_defines, .opt_level = tt::tt_metal::KernelBuildOptLevel::O3},
        .dfb_bindings = compute_bindings,
        .compile_time_args = compute_cta,
        .runtime_arg_schema = {.runtime_arg_names = compute_rta_names},
        .hw_config = compute_hw,
    };

    Group<TensorParameter> tensor_parameters = {
        TensorParameter{.unique_id = SRC, .spec = input_tensor.tensor_spec()},
        TensorParameter{.unique_id = DST, .spec = output_tensor.tensor_spec()}};
    if (has_mask) {
        tensor_parameters.push_back(TensorParameter{.unique_id = MASK, .spec = tensor_args.mask.value().tensor_spec()});
    }

    Group<WorkUnitSpec> work_units;
    work_units.push_back(WorkUnitSpec{.name = "wu", .kernels = {READER, WRITER, COMPUTE}, .target_nodes = all_cores});

    ProgramSpec spec{
        .name = "softmax_attention_optimized",
        .kernels = {reader, writer, compute},
        .dataflow_buffers = std::move(dfbs),
        .tensor_parameters = std::move(tensor_parameters),
        .work_units = std::move(work_units),
    };

    ProgramRunArgs run_args;
    KernelRunArgs reader_ra{.kernel = READER};
    KernelRunArgs writer_ra{.kernel = WRITER};
    KernelRunArgs compute_ra{.kernel = COMPUTE};

    const std::uint32_t scale_value = std::bit_cast<std::uint32_t>(attributes.scale.value_or(1.0f));

    std::uint32_t curr_row = 0;
    for (std::uint32_t i = 0; i < num_cores; ++i) {
        CoreCoord core = {i % grid_size.x, i / grid_size.x};
        std::uint32_t num_tile_rows_per_core = 0;
        if (core_group_1.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_1;
        } else if (core_group_2.contains(core)) {
            num_tile_rows_per_core = num_tile_rows_per_core_group_2;
        } else {
            TT_THROW("Core not in specified core ranges");
        }

        std::uint32_t tile_offset = curr_row * Wt;
        std::uint32_t curr_ht = curr_row % Ht;
        std::uint32_t mask_curr_ht = curr_ht % mask_Ht;
        std::uint32_t mask_offset = curr_row / Ht * mask_Ht * Wt;
        std::uint32_t mask_id = attributes.is_causal_mask ? ((mask_curr_ht * Wt) + mask_offset) : (curr_row / Ht * Wt);

        AddRuntimeArgsForNode(
            reader_ra.runtime_arg_values,
            core,
            {{"blk", block_size}, {"num_rows", num_tile_rows_per_core}, {"tile_offset", tile_offset}, {"Wt", Wt}});
        if (has_mask) {
            AddRuntimeArgsForNode(
                reader_ra.runtime_arg_values,
                core,
                {{"pre_scale", scale_value}, {"Ht", Ht}, {"start_ht", curr_ht}, {"start_mask_id", mask_id}});
            if (attributes.is_causal_mask) {
                AddRuntimeArgsForNode(
                    reader_ra.runtime_arg_values,
                    core,
                    {{"mask_start_ht", mask_curr_ht}, {"mask_offset", mask_offset}});
            }
        }

        AddRuntimeArgsForNode(
            writer_ra.runtime_arg_values,
            core,
            {{"num_tiles", num_tile_rows_per_core * Wt},
             {"tile_offset", tile_offset},
             {"blk", block_size},
             {"mask_padded_data", static_cast<std::uint32_t>(mask_padded_data)},
             {"Wt", Wt}});

        if (use_large_kernel) {
            AddRuntimeArgsForNode(
                compute_ra.runtime_arg_values,
                core,
                {{"num_rows", num_tile_rows_per_core},
                 {"Wt", Wt},
                 {"blk", block_size},
                 {"mask_padded_data", static_cast<std::uint32_t>(mask_padded_data)},
                 {"dfb_length", dfb_length}});
        } else {
            AddRuntimeArgsForNode(
                compute_ra.runtime_arg_values,
                core,
                {{"num_rows", num_tile_rows_per_core},
                 {"Ht", Ht},
                 {"Wt", Wt},
                 {"blk", block_size},
                 {"start_ht", curr_ht}});
        }

        curr_row += num_tile_rows_per_core;
    }

    run_args.kernel_run_args = {std::move(reader_ra), std::move(writer_ra), std::move(compute_ra)};
    run_args.tensor_args.emplace(SRC, input_tensor.mesh_tensor());
    run_args.tensor_args.emplace(DST, output_tensor.mesh_tensor());
    if (has_mask) {
        run_args.tensor_args.emplace(MASK, tensor_args.mask.value().mesh_tensor());
    }

    return ttnn::device_operation::ProgramArtifacts{.spec = std::move(spec), .run_params = std::move(run_args)};
}

}
