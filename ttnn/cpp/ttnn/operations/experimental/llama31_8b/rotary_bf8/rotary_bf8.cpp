// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b/rotary_bf8/rotary_bf8.hpp"

#include <tt-metalium/constants.hpp>

#include "ttnn/operations/copy/typecast/typecast.hpp"
#include "ttnn/operations/experimental/llama31_8b/rotary_bf8/device/rotary_bf8_device_operation.hpp"
#include "ttnn/operations/experimental/llama31_8b/rotary_bf8/device/rotary_bf8_table.hpp"
#include "ttnn/operations/experimental/transformer/rotary_embedding_llama/rotary_embedding_llama.hpp"

namespace ttnn::experimental::llama31_8b {

namespace {

bool dram_interleaved_bf16_tile(const Tensor& t) {
    return t.storage_type() == StorageType::DEVICE && t.dtype() == DataType::BFLOAT16 && t.layout() == Layout::TILE &&
           t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
           t.memory_config().buffer_type() == BufferType::DRAM;
}

std::optional<std::uint32_t> table_row(
    const Tensor& input,
    const Tensor& cos_cache,
    const Tensor& sin_cache,
    const Tensor& trans_mat,
    const MemoryConfig& memory_config,
    const DeviceComputeKernelConfig& config) {
    using tt::constants::TILE_HEIGHT;
    using tt::constants::TILE_WIDTH;
    if (input.device()->arch() != tt::ARCH::BLACKHOLE) {
        return std::nullopt;
    }
    for (const Tensor* t : {&input, &cos_cache, &sin_cache, &trans_mat}) {
        if (!dram_interleaved_bf16_tile(*t)) {
            return std::nullopt;
        }
    }
    if (memory_config.memory_layout() != TensorMemoryLayout::INTERLEAVED ||
        memory_config.buffer_type() != BufferType::DRAM) {
        return std::nullopt;
    }
    const auto& in = input.padded_shape();
    const auto& cs = cos_cache.padded_shape();
    if (in.rank() != 4 || cs.rank() != 4 || in[0] != 1 || cs[0] != 1 || cs[1] != 1 || cs[-1] != in[-1] ||
        sin_cache.padded_shape() != cs || trans_mat.padded_shape()[-1] != TILE_WIDTH ||
        trans_mat.padded_shape()[-2] != TILE_HEIGHT) {
        return std::nullopt;
    }
    const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
        get_compute_kernel_config_args(input.device()->arch(), config);
    if (math_fidelity != tt::tt_metal::MathFidelity::HiFi4 || fp32_dest_acc_en) {
        return std::nullopt;
    }
    const auto grid = input.device()->compute_with_storage_grid_size();
    const auto* e =
        rotary_bf8_table::find(grid.x, grid.y, in[1], in[2] / TILE_HEIGHT, in[3] / TILE_WIDTH, cs[2] / TILE_HEIGHT);
    if (e == nullptr) {
        return std::nullopt;
    }
    return static_cast<std::uint32_t>(e - rotary_bf8_table::ENTRIES.data());
}

}  // namespace

ttnn::Tensor rotary_bf8(
    const ttnn::Tensor& input,
    const ttnn::Tensor& cos_cache,
    const ttnn::Tensor& sin_cache,
    const ttnn::Tensor& trans_mat,
    DataType output_dtype,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<const DeviceComputeKernelConfig>& compute_kernel_config) {
    const auto config = init_device_compute_kernel_config(
        input.device()->arch(), compute_kernel_config, tt::tt_metal::MathFidelity::HiFi4, true, false, false);
    const auto out_memory_config = memory_config.value_or(input.memory_config());
    const auto row = table_row(input, cos_cache, sin_cache, trans_mat, out_memory_config, config);
    if (!row.has_value()) {
        auto rotated = ttnn::experimental::rotary_embedding_llama(
            input, cos_cache, sin_cache, trans_mat, false, out_memory_config, compute_kernel_config);
        if (output_dtype == rotated.dtype()) {
            return rotated;
        }
        return ttnn::typecast(rotated, output_dtype, out_memory_config);
    }
    return ttnn::prim::llama31_8b_rotary_bf8(
        input, cos_cache, sin_cache, trans_mat, output_dtype, out_memory_config, config, row.value());
}

}  // namespace ttnn::experimental::llama31_8b
