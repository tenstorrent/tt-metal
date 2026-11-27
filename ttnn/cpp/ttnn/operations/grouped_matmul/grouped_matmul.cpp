// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#include "grouped_matmul.hpp"

#include <algorithm>
#include <string>
#include <utility>

#include <fmt/format.h>

#include <tt-metalium/constants.hpp>
#include <tt_stl/assert.hpp>

#include "device/grouped_matmul_device_operation.hpp"

namespace ttnn {

namespace {

using operations::grouped_matmul::GroupedMatmulAssertion;
using operations::grouped_matmul::GroupedMatmulParams;

// The Python op's assert: the message is formatted only when the check fails (the entry runs on every call).
template <typename... Args>
void assert_that(bool condition, const char* format, Args&&... args) {
    if (!condition) {
        throw GroupedMatmulAssertion(fmt::format(fmt::runtime(format), std::forward<Args>(args)...));
    }
}

void check_interleaved_dram(const Tensor& t) {
    assert_that(t.layout() == Layout::TILE, "grouped_matmul: layout {} (TILE)", t.layout());
    const MemoryConfig& mc = t.memory_config();
    assert_that(
        mc.buffer_type() == BufferType::DRAM && mc.memory_layout() == TensorMemoryLayout::INTERLEAVED,
        "grouped_matmul: {} (DRAM INTERLEAVED)",
        mc);
}

void check_bank_w(const Tensor& w, const operations::grouped_matmul::BankLayout& layout) {
    const MemoryConfig& mc = w.memory_config();
    assert_that(
        w.layout() == Layout::TILE && mc.buffer_type() == BufferType::DRAM, "grouped_matmul: w {} {}", w.layout(), mc);
    assert_that(mc.memory_layout() == TensorMemoryLayout::WIDTH_SHARDED, "w_layout needs the bank tensor, got {}", mc);
    const auto& shard = mc.shard_spec();
    assert_that(
        shard.has_value() && shard->shape[0] == layout.tiles_per_bank() * tt::constants::TILE_HEIGHT &&
            shard->shape[1] == tt::constants::TILE_WIDTH,
        "w_layout: shard shape does not match the layout");
}

}  // namespace

Tensor grouped_matmul(
    const Tensor& x,
    const Tensor& w,
    const std::optional<DataType>& dtype,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config,
    const std::optional<Tensor>& output_tensor,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<CoreCoord>& max_grid,
    std::optional<uint32_t> block_w,
    uint32_t depth,
    uint32_t rt,
    const std::optional<operations::grouped_matmul::BankLayout>& w_layout,
    uint32_t w_inflight,
    uint32_t x_ahead,
    bool one_group,
    const std::vector<std::string>& diag) {
    check_interleaved_dram(x);
    if (w_layout.has_value()) {
        check_bank_w(w, *w_layout);
    } else {
        check_interleaved_dram(w);
    }

    GroupedMatmulParams params;
    params.output_dtype = dtype.value_or(x.dtype());

    const auto& xs = x.logical_shape();
    const auto& ws = w.logical_shape();
    // (the bank path's w is the layout's 2-D bank tensor: only the interleaved w gives the output's N)
    assert_that(
        xs.rank() == 4 && (w_layout.has_value() || ws.rank() == 4),
        "grouped_matmul: x {} and w {} must be rank 4",
        xs,
        ws);
    const uint32_t n = w_layout.has_value() ? w_layout->NT * tt::constants::TILE_WIDTH : ws[3];
    const ttnn::Shape out_shape({1, xs[1], xs[2], n});

    if (output_tensor.has_value()) {
        const Tensor& o = *output_tensor;
        assert_that(
            o.dtype() == params.output_dtype && o.layout() == Layout::TILE && o.logical_shape() == out_shape,
            "grouped_matmul: output_tensor {} {}",
            o.dtype(),
            o.logical_shape());
        assert_that(
            o.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED,
            "grouped_matmul: output_tensor {}",
            o.memory_config());
        params.output_memory_config = o.memory_config();
    } else {
        params.output_memory_config = memory_config.value_or(ttnn::DRAM_MEMORY_CONFIG);
    }

    for (const auto& d : diag) {
        if (d == "no_w") {
            params.diag_no_w = true;
        } else if (d == "no_x") {
            params.diag_no_x = true;
        } else if (d == "no_compute") {
            params.diag_no_compute = true;
        } else {
            throw GroupedMatmulAssertion(fmt::format("grouped_matmul: diag {}", d));
        }
    }

    // (the Python op read x.device() here and failed on None: a host or deallocated x)
    TT_FATAL(x.device() != nullptr, "grouped_matmul: x must be an allocated device tensor");
    if (compute_kernel_config.has_value()) {
        const auto [math_fidelity, math_approx_mode, fp32_dest_acc_en, packer_l1_acc, dst_full_sync_en] =
            get_compute_kernel_config_args(x.device()->arch(), *compute_kernel_config);
        params.math_fidelity = math_fidelity;
        params.fp32_dest_acc_en = fp32_dest_acc_en;
        params.math_approx_mode = math_approx_mode;
    }

    const CoreCoord grid = x.device()->compute_with_storage_grid_size();
    if (max_grid.has_value()) {
        params.grid_x = static_cast<uint32_t>(std::max<size_t>(1, std::min(grid.x, max_grid->x)));
        params.grid_y = static_cast<uint32_t>(std::max<size_t>(1, std::min(grid.y, max_grid->y)));
    } else {
        params.grid_x = static_cast<uint32_t>(grid.x);
        params.grid_y = static_cast<uint32_t>(grid.y);
    }
    params.block_w = block_w.value_or(0);
    params.depth = depth;
    params.rt = rt;
    if (w_layout.has_value()) {
        params.bank_layout = w_layout;
        params.w_inflight = w_inflight;
        params.x_ahead = x_ahead;
        params.one_group = one_group;
    }
    return ttnn::prim::grouped_matmul(x, w, output_tensor, params);
}

}  // namespace ttnn
