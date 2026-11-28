// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/grouped_matmul/device/grouped_matmul_device_operation_types.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

// x [1, G, M, K] @ w [1, G, K, N] -> [1, G, M, N] (TILE, DRAM interleaved), one grid row a group.
//   dtype                  output dtype (default x's)
//   compute_kernel_config  math fidelity, fp32 DEST, approx mode (default HiFi2, fp32 DEST, exact)
//   output_tensor          the caller's buffer of exactly the output spec, returned as is
//   memory_config          output placement when allocated (default DRAM interleaved)
//   max_grid               cores (0, 0) .. (max_grid - 1) (default the whole compute grid)
//   block_w, depth, rt     K tiles a block (default the widest that fits), K blocks buffered, row tiles a DST block
//   w_layout               bank layout: w is then the layout's WIDTH_SHARDED DRAM weight (bank path)
//   w_inflight, x_ahead, one_group   bank path knobs
//   diag                   diagnostic stubs: "no_w", "no_x", "no_compute"
// Refusals (the Python asserts) throw operations::grouped_matmul::GroupedMatmulAssertion.
Tensor grouped_matmul(
    const Tensor& x,
    const Tensor& w,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<CoreCoord>& max_grid = std::nullopt,
    std::optional<uint32_t> block_w = std::nullopt,
    uint32_t depth = 2,
    uint32_t rt = 1,
    const std::optional<operations::grouped_matmul::BankLayout>& w_layout = std::nullopt,
    uint32_t w_inflight = 1,
    uint32_t x_ahead = 1,
    bool one_group = false,
    const std::vector<std::string>& diag = {});

}  // namespace ttnn
