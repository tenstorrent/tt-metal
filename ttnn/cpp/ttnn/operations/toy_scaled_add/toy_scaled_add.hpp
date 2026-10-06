// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn {

// out = a + alpha * (b * gamma), gamma an optional row broadcast down the rows.
//
//   a, b                   tiled (32 x 32), same padded shape, bfloat16 or float32; both interleaved, or
//                          both height-sharded on L1 with one shard spec (shard width = the full row)
//   alpha                  per-call scalar; changing it reuses the cached program
//   gamma                  optional, tiled, interleaved, one tile-row: padded shape [..., 32, W]
//   dtype, memory_config   output dtype / placement (default: a's); a height-sharded config without a
//                          shard spec takes a's
//   compute_kernel_config  math fidelity, fp32 DEST accumulation, ...; default HiFi4, exact math, fp32
//                          DEST whenever an operand is float32
//   output_tensor          preallocated output; may be `a` itself (in place)
//
// Inputs outside the support contract throw UnsupportedAxisValue or ExcludedCell
// (device/toy_scaled_add_device_operation_types.hpp), raised in Python as the ttnn.operations._op_contract
// exceptions of the same name; inputs that do not fit together are TT_FATAL errors.
Tensor toy_scaled_add(
    const Tensor& a,
    const Tensor& b,
    float alpha = 1.0f,
    const std::optional<Tensor>& gamma = std::nullopt,
    const std::optional<DataType>& dtype = std::nullopt,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<DeviceComputeKernelConfig>& compute_kernel_config = std::nullopt,
    const std::optional<Tensor>& output_tensor = std::nullopt);

}  // namespace ttnn
