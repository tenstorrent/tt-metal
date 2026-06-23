// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/types.hpp"
#include <tt-metalium/sub_device_types.hpp>

#include "ttnn/operations/experimental/quasar/unary_lut/device/unary_lut_device_operation.hpp"

namespace ttnn::operations::experimental::quasar::unary_lut {

// Host front-end for the unary piecewise-LUT activation (Metal 2.0 / DataflowBuffer
// path). Applies an embedded piecewise-LUT activation to a fully height/block-sharded
// bf16 L1 input through the DFB framework.
//
// When `lut_config` is provided, the op bakes that per-activation LUT (boundaries +
// per-segment polynomial OR rational coefficients) into the compute kernel at JIT time
// — this is the GENERIC DFB eltwise flow (any activation, POLY or RATIONAL, driven by
// the fitter coefficient CSVs). When absent, the kernel's compile-time default (the
// proven deg-2 / 4-seg sigmoid, no range reduction) is used.
Tensor unary_lut(
    const Tensor& input_tensor,
    const std::optional<MemoryConfig>& memory_config = std::nullopt,
    const std::optional<Tensor>& output = std::nullopt,
    const std::optional<tt::tt_metal::SubDeviceId>& sub_device_id = std::nullopt,
    const std::optional<LutConfig>& lut_config = std::nullopt);

}  // namespace ttnn::operations::experimental::quasar::unary_lut
