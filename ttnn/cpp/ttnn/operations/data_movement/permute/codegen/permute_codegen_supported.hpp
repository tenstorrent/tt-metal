// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tt_stl/span.hpp>

namespace ttnn::operations::data_movement {

// Dispatch predicate deciding whether a call lands on PermuteCodegenDeviceOperation vs. the
// native prim. Must replicate PermuteCodegen.permute's ops/permute/spec.py gating (row-major only,
// and rejecting the fused-WH delegation to TransposeCodegen — see permute.yaml's scope=out cases).
bool supported_by_codegen(const Tensor& input_tensor, const ttsl::SmallVector<uint32_t>& dims);

}  // namespace ttnn::operations::data_movement
