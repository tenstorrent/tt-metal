// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Attribute / tensor-arg types for ttnn::prim::transpose_rm — swap the
// last two dims of a ROW_MAJOR fp32/bf16 tensor.  Living alongside the
// FFT op-types because the only consumer for now is the two-pass FFT
// composite (commit 3c); easy to promote to a top-level ttnn op if other
// consumers appear.

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct TransposeRmParams {
    // No attributes today — the op is fully described by the input tensor's
    // shape + dtype (which already feed the program hash).  Kept as a
    // struct rather than `std::monostate` so we can add attrs later
    // without breaking the device-op interface.
    uint32_t _reserved = 0;
};

struct TransposeRmTensorArgs {
    Tensor input;
};

}  // namespace ttnn::experimental::prim
