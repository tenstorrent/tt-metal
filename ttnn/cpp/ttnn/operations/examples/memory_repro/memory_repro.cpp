// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "memory_repro.hpp"

namespace ttnn {

Tensor memory_repro(const Tensor& input_tensor) { return prim::memory_repro(input_tensor); }

}  // namespace ttnn
