// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental {

// Chunked gated delta (GDN) recurrence. Boilerplate only — kernel not implemented.
Tensor chunked_gated_delta(const Tensor& g_exp, const Tensor& factor, const Tensor& bktv, const Tensor& state);

}  // namespace ttnn::experimental
