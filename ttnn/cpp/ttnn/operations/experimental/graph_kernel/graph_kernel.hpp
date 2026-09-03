// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <string>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::experimental {

// Experimental: runs a "graph kernel" over an arbitrary number of input tensors.
//
// `inputs` must be non-empty; every tensor must be interleaved and live on the same
// device. `text` is an opaque graph description carried as an operation attribute (it
// participates in the program-cache hash, so distinct texts yield distinct programs).
//
// Current basis behaviour: the output is a fresh tensor with the same TensorSpec as
// inputs[0], and the device program copies inputs[0] into it page by page. All inputs
// are bound into the program as tensor parameters so a future kernel can address them.
ttnn::Tensor graph_kernel(const std::vector<Tensor>& inputs, const std::string& text);

}  // namespace ttnn::operations::experimental
