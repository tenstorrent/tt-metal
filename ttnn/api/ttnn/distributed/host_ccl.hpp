// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

namespace ttnn::distributed::host_ccl {

// Performs an all-gather operation on the tensor shards.
Tensor all_gather(const Tensor& tensor);

}  // namespace ttnn::distributed::host_ccl
