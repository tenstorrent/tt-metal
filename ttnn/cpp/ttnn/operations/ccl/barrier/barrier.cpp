// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "barrier.hpp"

#include "ttnn/cpp/ttnn/operations/ccl/barrier/device/barrier_op.hpp"

namespace ttnn::operations::ccl {

ttnn::Tensor ExecuteBarrier::invoke(
    const ttnn::Tensor& input_tensor) {

    MemoryConfig out_memory_config = memory_config.value_or(input_tensor.memory_config());
    return ttnn::operations::ccl::barrier(input_tensor);
}

}  // namespace ttnn::operations::ccl