// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "hang_host_operation.hpp"
#include <tt-metalium/hal.hpp>

namespace ttnn::operations::experimental::test {

ttnn::Tensor ExecuteTestHangHostOperation::invoke(const ttnn::Tensor& input_tensor) {
#if TTNN_OPERATION_TIMEOUT_SECONDS > 0
    while (true);  // Ugly yet functional way to hang the operation
#else
    return input_tensor;
#endif
}
}  // namespace ttnn::operations::experimental::test
