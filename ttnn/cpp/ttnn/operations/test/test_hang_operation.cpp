// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/run_operation.hpp"
#include "ttnn/decorators.hpp"
#include "ttnn/operations/core/core.hpp"
#include "test_hang_operation.hpp"
#include <tt-metalium/hal.hpp>

namespace ttnn::operations::test {

ttnn::Tensor ExecuteTestHangOperation::invoke(const ttnn::Tensor& input_tensor) {
#ifdef TTNN_ENABLE_OPERATION_TIMEOUT
    while (true);  // Ugly yet functional way to hang the operation
#else
    return input_tensor;
#endif
}
}  // namespace ttnn::operations::test
