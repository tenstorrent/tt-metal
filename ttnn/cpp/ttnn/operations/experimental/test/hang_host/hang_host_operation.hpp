// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

namespace ttnn {
namespace operations::experimental::test {

struct ExecuteTestHangHostOperation {
    static ttnn::Tensor invoke(const ttnn::Tensor& input_tensor);
};

}  // namespace operations::experimental::test

namespace experimental::test {
constexpr auto hang_host_operation = ttnn::register_operation<
    "ttnn::experimental::test::test_hang_host_operation",
    ttnn::operations::experimental::test::ExecuteTestHangHostOperation>();
}  // namespace experimental::test
}  // namespace ttnn
