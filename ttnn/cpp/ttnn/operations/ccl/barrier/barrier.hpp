// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include "ttnn/operations/reduction/generic/generic_reductions.hpp"

#include "ttnn/cpp/ttnn/operations/ccl/ccl_host_types.hpp"

namespace ttnn::operations::ccl {

struct ExecuteBarrier {
    static ttnn::Tensor invoke(
        const ttnn::Tensor& input_tensor);
};

constexpr auto barrier =
    ttnn::register_operation<"ttnn::barrier", ttnn::operations::ccl::ExecuteBarrier>();

}  // namespace ttnn::operations::ccl