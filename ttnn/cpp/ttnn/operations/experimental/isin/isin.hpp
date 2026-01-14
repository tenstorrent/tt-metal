// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/decorators.hpp"

#include <optional>

namespace ttnn {
namespace operations::experimental {

using namespace tt;

struct IsInOperation {
    static Tensor invoke(
        const Tensor& elements,
        const Tensor& test_elements,
        bool assume_unique = false,
        bool invert = false,
        const std::optional<Tensor>& opt_out = std::nullopt);
};

}  // namespace operations::experimental

namespace experimental {
constexpr auto isin =
    ttnn::register_operation<"ttnn::experimental::isin", ttnn::operations::experimental::IsInOperation>();
}  // namespace experimental

}  // namespace ttnn
