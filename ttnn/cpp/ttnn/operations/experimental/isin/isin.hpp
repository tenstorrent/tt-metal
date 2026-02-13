// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

namespace ttnn {

using namespace tt;

namespace experimental {

Tensor isin(
    const Tensor& elements,
    const Tensor& test_elements,
    bool assume_unique = false,
    bool invert = false,
    const std::optional<Tensor>& opt_out = std::nullopt);

}  // namespace experimental

}  // namespace ttnn
