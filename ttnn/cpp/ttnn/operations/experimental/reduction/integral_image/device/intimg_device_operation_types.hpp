// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <tuple>

namespace ttnn::experimental::prim {

struct IntImgParams {
    static constexpr auto attribute_names = std::make_tuple();
    auto attribute_values() const { return std::make_tuple(); }
};

}  // namespace ttnn::experimental::prim
