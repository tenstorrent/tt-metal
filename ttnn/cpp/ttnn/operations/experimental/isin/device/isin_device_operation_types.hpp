// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"

#include <optional>
#include <tuple>

namespace ttnn::experimental::prim {

struct IsinParams {
    const bool assume_unique;
    const bool invert;
    const uint32_t single_fetch_subchunk_size;
};

struct IsinInputs {
    const Tensor elements_tensor;
    const Tensor test_elements_tensor;
    const std::optional<Tensor> optional_out;

    static constexpr auto attribute_names =
        std::forward_as_tuple("elements_tensor", "test_elements_tensor", "optional_out");
    auto attribute_values() const { return std::forward_as_tuple(elements_tensor, test_elements_tensor, optional_out); }
};

}  // namespace ttnn::experimental::prim
