// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/experimental/lazy/lazy_tensor.hpp"

namespace ttnn::experimental::lazy {

struct LazyOperation {
    LazyOperation() = default;
    virtual std::vector<tt::tt_metal::metal_tensor::Tensor> invoke() = 0;
    virtual std::string_view name() const = 0;
    virtual tt::stl::hash::hash_t operation_type_id() const = 0;
    // TODO: Do we need some attributes for serialization purposes?
    virtual ~LazyOperation() = default;
};

}  // namespace ttnn::experimental::lazy
