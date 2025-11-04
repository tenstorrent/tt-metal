// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/experimental/lazy/lazy_tensor.hpp"

namespace ttnn::experimental::lazy {

template <typename operation_t>
constexpr tt::stl::hash::hash_t get_operation_type_id() {
    return tt::stl::hash::type_hash<operation_t>;
}

struct LazyOperation {
    LazyOperation() = default;
    virtual std::vector<tt::tt_metal::metal_tensor::Tensor> invoke(
        const std::vector<tt::tt_metal::metal_tensor::Tensor>& input_tensors) = 0;
    virtual std::string_view name() const = 0;
    virtual tt::stl::hash::hash_t operation_type_id() const = 0;
    // TODO: Do we need some attributes for serialization purposes?
    virtual ~LazyOperation() = default;
};

}  // namespace ttnn::experimental::lazy
