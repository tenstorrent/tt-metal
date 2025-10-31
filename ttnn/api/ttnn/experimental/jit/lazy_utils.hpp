// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include <boost/pfr.hpp>
#include <optional>
#include <functional>
#include <tuple>
#include <utility>

namespace ttnn::experimental::jit {

template <typename object_t, typename T>
std::vector<T> object_to_vector(const object_t& object) {
    std::vector<T> vector;
    tt::stl::reflection::visit_object_of_type<T>([&](const auto& t) { vector.push_back(t); }, object);
    return vector;
}

}  // namespace ttnn::experimental::jit
