// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <memory>
#include <utility>

#include <nanobind/stl/unique_ptr.h>

namespace nanobind_helpers {

template <typename T>
using unique_ptr = std::unique_ptr<T, nanobind::deleter<T>>;

template <class T, class... Args>
auto make_unique(Args&&... args) {
    return unique_ptr<T>(new T(std::forward<Args>(args)...));
}

}  // namespace nanobind_helpers

namespace nbh = nanobind_helpers;
