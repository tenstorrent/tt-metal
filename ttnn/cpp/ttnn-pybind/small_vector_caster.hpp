// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <pybind11/operators.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <tt_stl/small_vector.hpp>

namespace PYBIND11_NAMESPACE {
namespace detail {
template <typename T, size_t PREALLOCATED_SIZE>
struct type_caster<ttnn::SmallVector<T, PREALLOCATED_SIZE>> : list_caster<ttnn::SmallVector<T, PREALLOCATED_SIZE>, T> {
};
}  // namespace detail
}  // namespace PYBIND11_NAMESPACE
