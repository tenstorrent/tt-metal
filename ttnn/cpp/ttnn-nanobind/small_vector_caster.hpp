// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/small_vector.hpp>
#include <nanobind/stl/detail/nb_list.h>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename T, size_t PREALLOCATED_SIZE>
struct type_caster<ttnn::SmallVector<T, PREALLOCATED_SIZE>>
    : list_caster<ttnn::SmallVector<T, PREALLOCATED_SIZE>, T> {};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)

