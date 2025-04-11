// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <nanobind/stl/detail/nb_list.h>

#include <tt_stl/small_vector.hpp>

NAMESPACE_BEGIN(NB_NAMESPACE)
NAMESPACE_BEGIN(detail)

template <typename T, size_t PREALLOCATED_SIZE>
struct type_caster<ttnn::SmallVector<T, PREALLOCATED_SIZE>>
    : list_caster<ttnn::SmallVector<T, PREALLOCATED_SIZE>, T> {};

NAMESPACE_END(detail)
NAMESPACE_END(NB_NAMESPACE)
