
// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <nanobind/stl/unique_ptr.h>

namespace nanobind_helpers {

template<typename T>
using unique_ptr = std::unique_ptr<T, nanobind::deleter<T>>;

}  // namespace nanobind

namespace nbh = nanobind_helpers;
