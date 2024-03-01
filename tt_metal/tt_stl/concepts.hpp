// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace tt::stl::concepts {

template <class>
inline constexpr bool always_false_v = false;

}  // namespace tt::stl::concepts
