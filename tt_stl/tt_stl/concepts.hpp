// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <reflect>

namespace ttsl::concepts {

template <typename... T>
inline constexpr bool always_false_v = false;

template <typename T>
concept Reflectable = (std::is_aggregate_v<std::decay_t<T>> and requires {
    reflect::for_each([]([[maybe_unused]] auto I) {}, std::declval<T>());
});

}  // namespace ttsl::concepts

namespace tt {
namespace [[deprecated("Use ttsl namespace instead")]] stl {
using namespace ::ttsl;
}  // namespace stl
}  // namespace tt
