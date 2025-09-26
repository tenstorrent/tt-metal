// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <type_traits>

namespace ttsl {

template <typename E>
    requires std::is_enum_v<E>
constexpr auto as_underlying_type(E e) {
    return static_cast<typename std::underlying_type<E>::type>(e);
}

}  // namespace ttsl
