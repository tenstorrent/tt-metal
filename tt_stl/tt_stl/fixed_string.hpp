// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <string_view>

namespace ttsl {

// A simple compile-time fixed string for use as a non-type template parameter.
// Replaces reflect::fixed_string.
template <std::size_t N>
struct fixed_string {
    char data[N]{};

    static constexpr auto size() { return N - 1; }

    consteval fixed_string() = default;

    consteval fixed_string(const char (&str)[N]) {
        for (std::size_t i = 0; i < N; ++i) {
            data[i] = str[i];
        }
    }

    constexpr operator std::string_view() const { return {data, N - 1}; }

    template <std::size_t M>
    constexpr bool operator==(const fixed_string<M>& other) const {
        if constexpr (N != M) {
            return false;
        } else {
            for (std::size_t i = 0; i < N; ++i) {
                if (data[i] != other.data[i]) {
                    return false;
                }
            }
            return true;
        }
    }
};

// CTAD guide
template <std::size_t N>
fixed_string(const char (&)[N]) -> fixed_string<N>;

}  // namespace ttsl

namespace tt {
namespace [[deprecated("Use ttsl namespace instead")]] stl {
using namespace ::ttsl;
}  // namespace stl
}  // namespace tt
