// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include <cstddef>

namespace ttsl {


// `ascii_caseless_comp_t` is a Functor that compares whether 2 ascii strings are equal.
//
// Example usage:
//
// ascii_caseless_comp_t{}(std::string_view("Hello"),std::string_view("heLLo"));
//

struct ascii_caseless_comp_t {
  template<typename String>
  constexpr bool operator()(const String& a, const String& b) const
  {
    if (a.size() != b.size())
      return false;
    constexpr auto tolower = [](const char c) { return (c >= 'A' && c <= 'Z') ? c + ('a' - 'A') : c; };
    for (std::size_t i = 0; i < a.size(); ++i)
      if (tolower(a[i]) != tolower(b[i]))
        return false;
    return true;
  }
};

inline constexpr ascii_caseless_comp_t ascii_caseless_comp{};

}