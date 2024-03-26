// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <boost/stacktrace.hpp>
#include <optional>

namespace tt::stl::stacktrace {

inline void print(std::optional<std::size_t> max_size = std::nullopt, std::ostream& os = std::cout) {
    const auto stacktrace = boost::stacktrace::stacktrace();
    std::size_t size = max_size.value_or(stacktrace.size());
    if (size > stacktrace.size()) {
        size = stacktrace.size();
    }
    for (std::size_t i = 1; i < size; ++i) {
        os << i << ": " << stacktrace[i] << '\n';
    }
}

}  // namespace tt::stl::stacktrace
