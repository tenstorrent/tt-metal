// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <functional>

namespace ttnn {

// clang-format off
/**
 * A guard that calls a function when it goes out of scope.
 * Return value: void
 * | Argument     | Description                                                                       | Type                          | Valid Range                        | Required |
 * |--------------|-----------------------------------------------------------------------------------|-------------------------------|------------------------------------|----------|
 * | close_func   | The function to call when the guard goes out of scope.                            | std::function<void()>         |                                    | Yes      |
 */
// clang-format on
template <typename CloseFunction>
class Guard {
    CloseFunction close_func_;

public:
    Guard(CloseFunction&& close_func) : close_func_(std::move(close_func)) {}

    Guard(const Guard&) = delete;
    Guard& operator=(const Guard&) = delete;
    Guard(Guard&& other) = delete;
    Guard& operator=(Guard&&) = delete;

    void release() { close_func_ = {}; }

    ~Guard() {
        if (close_func_) {
            close_func_();
        }
    }
};

using ScopeGuard = Guard<std::function<void()>>;

template <typename CloseFunction>
ScopeGuard make_guard(CloseFunction&& close_func) {
    return ScopeGuard(std::forward<CloseFunction>(close_func));
}

}  // namespace ttnn
