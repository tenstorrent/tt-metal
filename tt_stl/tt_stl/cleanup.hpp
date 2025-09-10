// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>

namespace ttsl {

// `Cleanup` is a RAII wrapper that calls a callable automatically upon leaving the scope of the object.
//
// Example usage:
//
// FILE* input_file = fopen(file_name.c_str(), "rb");
// TT_FATAL(input_file != nullptr, "Cannot open \"{}\"", file_name);
// auto cleanup = ttsl::make_cleanup([input_file]() { fclose(input_file); });
// ...
//
template <typename Callable>
class Cleanup final {
public:
    constexpr explicit Cleanup(Callable callable) : callable_(std::move(callable)) {}
    ~Cleanup() {
        if (callable_.has_value()) {
            (*callable_)();
        }
    }

    // Support move-only construction only, no re-assignment.
    Cleanup(const Cleanup&) = delete;
    Cleanup& operator=(const Cleanup&) = delete;
    Cleanup(Cleanup&& other) noexcept : callable_(std::move(other.callable_)) { other.callable_.reset(); }
    Cleanup& operator=(Cleanup&& other) noexcept = delete;

    // Cancels the cleanup.
    // Must be called on an rvalue-qualified object, to explicitly signal the cleanup instance should no longer be used;
    // use-after-move clang check can be used to enforce this.
    void cancel() && { callable_.reset(); }

private:
    std::optional<Callable> callable_;
};

template <typename Callable>
[[nodiscard]] auto make_cleanup(Callable&& callable) {
    return Cleanup(std::forward<Callable>(callable));
}

}  // namespace ttsl
