// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

namespace ttnn::experimental::jit {

namespace detail {
// Runtime flag that can be toggled
inline bool& get_runtime_flag() {
    static bool runtime_enabled = []() {
        const char* env = std::getenv("TTNN_LAZY_MODE");
        return env != nullptr && std::string(env) == "1";
    }();
    return runtime_enabled;
}
}  // namespace detail

// Check if lazy evaluation is enabled
inline bool is_lazy_enabled() { return detail::get_runtime_flag(); }

// Enable lazy evaluation
inline void enable() { detail::get_runtime_flag() = true; }

// Disable lazy evaluation
inline void disable() { detail::get_runtime_flag() = false; }

// RAII helper to temporarily disable lazy mode
class ScopedDisable {
public:
    ScopedDisable() : was_enabled_(is_lazy_enabled()) {
        if (was_enabled_) {
            disable();
        }
    }
    ~ScopedDisable() {
        if (was_enabled_) {
            enable();
        }
    }

private:
    bool was_enabled_;
};

}  // namespace ttnn::experimental::jit
