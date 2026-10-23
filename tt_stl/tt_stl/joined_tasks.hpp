// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstddef>
#include <future>
#include <vector>

namespace ttsl {

// Submitted tasks may borrow caller-owned buffers. Drain every task before
// returning, including when submission or execution throws.
template <typename Submit>
void run_joined_tasks(size_t count, const Submit& submit) {
    std::vector<std::shared_future<void>> pending(count);
    try {
        for (size_t i = 0; i < count; ++i) {
            pending[i] = submit(i);
        }
        for (auto& future : pending) {
            future.get();
        }
    } catch (...) {
        for (auto& future : pending) {
            if (future.valid()) {
                future.wait();
            }
        }
        throw;
    }
}

}  // namespace ttsl
