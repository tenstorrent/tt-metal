// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/graph_tracking.hpp>
#include <string_view>

namespace ttnn::graph {

class ScopedCompositeTrace {
public:
    template <class... Args>
    explicit ScopedCompositeTrace(std::string_view name, Args&&... args) {
        tt::tt_metal::GraphTracker::instance().track_function_start(name, std::forward<Args>(args)...);
    }
    ~ScopedCompositeTrace() { tt::tt_metal::GraphTracker::instance().track_function_end(); }
    ScopedCompositeTrace(const ScopedCompositeTrace&) = delete;
    ScopedCompositeTrace& operator=(const ScopedCompositeTrace&) = delete;
    ScopedCompositeTrace(ScopedCompositeTrace&&) = delete;
    ScopedCompositeTrace& operator=(ScopedCompositeTrace&&) = delete;
};

}  // namespace ttnn::graph
