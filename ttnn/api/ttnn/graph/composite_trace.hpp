// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <utility>
#include <string_view>
#include <tt-metalium/graph_tracking.hpp>
#include "graph_serialization.hpp"

namespace ttnn::graph {

class [[nodiscard]] FunctionScope {
public:
    template <class... Args>
    explicit FunctionScope(std::string_view name, Args&&... args) {
        tt::tt_metal::GraphTracker::instance().track_function_start(name, std::forward<Args>(args)...);
    }
    ~FunctionScope() { tt::tt_metal::GraphTracker::instance().track_function_end(); }
    FunctionScope(const FunctionScope&) = delete;
    FunctionScope& operator=(const FunctionScope&) = delete;
    FunctionScope(FunctionScope&&) = delete;
    FunctionScope& operator=(FunctionScope&&) = delete;
};

}  // namespace ttnn::graph

#define TT_OP_SCOPE(name, ...) ttnn::graph::FunctionScope _tt_op_scope(name, ##__VA_ARGS__)
