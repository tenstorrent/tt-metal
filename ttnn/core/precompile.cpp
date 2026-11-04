// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/precompile.hpp"

#include <tt-metalium/experimental/compile_session.hpp>
#include <tt-metalium/graph_tracking.hpp>
#include <tt_stl/assert.hpp>

#include "ttnn/graph/graph_processor.hpp"

namespace ttnn::precompile {

void begin_collect(tt::tt_metal::experimental::CompileSession& session, bool real_alloc) {
    // Inside another capture the hook already exists, and would neither block dispatch nor be restored after.
    TT_FATAL(
        tt::tt_metal::GraphTracker::instance().get_hook() == nullptr,
        "A precompile collect window cannot start inside another graph capture");
    ttnn::graph::GraphProcessor::begin_graph_capture(tt::tt_metal::IGraphProcessor::RunMode::NO_DISPATCH);
    if (real_alloc) {
        if (auto hook = tt::tt_metal::GraphTracker::instance().get_hook()) {
            if (auto* hooks = dynamic_cast<ttnn::graph::ProcessorHooks*>(hook.get())) {
                hooks->set_block_alloc(false);
            }
        }
    }
    session.set_deferral(true);
}

void end_collect(tt::tt_metal::experimental::CompileSession& session) {
    session.set_deferral(false);
    ttnn::graph::GraphProcessor::end_graph_capture();
}

}  // namespace ttnn::precompile
