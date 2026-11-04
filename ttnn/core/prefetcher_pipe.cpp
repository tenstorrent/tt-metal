// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/prefetcher_pipe.hpp"

#include <tt_stl/assert.hpp>
#include <tt-metalium/experimental/prefetcher_pipe.hpp>

namespace ttnn {

PrefetcherPipeRefList prefetcher_pipe_refs(const PrefetcherPipeList& prefetcher_pipes) {
    PrefetcherPipeRefList refs;
    refs.reserve(prefetcher_pipes.size());
    for (const auto& pipe : prefetcher_pipes) {
        TT_FATAL(pipe != nullptr, "PrefetcherPipe list holds a null pipe at index {}", refs.size());
        refs.emplace_back(*pipe);
    }
    return refs;
}

}  // namespace ttnn
