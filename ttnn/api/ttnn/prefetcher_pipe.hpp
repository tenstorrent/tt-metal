// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <memory>
#include <vector>

namespace tt::tt_metal::experimental {
class PrefetcherPipe;
}  // namespace tt::tt_metal::experimental

namespace ttnn {

// The PrefetcherPipes a ttnn caller shares, such as every pipe of one
// create_prefetcher_pipes_for_tensor_prefetcher call. Empty means none.
using PrefetcherPipeList = std::vector<std::shared_ptr<tt::tt_metal::experimental::PrefetcherPipe>>;

// The form the tt-metal PrefetcherPipe calls take: they borrow the pipes they read.
using PrefetcherPipeRefList = std::vector<std::reference_wrapper<const tt::tt_metal::experimental::PrefetcherPipe>>;

// Lends a ttnn caller's shared pipes to a tt-metal PrefetcherPipe call. TT_FATALs on a null pipe.
PrefetcherPipeRefList prefetcher_pipe_refs(const PrefetcherPipeList& prefetcher_pipes);

}  // namespace ttnn
