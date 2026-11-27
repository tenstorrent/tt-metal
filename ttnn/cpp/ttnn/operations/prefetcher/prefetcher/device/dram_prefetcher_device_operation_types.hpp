// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/prefetcher_pipe.hpp"
#include "ttnn/tensor/tensor.hpp"
#include <tt-metalium/experimental/prefetcher_pipe.hpp>
#include <tt-metalium/global_circular_buffer.hpp>

namespace ttnn::prim {

struct DramPrefetcherParams {
    uint32_t num_layers = 0;
    bool enable_performance_mode = false;
    std::optional<const tt::tt_metal::experimental::GlobalCircularBuffer> global_cb;
    // Alternative delivery target to `global_cb`: worker-sender PrefetcherPipes, one per reader core
    // (the pipe's sender). Empty means none; exactly one of the two is set. The program built against
    // them binds each pipe, so they must outlive every cached program that uses them.
    ttnn::PrefetcherPipeList prefetcher_pipes;
};

struct DramPrefetcherInputs {
    std::vector<Tensor> input_tensors;
};

}  // namespace ttnn::prim
