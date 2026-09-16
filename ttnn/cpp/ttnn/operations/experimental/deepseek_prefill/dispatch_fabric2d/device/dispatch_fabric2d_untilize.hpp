// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <memory>
#include <optional>

#include <tt-metalium/program_descriptors.hpp>

#include "dispatch_fabric2d_placement.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

// A device buffer the op allocates for itself, never initialises and never reads back on the host.
// The workload holds the owner so it survives a program-cache hit, which is what lets the kernels
// address it by a runtime argument the framework rewrites per dispatch.
struct OwnedScratch {
    std::shared_ptr<ttnn::Tensor> owner;
    tt::tt_metal::Buffer* buffer = nullptr;
};

// Where a TILE input's tokens end up, one row-major page each, so the stream cores address a token by
// page index exactly as they do a row-major input.
//
// Typed UINT32 rather than BFLOAT16 so the page is the token page EXACTLY rather than that rounded up
// to an alignment -- the untilizer writes rows at a token-page stride and the reader reads them at
// one, and a page that disagreed would shear every token after the first of a stripe.
OwnedScratch allocate_staging_buffer(ttnn::MeshDevice* mesh, uint32_t seq_len_per_chip, uint32_t token_bytes);

// Everything every core of the untilizer pool is told, which is everything except which stripes it
// takes. Built only for a TILE input; `plan_untilize` returns nothing for a row-major one, and that
// absence is what the rest of the program factory branches on.
struct UntilizePlan {
    uint32_t num_stripes = 0;
    uint32_t tiles_per_row = 0;
    uint32_t block_ct_dim = 0;
    uint32_t tile_bytes = 0;
    uint32_t token_bytes = 0;
    uint32_t sem_addr = 0;
    tt::DataFormat data_format = tt::DataFormat::Float16_b;
    tt::tt_metal::Buffer* input = nullptr;
    tt::tt_metal::Buffer* staging = nullptr;
};

std::optional<UntilizePlan> plan_untilize(
    const ttnn::Tensor& input,
    const ttnn::Tensor& out_payload,
    uint32_t seq_len_per_chip,
    uint32_t token_bytes,
    uint32_t sem_addr,
    tt::tt_metal::Buffer* staging);

// The pool that turns a TILE input into staging: every core of the op's universe no stream took, each
// running a reader / pack_untilize / writer trio over its round-robin share of the stripes.
//
// These cores run nothing else, which is what lets the untilize circular buffers take most of their
// L1 -- and it is why the pool is drawn from the universe rather than from the grid: on the model's
// split, everything outside it belongs to the shared expert running at the same time.
void add_untilizer_pool(
    tt::tt_metal::ProgramDescriptor& desc,
    const StreamPlacements& streams,
    const CoreRangeSet& universe,
    const UntilizePlan& plan);

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
