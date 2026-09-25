// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/program_descriptors.hpp>

#include "dispatch_fabric2d_placement.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

// Everything every core of the untilizer pool is told, which is everything except which tile rows it
// takes. Built only for a TILE input; `plan_untilize` returns nothing for a row-major one, and that
// absence is what the rest of the program factory branches on.
//
// The two formats are the two tensors: the tiles arrive in the input's and the rows leave in the
// payload's, and the packer converts between them as it writes. They are equal today because the op
// takes a BFLOAT16 input and pages a BFLOAT16 payload, and keeping them apart is what leaves room for
// the fp8 payload the sibling `dispatch` already packs this way.
struct UntilizePlan {
    uint32_t num_tile_rows = 0;
    uint32_t tiles_per_row = 0;
    uint32_t block_ct_dim = 0;
    uint32_t tile_bytes = 0;
    uint32_t token_bytes = 0;
    uint32_t sem_addr = 0;
    tt::DataFormat tile_format = tt::DataFormat::Float16_b;
    tt::DataFormat row_format = tt::DataFormat::Float16_b;
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

// Untilizers per link, a link being its two streams. Five per link covers the production shape (20
// tile rows at seq 640). The stream readers' `dspf2d_wait_untilize` zone rising off zero is the signal
// this is too low, and it is the only one.
constexpr uint32_t UNTILIZERS_PER_LINK = 5;

// Where the pool ended up relative to where it is designed to be.
enum class UntilizerPoolFallback : uint8_t {
    kNone,          // all of it in the row under the streams
    kRowTooNarrow,  // the row has fewer spare cores than the pool wants; the rest come from elsewhere
};

// The pool that turns a TILE input into staging: a bounded subset of the allowed cores' spare cores, each
// running a reader / pack_untilize / writer trio over its round-robin share of the tile rows.
//
// UNTILIZERS_PER_LINK per link, capped at one per tile row, in the row directly under the streams and
// spread across the streams' columns. The streams sit in the row under the eth cores; an untilizer on
// that same row puts its DRAM reads and staging writes on the NoC row the streams' own DRAM traffic
// (forwarding pages, output pages, staging reads) already fills, and the row below is the closest one
// that does not. A sub-device with no such row is refused. One where that row has too few spare cores
// tops the pool up from elsewhere, and the return value says so for the caller to report once per build.
//
// These cores run nothing else, which is what lets the untilize circular buffers take most of their
// L1 -- and it is why the pool is drawn from the allowed cores rather than from the grid: on the model's
// split, everything outside it belongs to the shared expert running at the same time.
UntilizerPoolFallback add_untilizer_pool(
    tt::tt_metal::ProgramDescriptor& desc,
    const StreamPlacements& streams,
    const CoreRangeSet& allowed_cores,
    const UntilizePlan& plan);

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
