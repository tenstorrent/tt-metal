// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/program_descriptors.hpp>

#include "dispatch_fabric2d_placement.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d {

// Settings shared by every core of the untilizer pool; only the tile rows each core takes differ. A tile
// row is 32 token rows, one tile high. `plan_untilize` returns nothing for a row-major input.
//
// Tiles arrive in the input's format and rows leave in the payload's; the packer converts between them as
// it writes. Both are BFLOAT16 today.
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

// Untilizers per link (two streams per link). Raise it if the stream readers' dspf2d_wait_untilize profiler
// zone is above zero.
constexpr uint32_t UNTILIZERS_PER_LINK = 5;

// Whether the whole pool fit in the core row under the streams.
enum class UntilizerPoolFallback : uint8_t {
    kNone,          // all of it in the core row under the streams
    kRowTooNarrow,  // the core row has fewer spare cores than the pool wants; the rest come from elsewhere
};

// Adds the pool that untilizes a TILE input into staging: spare cores, each running a reader,
// pack_untilize and a writer over its round-robin share of the tile rows.
//
// UNTILIZERS_PER_LINK per link, at most one per tile row, in the core row directly under the streams and
// spread across the streams' columns. The streams' own core row already carries their DRAM traffic
// (fwd_section pages, output pages, staging reads); the core row below is the closest one that does not.
// A sub-device without that core row is refused. If that core row has too few spare cores, the rest of the pool
// comes from other spare cores, and the return value says so, so the caller can warn once per build.
//
// Nothing else runs on these cores, so the untilize circular buffers can take most of their L1.
UntilizerPoolFallback add_untilizer_pool(
    tt::tt_metal::ProgramDescriptor& desc,
    const StreamPlacements& streams,
    const CoreRangeSet& allowed_cores,
    const UntilizePlan& plan);

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_fabric2d
