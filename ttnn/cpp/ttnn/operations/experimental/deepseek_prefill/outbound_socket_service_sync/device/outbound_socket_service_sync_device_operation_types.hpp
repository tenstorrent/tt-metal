// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include <fmt/format.h>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/services/stream_counter.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::experimental::prim {

struct OutboundSocketServiceSyncParams {
    // Uniform across the mesh (the sender backing shares one page-config on every
    // device).
    uint32_t page_size = 0;
    uint32_t num_pages = 0;
    // Page size / count of each extra D2D segment, in segment order (empty for D2H and
    // single-tensor D2D).
    std::vector<uint32_t> extra_page_sizes;
    std::vector<uint32_t> extra_num_pages;
    uint32_t scratch_cb_index = 0;
    uint32_t metadata_size_bytes = 0;  // 0 disables the metadata path
    bool metadata_only = false;        // true skip tensor copy
    tt::tt_metal::CoreRange worker_cores{tt::tt_metal::CoreCoord{0, 0}, tt::tt_metal::CoreCoord{0, 0}};

    // The service's words for every mesh coordinate, row-major. The sender service core (and thus the data-ready
    // counter / metadata L1) may differ per device, so each chip's program looks up its own
    // (create_workload_descriptor). The service builds the table once and shares it, so a launch copies a pointer.
    std::shared_ptr<const tt::tt_metal::StreamSenderCoordTable> coords;
    // Service-core record ring (D2H only; 1/0/0 keeps the single-slot write for D2D senders).
    uint32_t metadata_ring_slots = 1;
    uint32_t metadata_slot_stride = 0;
    uint32_t metadata_data_offset = 0;  // the ring also uses coords->termination_addrs
};

struct OutboundSocketServiceSyncInputs {
    // input: the producing stage's output hidden state (read source). Its address
    //   varies per dispatch -> registered as a BufferBinding.
    // backing: the sender service's persistent backing tensor (write dest). Shares
    //   `input`'s per-shard spec; also the op's (in-place) output.
    // metadata: optional [1,1,1,N] uint32 blob the designated worker forwards to the
    //   sender service core (metadata mode only).
    // Metadata-only (no payload): no input or backing, and the output is the metadata tensor. Each tensor is in one
    // slot, so a cache hit patches its address.
    std::optional<Tensor> input;
    std::optional<Tensor> backing;
    std::optional<Tensor> metadata;
    // Extra D2D segments: extra_inputs[i] is copied into extra_backings[i] page-for-page,
    // exactly like input -> backing.
    std::vector<Tensor> extra_inputs;
    std::vector<Tensor> extra_backings;
};

}  // namespace ttnn::experimental::prim

// The op's attributes print through reflection; the table prints as its hash.
template <>
struct fmt::formatter<std::shared_ptr<const tt::tt_metal::StreamSenderCoordTable>> : fmt::formatter<std::string_view> {
    auto format(
        const std::shared_ptr<const tt::tt_metal::StreamSenderCoordTable>& table, fmt::format_context& ctx) const {
        return table == nullptr ? fmt::format_to(ctx.out(), "StreamSenderCoordTable(null)")
                                : fmt::format_to(ctx.out(), "StreamSenderCoordTable(hash={:#x})", table->hash);
    }
};
