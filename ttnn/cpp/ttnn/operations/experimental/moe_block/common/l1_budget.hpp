// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The one rule every MoE block op uses to size the L1 it turns into program CBs (an arena, an engine's pools): the
// span from the allocator's L1 base up to the lowest occupied L1 address on any compute core (the persistent words /
// tensors sit at the top of the allocator, so the lowest of them bounds the CBs), less a margin for L1 the block
// allocates after the op is planned.

#pragma once

#include <cstdint>
#include <limits>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::operations::experimental::moe_block {

// L1 the block takes after an op is planned (the fused tail's kKeptLater).
inline constexpr uint32_t kL1KeptLater = 44 * 1024;

// Bytes of L1 left for program CBs on every compute core: lowest occupied L1 address - L1 base - margin_bytes, rounded
// down to 64 B, at most cap_bytes (0 if the margin does not fit). With nothing allocated, the top is the base plus
// the L1 size per core.
uint32_t free_cb_l1_bytes(
    const tt::tt_metal::distributed::MeshDevice& mesh_device,
    uint32_t margin_bytes = kL1KeptLater,
    uint32_t cap_bytes = std::numeric_limits<uint32_t>::max());

}  // namespace ttnn::operations::experimental::moe_block
