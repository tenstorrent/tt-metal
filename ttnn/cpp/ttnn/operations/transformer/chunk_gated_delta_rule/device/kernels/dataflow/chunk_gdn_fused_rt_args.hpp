// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Positions of the fused chunk_gdn program's runtime args that carry the head offset, the receiver's o
// address and its output row. Included by the host (the factory emits them, get_dynamic_runtime_args re-applies them on
// every cache hit) and by the three dataflow kernels that read them, so one definition covers all three places.
#pragma once

#include <cstdint>

namespace fused_rt {
constexpr uint32_t kPrepReaderHeadOffsetArg = 19;  // prep reader: head_offset, after the gates' tile sizes
constexpr uint32_t kReceiverOArg = 11;             // fused receiver: o's address, after the producer map
constexpr uint32_t kReceiverHeadOffsetArg = 12;    // fused receiver: head_offset, after o
constexpr uint32_t kReceiverORowArg = 13;          // fused receiver: the output row of its head (as the scan writer's)
constexpr uint32_t kScanWriterHeadArg = 0;         // scan writer: the output row of this receiver's head
}  // namespace fused_rt
