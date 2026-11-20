// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Positions of the fused chunk_gdn program's runtime args that carry the head offset. Included by the host
// (the factory emits them, get_dynamic_runtime_args re-applies them on every cache hit) and by the three
// dataflow kernels that read them, so one definition covers all three places.
#pragma once

#include <cstdint>

namespace fused_rt {
constexpr uint32_t kPrepReaderHeadOffsetArg = 19;  // prep reader: head_offset, after the gates' tile sizes
constexpr uint32_t kReceiverHeadOffsetArg = 11;    // fused receiver: head_offset, after the producer map
constexpr uint32_t kScanWriterHeadArg = 0;         // scan writer: the output row of this receiver's head
}  // namespace fused_rt
