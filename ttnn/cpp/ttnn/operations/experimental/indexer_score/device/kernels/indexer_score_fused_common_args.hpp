// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// Only fused-ring consumers use these tables. Unique slots remain reserved so
// the per-core schedule, multicast tuples, and band permutation keep their ABI.
namespace indexer_fused_common {
namespace reader {
enum : uint32_t { Q, K, W, KLocal, BatchOffset, KvLength, LocalBatchOffset, Count };
}
namespace writer {
enum : uint32_t { Output, KvLength, ChunkStart, StraddleQ, StraddleJump, Count };
}
namespace compute {
enum : uint32_t { KvLength, ChunkStart, StraddleQ, StraddleJump, Count };
}
}  // namespace indexer_fused_common
