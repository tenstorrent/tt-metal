// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0
#pragma once

#include <cstdint>

// Common runtime arguments shared by the factory and kernels. The only per-core argument is the
// row-major core identity; the reader's common arguments are followed by the physical X/Y multicast axes.
namespace indexer_common {
namespace reader {
enum : uint32_t { Q, K, W, BatchOffset, KvLength, Count };
}
namespace writer {
enum : uint32_t { Output, KvLength, ChunkStart, StraddleQ, StraddleJump, Count };
}
namespace compute {
enum : uint32_t { KvLength, ChunkStart, StraddleQ, StraddleJump, Count };
}
}  // namespace indexer_common
