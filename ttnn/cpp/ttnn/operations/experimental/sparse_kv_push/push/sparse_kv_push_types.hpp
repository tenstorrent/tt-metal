// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace ttnn::operations::experimental::sparse_kv_push {

// Which of a requester's needed rows one push sends. The prefix is the rows of the blocks before the one holding the
// plan's chunk start: rows earlier chunks wrote, so it can be pushed before this chunk's KV is written.
enum class SparseKvPushPart : uint32_t {
    kAll,     // every needed row
    kPrefix,  // the prefix
    kRest,    // every needed row after the prefix
};

}  // namespace ttnn::operations::experimental::sparse_kv_push
