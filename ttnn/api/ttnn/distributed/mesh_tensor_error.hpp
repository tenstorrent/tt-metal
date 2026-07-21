// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// Exception types for mesh tensor operations.
// Use these instead of asserting on exception message substrings in tests.
#pragma once
#include <stdexcept>
#include <string>

namespace ttnn::distributed {

// Thrown when tensor shards have mismatched tensor specs.
struct MeshTensorSpecMismatch : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Thrown when tensor shards are not on host storage.
struct MeshTensorShardsNotOnHost : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Thrown when shard count doesn't match mesh size.
struct MeshTensorShardCountMismatch : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Thrown when shards are not allocated on the same mesh buffer.
struct MeshTensorBufferMismatch : std::runtime_error {
    using std::runtime_error::runtime_error;
};

// Thrown when a tensor shard appears at a duplicate coordinate.
struct MeshTensorDuplicateCoordinate : std::runtime_error {
    using std::runtime_error::runtime_error;
};

}  // namespace ttnn::distributed
