// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0
//
// The pinned host memory window of the routed-expert weight upload (bank_major.py PinnedStager): tt-metal's
// PinnedMemoryCache keeps at most rtoptions' pinned_memory_cache_limit_bytes pinned (read on every pin request,
// tt_metal/distributed/pinned_memory_cache.cpp try_pin; default 4 GiB). The upload raises it for its staging buffers
// and restores it after, releasing (unpinning) the staging buffers' cache entries itself: lowering the limit alone
// unpins nothing (the cache evicts only when a later pin needs room).

#pragma once

#include <cstddef>

namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace ttnn {
class Tensor;
}  // namespace ttnn

namespace ttnn::operations::bringup::flat_routed_expert {

// The pinned-memory cache limit of the mesh device's metal environment (bytes).
size_t pinned_memory_limit(tt::tt_metal::distributed::MeshDevice* mesh);
// Sets it; returns the previous value. Takes effect at the next pin request.
size_t set_pinned_memory_limit(tt::tt_metal::distributed::MeshDevice* mesh, size_t limit_bytes);
// Drops the pinned-memory cache's entries of every shard of a host tensor (each shard's host address): the pins are
// undone now (or, if a transfer still holds one, when it lets go). Returns the bytes of the shards released.
size_t release_pinned(const ttnn::Tensor& host_tensor);
// The number of entries the pinned-memory cache holds (all devices).
size_t pinned_cache_entries();

}  // namespace ttnn::operations::bringup::flat_routed_expert
