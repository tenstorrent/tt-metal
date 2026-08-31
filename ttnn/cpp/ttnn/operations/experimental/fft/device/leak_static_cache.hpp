// SPDX-FileCopyrightText: © 2026 Tenstorrent Inc.
// SPDX-License-Identifier: Apache-2.0
//
// Process-exit safety for FFT host-side plan caches that own MeshBuffers or
// device tensors.  Their destructors consult GraphTracker / deallocate device
// memory; if the static map is destroyed after Metal teardown, exit segfaults.
// Heap-allocate and never destroy these maps (OS reclaims at exit).

#pragma once

namespace ttnn::experimental::prim::fft_cache {

template <typename Map>
Map& leak_static_map() {
    static auto* map = new Map();
    return *map;
}

}  // namespace ttnn::experimental::prim::fft_cache
