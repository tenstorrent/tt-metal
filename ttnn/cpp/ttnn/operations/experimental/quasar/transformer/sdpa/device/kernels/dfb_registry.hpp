// SPDX-FileCopyrightText: © 2026 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <new>

#include "api/dataflow/dataflow_buffer.h"

// Kernel-scope DataflowBuffer registry for the SDPA kernels.
//
// On Quasar, ~DataflowBuffer() drains the buffer (spins until every pushed entry has been consumed), and only
// the object a DFB was originally constructed from drains: copies of it never do. The SDPA helpers in
// compute_common.hpp / dataflow_common.hpp build their DataflowBuffer objects per call from a DFB id, so each
// helper call used to end in a drain that waits for the peer core thread -- which deadlocks as soon as the
// peer needs a later push from the same kernel (reader draining K before it has pushed Q, writer draining the
// identity-scale tile that compute keeps for the whole kernel, every per-block compute helper, ...).
//
// Each kernel obtains its originals through original(id) and helpers take view(id), a copy of the original, so
// the only drains are the explicit finish_all() at kernel exit. An unregistered id falls back to a fresh
// (draining) object, i.e. the legacy behaviour.
//
// The originals live in static storage (kernel .bss, in L1), not on the stack: a TRISC has 4 KB of local
// memory for TLS + stack, and ~20 kernel-scope objects in kernel_main overflowed it into the firmware's TLS
// (g_dfb_config_base_addr got clobbered; seen on craq-sim). Plain statics, deliberately not thread_local.
namespace sdpa_dfb {

inline constexpr uint32_t kMaxDfbs = 32;

struct Slot {
    alignas(DataflowBuffer) unsigned char bytes[sizeof(DataflowBuffer)];
};
static Slot g_storage[kMaxDfbs];
static DataflowBuffer* g_originals[kMaxDfbs] = {};

// The kernel-scope original for `id`; constructed on first use.
inline DataflowBuffer& original(uint32_t id) {
    if (g_originals[id] == nullptr) {
        g_originals[id] = new (g_storage[id].bytes) DataflowBuffer(static_cast<uint16_t>(id));
    }
    return *g_originals[id];
}

// Non-draining handle for helpers: a copy of the original (legacy draining object if there is none).
inline DataflowBuffer view(uint32_t id) {
    if (id < kMaxDfbs && g_originals[id] != nullptr) {
        return *g_originals[id];
    }
    return DataflowBuffer(static_cast<uint16_t>(id));
}

// Destroy (and thereby drain) every original. Call once at the end of kernel_main, after every entry the kernel
// kept fronted has been popped.
inline void finish_all() {
    for (uint32_t id = 0; id < kMaxDfbs; ++id) {
        if (g_originals[id] != nullptr) {
            g_originals[id]->~DataflowBuffer();
            g_originals[id] = nullptr;
        }
    }
}

}  // namespace sdpa_dfb
