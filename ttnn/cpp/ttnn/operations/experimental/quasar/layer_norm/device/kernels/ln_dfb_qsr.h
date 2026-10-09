// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "api/dataflow/dataflow_buffer.h"

// Quasar (Gen2) DataflowBuffer handles drain in their destructor: a DM handle spins until the consumer
// acked every entry it pushed, and an UNPACK / PACK handle spins until the tile counter it owns reads
// empty. The layernorm kernels keep several buffers resident for the whole kernel (reduce scaler, eps,
// gamma, beta, column mask ...) and pop them late or never, so a draining handle that goes out of scope
// early deadlocks the kernel, and a draining handle at kernel exit races the other TRISC's late pops.
// LN_RESIDENT_DFB declares a handle whose destructor does not run; the next program re-initializes
// every tile counter anyway. On WH/BH it is a plain DataflowBuffer.
#ifdef ARCH_QUASAR
union LnNoDrainDfb {
    DataflowBuffer dfb;
    explicit LnNoDrainDfb(uint32_t id) : dfb(static_cast<uint16_t>(id)) {}
    ~LnNoDrainDfb() {}
};
#define LN_RESIDENT_DFB(name, id)   \
    LnNoDrainDfb name##_holder(id); \
    DataflowBuffer& name = name##_holder.dfb
#else
#define LN_RESIDENT_DFB(name, id) DataflowBuffer name(id)
#endif
