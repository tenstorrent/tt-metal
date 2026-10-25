// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/common.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"

namespace kda {

// What the unpacker and the math engine were last configured for: the operand each source's data format came from,
// and the block matmul geometry the last init left in place. A kernel that alternates short stages pays a
// reconfiguration and an init per stage; with this it reconfigures a source only when its format changes and re-runs a
// matmul init only after another op's init, or for a different geometry.
//
// Every op init other than matmul_init must be followed by other_init(), and every source reconfiguration must go
// through sources() (or the srca/srcb forms), so the tracked state matches the hardware.
class EngineState {
public:
    // The sources as compute_kernel_hw_startup left them: srcA from src_a, srcB from src_b.
    constexpr EngineState(uint32_t src_a, uint32_t src_b) : src_a_(src_a), src_b_(src_b) {}

    FORCE_INLINE void srca(uint32_t operand) {
        reconfig_data_format_srca(src_a_, operand);
        src_a_ = operand;
    }

    FORCE_INLINE void srcb(uint32_t operand) {
        reconfig_data_format_srcb(src_b_, operand);
        src_b_ = operand;
    }

    FORCE_INLINE void sources(uint32_t src_a, uint32_t src_b) {
        srca(src_a);
        srcb(src_b);
    }

    // Configure a block matmul of in0 (srcB) by in1 (srcA). The tiles of every operand passed through here are full
    // 32x32 tiles, so the init depends only on the geometry.
    FORCE_INLINE void matmul_init(uint32_t in0, uint32_t in1, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim) {
        sources(in1, in0);
        const uint32_t geometry = 1u << 31 | ct_dim << 20 | rt_dim << 10 | kt_dim;
        if (geometry != matmul_) {
            matmul_block_init(in0, in1, false, ct_dim, rt_dim, kt_dim);
            matmul_ = geometry;
        }
    }

    // Another op's init ran: the next matmul must re-initialize.
    FORCE_INLINE void other_init() { matmul_ = 0; }

private:
    uint32_t src_a_;
    uint32_t src_b_;
    uint32_t matmul_ = 0;
};

}  // namespace kda
