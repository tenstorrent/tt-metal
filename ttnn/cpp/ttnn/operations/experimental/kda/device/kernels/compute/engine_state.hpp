// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/compute/bcast.h"
#include "api/compute/common.h"
#include "api/compute/eltwise_binary.h"
#include "api/compute/matmul.h"
#include "api/compute/reconfig_data_format.h"
#include "api/compute/tile_move_copy.h"

namespace kda {

// What the unpacker and the math engine were last configured for: the operand each source's data format came from,
// and the block matmul geometry the last init left in place. A kernel that alternates short stages pays a
// reconfiguration and an init per stage; with this it reconfigures a source only when its format changes and re-runs a
// matmul init only after another op's init, or for a different geometry.
//
// A kernel that uses matmul_init runs every op init and source reconfiguration through this class, so the tracked
// state matches the hardware.
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

    // Reconfigure srcA even when its format is unchanged. After an SFPU or transpose pass a skipped reconfiguration
    // changed bits in otherwise-unspecified output lanes; kept so every lane stays bit-identical.
    FORCE_INLINE void reset_srca(uint32_t operand) {
        reconfig_data_format_srca(operand);
        src_a_ = operand;
    }

    // Configure a block matmul of in0 (srcB) by in1 (srcA). The tiles of every operand passed through here are full,
    // untransposed 32x32 tiles, so the init depends only on the geometry: a caller that needs in1 transposed must
    // bypass this class.
    FORCE_INLINE void matmul_init(uint32_t in0, uint32_t in1, uint32_t ct_dim, uint32_t rt_dim, uint32_t kt_dim) {
        sources(in1, in0);
        const uint32_t geometry = 1u << 31 | ct_dim << 20 | rt_dim << 10 | kt_dim;
        if (geometry != matmul_) {
            matmul_block_init(in0, in1, false, ct_dim, rt_dim, kt_dim);
            matmul_ = geometry;
        }
    }

    // The other ops' inits: each sets its sources' formats, and the next matmul must re-initialize.
    FORCE_INLINE void copy_init(uint32_t input) {
        srca(input);
        ckernel::copy_init(input);
        matmul_ = 0;
    }

    template <EltwiseBinaryType Op>
    FORCE_INLINE void binary_init(uint32_t a, uint32_t b) {
        sources(a, b);
        if constexpr (Op == EltwiseBinaryType::ELWADD) {
            add_init(a, b);
        } else {
            sub_init(a, b);
        }
        matmul_ = 0;
    }

    FORCE_INLINE void mul_bcast_cols_init(uint32_t a, uint32_t column) {
        sources(a, column);
        ckernel::mul_bcast_cols_init(a, column);
        matmul_ = 0;
    }

    // Op(operand, DST) with DST moved to srcB, which takes srcb_operand's format.
    template <EltwiseBinaryType Op>
    FORCE_INLINE void reuse_dest_init(uint32_t operand, uint32_t srcb_operand) {
        sources(operand, srcb_operand);
        if constexpr (Op == EltwiseBinaryType::ELWADD) {
            add_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(operand);
        } else {
            sub_reuse_dest_init<EltwiseBinaryReuseDestType::DEST_TO_SRCB>(operand);
        }
        matmul_ = 0;
    }

private:
    uint32_t src_a_;
    uint32_t src_b_;
    uint32_t matmul_ = 0;
};

}  // namespace kda
