// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

struct ExpRingJointWriterPolicy {
    static constexpr uint32_t kKWriterAliasCb = 14;  // tt::CBIndex::c_14
    static constexpr uint32_t kVWriterAliasCb = 15;  // tt::CBIndex::c_15
    static constexpr bool kPassOuterRing = false;
    static constexpr uint32_t kColIdentityCb = 8;   // tt::CBIndex::c_8
    static constexpr uint32_t kReduceScalerCb = 5;  // tt::CBIndex::c_5
    static constexpr bool kGeneratesScaleTile = true;
    static constexpr bool kGeneratesMaskTiles = true;
    static constexpr bool kDrainPhaseAlignmentAfterOutput = false;
};

#include "exp_ring_joint_writer_impl.hpp"
