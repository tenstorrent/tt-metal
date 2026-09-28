// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

struct ExpRingJointReaderPolicy {
    static constexpr uint32_t kKWriterAliasCb = 14;  // tt::CBIndex::c_14
    static constexpr uint32_t kVWriterAliasCb = 15;  // tt::CBIndex::c_15
    static constexpr bool kPartialQSubblocks = false;
    static constexpr bool kPassOuterRing = false;
    static constexpr bool kCreditKAfterQ = false;
    static constexpr uint32_t kDerivedCb = 13;  // tt::CBIndex::c_13
};

#include "exp_ring_joint_reader_impl.hpp"
