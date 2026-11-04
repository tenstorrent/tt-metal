// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

struct RingJointWriterPolicy {
    static constexpr bool kResidentRingState = false;
    static constexpr uint32_t kCommonArgCount = 0;
};

#include "ring_joint_writer_impl.hpp"
