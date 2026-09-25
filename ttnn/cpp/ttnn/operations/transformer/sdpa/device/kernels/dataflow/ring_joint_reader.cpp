// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

struct RingJointReaderPolicy {
    static constexpr bool kPartialQSubblocks = false;
    static constexpr uint32_t kKVStagingSlots = 2;
};

#include "ring_joint_reader_impl.hpp"
