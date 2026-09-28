// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdint>

struct RingJointReaderPolicy {
    static constexpr bool kPartialQSubblocks = false;
};

#include "ring_joint_reader_impl.hpp"
