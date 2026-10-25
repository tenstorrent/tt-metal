// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace kda_select_final_carry {

// How this rank produces the final carry; every rank derives the same chronology.
enum Mode : uint32_t {
    copy_prefix = 0,  // Unsplit interval: the replicated prefix carry is already the answer.
    broadcast = 1,    // Split interval, and this rank owns the tail: send its final state to the line.
    receive = 2,      // Split interval, owned elsewhere: wait for the owner's state.
};

}  // namespace kda_select_final_carry
