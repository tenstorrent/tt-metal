// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

namespace kda_select_final_carry {

// The reader publishes to the writer, in mode_cb's page: the mode, then the first output page and whether it writes.
constexpr uint32_t mode_word = 0;
constexpr uint32_t output_base_word = 1;
constexpr uint32_t write_word = 2;

// How this rank produces the final carry; every rank derives the same chronology.
enum Mode : uint32_t {
    broadcast = 0,  // This rank holds the final state (the split tail's owner, else the last rank): send it.
    receive = 1,    // Another rank holds it: wait for its arrival.
};

}  // namespace kda_select_final_carry
