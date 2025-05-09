// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include "hostdevcommon/kernel_structs.h"

namespace ttnn::operations::conv {
namespace conv2d {

using namespace tt;

// In order to make circular buffer indicies sequential, we use variable to keep track of the next available index.
// Circular buffer indices should be assigned right before their creation.
struct CBIndices {
    // Invalid value for cb id is 32, number greater than the maximum number of index circular buffer can have.
    // Not assigning get_next_cb_index() value before creating cb will throw exception in circular_buffer_config.cpp
    // which can be used as a reminder.
    uint32_t weight_cb = 32;
    uint32_t tilize_mode_tilized_act_cb = 32;
    uint32_t act_cb = 32;
    uint32_t bias_cb = 32;
    uint32_t sharded_act_cb = 32;
    uint32_t cb_for_reader_indices = 32;
    uint32_t cb_for_l1_array = 32;
    uint32_t act_cb_row_major_bfloat16 = 32;
    uint32_t act_cb_second_reader = 32;
    uint32_t matmul_partials_cb = 32;
    uint32_t untilize_mode_reblock_cb = 32;
    uint32_t out0_cb = 32;
    uint32_t temp_sum_cb = 32;

    uint32_t get_next_cb_index();

private:
    uint32_t next_cb_index = tt::CBIndex::c_0;
};
}  // namespace conv2d
}  // namespace ttnn::operations::conv
