// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

// Argument layout shared by the deepseek_v4_prefill_write_rows program factory and its kernel.
namespace deepseek_v4_prefill_write_rows {

// Rows one core moves per NoC batch.
constexpr uint32_t BATCH_ROWS = 16;

namespace ct_arg {
enum : uint32_t {
    ROW_BYTES = 0,
    NUM_ROWS,  // rows of the input (and entries of the row index)
    CB_ROWS,   // L1 scratch for the row index
    CB_DATA,   // L1 scratch for BATCH_ROWS rows
    END,       // the input, output and row-index accessor args follow
};
}  // namespace ct_arg

namespace common_arg {
enum : uint32_t {
    INPUT_ADDRESS = 0,
    OUTPUT_ADDRESS,
    ROWS_ADDRESS,
    OUTPUT_ROWS,  // rows of the output: an index at or past it is skipped
    END,
};
}  // namespace common_arg

namespace core_arg {
enum : uint32_t {
    FIRST_ROW = 0,
    NUM_ROWS,
    END,
};
}  // namespace core_arg

}  // namespace deepseek_v4_prefill_write_rows
