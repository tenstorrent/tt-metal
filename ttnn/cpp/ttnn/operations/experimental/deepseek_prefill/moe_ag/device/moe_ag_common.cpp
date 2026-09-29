// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "moe_ag_common.hpp"

#include <tt_stl/assert.hpp>

namespace ttnn::operations::experimental::deepseek_prefill::moe_ag {

using namespace tt::tt_metal;

void check_dram_interleaved(const Tensor& t, const char* op, const char* name) {
    TT_FATAL(t.storage_type() == StorageType::DEVICE, "{}: {} must be on device", op, name);
    TT_FATAL(t.buffer() != nullptr, "{}: {} must be allocated", op, name);
    TT_FATAL(
        t.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
            t.memory_config().buffer_type() == BufferType::DRAM,
        "{}: {} must be DRAM interleaved",
        op,
        name);
}

void check_row_major(const Tensor& t, DataType dtype, const char* op, const char* name) {
    check_dram_interleaved(t, op, name);
    TT_FATAL(t.layout() == Layout::ROW_MAJOR, "{}: {} must be ROW_MAJOR", op, name);
    TT_FATAL(t.dtype() == dtype, "{}: {} must be {}, got {}", op, name, dtype, t.dtype());
}

void check_single_row(const Tensor& t, const char* op, const char* name) {
    TT_FATAL(
        rm_rows(t) == 1, "{}: {} must be a single row [.., 1, X] (one DRAM page), got {}", op, name, t.logical_shape());
}

uint32_t rm_rows(const Tensor& t) {
    const auto& s = t.logical_shape();
    uint32_t rows = 1;
    for (int i = 0; i + 1 < static_cast<int>(s.rank()); ++i) {
        rows *= s[i];
    }
    return rows;
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::moe_ag
