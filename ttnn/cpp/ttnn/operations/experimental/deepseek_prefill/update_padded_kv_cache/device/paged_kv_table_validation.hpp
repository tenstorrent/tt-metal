// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "ttnn/tensor/tensor.hpp"

// Host-side checks of a paged KV page table (kernels/paged_kv_table.hpp), shared by the ops that write and read it.
namespace ttnn::operations::experimental::paged_kv {

inline uint32_t page_table_width(const Tensor& table) { return table.logical_shape()[-1]; }

inline uint32_t page_table_rows(const Tensor& table) { return table.logical_volume() / page_table_width(table); }

// The container the kernels read: rows of INT32 entries, interleaved in DRAM on the pool's device.
inline void validate_page_table(const Tensor& table, const Tensor& pool) {
    TT_FATAL(
        table.storage_type() == StorageType::DEVICE && table.device() == pool.device(),
        "page_table must be on the device of the paged cache");
    TT_FATAL(
        table.dtype() == DataType::INT32 && table.layout() == Layout::ROW_MAJOR,
        "page_table must be INT32 ROW_MAJOR (got {} {})",
        table.dtype(),
        table.layout());
    TT_FATAL(
        table.memory_config().memory_layout() == TensorMemoryLayout::INTERLEAVED &&
            table.memory_config().buffer_type() == BufferType::DRAM,
        "page_table must be DRAM interleaved");
    TT_FATAL(
        page_table_width(table) > 0 && table.padded_shape()[-1] == page_table_width(table),
        "page_table rows must be unpadded (width {}, padded {})",
        page_table_width(table),
        table.padded_shape()[-1]);
}

}  // namespace ttnn::operations::experimental::paged_kv
