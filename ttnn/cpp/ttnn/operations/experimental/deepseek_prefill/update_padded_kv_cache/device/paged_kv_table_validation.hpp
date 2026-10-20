// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <algorithm>
#include <cstdint>
#include <variant>

#include <tt-metalium/constants.hpp>
#include <tt_stl/fmt.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

// Host-side checks of a paged KV pool and its page table (kernels/paged_kv_table.hpp).
namespace ttnn::operations::experimental::paged_kv {

inline uint32_t page_table_width(const Tensor& table) { return table.logical_shape()[-1]; }

inline uint32_t page_table_rows(const Tensor& table) { return table.logical_volume() / page_table_width(table); }

// A pool of single-head pages [entries, 1, page, D] per chip, the page a positive whole number of 32-row tiles.
inline void validate_page_pool(const Tensor& pool) {
    const auto& shape = pool.padded_shape();
    TT_FATAL(
        shape.rank() == 4 && shape[1] == 1,
        "a paged KV pool holds single-head pages [entries, 1, page, D]; got {}",
        shape);
    TT_FATAL(
        shape[2] > 0 && shape[2] % tt::constants::TILE_HEIGHT == 0,
        "a paged KV pool's page ({} tokens, its seq dim) must be a positive multiple of {}",
        shape[2],
        tt::constants::TILE_HEIGHT);
}

// The container the kernels read: rows of INT32 entries, interleaved in DRAM on the pool's device, replicated (each
// chip reads the request's row from its own copy).
inline void validate_page_table(const Tensor& table, const Tensor& pool) {
    TT_FATAL(
        table.storage_type() == StorageType::DEVICE && table.device() == pool.device(),
        "page_table must be on the device of the paged cache");
    const auto& placements = table.tensor_topology().placements();
    TT_FATAL(
        std::all_of(
            placements.begin(),
            placements.end(),
            [](const auto& placement) {
                return std::holds_alternative<tt::tt_metal::distributed::MeshMapperConfig::Replicate>(placement);
            }),
        "page_table must be replicated across the mesh");
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
