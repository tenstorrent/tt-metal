// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "api/core_local_mem.h"
#include "api/dataflow/dataflow_api.h"

// Tensor accessor for a uint16 logical-page -> physical-bundle table. Physical
// cache pages are flattened as [bundle][layer][head], and each page contains
// page_size_rows sequence rows. Supports random logical rows and a sequential
// cursor that caches the current bundle-table entry.
template <typename ReaderType>
struct PagedKVAccessor {
private:
    ReaderType reader_;

public:
    uint32_t bundle_ids_l1_addr;
    uint32_t page_size_rows;
    uint32_t num_layers;
    uint32_t num_heads;
    uint32_t layer_idx;

    PagedKVAccessor(
        const ReaderType& tensor_reader,
        uint32_t table_l1_addr,
        uint32_t rows_per_page,
        uint32_t layers = 1,
        uint32_t heads = 1,
        uint32_t selected_layer = 0) :
        reader_(tensor_reader),
        bundle_ids_l1_addr(table_l1_addr),
        page_size_rows(rows_per_page),
        num_layers(layers),
        num_heads(heads),
        layer_idx(selected_layer) {}

    struct Cursor {
        uint32_t bundle_ids_l1_addr = 0;
        uint32_t page_size_rows = 1;
        uint32_t pages_per_bundle = 1;
        uint32_t layer_head_offset = 0;
        uint32_t logical_bundle = 0;
        uint32_t row_in_bundle = 0;
        uint32_t physical_bundle_id = 0;

        void reset(
            uint32_t table_l1_addr,
            uint32_t logical_row,
            uint32_t rows_per_page,
            uint32_t bundle_page_stride,
            uint32_t page_offset_in_bundle) {
            bundle_ids_l1_addr = table_l1_addr;
            page_size_rows = rows_per_page;
            pages_per_bundle = bundle_page_stride;
            layer_head_offset = page_offset_in_bundle;
            logical_bundle = logical_row / rows_per_page;
            row_in_bundle = logical_row % rows_per_page;
            load_bundle();
        }

        void load_bundle() { physical_bundle_id = CoreLocalMem<volatile uint16_t>(bundle_ids_l1_addr)[logical_bundle]; }

        uint32_t physical_bundle() const { return physical_bundle_id; }

        uint32_t physical_page() const { return physical_bundle() * pages_per_bundle + layer_head_offset; }

        uint32_t physical_row() const { return physical_page() * page_size_rows + row_in_bundle; }

        // has_next_row avoids reading one table entry beyond the final traversal row.
        void advance_row(bool has_next_row) {
            if (++row_in_bundle == page_size_rows) {
                row_in_bundle = 0;
                ++logical_bundle;
                if (has_next_row) {
                    load_bundle();
                }
            }
        }
    };

    uint32_t pages_per_bundle() const { return num_layers * num_heads; }

    uint32_t layer_head_offset(uint32_t head_idx = 0) const { return layer_idx * num_heads + head_idx; }

    uint32_t physical_bundle_for_row(uint32_t logical_row) const {
        return CoreLocalMem<volatile uint16_t>(bundle_ids_l1_addr)[logical_row / page_size_rows];
    }

    uint32_t physical_page(uint32_t logical_row, uint32_t head_idx = 0) const {
        return physical_bundle_for_row(logical_row) * pages_per_bundle() + layer_head_offset(head_idx);
    }

    uint32_t physical_row(uint32_t logical_row, uint32_t head_idx = 0) const {
        return physical_page(logical_row, head_idx) * page_size_rows + logical_row % page_size_rows;
    }

    uint32_t tensor_page_size() const {
        if constexpr (has_get_aligned_page_size_v<ReaderType>) {
            return reader_.get_aligned_page_size();
        } else {
            return reader_.page_size;
        }
    }

    Cursor cursor(uint32_t logical_row, uint32_t head_idx = 0) const {
        Cursor result;
        result.reset(bundle_ids_l1_addr, logical_row, page_size_rows, pages_per_bundle(), layer_head_offset(head_idx));
        return result;
    }

    template <typename NocType, typename LocalEndpoint>
    void async_read_page(NocType& noc, const LocalEndpoint& local_l1, uint32_t page_id, uint32_t bytes) const {
        noc.async_read(reader_, local_l1, bytes, {.page_id = page_id}, {});
    }

    template <typename NocType>
    void async_read_pages(
        NocType& noc,
        uint32_t base_page_id,
        uint32_t row_stride,
        uint32_t num_rows,
        uint32_t cols,
        uint32_t dst_row_origin,
        uint32_t dst_addr,
        uint32_t outer_stride,
        uint32_t inner_stride,
        uint32_t barrier_threshold,
        uint32_t& barrier_count) const {
        const uint32_t page_bytes = tensor_page_size();
        uint32_t page_id = base_page_id;
        for (uint32_t row = 0; row < num_rows; ++row) {
            uint32_t dst = dst_addr + (dst_row_origin + row) * outer_stride;
            for (uint32_t col = 0; col < cols; ++col) {
                async_read_page(noc, CoreLocalMem<uint32_t>(dst), page_id++, page_bytes);
                dst += inner_stride;
                if (barrier_threshold > 0 && ++barrier_count == barrier_threshold) {
                    noc.async_read_barrier();
                    barrier_count = 0;
                }
            }
            page_id += row_stride - cols;
        }
    }

    template <typename NocType, typename LocalEndpoint>
    void async_read_row(
        NocType& noc,
        const LocalEndpoint& local_l1,
        uint32_t logical_row,
        uint32_t head_idx,
        uint32_t bytes,
        uint32_t dst_l1) const {
        noc.async_read(reader_, local_l1, bytes, {.page_id = physical_row(logical_row, head_idx)}, {.addr = dst_l1});
    }

    uint64_t get_shard_row_noc_addr(const Cursor& position, uint32_t byte_offset) const {
        return reader_.get_shard_noc_addr(position.physical_page(), byte_offset);
    }
};
