// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compress_pool_decode.hpp"

#include "device/compress_pool_decode_device_operation.hpp"

namespace ttnn::experimental::deepseek {

std::vector<Tensor> compress_pool_decode(
    const Tensor& kv,
    const Tensor& norm_weight,
    float epsilon,
    const std::optional<Tensor>& gate,
    const std::optional<Tensor>& win_kv,
    const std::optional<Tensor>& win_gate,
    const std::optional<Tensor>& win_slot,
    const std::optional<Tensor>& prev_slot,
    const std::optional<Tensor>& cos,
    const std::optional<Tensor>& sin,
    bool do_pool,
    const std::optional<Tensor>& index_wk,
    const std::optional<Tensor>& index_norm_weight,
    const std::optional<Tensor>& idx_key_cache,
    const std::optional<Tensor>& comp_kv,
    const std::optional<Tensor>& page_table,
    const std::optional<Tensor>& entry,
    const std::optional<Tensor>& page_row,
    const std::optional<Tensor>& dense_kv,
    const std::optional<Tensor>& dense_entry,
    const std::optional<Tensor>& key_row) {
    return ttnn::prim::compress_pool_decode(
        kv,
        norm_weight,
        epsilon,
        gate,
        win_kv,
        win_gate,
        win_slot,
        prev_slot,
        cos,
        sin,
        do_pool,
        index_wk,
        index_norm_weight,
        idx_key_cache,
        comp_kv,
        page_table,
        entry,
        page_row,
        dense_kv,
        dense_entry,
        key_row);
}

}  // namespace ttnn::experimental::deepseek
