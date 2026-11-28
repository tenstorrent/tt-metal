// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/experimental/llama31_8b_v3/kv_rows_update/kv_rows_update.hpp"

#include "ttnn/operations/experimental/llama31_8b_v3/kv_rows_update/device/kv_rows_update_device_operation.hpp"

namespace ttnn::experimental::llama31_8b_v3 {

ttnn::Tensor kv_rows_update(
    const ttnn::Tensor& keys,
    const ttnn::Tensor& k,
    const ttnn::Tensor& values,
    const ttnn::Tensor& v,
    const std::vector<ttnn::Tensor>& row_positions,
    const ttnn::Tensor& page_table) {
    return ttnn::prim::llama31_8b_v3_kv_rows_update(keys, k, values, v, row_positions, page_table);
}

}  // namespace ttnn::experimental::llama31_8b_v3
