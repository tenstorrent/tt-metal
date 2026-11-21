// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <vector>

#include "ttnn/types.hpp"

namespace ttnn::experimental::llama31_8b_v3 {

ttnn::Tensor kv_rows_update(
    const ttnn::Tensor& keys,
    const ttnn::Tensor& k,
    const ttnn::Tensor& values,
    const ttnn::Tensor& v,
    const std::vector<ttnn::Tensor>& row_positions,
    const ttnn::Tensor& page_table);

bool kv_rows_update_supported(
    const ttnn::Tensor& keys,
    const ttnn::Tensor& k,
    const ttnn::Tensor& values,
    const ttnn::Tensor& v,
    const std::vector<ttnn::Tensor>& row_positions,
    const ttnn::Tensor& page_table);

}  // namespace ttnn::experimental::llama31_8b_v3
