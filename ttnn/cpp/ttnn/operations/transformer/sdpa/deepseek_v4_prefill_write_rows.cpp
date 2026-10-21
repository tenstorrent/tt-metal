// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/operations/transformer/sdpa/deepseek_v4_prefill_write_rows.hpp"
#include "ttnn/operations/transformer/sdpa/device/deepseek_v4_prefill_write_rows_device_operation.hpp"

namespace ttnn::transformer {

ttnn::Tensor deepseek_v4_prefill_write_rows(
    const ttnn::Tensor& input, const ttnn::Tensor& output, const ttnn::Tensor& rows) {
    return ttnn::prim::deepseek_v4_prefill_write_rows(input, output, rows);
}

}  // namespace ttnn::transformer
