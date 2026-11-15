// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "step_context.hpp"

#include "device/step_context_device_operation.hpp"

namespace ttnn::experimental::deepseek {

std::vector<Tensor> decode_step_context(
    const Tensor& packet,
    const std::vector<Tensor>& rope_tables,
    const std::vector<std::vector<uint32_t>>& int_shapes,
    const std::vector<DataType>& int_dtypes,
    const std::vector<std::vector<int32_t>>& int_terms,
    const std::vector<uint32_t>& rope_table_ids,
    const std::vector<uint32_t>& rope_fields,
    const std::vector<int32_t>& rope_offsets) {
    return ttnn::prim::decode_step_context(
        packet, rope_tables, int_shapes, int_dtypes, int_terms, rope_table_ids, rope_fields, rope_offsets);
}

}  // namespace ttnn::experimental::deepseek
