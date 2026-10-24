// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "indexed_entry_copy_device_operation_types.hpp"
#include "indexed_entry_copy_program_factory.hpp"

namespace ttnn::experimental::prim {

struct IndexedEntryCopyDeviceOperation {
    using operation_attributes_t = IndexedEntryCopyParams;
    using tensor_args_t = IndexedEntryCopyInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<IndexedEntryCopyProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

// Pages of one entry along dim 0: tiles for TILE layout, rows for ROW_MAJOR.
uint32_t indexed_entry_copy_num_pages(const Tensor& tensor);

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor indexed_entry_copy(
    const Tensor& source, const Tensor& destination, const Tensor& index, uint32_t index_scale, uint32_t index_offset);
}  // namespace ttnn::prim
