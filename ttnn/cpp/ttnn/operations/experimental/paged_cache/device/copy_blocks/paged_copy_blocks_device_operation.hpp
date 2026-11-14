// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include "ttnn/tensor/tensor.hpp"

#include "paged_copy_blocks_device_operation_types.hpp"
#include "paged_copy_blocks_program_factory.hpp"

namespace ttnn::experimental::prim {

// In place: returns the caches.
struct PagedCopyBlocksDeviceOperation {
    using operation_attributes_t = PagedCopyBlocksParams;
    using tensor_args_t = PagedCopyBlocksInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<PagedCopyBlocksProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {

std::vector<Tensor> paged_copy_blocks(const std::vector<Tensor>& caches, const Tensor& src_dst);

}  // namespace ttnn::prim
