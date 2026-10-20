// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include "ttnn/tensor/tensor.hpp"
#include "paged_pool_gather_device_operation_types.hpp"
#include "paged_pool_gather_program_factory.hpp"

namespace ttnn::operations::experimental::paged_pool_gather {

struct PagedPoolGatherDeviceOperation {
    using operation_attributes_t = paged_pool_gather::operation_attributes_t;
    using tensor_args_t = paged_pool_gather::tensor_args_t;
    using spec_return_value_t = paged_pool_gather::spec_return_value_t;
    using tensor_return_value_t = paged_pool_gather::tensor_return_value_t;
    using program_factory_t = std::variant<PagedPoolGatherProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::paged_pool_gather

namespace ttnn::experimental {

Tensor paged_pool_gather(
    const Tensor& cache,
    const Tensor& page_table,
    const Tensor& pool_ids,
    uint32_t pool_size = 4,
    const std::optional<tt::tt_metal::MemoryConfig>& memory_config = std::nullopt);

}  // namespace ttnn::experimental
