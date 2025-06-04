// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "gather_device_operation_types.hpp"
#include "gather_program_factory.hpp"

#include "ttnn/decorators.hpp"

#include <optional>

namespace ttnn::operations::experimental::gather {

struct GatherDeviceOperation {
    using operation_attributes_t = gather::operation_attributes_t;
    using tensor_args_t = gather::tensor_args_t;
    using spec_return_value_t = gather::spec_return_value_t;
    using tensor_return_value_t = gather::tensor_return_value_t;
    using program_factory_t = std::variant<
        gather::program::GatherProgramFactorySingleRowSingleCore,
        gather::program::GatherProgramFactorySingleRowMultiCore>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static std::tuple<operation_attributes_t, tensor_args_t> invoke(
        const Tensor& input_tensor,
        const int8_t dim,
        const Tensor& input_index_tensor,
        const bool sparse_grad,
        const MemoryConfig& output_memory_config,
        const std::optional<Tensor>& output_tensors);
};

}  // namespace ttnn::operations::experimental::gather

namespace ttnn::prim {

constexpr auto gather =
    ttnn::register_operation<"ttnn::prim::gather", ttnn::operations::experimental::gather::GatherDeviceOperation>();

}  // namespace ttnn::prim
