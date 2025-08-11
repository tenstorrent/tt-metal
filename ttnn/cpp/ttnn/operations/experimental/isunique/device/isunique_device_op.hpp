// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "../isunique_common.hpp"
#include "isunique_device_op_types.hpp"
#include "isunique_program_factory.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::isunique {

using namespace common;
using namespace tt;
using namespace tt::tt_metal;

struct IsUniqueDeviceOperation {
    using operation_attributes_t = operation_attributes_t;
    using tensor_args_t = tensor_args_t;
    using spec_return_value_t = spec_return_value_t;
    using tensor_return_value_t = tensor_return_value_t;
    using program_factory_t = std::variant<IsUniqueProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static operation::Hash compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    using invocation_result_t = std::tuple<operation_attributes_t, tensor_args_t>;
    static invocation_result_t invoke(
        const Tensor& input_tensor,
        const Tensor& index_hint_tensor,
        const bool& invert,
        const std::optional<int32_t>& dim,
        const OptimalHeuristic& optimal_heuristic,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<Tensor>& first_occurrences_tensor,
        const std::optional<Tensor>& optional_out,
        const QueueId& queue_id = DefaultQueueId);
};

}  // namespace ttnn::operations::experimental::isunique

namespace ttnn::prim {

constexpr auto isunique = ttnn::
    register_operation<"ttnn::prim::isunique", ttnn::operations::experimental::isunique::IsUniqueDeviceOperation>();

}
