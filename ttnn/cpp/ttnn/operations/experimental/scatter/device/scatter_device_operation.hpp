// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/host_api.hpp>
#include <tt-metalium/constants.hpp>
#include <tt-metalium/util.hpp>
#include <tt-metalium/work_split.hpp>

#include "ttnn/decorators.hpp"

#include "../scatter_enums.hpp"
#include "scatter_device_operation_types.hpp"

#include "scatter_program_factory.hpp"

namespace ttnn::operations::experimental::scatter {

using namespace tt::tt_metal;

struct ScatterDeviceOperation {
    using operation_attributes_t = scatter::operation_attributes_t;
    using tensor_args_t = scatter::tensor_args_t;
    using spec_return_value_t = scatter::spec_return_value_t;
    using tensor_return_value_t = scatter::tensor_return_value_t;
    using program_factory_t = std::variant<scatter::ScatterProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static operation::Hash compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    using invocation_result_t = std::tuple<operation_attributes_t, tensor_args_t>;

    static invocation_result_t invoke(
        const Tensor& input_tensor,
        const int32_t& dim,
        const Tensor& index_tensor,
        const Tensor& source_tensor,
        const MemoryConfig& output_memory_config,
        const std::optional<ScatterReductionType>& opt_reduction,
        const QueueId& queue_id = DefaultQueueId);
};

}  // namespace ttnn::operations::experimental::scatter

namespace ttnn::prim {
constexpr auto scatter =
    ttnn::register_operation<"ttnn::prim::scatter", ttnn::operations::experimental::scatter::ScatterDeviceOperation>();
}
