// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "isin_device_op_types.hpp"
#include "isin_program_factory.hpp"

#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::isin {

using namespace common;
using namespace tt;
using namespace tt::tt_metal;

struct IsInDeviceOperation {
    using operation_attributes_t = operation_attributes_t;
    using tensor_args_t = tensor_args_t;
    using spec_return_value_t = spec_return_value_t;
    using tensor_return_value_t = tensor_return_value_t;
    using program_factory_t = std::variant<IsInProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static operation::Hash compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    using invocation_result_t = std::tuple<operation_attributes_t, tensor_args_t>;
    static invocation_result_t invoke(
        const Tensor& elements,
        const Tensor& test_elements,
        const uint32_t& single_fetch_chunk_size,
        const bool& assume_unique,
        const bool& invert,
        const std::optional<MemoryConfig>& memory_config,
        const std::optional<Tensor>& optional_out,
        const QueueId& queue_id = DefaultQueueId);
};

}  // namespace ttnn::operations::experimental::isin

namespace ttnn::prim {

constexpr auto isin =
    ttnn::register_operation<"ttnn::prim::isin", ttnn::operations::experimental::isin::IsInDeviceOperation>();

}
