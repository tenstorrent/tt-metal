// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "unique_device_op_types.hpp"
#include "unique_program_factory.hpp"

#include "ttnn/decorators.hpp"

namespace ttnn::operations::experimental::unique {

using namespace common;
using namespace tt;
using namespace tt::tt_metal;

struct UniqueDeviceOperation {
    using operation_attributes_t = operation_attributes_t;
    using tensor_args_t = tensor_args_t;
    using spec_return_value_t = spec_return_value_t;
    using tensor_return_value_t = tensor_return_value_t;
    using program_factory_t = std::variant<UniqueProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    using invocation_result_t = std::tuple<operation_attributes_t, tensor_args_t>;
    static invocation_result_t invoke(
        const Tensor& input_tensor,
        const Tensor& first_occurrences_tensor,
        const uint32_t& single_fetch_subchunk_size,
        const bool& sorted,
        const bool& return_inverse,
        const bool& return_counts,
        const std::optional<int32_t>& dim,
        const std::optional<MemoryConfig>& memory_config);
};

}  // namespace ttnn::operations::experimental::unique

namespace ttnn::prim {

constexpr auto unique =
    ttnn::register_operation<"ttnn::prim::unique", ttnn::operations::experimental::unique::UniqueDeviceOperation>();

}
