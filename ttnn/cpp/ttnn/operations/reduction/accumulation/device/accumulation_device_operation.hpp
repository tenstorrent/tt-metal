// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include "accumulation_program_factory.hpp"

#include <optional>
#include <type_traits>
#include <variant>

#include "hostdevcommon/kernel_structs.h"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"
#include "ttnn/decorators.hpp"

namespace ttnn::operations::reduction::accumulation {

using namespace tt::tt_metal;
using namespace tt::stl;

struct AccumulationDeviceOperation {
    using operation_attributes_t = accumulation::operation_attributes_t;
    using tensor_args_t = accumulation::tensor_args_t;
    using spec_return_value_t = accumulation::spec_return_value_t;
    using tensor_return_value_t = accumulation::tensor_return_value_t;
    using program_factory_t = std::variant<accumulation::AccumulationProgramFactory>;

    using invocation_result_t = std::tuple<operation_attributes_t, tensor_args_t>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);

    static operation::Hash compute_program_hash(const operation_attributes_t&, const tensor_args_t&);

    static invocation_result_t invoke(
        const Tensor& input_tensor,
        const int32_t& dim,
        const std::optional<DataType>& dtype,
        const bool& reverse_order,
        std::optional<Tensor> optional_out,
        const std::optional<MemoryConfig>& memory_config,
        AccumulationOp op);
};

}  // namespace ttnn::operations::reduction::accumulation

namespace ttnn::prim {
constexpr auto accumulation = ttnn::register_operation<
    "ttnn::prim::accumulation",
    ttnn::operations::reduction::accumulation::AccumulationDeviceOperation>();
}  // namespace ttnn::prim
