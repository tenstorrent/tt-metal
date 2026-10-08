// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/operation.hpp"
#include "exchange_histories_program_factory.hpp"

namespace ttnn::experimental::prim {

struct ExchangeHistoriesOperation {
    using operation_attributes_t = ExchangeHistoriesParams;
    using tensor_args_t = ExchangeHistoriesInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<ExchangeHistoriesProgramFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::vector<Tensor> exchange_histories(
    const Tensor& projected,
    const tt::tt_metal::MemoryConfig&,
    const Tensor& actual_start,
    const std::optional<Tensor>& actual_end,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    uint32_t width,
    tt::tt_fabric::Topology topology);

}  // namespace ttnn::experimental::prim
