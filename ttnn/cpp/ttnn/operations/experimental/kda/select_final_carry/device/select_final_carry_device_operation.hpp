// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/operation.hpp"
#include "select_final_carry_program_factory.hpp"

namespace ttnn::experimental::prim {

struct SelectFinalCarryOperation {
    using operation_attributes_t = SelectFinalCarryParams;
    using tensor_args_t = SelectFinalCarryInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<SelectFinalCarryProgramFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor select_final_carry(
    const Tensor& rank_final,
    const Tensor& prefix_final,
    const tt::tt_metal::MemoryConfig&,
    const Tensor& actual_start,
    const std::optional<Tensor>& actual_end,
    uint32_t sequence_parallel_axis,
    uint32_t local_rows,
    uint32_t num_links,
    tt::tt_fabric::Topology topology);

}  // namespace ttnn::experimental::prim
