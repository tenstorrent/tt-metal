// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "ttnn/operation.hpp"
#include "select_tile_rows_program_factory.hpp"

namespace ttnn::experimental::prim {

struct SelectTileRowsOperation {
    using operation_attributes_t = SelectTileRowsParams;
    using tensor_args_t = SelectTileRowsInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<SelectTileRowsProgramFactory>;
    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::vector<Tensor> select_tile_rows(
    const Tensor& input,
    const std::optional<Tensor>& indices,
    uint32_t width,
    const tt::tt_metal::MemoryConfig& memory_config,
    std::optional<uint32_t> record = std::nullopt,
    const std::optional<Tensor>& actual_start = std::nullopt,
    const std::optional<Tensor>& actual_end = std::nullopt,
    uint32_t sequence_parallel_axis = 0,
    uint32_t local_rows = 0,
    uint32_t rows_per_output = 0);

}  // namespace ttnn::experimental::prim
