// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>

#include <tt-metalium/core_coord.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"
#include "ttnn/types.hpp"
#include "tilize_codegen_program_factory.hpp"
#include "tilize_codegen_supported.hpp"

namespace ttnn::prim {

struct TilizeCodegenTensorArgs {
    Tensor input;
};

struct TilizeCodegenOperationAttributes {
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    bool use_multicore = true;
    bool use_low_perf = false;
    std::optional<CoreRangeSet> sub_core_grids = std::nullopt;
    ttnn::operations::data_movement::tilize_codegen::TilizeCodegenPlan plan;
};

using TilizeCodegenTensorReturnValue = Tensor;
using TilizeCodegenSpecReturnValue = tt::tt_metal::TensorSpec;

struct TilizeCodegenDeviceOperation {
    using operation_attributes_t = TilizeCodegenOperationAttributes;
    using tensor_args_t = TilizeCodegenTensorArgs;
    using spec_return_value_t = TilizeCodegenSpecReturnValue;
    using tensor_return_value_t = TilizeCodegenTensorReturnValue;
    using program_factory_t = std::variant<TilizeCodegenProgramFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

Tensor tilize_codegen(
    const Tensor& input,
    tt::tt_metal::MemoryConfig output_mem_config,
    tt::tt_metal::DataType output_dtype,
    bool use_multicore,
    bool use_low_perf,
    const std::optional<CoreRangeSet>& sub_core_grids,
    const ttnn::operations::data_movement::tilize_codegen::TilizeCodegenPlan& plan);

}  // namespace ttnn::prim
