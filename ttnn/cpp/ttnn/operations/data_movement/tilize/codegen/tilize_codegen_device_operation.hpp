// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include <tt-metalium/tile.hpp>

#include "ttnn/operation.hpp"
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/tensor/memory_config/memory_config.hpp"
#include "ttnn/types.hpp"
#include "tilize_codegen_program_factory.hpp"

namespace ttnn::prim {

struct TilizeCodegenTensorArgs {
    Tensor input;
};

struct TilizeCodegenOperationAttributes {
    tt::tt_metal::MemoryConfig output_mem_config;
    tt::tt_metal::DataType output_dtype;
    tt::tt_metal::Tile tile;
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
    tt::tt_metal::Tile tile);

}  // namespace ttnn::prim
