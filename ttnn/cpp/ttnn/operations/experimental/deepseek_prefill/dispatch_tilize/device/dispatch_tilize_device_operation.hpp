// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>

#include "dispatch_tilize_types.hpp"
#include "dispatch_tilize_program_factory.hpp"

#include "ttnn/device_operation.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::dispatch_tilize {

struct DispatchTilizeDeviceOperation {
    using operation_attributes_t = DispatchTilizeParams;
    using tensor_args_t = DispatchTilizeInputs;
    using spec_return_value_t = ttnn::TensorSpec;
    using tensor_return_value_t = ttnn::Tensor;
    using program_factory_t = std::variant<DispatchTilizeProgramFactory>;

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::dispatch_tilize

namespace ttnn::prim {

ttnn::Tensor dispatch_tilize(
    const ttnn::Tensor& input_tensor,
    const std::optional<ttnn::Tensor>& total_counts_per_expert,
    tt::tt_metal::DataType output_dtype,
    uint32_t experts_per_chip,
    const tt::tt_metal::MemoryConfig& output_memory_config);

}  // namespace ttnn::prim
