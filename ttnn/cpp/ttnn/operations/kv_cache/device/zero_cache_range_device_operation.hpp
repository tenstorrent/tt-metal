// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/device_operation.hpp"
#include <tt-metalium/program_descriptors.hpp>
#include "zero_cache_range_device_operation_types.hpp"

namespace ttnn::prim {

struct ZeroCacheRangeOperation {
    using operation_attributes_t = ZeroCacheRangeParams;
    using tensor_args_t = ZeroCacheRangeInputs;
    using spec_return_value_t = TensorSpec;
    using tensor_return_value_t = Tensor;

    // Direct descriptor (no program_factory_t / select_program_factory): single program path,
    // built declaratively via create_descriptor. The framework handles caching, CB address
    // patching, and runtime arg copy on cache hits.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes,
        const tensor_args_t& tensor_args,
        tensor_return_value_t& tensor_return_value);

    static void validate_on_program_cache_miss(const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static spec_return_value_t compute_output_specs(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
    static tt::tt_metal::operation::Hash compute_program_hash(
        const operation_attributes_t& args, const tensor_args_t& tensor_args);
};

Tensor zero_cache_range(const Tensor& cache, uint32_t start_page, uint32_t end_page);

}  // namespace ttnn::prim
