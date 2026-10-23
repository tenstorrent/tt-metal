// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "nlp_kv_cache_load_slice_device_operation_types.hpp"
#include "ttnn/device_operation.hpp"
#include "ttnn/types.hpp"

namespace ttnn::experimental::prim {

struct NlpKVCacheLoadSliceDeviceOperation {
    using operation_attributes_t = NlpKvCacheLoadSliceParams;
    using tensor_args_t = NlpKvCacheLoadSliceInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;

    // The slice window (output_tensor_start/end) is an attribute and therefore part of the program
    // hash, so every per-core start tile id is structural. The only per-dispatch state is the input
    // address (a reader runtime-arg binding) and the output shard buffer backing CB c_0 (a CB
    // binding); those bindings are the whole cache-hit refresh.
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const operation_attributes_t& operation_attributes, const tensor_args_t& tensor_args, Tensor& output);

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim

namespace ttnn::prim {
Tensor nlp_kv_cache_load_slice(
    const Tensor& input_tensor,
    uint32_t seq_len_start,
    uint32_t seq_len_end,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& preallocated_output);
}  // namespace ttnn::prim
