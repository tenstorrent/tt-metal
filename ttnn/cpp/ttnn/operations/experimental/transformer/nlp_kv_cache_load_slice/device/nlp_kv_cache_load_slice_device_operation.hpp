// SPDX-FileCopyrightText: © 2025 Tenstorrent AI ULC
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/decorators.hpp"
#include "nlp_kv_cache_load_slice_device_operation_types.hpp"
#include "nlp_kv_cache_load_slice_program_factory.hpp"

namespace ttnn::operations::experimental::transformer::nlp_kv_cache_load_slice {

struct NlpKVCacheLoadSliceDeviceOperation {
    using operation_attributes_t = NlpKvCacheLoadSliceParams;
    using tensor_args_t = NlpKvCacheLoadSliceInputs;
    using spec_return_value_t = nlp_kv_cache_load_slice::spec_return_value_t;
    using tensor_return_value_t = nlp_kv_cache_load_slice::tensor_return_value_t;
    using program_factory_t = std::variant<program::NlpKVCacheLoadSliceProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);

    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);

    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);

    static tensor_return_value_t create_output_tensors(
        const operation_attributes_t& operation_attributes, const tensor_args_t&);
};

}  // namespace ttnn::operations::experimental::transformer::nlp_kv_cache_load_slice

namespace ttnn::prim {
ttnn::operations::experimental::transformer::nlp_kv_cache_load_slice::tensor_return_value_t nlp_kv_cache_load_slice(
    const Tensor& input_tensor,
    uint32_t seq_len_start,
    uint32_t seq_len_end,
    const std::optional<MemoryConfig>& memory_config,
    const std::optional<Tensor>& preallocated_output);
}  // namespace ttnn::prim
