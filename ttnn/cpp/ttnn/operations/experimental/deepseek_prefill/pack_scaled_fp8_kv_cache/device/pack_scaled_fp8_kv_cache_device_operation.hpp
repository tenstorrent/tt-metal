// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "pack_scaled_fp8_kv_cache_device_operation_types.hpp"
#include "pack_scaled_fp8_kv_cache_program_factory.hpp"

namespace ttnn::experimental::prim::pack_scaled_fp8_kv_cache {

struct PackScaledFp8KvCacheDeviceOperation {
    using operation_attributes_t = PackScaledFp8KvCacheParams;
    using tensor_args_t = PackScaledFp8KvCacheInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<PackScaledFp8KvCacheProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

}  // namespace ttnn::experimental::prim::pack_scaled_fp8_kv_cache

namespace ttnn::prim {
ttnn::Tensor pack_scaled_fp8_kv_cache(
    const Tensor& latent,
    const Tensor& scales,
    const Tensor& rope,
    const tt::tt_metal::MemoryConfig& output_memory_config);
}
