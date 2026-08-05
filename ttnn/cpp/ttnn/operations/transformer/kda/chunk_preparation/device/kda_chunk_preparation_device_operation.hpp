// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "kda_chunk_preparation_device_operation_types.hpp"
#include "kda_chunk_preparation_program_factory.hpp"

namespace ttnn::prim {

struct KdaChunkPreparationOperation {
    using operation_attributes_t = KdaChunkPreparationParams;
    using tensor_args_t = KdaChunkPreparationInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<KdaChunkPreparationProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::vector<Tensor> kda_chunk_preparation(
    const Tensor& q,
    const Tensor& k,
    const Tensor& v,
    const Tensor& g,
    const Tensor& beta,
    const Tensor& eye,
    const Tensor& tril,
    const Tensor& ones,
    const Tensor& masks,
    uint32_t chunk_size,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool v_flat,
    uint32_t value_heads,
    bool normalize_qk,
    float scale,
    bool qk_flat,
    uint32_t key_heads,
    bool gate_flat,
    uint32_t output_bf16_mask);

}  // namespace ttnn::prim
