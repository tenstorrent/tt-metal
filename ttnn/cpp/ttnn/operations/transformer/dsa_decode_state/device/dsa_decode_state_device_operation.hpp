// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include "ttnn/tensor/tensor.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"

#include "dsa_decode_state_device_operation_types.hpp"
#include "dsa_decode_state_program_factory.hpp"

namespace ttnn::prim {

// Device operation returning {q_resid, latent_cache, ring_k, ring_g, pool_cache} (the last four aliased).
struct DsaDecodeStateDeviceOperation {
    using operation_attributes_t = DsaDecodeStateParams;
    using tensor_args_t = DsaDecodeStateInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<DsaDecodeStateProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::vector<Tensor> dsa_decode_state(
    const DsaDecodeStateInputs& inputs,
    uint32_t qa_off,
    uint32_t kva_off,
    uint32_t k_off,
    uint32_t g_off,
    uint32_t Lq,
    uint32_t L,
    uint32_t D,
    uint32_t P,
    uint32_t pool_blocks,
    float eps_rms,
    float eps_ln,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config);

}  // namespace ttnn::prim
