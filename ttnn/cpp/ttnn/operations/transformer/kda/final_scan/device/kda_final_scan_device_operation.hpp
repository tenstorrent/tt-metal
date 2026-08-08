// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <variant>

#include "kda_final_scan_device_operation_types.hpp"
#include "kda_final_scan_program_factory.hpp"

namespace ttnn::prim {

struct KdaFinalChunkScanOperation {
    using operation_attributes_t = KdaFinalChunkScanParams;
    using tensor_args_t = KdaFinalChunkScanInputs;
    using spec_return_value_t = std::vector<tt::tt_metal::TensorSpec>;
    using tensor_return_value_t = std::vector<Tensor>;
    using program_factory_t = std::variant<KdaFinalChunkScanProgramFactory>;

    static program_factory_t select_program_factory(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
};

std::vector<Tensor> kda_final_chunk_scan(
    const Tensor& v_beta,
    const Tensor& kd,
    const Tensor& q_decay,
    const Tensor& intra,
    const Tensor& k_dec_t,
    const Tensor& final_decay,
    const Tensor& t_inv,
    const std::optional<Tensor>& initial_state,
    uint32_t chunk_size,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const DeviceComputeKernelConfig& compute_kernel_config,
    bool state_only,
    const std::optional<Tensor>& identity_tile,
    bool summary_pair,
    bool output_bf16);

}  // namespace ttnn::prim
