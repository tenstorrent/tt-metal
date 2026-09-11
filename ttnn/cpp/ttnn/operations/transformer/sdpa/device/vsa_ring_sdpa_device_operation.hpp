// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <variant>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include "ttnn/operations/ccl/ccl_host_types.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/transformer/sdpa/device/vsa_ring_sdpa_device_operation_types.hpp"
#include "ttnn/operations/transformer/sdpa/device/vsa_ring_sdpa_program_factory.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::prim {

// VSA fine stage fused with the SP-ring all-gather of K/V; one program per mesh coordinate. Semantics equal
// vsa_sdpa(q, all_gather(k), all_gather(v), ...) up to bf16 rounding order (VSA_RING_SDPA_SPEC.md).
struct VsaRingSdpaOperation {
    using operation_attributes_t = VsaRingSdpaParams;
    using tensor_args_t = VsaRingSdpaInputs;
    using spec_return_value_t = tt::tt_metal::TensorSpec;
    using tensor_return_value_t = Tensor;
    using program_factory_t = std::variant<VsaRingSdpaMeshWorkloadFactory>;

    static void validate_on_program_cache_miss(const operation_attributes_t&, const tensor_args_t&);
    static void validate_on_program_cache_hit(const operation_attributes_t&, const tensor_args_t&);
    static spec_return_value_t compute_output_specs(const operation_attributes_t&, const tensor_args_t&);
    static tensor_return_value_t create_output_tensors(const operation_attributes_t&, const tensor_args_t&);
    static ttsl::hash::hash_t compute_program_hash(const operation_attributes_t&, const tensor_args_t&);
};

Tensor vsa_ring_sdpa(
    const Tensor& q,
    const Tensor& kv,  // local concatenated K/V shard [1, 2H, T_local, d]
    const Tensor& indices,
    const Tensor& block_counts,
    const Tensor& persistent_output_buffer_kv,  // [1, 2H, T_local*ring_size, d]
    float scale,
    uint32_t block_size,
    ttnn::DeviceComputeKernelConfig compute_kernel_config,
    uint32_t list_len,
    std::vector<uint32_t> exempt_ids,
    std::optional<Tensor> dense_row_mask,
    uint32_t coarse_slots_shift,
    uint32_t coarse_real_per_shard,
    std::vector<uint32_t> dense_row_hint,
    const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    uint32_t num_links,
    uint32_t cluster_axis,
    const MeshDevice& mesh_device,
    ttnn::ccl::Topology topology,
    uint32_t num_workers_per_link,
    std::optional<tt::tt_metal::SubDeviceId> subdevice_id);

}  // namespace ttnn::prim
