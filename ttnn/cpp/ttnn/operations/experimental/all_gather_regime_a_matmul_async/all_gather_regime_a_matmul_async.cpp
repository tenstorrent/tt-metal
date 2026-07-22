// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "all_gather_regime_a_matmul_async.hpp"

#include <string>

#include "device/all_gather_regime_a_matmul_async_plan.hpp"
#include "ttnn/operations/experimental/regime_a_matmul/regime_a_matmul.hpp"

using namespace tt::tt_metal;
namespace pl = ttnn::operations::experimental::agmm::plan;

namespace ttnn::experimental {

ttnn::Tensor all_gather_regime_a_matmul_async(
    const ttnn::Tensor& input_tensor,
    const ttnn::Tensor& weight_tensor,
    const std::optional<const ttnn::experimental::prim::RegimeAMatmulConfig>& config,
    [[maybe_unused]] std::optional<uint32_t> cluster_axis,
    ttnn::ccl::Topology topology,
    uint32_t num_links,
    uint32_t num_workers_per_link,
    [[maybe_unused]] uint32_t num_buffers_per_channel,
    [[maybe_unused]] const std::vector<GlobalSemaphore>& multi_device_global_semaphore,
    [[maybe_unused]] const std::optional<GlobalSemaphore>& barrier_semaphore,
    [[maybe_unused]] const std::optional<ttnn::Tensor>& persistent_output_buffer,
    const std::optional<MemoryConfig>& memory_config,
    std::optional<const DataType> dtype,
    std::optional<DeviceComputeKernelConfig> compute_kernel_config) {
    const auto& a = input_tensor.logical_shape();
    const auto& w = weight_tensor.logical_shape();
    TT_FATAL(a.rank() >= 2 && w.rank() >= 2, "all_gather_regime_a_matmul_async expects rank >= 2 tensors");

    // v1 constraints: bf16, no transpose/batching, no epilogues (the epilogue args are not exposed yet).
    TT_FATAL(
        input_tensor.dtype() == DataType::BFLOAT16 && weight_tensor.dtype() == DataType::BFLOAT16,
        "all_gather_regime_a_matmul_async v1 supports only BFLOAT16 inputs");
    for (int i = 0; i < static_cast<int>(a.rank()) - 2; ++i) {
        TT_FATAL(a[i] == 1, "all_gather_regime_a_matmul_async input must be 1 in all dims < -2 (no batching)");
    }

    // D is inferred from the K-shard ratio: in0 owns K_local, in1 carries the full K_global.
    const uint32_t K_local = a[-1];
    const uint32_t K_global = w[-2];
    const uint32_t M = a[-2];
    const uint32_t N = w[-1];
    TT_FATAL(K_local > 0 && K_global > 0, "all_gather_regime_a_matmul_async K dims must be positive");
    TT_FATAL(
        K_global % K_local == 0,
        "all_gather_regime_a_matmul_async: in1 K ({}) must be a multiple of in0 K ({})",
        K_global,
        K_local);
    const uint32_t D = K_global / K_local;

    if (D == 1) {
        // Behaviorally identical to production regime_a_matmul (no fabric).
        return regime_a_matmul(
            input_tensor,
            weight_tensor,
            config,
            /*bias_tensor=*/std::nullopt,
            /*fused_activation=*/std::nullopt,
            /*fused_ternary_scalar=*/std::nullopt,
            /*fused_ternary_input_a=*/std::nullopt,
            /*fused_ternary_input_b=*/std::nullopt,
            memory_config,
            dtype,
            compute_kernel_config);
    }

    // D>1: build and validate the host plan, then defer the streaming path to Task 3.
    pl::AgmmPlanConfig pcfg;
    pcfg.M = M;
    pcfg.K = K_global;
    pcfg.N = N;
    pcfg.D = D;
    pcfg.topology = (topology == ttnn::ccl::Topology::Ring) ? pl::Topology::Ring : pl::Topology::Linear;
    pcfg.num_links = num_links;
    pcfg.num_workers_per_link = num_workers_per_link;
    if (config.has_value()) {
        pcfg.Ns = config->n_slices;
        pcfg.Pk = config->k_slices;
        pcfg.Sm = config->m_slices;
        pcfg.kb = config->k_block_tiles;
        pcfg.nsb = config->n_subblock_tiles;
    }
    const auto p = pl::build_plan(pcfg);
    if (!p.valid) {
        std::string msg = "all_gather_regime_a_matmul_async plan invalid:";
        for (const auto& e : p.errors) {
            msg += "\n  - " + e;
        }
        TT_THROW("{}", msg);
    }
    TT_THROW(
        "all_gather_regime_a_matmul_async: D={} streaming path is implemented in Task 3 "
        "(host plan validated: {} global K-blocks, {} regime_a cores + {} reserved fabric cores).",
        D,
        p.global_k_blocks,
        p.regime_a_cores,
        p.reserved_fabric_cores);
    return input_tensor;  // unreachable
}

}  // namespace ttnn::experimental
