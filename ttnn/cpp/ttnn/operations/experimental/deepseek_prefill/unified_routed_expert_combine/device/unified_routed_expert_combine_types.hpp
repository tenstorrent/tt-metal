// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <optional>
#include <tuple>
#include <vector>

#include "ttnn/global_semaphore.hpp"
#include "ttnn/operations/core/compute_kernel/compute_kernel_config.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/unified_routed_expert_ffn/device/unified_routed_expert_ffn_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine {

using ::ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_ffn::RoutedExpertActivation;

struct UnifiedRoutedExpertCombineParams {
    // The routed expert: unified_routed_expert_moe's wave-planned program, every expert, no active-token band.
    uint32_t m_tiles = 0;
    uint32_t experts_per_chip = 1;
    RoutedExpertActivation activation = RoutedExpertActivation::Silu;
    bool fuse_bias = false;
    std::optional<ttnn::DeviceComputeKernelConfig> compute_kernel_config;
    // Experts above this many tokens run in a first walk, the rest in a second (0: one walk in local order).
    // Either 0 or at least 96.
    uint32_t big_first_threshold = 0;
    // The L1 plan (per_core_M, gate/up K-block width) the routed expert sizes its circular buffers with, picked from
    // the free L1 when the op is called. Hashed, and combine predicts the routed expert's finish order with it.
    uint32_t wave_per_core_M = 0;
    uint32_t wave_in0_block_w = 0;

    // Overlapped with combine_fabric2d in the same program: the routed expert runs on rows 2-9, combine on rows 0-1
    // takes each expert as soon as it is written, and the op returns combine's output. The fields below are
    // combine's and are read only in this mode.
    bool overlap_combine = false;
    // The lowest occupied L1 address when the op is called. There is no L1 arena: combine lays its rings, control
    // tables and the routed expert's L1 slots out below it, by hand, so they must stay under every buffer that holds
    // data across launches -- including persistent state another op allocates after this one's first call. Hashed,
    // so such an allocation rebuilds the program over the smaller region instead of a cached program writing over
    // it.
    uint32_t combine_l1_floor = 0;
    uint32_t combine_axis = 0;
    uint32_t combine_num_links = 2;
    uint32_t num_experts_per_tok = 0;
    uint32_t seq_len_per_chip = 0;

    // The overlap's counters: combine's per-stream receive counts, the routed expert's start signal and combine's
    // per-channel credits (overlap::channel_semaphore_count of them). They must outlive every launch -- an upstream
    // chip may bump one for the next launch before this chip starts it -- so the caller owns them and passes the same
    // ones every call.
    std::optional<GlobalSemaphore> fwd_arrived;
    std::optional<GlobalSemaphore> final_arrived;
    std::optional<GlobalSemaphore> expert_go;
    std::vector<GlobalSemaphore> channel_semaphores;
    // Every address combine bakes into its compile-time arguments. Hashed, so a cached program is only ever reused
    // over the same semaphores.
    uint32_t fwd_arrived_addr = 0;
    uint32_t final_arrived_addr = 0;
    uint32_t expert_go_addr = 0;
    std::vector<uint32_t> channel_semaphore_addrs;

    static constexpr auto attribute_names = std::forward_as_tuple(
        "m_tiles",
        "experts_per_chip",
        "activation",
        "fuse_bias",
        "compute_kernel_config",
        "big_first_threshold",
        "wave_per_core_M",
        "wave_in0_block_w",
        "overlap_combine",
        "combine_l1_floor",
        "combine_axis",
        "combine_num_links",
        "num_experts_per_tok",
        "seq_len_per_chip",
        "fwd_arrived_addr",
        "final_arrived_addr",
        "expert_go_addr",
        "channel_semaphore_addrs");

    auto attribute_values() const {
        return std::forward_as_tuple(
            m_tiles,
            experts_per_chip,
            activation,
            fuse_bias,
            compute_kernel_config,
            big_first_threshold,
            wave_per_core_M,
            wave_in0_block_w,
            overlap_combine,
            combine_l1_floor,
            combine_axis,
            combine_num_links,
            num_experts_per_tok,
            seq_len_per_chip,
            fwd_arrived_addr,
            final_arrived_addr,
            expert_go_addr,
            channel_semaphore_addrs);
    }
};

struct UnifiedRoutedExpertCombineInputs {
    // The routed expert's: the bfloat16 ROW_MAJOR dispatched buffer, the weights, this device's counts and index
    // table slice, and the per-expert region offsets.
    Tensor x;
    std::vector<Tensor> gate_projs;
    std::vector<Tensor> up_projs;
    std::vector<Tensor> down_projs;
    Tensor counts;
    Tensor global_expert_idx_table;
    Tensor expert_region_offsets;
    // Either all three hold one bias per local expert or all three are empty.
    std::vector<Tensor> gate_biases;
    std::vector<Tensor> up_biases;
    std::vector<Tensor> down_biases;
    // The routed expert's bfloat8_b TILE output: what the op returns alone, and combine's dispatched buffer when
    // overlapped.
    Tensor output;

    // Combine's, present exactly when overlap_combine is set. The index table is the full one, replicated on every
    // device.
    std::optional<Tensor> dispatched_metadata;
    std::optional<Tensor> expert_offsets;
    std::optional<Tensor> replicated_global_expert_idx_table;
    // Combine's DRAM forwarding buffer, of overlap::forwarding_buffer_spec. Absent: combine allocates its own at cache
    // miss and the cached program keeps it.
    std::optional<Tensor> forwarding_buffer;
};

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine
