// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/combine_fabric2d_placement.hpp"
#include "ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/combine_fabric2d_types.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine {

using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::CombineFabric2dParams;
using ::ttnn::operations::experimental::deepseek_prefill::combine_fabric2d::DevicePlacement;

// Adds the collector to one chip's program: the kernel on `placement`'s collector core that folds the routed
// expert's per-writer reports into the `ready` count, and the per-chunk `progress` release, that combine's
// untilizers wait on before reading routed-expert output. The writers bump the per-step count array at
// `counts_addr` on that core; `ready_sem` and `progress_sem` are the program semaphores it publishes.
//
// Without `gate_untilizers` the collector publishes to no core, for a program that builds no untilizers: it
// still releases the routed expert and takes its reports, which the routed expert waits on either way.
void add_collector(
    tt::tt_metal::ProgramDescriptor& desc,
    const CombineFabric2dParams& args,
    const DevicePlacement& placement,
    uint32_t counts_addr,
    uint32_t ready_sem,
    uint32_t progress_sem,
    bool gate_untilizers);

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine
