// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "combine_fabric2d_collector.hpp"

#include <string>
#include <utility>

#include <tt_stl/assert.hpp>

#include "ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/combine_fabric2d_collector_ct_args.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine {

void add_collector(
    tt::tt_metal::ProgramDescriptor& desc,
    const CombineFabric2dParams& args,
    const DevicePlacement& placement,
    uint32_t counts_addr,
    uint32_t ready_sem,
    uint32_t progress_sem,
    bool gate_untilizers) {
    cmbf2d::op::CollectorPlan plan{.counts_addr = counts_addr, .ready_sem = ready_sem};
    if (gate_untilizers) {
        // Only the untilizers: they are the cores that read routed-expert output from DRAM, so only they can
        // run ahead of it. The readers consume what the untilizers staged in L1 and are ordered behind them.
        for (const auto& group : placement.untilizers) {
            for (const auto& untilizer : group) {
                plan.waiting.push_back(untilizer.worker_virtual);
            }
        }
        // An empty list would leave every DRAM reader ungated, which reads stale rows rather than failing.
        TT_FATAL(!plan.waiting.empty(), "combine_fabric2d: overlapped with the routed expert but no core gates on it");
    }

    tt::tt_metal::KernelDescriptor kernel;
    kernel.kernel_source =
        "ttnn/cpp/ttnn/operations/experimental/deepseek_prefill/combine_fabric2d/device/kernels/dataflow/"
        "collector_combine_fabric2d.cpp";
    kernel.source_type = tt::tt_metal::KernelDescriptor::SourceType::FILE_PATH;
    kernel.defines.emplace_back("CMBF2D_PROGRESS_SEM", std::to_string(progress_sem));
    kernel.core_ranges = CoreRangeSet(CoreRange(placement.collector.value()));
    kernel.compile_time_args = cmbf2d::CollectorCtArgs(args, plan).to_ct_word_arr();
    kernel.config = tt::tt_metal::DataMovementConfigDescriptor{
        .processor = tt::tt_metal::DataMovementProcessor::RISCV_1,
        .noc = tt::tt_metal::NOC::NOC_0,
    };
    desc.kernels.push_back(std::move(kernel));
}

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn::combine
