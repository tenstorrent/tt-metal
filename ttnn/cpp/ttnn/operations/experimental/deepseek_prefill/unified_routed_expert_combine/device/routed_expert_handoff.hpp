// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <map>
#include <optional>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>
#include <tt-metalium/workload_descriptor.hpp>
#include <ttnn/global_semaphore.hpp>

#include "combine_overlap_program_factory.hpp"
#include "ttnn/distributed/types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine::overlap {

// The routed expert's side of a program it shares with combine_fabric2d: what its writers report to combine's
// collector, and how its descriptor is folded into each chip's combine program (unified_routed_expert_moe_combine).

// Combine takes grid rows 0-1; the routed expert runs on the rows below.
inline constexpr uint32_t kRoutedExpertOriginY = 2;

// The combine core every routed-expert writer reports to, once per expert index it walks, and the global
// semaphore on the writer cores that says it may start. See kernels/expert_done.hpp for the device side.
struct ExpertDoneSignal {
    uint32_t collector_noc_x = 0;
    uint32_t collector_noc_y = 0;
    uint32_t collector_counts_addr = 0;
    uint32_t go_addr = 0;
};

// Turns the hook on in the routed expert's descriptor. The signal's block goes after every other runtime argument, on
// every writer core, so no argument layout moves.
void append_expert_done_signal(tt::tt_metal::ProgramDescriptor& desc, const ExpertDoneSignal& signal);

// How many writer cores report through that hook, each once per expert index it walks.
uint32_t expert_done_writer_count(const tt::tt_metal::ProgramDescriptor& desc);

// Folds `re_descriptor` into every chip's program of `workload` (create_combine_workload's, with `collectors` from
// the same call): each chip's copy reports to that chip's collector, reads where combine's walks open on that chip,
// and writes its last tile-rows to that chip's L1 slots when combine planned any.
void merge_routed_expert(
    tt::tt_metal::WorkloadDescriptor& workload,
    const std::map<ttnn::MeshCoordinate, CollectorTarget>& collectors,
    const tt::tt_metal::ProgramDescriptor& re_descriptor,
    const CombineFabric2dParams& combine_args,
    tt::tt_metal::Buffer* expert_offsets,
    uint32_t go_addr);

// The caller-owned global semaphores of the overlap, and the addresses its attributes hash. `op_name` prefixes
// every message.
void validate_overlap_semaphores(
    const char* op_name,
    ttnn::MeshDevice* device,
    uint32_t combine_axis,
    const std::optional<GlobalSemaphore>& fwd_arrived,
    const std::optional<GlobalSemaphore>& final_arrived,
    const std::optional<GlobalSemaphore>& expert_go,
    const std::vector<GlobalSemaphore>& channel_semaphores,
    uint32_t fwd_arrived_addr,
    uint32_t final_arrived_addr,
    uint32_t expert_go_addr,
    const std::vector<uint32_t>& channel_semaphore_addrs);

// The caller's forwarding buffer, when it passes one, against the spec combine needs for these arguments.
void validate_forwarding_buffer(
    const char* op_name,
    const std::optional<ttnn::Tensor>& forwarding_buffer,
    const CombineFabric2dParams& combine_args,
    const CombineFabric2dInputs& combine_inputs);

}  // namespace ttnn::operations::experimental::deepseek_prefill::unified_routed_expert_combine::overlap
