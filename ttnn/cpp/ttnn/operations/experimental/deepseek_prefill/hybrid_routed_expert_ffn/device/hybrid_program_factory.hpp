// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <tt-metalium/program_descriptors.hpp>

#include "hybrid_routed_expert_ffn_types.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn {

// Everything the merged op rejects, checked before anything is placed. Each half also validates
// what it will actually be handed, so a merged dispatch cannot pass a configuration either
// implementation would reject on its own.
void validate_arguments(const HybridRoutedExpertFfnParams& op, const HybridRoutedExpertFfnInputs& t);

// The merged op's whole program, from one call.
//
// Which implementations are carried depends on the threshold: with no expert below it the fused
// half is not built at all, and the result is a plain unified program. Otherwise both are built
// and folded into one binary per RISC-V that runs them as ordered passes.
tt::tt_metal::ProgramDescriptor create_hybrid_program_descriptor(
    const HybridRoutedExpertFfnParams& op, const HybridRoutedExpertFfnInputs& t, ttnn::Tensor& output);

// The same, with both halves' circular buffers laid over `l1_arena` instead of the caller's t.l1_arena.
tt::tt_metal::ProgramDescriptor create_hybrid_program_descriptor(
    const HybridRoutedExpertFfnParams& op,
    const HybridRoutedExpertFfnInputs& t,
    ttnn::Tensor& output,
    tt::tt_metal::Buffer* l1_arena);

// The combine core every routed-expert writer reports to, once per expert index it walks, and the global
// semaphore on the writer cores that says it may start. See kernels/hybrid_expert_done.hpp for the device side.
struct ExpertDoneSignal {
    uint32_t collector_noc_x = 0;
    uint32_t collector_noc_y = 0;
    uint32_t collector_counts_addr = 0;
    uint32_t go_addr = 0;
};

// Turns the hook on in a descriptor create_hybrid_program_descriptor built, whichever shape it has: the
// merged union writer or the unified writer alone. The signal's block goes after every other runtime
// argument, on every writer core, so neither half's argument layout moves.
void append_expert_done_signal(tt::tt_metal::ProgramDescriptor& desc, const ExpertDoneSignal& signal);

// How many writer cores report through that hook, each once per expert index it walks.
uint32_t expert_done_writer_count(const tt::tt_metal::ProgramDescriptor& desc);

}  // namespace ttnn::operations::experimental::deepseek_prefill::hybrid_routed_expert_ffn
