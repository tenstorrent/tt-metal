// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <vector>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

// Default read_batch==write_batch for the non-width RM builders
// (build_concat_rm / build_concat_rm_nonwidth_nway).
inline constexpr uint32_t kConcatNonWidthBatch = 4;
// Default write_batch for the width RM builders (build_concat_rm_width /
// build_concat_rm_width_nway); their reader is unbatched (2-tensor) or reads
// with a separate, unscaled read_batch=1 (N-way), so only write_batch (and the
// CB depth it drives) is scaled for L1 fit.
inline constexpr uint32_t kConcatWidthWriteBatch = 4;
// The non-width N-way reader spends 3 + 3*N runtime-arg words and the width N-way
// reader 2 + 3*N in the same order, so at N=64 both sit comfortably under the
// portable runtime-argument bound. One ceiling covers both N-way builders.
inline constexpr uint32_t kConcatMaxNwayInputs = 64;

struct ConcatCbPlan {
    uint32_t batch;
    uint32_t depth;
};

// Largest feasible (batch, depth) for a `page_size`-byte-page CB under
// `l1_budget_bytes`: prefers double-buffered depth=2*batch (batch <=
// max_batch), and falls back to a single-buffered batch=1/depth=1 plan when
// even depth=2 at batch=1 does not fit -- mirroring
// concat_program_factory.cpp's native depth-2-to-depth-1 fallback. The
// reader/writer kernels' BATCH<=1 path (writer_interleaved.cpp's `else`
// branch; every ported reader's own batch=1 loop) only waits on depth>=1, so
// forcing batch to 1 is sufficient to make depth=1 correct. nullopt only when
// even a single page does not fit. Single source shared by the routing gate
// (concat_codegen_supported.cpp) and the factory, so they cannot drift.
std::optional<ConcatCbPlan> plan_concat_cb(uint32_t page_size, uint32_t max_batch, uint64_t l1_budget_bytes);

struct ConcatCodegenParams {
    uint32_t dim{};
    uint32_t num_inputs{};
    uint32_t stick_size{};
    uint32_t total_out_sticks{};
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct ConcatCodegenInputs {
    std::vector<Tensor> input_tensors;
};

struct ConcatCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const ConcatCodegenParams& operation_attributes,
        const ConcatCodegenInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
