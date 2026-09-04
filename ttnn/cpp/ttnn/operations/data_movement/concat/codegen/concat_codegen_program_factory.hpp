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
// Bounded by dataflow-RISC stack, not by runtime-arg words. The width N-way reader
// holds four uint32_t[N] plus a bool[N] = 17*N bytes of frame, against a guaranteed
// MEM_{BRISC,NCRISC}_STACK_MIN_SIZE of 256 B on both Blackhole and Wormhole; the
// linker enforces that minimum against static .data/.bss only, never against runtime
// frames. 8 inputs is 136 B, half the guarantee. Wider concat routes to native.
inline constexpr uint32_t kConcatMaxNwayInputs = 8;

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

// The CB geometry every builder actually programs, under an explicit L1 budget.
// Reproduces each of the four builders' page/scratch arithmetic exactly once, so the
// routing gate and the factory cannot disagree about what will fit. `scratch_page` is 0
// for the non-width builders, which have no scratch CB.
struct ConcatCbSelection {
    uint32_t cb_page;
    uint32_t scratch_page;
    uint32_t batch;
    uint32_t depth;
};

// Single authority for the output spec. The gate, the hash and compute_output_specs all
// need the output's page size before the output buffer exists, so they project it from
// here rather than from a Buffer.
tt::tt_metal::TensorSpec concat_output_spec(
    const std::vector<Tensor>& input_tensors, uint32_t dim, const tt::tt_metal::MemoryConfig& output_mem_config);

// nullopt when the selected builder's CBs do not fit `l1_budget_bytes`. The budget is the
// caller's choice of frontier: the routing gate passes the STATIC per-core budget so its
// answer cannot move under it, the factory passes the LIVE one so the program it builds
// fits what is actually free. Both get the same arithmetic.
std::optional<ConcatCbSelection> plan_concat_cbs(
    const std::vector<Tensor>& input_tensors,
    uint32_t dim,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    uint64_t l1_budget_bytes);

struct ConcatCodegenParams {
    uint32_t dim{};
    uint32_t num_inputs{};
    uint32_t stick_size{};
    uint32_t total_out_sticks{};
    tt::tt_metal::MemoryConfig output_mem_config;
};

// Single authority for the prim's derived attributes. The routing sites build them from the
// tensors and validate recomputes them from the same function, so a disagreement means the
// attributes were fabricated rather than derived.
ConcatCodegenParams concat_codegen_params(
    const std::vector<Tensor>& input_tensors, uint32_t dim, const tt::tt_metal::MemoryConfig& output_mem_config);

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
