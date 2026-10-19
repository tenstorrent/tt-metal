// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <array>
#include <cstdint>
#include <optional>

#include <tt-metalium/program_descriptors.hpp>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::prim {

inline constexpr uint32_t kPermuteCodegenMaxDims = 8;

// Axis positions counted from the end of the shape.
inline constexpr uint32_t kPermuteWidthAxisFromEnd = 1;
inline constexpr uint32_t kPermuteHeightAxisFromEnd = 2;

// Largest number of sticks the row-invariant reader and writer move per turn. The CB holds two
// turns, so the writer can drain one while the reader fills the next.
inline constexpr uint32_t kPermuteRmMaxBatch = 4;
inline constexpr uint32_t kPermuteRmCbTurns = 2;

struct PermuteRmCbPlan {
    uint32_t batch = 0;
    uint32_t depth = 0;
};

// Largest batch (halving from kPermuteRmMaxBatch) whose CB of `kPermuteRmCbTurns * batch` slots of
// `slot_bytes` fits in `l1_budget`; nullopt when even a single-stick turn does not fit.
std::optional<PermuteRmCbPlan> plan_permute_rm_cb(uint32_t slot_bytes, uint64_t l1_budget);

// L1 bytes the blocked kernels hold in CBs for one core. The block edge is the tile edge, so this
// does not scale with the budget and the gate can only accept or reject it.
uint64_t permute_blocked_cb_bytes(uint32_t element_size);

// Per-core L1 the CBs may use: the window above the allocator base, less the per-bank share of an
// interleaved-L1 input or output. Depends only on device and operand specs, so it is the static
// bound the support gate checks.
uint64_t permute_cb_l1_budget(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec);

// The static budget clamped to the L1 left below the lowest occupied address right now. This is
// what sizes the row-invariant CB, so the descriptor also depends on the live allocator state.
uint64_t permute_live_cb_l1_budget(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec);

// Row-invariant CB plan under the live budget; nullopt when even a single-stick turn does not fit.
// The factory and the program hash both read it, so the cache key tracks the plan the factory builds.
std::optional<PermuteRmCbPlan> permute_rm_cb_plan(const Tensor& input, const tt::tt_metal::TensorSpec& output_spec);

struct PermuteCodegenParams {
    // Zero-padded past the input rank.
    std::array<uint32_t, kPermuteCodegenMaxDims> dims{};
    tt::tt_metal::MemoryConfig output_mem_config;
};

struct PermuteCodegenInputs {
    Tensor input;
    std::optional<Tensor> optional_output_tensor;
};

struct PermuteCodegenProgramFactory {
    static tt::tt_metal::ProgramDescriptor create_descriptor(
        const PermuteCodegenParams& operation_attributes,
        const PermuteCodegenInputs& tensor_args,
        Tensor& tensor_return_value);
};

}  // namespace ttnn::prim
