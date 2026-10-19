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

// CB slot of the row-invariant path: the larger aligned stick of the input buffer and of the output.
// The last axis is unchanged, so it follows from the input and the output buffer type alone.
uint32_t permute_rm_slot_bytes(const Tensor& input, const tt::tt_metal::MemoryConfig& output_mem_config);

// L1 bytes the blocked kernels hold in CBs for one core. The block edge is the tile edge, so this
// does not scale with the budget and the gate can only accept or reject it.
uint64_t permute_blocked_cb_bytes(uint32_t element_size);

// Per-core L1 the CBs may use: the window above the allocator base, less the per-bank share of an
// interleaved-L1 input. Depends only on device and input spec, so it is the static answer the
// support gate gives; the factory plans against permute_live_cb_l1_budget instead.
uint64_t permute_cb_l1_budget(const Tensor& input);

// permute_cb_l1_budget further clamped to the lowest occupied compute L1 address, which program
// enqueue re-checks the CB region against on every call.
uint64_t permute_live_cb_l1_budget(const Tensor& input);

// Row-invariant batch the live frontier allows, or 0 when nothing fits, the last axis moves, or the
// tensor is not a device tensor yet (validation reports that).
uint32_t permute_live_rm_cb_batch(
    const Tensor& input,
    const tt::tt_metal::MemoryConfig& output_mem_config,
    const std::array<uint32_t, kPermuteCodegenMaxDims>& dims);

struct PermuteCodegenParams {
    // Zero-padded past the input rank.
    std::array<uint32_t, kPermuteCodegenMaxDims> dims{};
    tt::tt_metal::MemoryConfig output_mem_config;
    // Derived from the live L1 frontier before dispatch so a changed frontier misses the program cache.
    uint32_t rm_cb_batch = 0;
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
