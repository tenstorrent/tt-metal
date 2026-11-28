// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <tuple>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"

namespace ttnn::operations::chunk_program {

// The per-chunk prepared programs of deepseek_v41_flash_d_p (NC.7), one program kind each (io and per-call values:
// chunk_program.hpp).
enum class ChunkKind : uint8_t {
    PAGE_COPY = 0,    // tt/page_copy.PageCopy
    GLUED_ROWS = 1,   // tt/lane_rows.GluedRows (NPAIR tables of one piece list: RoPE cos + sin fused)
    IDS_REORDER = 2,  // tt/page_copy.IdsReorder (the kernel after the SP all-gather)
    GLUED_IDS = 3,    // tt/page_copy.GluedIdsReorder (the kernel after the SP all-gather, if any)
    DERIVE = 4,       // tt/glued_derive.Derive (one or more sections in one program, each on its own cores)
    OWNED_SEND = 5,   // tt/owned_gather.OwnedSend
};

// Everything structural about one Python program instance, made once by the model and kept; its hash is computed
// once. ``consts`` per kind (chunk_program_program_factory.cpp); ``groups`` per kind: DERIVE its sections, OWNED_SEND
// per chip (chip-linear) its ports; ``sems``: global semaphore addresses (OWNED_SEND).
struct ChunkPlan {
    ChunkKind kind = ChunkKind::PAGE_COPY;
    uint32_t rows = 0, cols = 0;
    std::vector<uint32_t> consts;
    std::vector<std::vector<uint32_t>> groups;
    std::vector<uint32_t> sems;
    uint64_t hash = 0;
};

struct ChunkProgramParams {
    std::shared_ptr<const ChunkPlan> plan;
    // Per-call values (never in the program key: written into the cached programs' runtime args on every call).
    std::vector<uint32_t> values;

    static constexpr auto attribute_names = std::forward_as_tuple("kind", "plan_hash");
    auto attribute_values() const {
        return std::make_tuple(static_cast<uint32_t>(plan->kind), static_cast<uint64_t>(plan->hash));
    }
};

struct ChunkProgramInputs {
    std::vector<Tensor> io;
};

}  // namespace ttnn::operations::chunk_program
