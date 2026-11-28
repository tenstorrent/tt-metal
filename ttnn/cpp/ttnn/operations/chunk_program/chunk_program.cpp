// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#include "chunk_program.hpp"

#include "device/chunk_program_device_operation.hpp"

namespace ttnn {

std::shared_ptr<const operations::chunk_program::ChunkPlan> chunk_program_plan(
    uint32_t kind,
    uint32_t rows,
    uint32_t cols,
    const std::vector<uint32_t>& consts,
    const std::vector<std::vector<uint32_t>>& groups,
    const std::vector<uint32_t>& sems) {
    namespace cp = operations::chunk_program;
    TT_FATAL(kind <= static_cast<uint32_t>(cp::ChunkKind::OWNED_SEND), "chunk_program_plan: unknown kind {}", kind);
    auto p = std::make_shared<cp::ChunkPlan>();
    p->kind = static_cast<cp::ChunkKind>(kind);
    p->rows = rows;
    p->cols = cols;
    p->consts = consts;
    p->groups = groups;
    p->sems = sems;
    std::vector<uint32_t> flat;
    for (const auto& g : groups) {
        flat.push_back(static_cast<uint32_t>(g.size()));
        flat.insert(flat.end(), g.begin(), g.end());
    }
    p->hash = ttsl::hash::hash_objects_with_default_seed(
        ttsl::hash::type_hash<cp::ChunkProgramDeviceOperation>, kind, rows, cols, consts, flat, sems);
    return p;
}

Tensor chunk_program(
    const std::vector<Tensor>& io,
    const std::shared_ptr<const operations::chunk_program::ChunkPlan>& plan,
    const std::vector<uint32_t>& values) {
    return ttnn::prim::chunk_program(io, plan, values);
}

}  // namespace ttnn
