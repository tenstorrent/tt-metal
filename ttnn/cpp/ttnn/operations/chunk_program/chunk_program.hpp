// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <vector>

#include "ttnn/tensor/tensor.hpp"
#include "ttnn/types.hpp"
#include "device/chunk_program_device_operation_types.hpp"

namespace ttnn {

// The plan of one per-chunk program instance (made once): kind (operations::chunk_program::ChunkKind), the mesh shape,
// the kind's constants, groups (DERIVE: sections; OWNED_SEND: per chip its ports) and global semaphore addresses.
std::shared_ptr<const operations::chunk_program::ChunkPlan> chunk_program_plan(
    uint32_t kind,
    uint32_t rows,
    uint32_t cols,
    const std::vector<uint32_t>& consts,
    const std::vector<std::vector<uint32_t>>& groups,
    const std::vector<uint32_t>& sems);

// One call of a per-chunk program of deepseek_v41_flash_d_p (in place; returns io[0]):
//   PAGE_COPY    io [src, dst]                          values [s0, d0, n, nb, ss, ds]
//   GLUED_ROWS   io (table, out) x pairs                values every mesh row's pieces (GWORDS a row)
//   IDS_REORDER  io [gathered input, dst]               values [iu, cs, valid]
//   GLUED_IDS    io [input, dst]                        values []
//   DERIVE       io [header, the sections' outputs]     values [header byte offset, the sections' values]
//   OWNED_SEND   io [out, a record buffer]              values [record buffers, lo, hi, map words]
//   PAGE_MULTI   io (src, dst) x segments               values (s0, d0, n) x segments (NC.9)
Tensor chunk_program(
    const std::vector<Tensor>& io,
    const std::shared_ptr<const operations::chunk_program::ChunkPlan>& plan,
    const std::vector<uint32_t>& values);

}  // namespace ttnn
