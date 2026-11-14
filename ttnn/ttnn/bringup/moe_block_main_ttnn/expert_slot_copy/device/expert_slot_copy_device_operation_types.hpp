// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstdint>
#include <memory>
#include <optional>
#include <tuple>
#include <vector>

#include <tt_stl/reflection.hpp>

#include "../../common/mesh_routing.hpp"
#include "../../common/port_placement.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace ttnn::operations::bringup::moe_block_main::expert_slot_copy {

inline constexpr uint32_t kMaxJobs = 32;  // jobs a port core a call (ESC_MAXJ)
inline constexpr uint32_t kMaxLinks = 4;  // sender cores a copy (ESC_MAXL)
inline constexpr uint32_t kLocal = 0xFF;
inline constexpr uint32_t kRunWords = 6;
inline constexpr uint32_t kSemStride = 16;

// The mesh wiring and the port cores the program is built for (from MoEBlockResources, computed once, shared).
struct SlotCopyPorts {
    MeshPlan plan;
    PortCores cores;
};

// One copy: chip `src` reads run table `table` (its home slot's runs) and writes it into chip `dst` (chip index r C +
// c), through the fabric. A local job (`dst` == `src`): the chip copies within its own DRAM (src tensors -> dst
// tensors), e.g. a re-layout at load.
struct SlotCopyJob {
    uint32_t src = 0;
    uint32_t dst = 0;
    uint32_t table = 0;
};

struct SlotCopyParams {
    // Structural, hashed.
    ttsl::hash::hash_t plan_key = 0;  // MoEBlockResources::plan_key(): determines `ports`
    uint32_t chunk_bytes = 0;         // a fabric packet's payload (<= the fabric's max), 64 B multiple
    uint32_t depth = 8;               // CB_DATA pages
    uint32_t batch = 4;               // chunks a read barrier
    uint32_t links = 1;               // sender cores (links of the first hop's direction) a copy
    // Per call, not hashed: the jobs (runtime args only) and the shared plan.
    std::vector<SlotCopyJob> jobs;
    std::shared_ptr<const SlotCopyPorts> ports;

    static constexpr auto attribute_names = std::forward_as_tuple("plan_key", "chunk_bytes", "depth", "batch", "links");
    auto attribute_values() const { return std::forward_as_tuple(plan_key, chunk_bytes, depth, batch, links); }
};

struct SlotCopyInputs {
    Tensor src_gu;  // the weights read (DRAM, sharded over the banks as the flat expert's: bank b's shard at the
    Tensor src_d;   //  buffer address in bank b)
    Tensor dst_gu;  // the weights written (the same tensors as src_* for a fill)
    Tensor dst_d;
    Tensor runs;  // [1, 1, tables, W] uint32 ROW_MAJOR DRAM interleaved: a run table a page (slot_copy_common.hpp)
    Tensor
        sems;  // persistent words over the port cores (MoEBlockResources "expert_slot_copy/sems"), zero between calls
};

}  // namespace ttnn::operations::bringup::moe_block_main::expert_slot_copy
