// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

// The flat expert's DRISC gate/up weight senders (kernels/se_drisc.hpp): persistent kernels on the Blackhole
// programmable DRAM cores, launched once per mesh device through tt-metalium/experimental/dram_core_kernels.hpp and
// left running across flat programs; FlatDriscSenders owns them (its destruction or stop() stops them, and the mesh
// device stops them at close). Also the sender topology the plan and the program factory share: which DRAM tile
// sends weight slot r, on which NoC, with which job index.

#pragma once

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/dram_core_kernels.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace tt::tt_metal::distributed {
class MeshDevice;
}

namespace ttnn::operations::experimental::moe_block::flat_routed_expert {

// The DRAM cores as the flat expert uses them, read once per mesh device (its first local chip, as the plan).
// Weight slot r lives in bank r % banks. Each bank has a "free" tile (no view's endpoint: both NIUs in stream mode)
// and, in switch mode, a bank's second slot (r >= banks) is sent by the bank's other Metal tile (its NOC1 endpoint,
// on NOC0) while the free tile's job 1 is skipped.
struct FlatDriscTopology {
    uint32_t banks = 0;
    uint32_t base = 0;  // the senders' working region in DRAM-core L1 (se_drisc.hpp offsets are from here)
    uint32_t size = 0;
    tt::tt_metal::distributed::MeshCoordinate coord{0, 0};                     // the chip the topology was read from
    std::vector<std::vector<tt::tt_metal::experimental::DramCoreInfo>> cores;  // per bank

    static FlatDriscTopology of(tt::tt_metal::distributed::MeshDevice& mesh_device);

    // the bank's free tile index
    uint32_t free_tile(uint32_t bank) const;
    // reader slot r's DRISC (logical): the free tile of bank r % banks
    tt::tt_metal::CoreCoord logical(uint32_t r) const;
    // the job index of slot r on its DRISC
    static uint32_t job(uint32_t r, uint32_t banks) { return r / banks; }
    // the NoC the free tile writes slot r on: job 0 on NOC1, job 1 on NOC0
    static uint32_t noc(uint32_t r, uint32_t banks) { return r / banks == 0 ? 1 : 0; }
    // switch mode: slot r's alternate {logical tile, job}; job ~0u: none
    std::pair<tt::tt_metal::CoreCoord, uint32_t> alt(uint32_t r) const;
    // slot r's sending tile (NOC0 grid coordinates) and NoC
    std::pair<tt::tt_metal::CoreCoord, uint32_t> sender(uint32_t r) const;
    // NoC address of a logical DRAM tile, packed x << 16 | y
    uint32_t virtual_xy(const tt::tt_metal::CoreCoord& logical) const;
    // a DRAM tile's info
    const tt::tt_metal::experimental::DramCoreInfo& info(const tt::tt_metal::CoreCoord& logical) const;
};

// The running senders of slots 0 .. n_slots - 1 on every chip of one mesh device.
class FlatDriscSenders {
public:
    // The mesh device's senders: launched on first use, shared while anyone holds them (one set a mesh: they own the
    // DRAM cores). A second request must ask for the same slot count.
    static std::shared_ptr<FlatDriscSenders> get(tt::tt_metal::distributed::MeshDevice& mesh_device, uint32_t n_slots);

    // Whether the mesh device's senders of `n_slots` slots are running (a flat program needs them).
    static bool running_on(const tt::tt_metal::distributed::MeshDevice& mesh_device, uint32_t n_slots);

    ~FlatDriscSenders();
    FlatDriscSenders(const FlatDriscSenders&) = delete;
    FlatDriscSenders& operator=(const FlatDriscSenders&) = delete;

    // Finishes the mesh's command queue (no flat program in flight), then stops the senders and waits for them.
    // A no-op once stopped or once the mesh device has closed.
    void stop();
    bool running() const;
    uint32_t n_slots() const { return n_slots_; }
    const tt::tt_metal::distributed::MeshDevice* mesh_device() const { return mesh_device_; }

private:
    FlatDriscSenders(tt::tt_metal::distributed::MeshDevice& mesh_device, uint32_t n_slots);

    tt::tt_metal::distributed::MeshDevice* mesh_device_;
    uint32_t n_slots_;
    FlatDriscTopology topo_;
    std::vector<tt::tt_metal::CoreCoord> cores_;  // logical DRAM tiles that run a sender
    std::unique_ptr<tt::tt_metal::experimental::DramCoreKernels> kernels_;
};

// The NOC0 grid coordinates of a worker core (what its kernels read as their own NoC position).
tt::tt_metal::CoreCoord flat_worker_noc0(
    tt::tt_metal::distributed::MeshDevice& mesh_device, const tt::tt_metal::CoreCoord& logical);

// The first chip this rank drives (the plan and the topology are read from it).
tt::tt_metal::distributed::MeshCoordinate flat_reference_coord(tt::tt_metal::distributed::MeshDevice& mesh_device);

}  // namespace ttnn::operations::experimental::moe_block::flat_routed_expert
