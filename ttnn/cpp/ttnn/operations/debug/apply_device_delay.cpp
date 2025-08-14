// SPDX-FileCopyrightText: © 2025 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "apply_device_delay.hpp"

#include <cstddef>
#include <cstdint>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/mesh_coord.hpp>

namespace ttnn::operations::debug {

using namespace tt::tt_metal;
using namespace tt::tt_metal::distributed;

namespace {

// Select a single logical worker core for the given subdevice. Fallback to {0,0} if empty.
static CoreCoord select_single_worker_core(const CoreRangeSet& worker_cores) {
    if (worker_cores.ranges().empty()) {
        return CoreCoord{0, 0};
    }
    const auto& r = worker_cores.ranges().front();
    return r.start_coord;  // choose the first core in the first range
}

}  // namespace

void apply_device_delay(
    MeshDevice* mesh_device,
    QueueId queue_id,
    const std::optional<SubDeviceId>& subdevice_id,
    const std::vector<std::vector<uint32_t>>& delays) {
    TT_FATAL(mesh_device != nullptr, "MeshDevice is null");

    const auto& view = mesh_device->get_view();
    TT_FATAL(view.is_mesh_2d(), "apply_device_delay currently supports only 2D mesh");
    TT_FATAL(
        delays.size() == view.num_rows(),
        "delays rows ({} ) must match mesh rows ({})",
        delays.size(),
        view.num_rows());
    for (size_t r = 0; r < delays.size(); ++r) {
        TT_FATAL(
            delays[r].size() == view.num_cols(),
            "delays cols for row {} ({} ) must match mesh cols ({})",
            r,
            delays[r].size(),
            view.num_cols());
    }

    // Build a MeshWorkload with one Program per device coordinate.
    MeshWorkload workload = CreateMeshWorkload();

    // Iterate through mesh coordinates and stitch per-device programs with the compiled delay kernel.
    for (size_t row = 0; row < view.num_rows(); ++row) {
        for (size_t col = 0; col < view.num_cols(); ++col) {
            const auto delay_cycles = delays[row][col];
            const MeshCoordinate coord{static_cast<uint32_t>(row), static_cast<uint32_t>(col)};

            // Create a minimal program that launches a single kernel on one worker core.
            Program program{};

            // Determine worker core range set for the subdevice id (or default first one).
            SubDeviceId sd = subdevice_id.has_value() ? subdevice_id.value() : mesh_device->get_sub_device_ids().at(0);
            auto worker_set = mesh_device->worker_cores(HalProgrammableCoreType::TENSIX, sd);
            const CoreCoord chosen_core = select_single_worker_core(worker_set);

            // Compile-time args: [0] = delay cycles
            std::vector<uint32_t> ct_args = {delay_cycles};

            // Single DM kernel that spins for delay.
            const std::string kernel_path = "ttnn/cpp/ttnn/operations/debug/kernels/dataflow/device_delay_spin.cpp";

            // Place kernel on chosen core for this single device.
            auto kernel_id = CreateKernel(
                program,
                kernel_path,
                CoreRangeSet(CoreRange(chosen_core)),
                DataMovementConfig{.compile_args = ct_args});

            // No runtime args needed; compile-time delay is sufficient.

            // Map this program to the single device coordinate.
            AddProgramToMeshWorkload(workload, std::move(program), MeshCoordinateRange(coord));
        }
    }

    // Enqueue the workload on the provided queue.
    mesh_device->mesh_command_queue(*queue_id).enqueue_mesh_workload(workload, /*blocking=*/false);
}

}  // namespace ttnn::operations::debug
