// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/worker_cores.hpp"

#include <algorithm>

#include <utility>
#include <vector>

#include <tt-metalium/experimental/kernel_cores.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/mesh_workload.hpp>
#include <tt_stl/assert.hpp>
#include <tt_stl/overloaded.hpp>

namespace ttnn {

using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;

namespace {

std::vector<WorkerCoresScope>& scopes_for_thread() {
    thread_local std::vector<WorkerCoresScope> scopes;
    return scopes;
}

CoreRangeSet compute_grid(const tt::tt_metal::distributed::MeshDevice& mesh_device) {
    const auto grid = mesh_device.compute_with_storage_grid_size();
    return CoreRangeSet(CoreRange({0, 0}, {grid.x - 1, grid.y - 1}));
}

}  // namespace

void push_worker_cores_scope_for_thread(WorkerCoresScope scope) {
    if (const auto* cores = std::get_if<CoreRangeSet>(&scope)) {
        TT_FATAL(!cores->empty(), "A worker-cores scope needs at least one core");
    }
    scopes_for_thread().push_back(std::move(scope));
}

void pop_worker_cores_scope_for_thread() {
    auto& scopes = scopes_for_thread();
    TT_FATAL(!scopes.empty(), "No worker-cores scope to pop on this thread");
    scopes.pop_back();
}

ScopeGuard with_sub_device(tt::tt_metal::SubDeviceId sub_device_id) {
    push_worker_cores_scope_for_thread(sub_device_id);
    return make_guard([] { pop_worker_cores_scope_for_thread(); });
}

ScopeGuard with_worker_cores(CoreRangeSet worker_cores) {
    push_worker_cores_scope_for_thread(std::move(worker_cores));
    return make_guard([] { pop_worker_cores_scope_for_thread(); });
}

std::optional<CoreRangeSet> get_scoped_worker_cores(const tt::tt_metal::distributed::MeshDevice& mesh_device) {
    const auto& scopes = scopes_for_thread();
    if (scopes.empty()) [[likely]] {
        return std::nullopt;
    }
    CoreRangeSet cores = std::visit(
        ttsl::overloaded{
            [&](tt::tt_metal::SubDeviceId sub_device_id) {
                return mesh_device.worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, sub_device_id);
            },
            [](const CoreRangeSet& worker_cores) { return worker_cores; }},
        scopes.back());
    // A sub-device's cores may lie outside the compute grid the device reports, when that grid is narrowed to another
    // sub-device (experimental::SetReservedComputeColumns).
    const auto within_a_sub_device = [&] {
        return std::ranges::any_of(mesh_device.get_sub_device_ids(), [&](tt::tt_metal::SubDeviceId id) {
            return mesh_device.worker_cores(tt::tt_metal::HalProgrammableCoreType::TENSIX, id).contains(cores);
        });
    };
    TT_FATAL(
        compute_grid(mesh_device).contains(cores) || within_a_sub_device(),
        "Worker-cores scope {} reaches outside the compute grid {} of mesh device {}",
        cores,
        mesh_device.compute_with_storage_grid_size(),
        mesh_device.id());
    return cores;
}

CoreRangeSet get_worker_cores(const tt::tt_metal::distributed::MeshDevice& mesh_device) {
    auto cores = get_scoped_worker_cores(mesh_device);
    return cores.has_value() ? std::move(*cores) : compute_grid(mesh_device);
}

}  // namespace ttnn

namespace ttnn::device_operation::detail {

using tt::tt_metal::CoreRangeSet;

void throw_factory_does_not_take_worker_cores(
    std::string_view operation_name, std::string_view factory_name, const CoreRangeSet& worker_cores) {
    TT_THROW(
        "{} cannot run inside a worker-cores scope ({}): its program factory {} does not take worker cores, so it "
        "would place kernels wherever it chooses. Launch it outside the scope, or give {} a worker_cores parameter.",
        operation_name,
        worker_cores,
        factory_name,
        factory_name);
}

void validate_workload_on_worker_cores(
    const tt::tt_metal::distributed::MeshWorkload& workload,
    const CoreRangeSet& worker_cores,
    std::string_view operation_name,
    std::string_view factory_name) {
    for (const auto& [range, program] : workload.get_programs()) {
        const auto kernel_cores =
            tt::tt_metal::experimental::GetKernelCores(program, tt::tt_metal::HalProgrammableCoreType::TENSIX);
        TT_FATAL(
            worker_cores.contains(kernel_cores),
            "{}: program factory {} placed kernels on {} for {}, outside the worker-cores scope {}",
            operation_name,
            factory_name,
            kernel_cores.subtract(worker_cores),
            range,
            worker_cores);
    }
}

}  // namespace ttnn::device_operation::detail
