// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <optional>
#include <string_view>
#include <variant>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/sub_device_types.hpp>

#include "ttnn/common/guard.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshWorkload;
}  // namespace tt::tt_metal::distributed

namespace ttnn {

// Confines the device operations a thread launches to a set of Tensix worker cores.
//
// Inside a scope, launch() gives the scope's cores to the operation's program factory, refuses a factory that does not
// take worker cores rather than let it place kernels anywhere on the device, and checks that the program it builds
// stays on those cores. A program factory takes worker cores by declaring a trailing `const CoreRangeSet&
// worker_cores` parameter on its create function (see ttnn/device_operation.hpp); outside any scope it is given the
// device's whole compute grid, so its programs do not change.
//
// A sub-device scope stands for the Tensix cores of that sub-device of the operation's device, resolved when the
// operation launches; a core scope names the cores directly. Scopes nest and the innermost one applies.
using WorkerCoresScope = std::variant<tt::tt_metal::SubDeviceId, tt::tt_metal::CoreRangeSet>;

void push_worker_cores_scope_for_thread(WorkerCoresScope scope);
void pop_worker_cores_scope_for_thread();

// Scopes for the lifetime of the returned guard.
ScopeGuard with_sub_device(tt::tt_metal::SubDeviceId sub_device_id);
ScopeGuard with_worker_cores(tt::tt_metal::CoreRangeSet worker_cores);

// The Tensix cores of the innermost scope on `mesh_device`, or std::nullopt outside any scope.
std::optional<tt::tt_metal::CoreRangeSet> get_scoped_worker_cores(
    const tt::tt_metal::distributed::MeshDevice& mesh_device);

// The Tensix cores operations launched now may place kernels on: the innermost scope's, or the whole compute grid of
// `mesh_device` outside any scope. For code that sizes a program before its factory runs, such as a default program
// config or an explicit core grid.
tt::tt_metal::CoreRangeSet get_worker_cores(const tt::tt_metal::distributed::MeshDevice& mesh_device);

}  // namespace ttnn

namespace ttnn::device_operation::detail {

// Raised when a scope is active and the selected program factory does not take worker cores.
[[noreturn]] void throw_factory_does_not_take_worker_cores(
    std::string_view operation_name, std::string_view factory_name, const tt::tt_metal::CoreRangeSet& worker_cores);

// Raises unless every Tensix kernel of every program in `workload` runs on `worker_cores`.
void validate_workload_on_worker_cores(
    const tt::tt_metal::distributed::MeshWorkload& workload,
    const tt::tt_metal::CoreRangeSet& worker_cores,
    std::string_view operation_name,
    std::string_view factory_name);

}  // namespace ttnn::device_operation::detail
