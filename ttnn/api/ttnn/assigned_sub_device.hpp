// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <concepts>
#include <cstdint>
#include <optional>
#include <string_view>

#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/sub_device_types.hpp>
#include <tt_stl/reflection.hpp>

#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshWorkload;
}  // namespace tt::tt_metal::distributed

namespace ttnn {

// The sub-device an operation runs on.
//
// A sub-device is a dispatch lane of its device: the programs on it are dispatched, completed and synchronized in its
// order, and run on its cores. An operation opts in to sub-devices by recording the one it is assigned in its
// attributes, from assign_sub_device() before it sizes anything, and exposing the record through a static hook:
//
//     static const ttnn::AssignedSubDevice& assigned_sub_device(const operation_attributes_t& attributes);
//
// Its validation, output specs, factory selection and program factories then take placement only from the record:
// Tensix kernels, circular/dataflow buffers and semaphores go on `worker_cores`, and `id` names the lane for what is
// keyed by it (host synchronization, semaphores). Which of those cores the operation uses is its own choice. The record
// says nothing about where buffers are allocated.
//
// Launch refuses, inside a sub-device scope (ttnn::with_sub_device), an operation that records no sub-device or another
// one; refuses a program the operation builds whose Tensix kernels leave `worker_cores`; and keys the operation's
// programs by the record. Metal, with a sub-device manager loaded, refuses a program whose semaphores or
// circular/dataflow buffers reach another sub-device's cores, since fast dispatch writes their initial values and
// configs to the cores they are on. Between the two, nothing dispatch writes for the operation lands on another
// sub-device. Where the operation's kernels write over the NoC is its own promise: a multicast or semaphore set must
// land only where the operation placed the object it targets.
struct AssignedSubDevice {
    tt::tt_metal::SubDeviceId id;
    // The lane's Tensix cores, as the device reported them when the operation was assigned it. Part of the program
    // cache key, so a program is reused only on the cores it was built for. Empty when unassigned.
    tt::tt_metal::CoreRangeSet worker_cores;

    bool assigned() const { return !worker_cores.empty(); }
    bool holds(const tt::tt_metal::CoreRangeSet& cores) const { return worker_cores.contains(cores); }
};

// What names the sub-device an operation runs on, besides a sub-device scope.
struct SubDeviceRequest {
    // The sub-device the caller named (an operation's sub_device_id argument).
    std::optional<tt::tt_metal::SubDeviceId> sub_device;
    // Cores the caller narrowed the operation to (its sub_core_grids, the cores of a program config the caller wrote).
    // They must lie in the assigned sub-device.
    std::optional<tt::tt_metal::CoreRangeSet> cores;
    // Cores the operation's L1-sharded data lives on (see l1_shard_cores). They name the sub-device that holds them
    // when nothing else names one; the operation reads and writes data elsewhere over the NoC, or refuses it where it
    // can only use it in place.
    std::optional<tt::tt_metal::CoreRangeSet> data;
};

// Assigns an operation about to run on `mesh_device` its sub-device, from the first of:
//   1. the innermost sub-device scope of the calling thread, which `request.sub_device` must agree with;
//   2. `request.sub_device`;
//   3. the sub-device holding `request.cores`;
//   4. the sub-device holding `request.data`;
//   5. the device's only sub-device.
// Raises when several sub-devices are loaded and none of these names one, when `request.cores` do not lie in the
// sub-device assigned, and when `request.data` spans sub-devices and is what would name one. Call it once, before
// anything is sized from it. With no device (the operation's tensor is on the host or deallocated), returns an
// unassigned record and leaves the refusal to the operation's own validation.
AssignedSubDevice assign_sub_device(
    const tt::tt_metal::distributed::MeshDevice* mesh_device,
    const SubDeviceRequest& request,
    std::string_view operation_name);

// For an operation that launches other operations (a composite), while the guard lives. Inside a sub-device scope, the
// constructor checks `request` against the scope (assign_sub_device), so arguments that contradict it are refused
// before the composite launches anything. Outside one, on a device with several sub-devices, the operations it launches
// that record a sub-device run on the one assign_sub_device() assigns `request`; those that record none are launched as
// they would be without it. assigned() is that sub-device (resolved on first use when there is neither).
//
// Composites are not transactional: each operation a composite launches is checked as it launches, and when one is
// refused, what the composite launched before is not undone. Inside a scope, the refusal of an operation that records
// no sub-device names the outermost composite it was launched within. `operation_name` must outlive the guard.
class [[nodiscard]] CompositeSubDevice {
public:
    CompositeSubDevice(
        const tt::tt_metal::distributed::MeshDevice* mesh_device,
        SubDeviceRequest request,
        std::string_view operation_name);
    ~CompositeSubDevice() noexcept;
    CompositeSubDevice(const CompositeSubDevice&) = delete;
    CompositeSubDevice& operator=(const CompositeSubDevice&) = delete;
    CompositeSubDevice(CompositeSubDevice&&) = delete;
    CompositeSubDevice& operator=(CompositeSubDevice&&) = delete;

    const AssignedSubDevice& assigned() const;

private:
    const tt::tt_metal::distributed::MeshDevice* mesh_device_;
    SubDeviceRequest request_;
    std::string_view operation_name_;
    mutable std::optional<AssignedSubDevice> assigned_;
    std::optional<std::uint64_t> token_;
    bool outermost_ = false;  // it named the call on this thread, so its destructor clears the name
};

// The cores an L1-sharded memory config's shards live on; nullopt for DRAM, interleaved L1, or a sharded config that
// names no grid.
std::optional<tt::tt_metal::CoreRangeSet> l1_shard_grid(const tt::tt_metal::MemoryConfig& config);

// The cores holding the L1 shards of every tensor in `tensors` (an operation's tensor arguments, or a tuple of
// tensors), and those an L1-sharded `output` memory config names. Nullopt when there are none: DRAM and interleaved
// tensors live on no particular core.
template <typename Tensors>
std::optional<tt::tt_metal::CoreRangeSet> l1_shard_cores(
    const Tensors& tensors, const std::optional<tt::tt_metal::MemoryConfig>& output = std::nullopt) {
    std::optional<tt::tt_metal::CoreRangeSet> cores;
    const auto add = [&cores](const tt::tt_metal::MemoryConfig& config) {
        if (auto grid = l1_shard_grid(config)) {
            if (!cores.has_value() || grid->contains(*cores)) {
                cores = std::move(*grid);
            } else if (!cores->contains(*grid)) {
                cores = cores->merge(*grid);
            }
        }
    };
    ttsl::reflection::visit_object_of_type<Tensor>(
        [&add](const Tensor& tensor) { add(tensor.memory_config()); }, tensors);
    if (output.has_value()) {
        add(*output);
    }
    return cores;
}

// The largest rectangle of `cores`; of equal ones, the first in row-major order of its start.
tt::tt_metal::CoreRange rectangle_in(const tt::tt_metal::CoreRangeSet& cores);

}  // namespace ttnn

namespace ttnn::device_operation {

// An operation records the sub-device it runs on (see ttnn::AssignedSubDevice).
template <typename device_operation_t>
concept RecordsSubDevice = requires(const typename device_operation_t::operation_attributes_t& attributes) {
    { device_operation_t::assigned_sub_device(attributes) } -> std::same_as<const ttnn::AssignedSubDevice&>;
};

}  // namespace ttnn::device_operation

namespace ttnn::device_operation::detail {

// Inside a sub-device scope, refuses an operation that records no sub-device (`recorded` null) or another one. Inside a
// composite operation's sub-device, refuses only another one. The refusal of an unrecorded operation names the
// outermost composite that built a ttnn::CompositeSubDevice.
void check_sub_device_scope(const ttnn::AssignedSubDevice* recorded, std::string_view operation_name);

// Refuses an operation launched without a sub-device assigned.
void check_assigned(const ttnn::AssignedSubDevice& recorded, std::string_view operation_name);

// What happens to an output an operation allocates in L1 other than sharded within its sub-device, on a device with
// several sub-devices. Interleaved L1 spans the cores of every sub-device, where programs running concurrently on the
// others keep their circular buffers, and Metal checks the two against each other only when a program is enqueued.
// Inside a sub-device scope such an output is refused; outside one, a warning is logged once per operation.
enum class L1OutputCheck : std::uint8_t { none, warn, refuse };
L1OutputCheck l1_output_check(const tt::tt_metal::distributed::MeshDevice& mesh_device);

void check_l1_output_in_sub_device(
    const ttnn::AssignedSubDevice& recorded,
    const Tensor& output,
    std::string_view operation_name,
    L1OutputCheck check);

// Inside a composite operation's sub-device (see CompositeSubDevice), logs once per operation an operation that records
// no sub-device and whose kernels landed outside the composite's: nothing orders it against the call's other programs.
void check_unrecorded_workload_in_composite(
    const tt::tt_metal::distributed::MeshWorkload& workload,
    const tt::tt_metal::distributed::MeshDevice& mesh_device,
    std::string_view operation_name);

// Raises unless every Tensix kernel of every program in `workload` runs on the cores of `sub_device`.
void check_workload_on_sub_device(
    const tt::tt_metal::distributed::MeshWorkload& workload,
    const ttnn::AssignedSubDevice& sub_device,
    std::string_view operation_name);

}  // namespace ttnn::device_operation::detail
