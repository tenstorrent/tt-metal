// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <functional>
#include <optional>
#include <span>
#include <string_view>

#include <tt-metalium/sub_device_types.hpp>

#include "ttnn/common/queue_id.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshEvent;
class MeshWorkload;
}  // namespace tt::tt_metal::distributed

namespace ttnn {
class Tensor;
}  // namespace ttnn

// Orders device memory accesses across streams.
//
// A stream is what one command queue runs on one sub-device, or the queue's host transfers. A queue runs its programs
// on a sub-device in order, and its transfers in order and after its programs on the stall group; but the programs it
// runs on different sub-devices, and anything two queues run, may overlap. An access to memory that an access on
// another stream may still be racing with (both touch it, one writes) needs an ordering edge: a wait for an event or a
// fence, or the host waiting for the device.
//
// Without one, the access is ordered or refused before it is enqueued:
// - Memory the allocator handed out again (another buffer where a freed one was) is ordered: the stream waits for
//   exactly the old buffer's accesses it would race with. The allocator picked the address, so the program could not.
// - The same buffer is refused under Mode::check, naming both operations, and ordered under Mode::order.
//
// Waits are fences (tt-metalium/experimental/queue_fence.hpp): a stream waits for the accesses it races with, not for
// whatever their streams enqueued after them. A wait on another queue uses the earliest event recorded there at or
// after the access, if there is one; otherwise an event is recorded when the wait is needed, and that sits behind
// everything the queue was given since.
//
// Off unless TTNN_CHECK_QUEUE_HAZARDS is 1 (check) or "order", or set_mode() turns it on.
// TTNN_QUEUE_ORDERING_LOG=<file> writes a line per launch, host transfer, ordering edge and enqueued wait.
namespace ttnn::queue_hazards {

enum class Mode { off, check, order };

Mode mode();
// Starts over: history from before is dropped.
void set_mode(Mode mode);
inline bool enabled() { return mode() != Mode::off; }

// Runs `enqueue`, which enqueues `workload` on the thread's current command queue, after ordering the workload's reads
// and writes against other streams; then records them.
void launch(
    tt::tt_metal::distributed::MeshDevice& device,
    const tt::tt_metal::distributed::MeshWorkload& workload,
    std::string_view operation,
    std::span<const Tensor* const> reads,
    std::span<const Tensor* const> writes,
    const std::function<void()>& enqueue);

// Runs `run`, a host transfer on `queue` that writes or reads `tensor`'s device memory, after ordering it; then
// records it. A blocking read also tells the host that the queue's earlier work on its stall group is done.
void transfer(
    tt::tt_metal::distributed::MeshDevice& device,
    QueueId queue,
    std::string_view what,
    const Tensor& tensor,
    bool writes,
    bool blocking,
    const std::function<void()>& run);

// Makes `queue` wait for every access to `tensor` on another stream, so what it enqueues next may read or write it.
void wait_for(const Tensor& tensor, QueueId queue);

// Collectives whose completion on any member proves that every member has started them ("full" collectives), across a
// group: a mesh axis, or the whole mesh (nullopt).
//
// Whether a full collective about to run on (`queue`, `sub_device`) across `group` may write into its peers' copies of
// `targets` without an entry handshake: true when every earlier access to them is ordered before the start of an
// earlier full collective on the same group and stream. That collective has completed on the writer before it starts
// this one, which proves every peer had started it, and so had finished those accesses. False while ordering is off,
// since nothing is known then.
bool peers_may_write_without_handshake(
    tt::tt_metal::distributed::MeshDevice& device,
    QueueId queue,
    tt::tt_metal::SubDeviceId sub_device,
    std::optional<uint32_t> group,
    std::span<const Tensor* const> targets);

// Runs `launch`, which enqueues a full collective on (`queue`, `sub_device`) across `group`, and records it. If the
// previous collective on the same links ran on another stream, this one waits for it first: collectives from two
// streams over the same fabric links at once deadlock. A mesh axis's collectives share its links; a whole-mesh
// collective shares every axis's.
void full_collective(
    tt::tt_metal::distributed::MeshDevice& device,
    QueueId queue,
    tt::tt_metal::SubDeviceId sub_device,
    std::optional<uint32_t> group,
    const std::function<void()>& launch);

// `sub_device_ids` as passed to the record: empty means the stall group.
void on_record_event(
    const tt::tt_metal::distributed::MeshEvent& event, std::span<const tt::tt_metal::SubDeviceId> sub_device_ids);
void on_wait_for_event(QueueId waiting_queue, const tt::tt_metal::distributed::MeshEvent& event);
// The host waited for `event`, so whatever any stream does from now on follows it.
void on_event_synchronize(const tt::tt_metal::distributed::MeshEvent& event);
// The host waited for `queue`, or every queue of `device`, on `sub_device_ids` (empty: the stall group).
void on_synchronize(
    tt::tt_metal::distributed::MeshDevice& device,
    std::optional<QueueId> queue,
    std::span<const tt::tt_metal::SubDeviceId> sub_device_ids);

}  // namespace ttnn::queue_hazards
