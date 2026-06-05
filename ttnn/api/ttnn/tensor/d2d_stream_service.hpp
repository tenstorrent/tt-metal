// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_coord.hpp>

#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal {

// Persistent device-to-device streaming service backed by a fixed device tensor
// on each side and a MeshSocket running over tt-fabric. The D2D analog of
// H2DStreamService: where H2D drains a PCIe-pinned host FIFO into a single
// backing tensor, D2D drains a worker-produced backing tensor on a SENDER mesh
// into a backing tensor on a RECEIVER mesh.
//
// The data path is fully device-side after construction: one persistent kernel
// per side per participating coord is launched at create_pair and forwards over
// fabric for the lifetime of the service. Host involvement is limited to
// building the pair (create_pair) and tearing it down (the handle destructors).
// Producer/consumer worker ops synchronize with the service through the
// per-handle getters (data_ready / consumed semaphores + counters).

// Configuration for a D2DStreamService pair. The same per-shard spec & topology
// (derived from `global_spec` + `mapper`) is allocated on both the sender and
// receiver mesh; `sender_mesh->shape() == receiver_mesh->shape()` is a hard
// precondition wired 1:1 by coord.
struct D2DStreamConfig {
    // Logical shape & layout of the un-sharded global tensor. The mapper runs
    // once on a zero host tensor of this spec to derive the per-shard
    // TensorSpec + TensorTopology shared by both sides. V0 supports UINT32,
    // ROW_MAJOR, DRAM-interleaved.
    TensorSpec global_spec;

    // Required. Same mapper describes both sides. Ownership is transferred into
    // the service (move). Construct via ttnn::distributed::create_mesh_mapper.
    std::unique_ptr<ttnn::distributed::TensorToMesh> mapper;

    // Forwarded (mostly verbatim) to MeshSocket::create_socket_pair. Controls
    // socket_storage_type (L1 or DRAM for the receiver-side FIFO), fifo_size,
    // and any sub-device fields. V0 recommends L1.
    distributed::SocketMemoryConfig socket_mem_config;

    // Worker grid on the sender mesh that produces into the sender backing
    // tensor. Uniform across every participating sender device.
    CoreRange sender_worker_cores;

    CoreRange receiver_worker_cores;
};

// Sender-side handle. Owns: the sender backing tensor, one claimed service core
// per participating coord on the sender mesh, the sender MeshSocket endpoint,
// the sender-side worker-sync resources, a per-side termination semaphore, and
// the persistent sender MeshWorkload.
//
// Non-copyable and non-movable (holds a persistent MeshWorkload, claimed
// service-core slots, and a MeshSocket endpoint).
class D2DStreamServiceSender {
public:
    ~D2DStreamServiceSender();

    D2DStreamServiceSender(const D2DStreamServiceSender&) = delete;
    D2DStreamServiceSender& operator=(const D2DStreamServiceSender&) = delete;
    D2DStreamServiceSender(D2DStreamServiceSender&&) = delete;
    D2DStreamServiceSender& operator=(D2DStreamServiceSender&&) = delete;

    // Sender-side workers write per-shard slices here.
    const Tensor& get_backing_tensor() const;

    // Per-shard TensorSpec; same as get_backing_tensor().tensor_spec().
    const TensorSpec& get_per_shard_spec() const;

    // Echo of Config::sender_worker_cores.
    CoreRange get_worker_cores() const;

    // Logical CoreCoord of this coord's sender service core. Workers must
    // convert via worker_core_from_logical_core before using as a NoC target.
    CoreCoord get_service_core(const distributed::MeshCoordinate& coord) const;

    // Service-core L1 slot. Sender workers atomic-inc here once per iter. Per-
    // coord because each device's service core is independent.
    DeviceAddr get_data_ready_counter_addr(const distributed::MeshCoordinate& coord) const;

    // Worker-L1 GlobalSemaphore. Sender workers spin on the local copy after
    // each produce; the service kernel multicast-incs it once per drained iter.
    // Same address across (device, worker core).
    DeviceAddr get_consumed_sem_addr() const;

private:
    friend class D2DStreamService;

    struct Impl;
    explicit D2DStreamServiceSender(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

// Receiver-side handle. Owns: the receiver backing tensor, one claimed service
// core per participating coord on the receiver mesh, the receiver MeshSocket
// endpoint, the receiver-side worker-sync resources, a per-side termination
// semaphore, and the persistent receiver MeshWorkload.
//
// Non-copyable and non-movable (same rationale as the sender handle).
class D2DStreamServiceReceiver {
public:
    ~D2DStreamServiceReceiver();

    D2DStreamServiceReceiver(const D2DStreamServiceReceiver&) = delete;
    D2DStreamServiceReceiver& operator=(const D2DStreamServiceReceiver&) = delete;
    D2DStreamServiceReceiver(D2DStreamServiceReceiver&&) = delete;
    D2DStreamServiceReceiver& operator=(D2DStreamServiceReceiver&&) = delete;

    // Receiver-side workers read per-shard slices here.
    const Tensor& get_backing_tensor() const;

    // Per-shard TensorSpec; same as get_backing_tensor().tensor_spec().
    const TensorSpec& get_per_shard_spec() const;

    // Echo of Config::receiver_worker_cores.
    CoreRange get_worker_cores() const;

    // Logical CoreCoord of this coord's receiver service core. Workers must
    // convert via worker_core_from_logical_core before using as a NoC target.
    CoreCoord get_service_core(const distributed::MeshCoordinate& coord) const;

    // Worker-L1 GlobalSemaphore. Receiver workers spin on the local copy each
    // iter; the service kernel multicast-incs after the transfer has landed.
    // Same address across (device, worker core).
    DeviceAddr get_data_ready_sem_addr() const;

    // Service-core L1 slot. Receiver workers atomic-inc here once per iter. Per-
    // coord because each device's service core is independent.
    DeviceAddr get_consumed_counter_addr(const distributed::MeshCoordinate& coord) const;

private:
    friend class D2DStreamService;

    struct Impl;
    explicit D2DStreamServiceReceiver(std::unique_ptr<Impl> impl);

    std::unique_ptr<Impl> impl_;
};

// Factory. create_pair is the only entry point; D2DStreamService itself holds
// no state. The two returned handles can be destroyed independently in any
// order (each owns its own termination semaphore).
//
// V0 is SINGLE-HOST only: create_pair takes both MeshDevice handles, so the
// calling process must own both the sender and receiver meshes. This works for
// a single Galaxy (submeshes carved from one device, fabric routes intra-
// Galaxy) and for any single host that owns the full topology (e.g. a big mesh
// spanning Galaxies). Internally it uses MeshSocket::create_socket_pair, which
// derives fabric mesh-ids locally from both device handles.
//
// TODO: multi-host: the usual multi-Galaxy deployment is 1 host per Galaxy, so
// galaxy_N and galaxy_N+1 live in different processes/ranks and a single
// process cannot hold both handles. Supporting that requires:
//   * the per-endpoint MeshSocket(device, config) ctor with sender_rank /
//     receiver_rank + a DistributedContext, instead of create_socket_pair, and
//   * splitting this single-process factory into per-side factories joined by a
//     host-side rendezvous (one host builds the sender endpoint, the other the
//     receiver; handshake over the DistributedContext / MPI). Mirror
//     H2DStreamService::export_descriptor / connect, which already does the
//     cross-process rendezvous for H2D (over /dev/shm there; over the
//     DistributedContext here).
class D2DStreamService {
public:
    static std::pair<std::unique_ptr<D2DStreamServiceSender>, std::unique_ptr<D2DStreamServiceReceiver>> create_pair(
        const std::shared_ptr<distributed::MeshDevice>& sender_mesh,
        const std::shared_ptr<distributed::MeshDevice>& receiver_mesh,
        D2DStreamConfig cfg);
};

}  // namespace tt::tt_metal
