// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <tt_stl/span.hpp>

#include <optional>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/mesh_workload.hpp>

#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal {

// Persistent host-to-device streaming service backed by a fixed device tensor.
//
// At construction the service:
//   * takes ownership of the caller-provided `Config::mapper` (or synthesises a
//     replicate-on-every-mesh-dim default if none is supplied),
//   * runs the mapper once on a zero-filled host tensor with `Config::global_spec`
//     to obtain both the per-shard TensorSpec and the TensorTopology (which mesh
//     coords participate and how the placement looks),
//   * allocates the device tensor with that derived per-shard spec & topology,
//   * creates one H2DSocket per participating mesh coord, each targeting
//     `Config::recv_core`, and sets each socket's page size,
//   * allocates a single global termination semaphore on the union of recv cores,
//   * builds one persistent receiver Program per recv core (fixed-shape, fixed
//     output address, fixed chunking — see persistent_h2d_receiver.cpp), bundles
//     them into one MeshWorkload, and enqueues it non-blocking. The kernels then
//     run for the lifetime of the service, draining one full tensor's worth of
//     data from their socket on every outer-loop iteration.
//
// `forward_to_tensor` calls only need to push bytes into the FIFOs; no per-call
// program build, dispatch, or kernel launch.
//
// At destruction the service:
//   * `barrier()`s every socket so no host writes are still in flight,
//   * flips the termination semaphore to 1, which kicks the kernels out of their
//     socket-wait poll loops,
//   * drains the mesh CQ so the workload actually completes before we tear down
//     the sockets / device tensor.
//
// Two write paths are exposed:
//   * forward_to_tensor(span<const std::byte>) treats the bytes as the GLOBAL
//     un-sharded tensor and uses the mapper to split / replicate before
//     streaming per-shard bytes through the sockets.
//   * forward_to_tensor(const Tensor&) takes an already-distributed host tensor
//     whose spec matches the backing tensor; the per-coord shards are streamed
//     through the sockets verbatim.
class H2DStreamService {
public:
    struct Config {
        // Logical shape & layout of the un-sharded source tensor. Drives the
        // mapper input shape, the size check in the raw-bytes write path, AND
        // the per-shard device tensor's layout (the mapper preserves layout and
        // only resizes the shape).
        TensorSpec global_spec;

        // Pre-built TensorToMesh describing how the global tensor is split /
        // replicated across the mesh device. Ownership is transferred into the
        // service at construction time; the per-shard TensorSpec is derived
        // from `global_spec` + this mapper by running it once on a dummy host
        // tensor.
        //
        // Optional: if left null, defaults to replicate-on-every-mesh-dim,
        // which is the identity on a 1x1 mesh and "full tensor on every device"
        // on a larger mesh. Sharded distributions must supply a mapper
        // explicitly. Construct via
        // `ttnn::distributed::create_mesh_mapper(mesh_device, mapper_config)`.
        std::unique_ptr<ttnn::distributed::TensorToMesh> mapper;

        // Logical core on every participating mesh coord that hosts the receiver
        // kernel and the device side of the H2D socket. Same coord on every
        // device for now; per-coord overrides can be added later.
        CoreCoord recv_core{0, 0};

        // Socket / scratch CB sizing. All required.
        BufferType socket_buffer_type = BufferType::L1;
        uint32_t fifo_size_bytes = 0;
        uint32_t scratch_cb_size_bytes = 0;
        distributed::H2DMode socket_mode = distributed::H2DMode::DEVICE_PULL;
    };

    H2DStreamService(const std::shared_ptr<distributed::MeshDevice>& mesh_device, Config cfg);
    ~H2DStreamService();

    // Non-copyable and non-movable: H2DSocket itself deletes copy & implicitly
    // deletes move, and we own a vector of them.
    H2DStreamService(const H2DStreamService&) = delete;
    H2DStreamService& operator=(const H2DStreamService&) = delete;
    H2DStreamService(H2DStreamService&&) = delete;
    H2DStreamService& operator=(H2DStreamService&&) = delete;

    // Raw bytes path. `bytes` must equal `Config::global_spec.compute_packed_buffer_size_bytes()`.
    // Stubbed for now.
    void forward_to_tensor(ttsl::Span<const std::byte> bytes);

    // Distributed host tensor path. `host_tensor` must:
    //   * be a host tensor (storage_type == HOST),
    //   * have `tensor_spec() == get_per_shard_spec()` (already distributed by a
    //     mapper equivalent to the one passed via `Config::mapper`),
    //   * have a populated shard at every mesh coord this service covers.
    //
    // Streams the per-coord shards through the sockets verbatim. Returns once
    // all bytes are in the socket FIFOs; the caller must `barrier()` (or
    // destruct the service) to know the kernels have drained them.
    void forward_to_tensor(const Tensor& host_tensor);

    // Block until every in-flight host->socket write has been ACKed by the
    // device-side kernel. Call before reading the backing tensor, before
    // destruction, or any time a caller needs flow-control synchronisation.
    void barrier();

    const Tensor& get_backing_tensor() const { return device_tensor_; }

    // The per-shard TensorSpec produced by the mapper. This is the single source
    // of truth for the device tensor's per-coord spec; same as
    // `get_backing_tensor().tensor_spec()`.
    const TensorSpec& get_per_shard_spec() const { return device_tensor_.tensor_spec(); }

    std::vector<distributed::H2DSocket*> get_sockets() const;

private:
    // Flip the termination semaphore from 0 to 1, kicking every persistent
    // receiver kernel out of its socket-wait poll loop on the next iteration.
    // Idempotent — safe to call multiple times.
    void signal_termination();

    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    Config cfg_;

    std::unique_ptr<ttnn::distributed::TensorToMesh> mapper_;
    Tensor device_tensor_;
    std::vector<std::unique_ptr<distributed::H2DSocket>> sockets_;

    // Termination signal for the persistent receiver kernels. Allocated on the
    // CoreRangeSet covering every recv core. `std::optional` is required only
    // because GlobalSemaphore has no default constructor.
    std::optional<GlobalSemaphore> termination_semaphore_;

    // Persistent receiver workload — built and enqueued once in the ctor,
    // drained in the dtor after termination is signalled.
    distributed::MeshWorkload workload_;

    // Chunk plan, cached in the ctor and consumed by every `forward_to_tensor`
    // call. The same values are baked into the kernels' CT args so they must
    // stay constant for the service's lifetime.
    uint32_t socket_page_size_ = 0;
    uint32_t num_socket_pages_ = 0;
};

}  // namespace tt::tt_metal
