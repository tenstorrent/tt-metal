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

#include <map>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/global_semaphore.hpp>
#include <tt-metalium/hal_types.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_workload.hpp>

#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/tensor/tensor.hpp"

namespace tt::tt_metal::distributed {
class MeshDevice;
class MeshBuffer;
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
//   * claims one service core per participating device via
//     `tt::tt_metal::internal::ServiceCoreManager` — each device's persistent
//     receiver kernel, socket FIFO, and termination semaphore live on that
//     core, off the worker grid,
//   * creates one H2DSocket per mesh coord pointing at that coord's service
//     core (the socket auto-detects service cores and allocates its config /
//     data buffers from the per-core service allocator),
//   * allocates a per-device termination word (uint32) at a service-core L1
//     address via `ServiceCoreManager::allocate_l1` and zero-initialises it,
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

        // Socket / scratch CB sizing. All required.
        BufferType socket_buffer_type = BufferType::L1;
        uint32_t fifo_size_bytes = 0;
        uint32_t scratch_cb_size_bytes = 0;
        distributed::H2DMode socket_mode = distributed::H2DMode::DEVICE_PULL;

        // Optional worker-core sync handshake. When set, after each transfer
        // the persistent receiver kernel:
        //   1. Multicasts a `noc_semaphore_inc_multicast` (inc=1, num_dests =
        //      number of cores in `worker_cores`) to a GlobalSemaphore on
        //      `worker_cores` (workers poll their local copy).
        //   2. Waits for every worker to ack consumption via an atomic-inc to
        //      a per-coord L1 counter on the service core, then proceeds to
        //      drain the next transfer.
        // The CoreRange must be uniform across every participating device
        // (same logical cores, same NoC layout). When unset, the kernel
        // bypasses the sync block entirely via a `worker_sync_enabled = 0`
        // compile-time arg — no host-side allocations either.
        std::optional<CoreRange> worker_cores;

        // Optional inline metadata multicast. When non-zero, every transfer
        // (every `forward_to_tensor` call) ships an extra `metadata_size_bytes`
        // worth of caller-defined bytes appended to the data stream. The
        // service kernel multicasts those bytes to a fixed L1 address on every
        // worker core in `worker_cores` (allocated by the service as an
        // L1-sharded Buffer across the worker grid; address retrievable via
        // `get_metadata_addr()`). The metadata multicast lands BEFORE the
        // data_ready_sem flip, so when workers observe data_ready, both the
        // backing tensor (DRAM) and the metadata (L1) are valid.
        //
        // Constraints (enforced at construction):
        //   * metadata_size_bytes > 0 requires `worker_cores` to be set.
        //   * metadata_size_bytes <= derived socket_page_size (single metadata
        //     page on the wire; multi-page metadata is a future extension).
        //
        // Per-call: `forward_to_tensor`'s `metadata` arg size must equal
        // `metadata_size_bytes` exactly when enabled, empty otherwise. The
        // host transparently pads the metadata to socket_page_size before
        // pushing — the caller never sees the padding.
        uint32_t metadata_size_bytes = 0;
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
    // `metadata` is required to be exactly `Config::metadata_size_bytes` bytes
    // long when metadata is enabled, empty otherwise.
    void forward_to_tensor(
        ttsl::Span<const std::byte> bytes,
        ttsl::Span<const std::byte> metadata = {});

    // Distributed host tensor path. `host_tensor` must:
    //   * be a host tensor (storage_type == HOST),
    //   * have `tensor_spec() == get_per_shard_spec()` (already distributed by a
    //     mapper equivalent to the one passed via `Config::mapper`),
    //   * have a populated shard at every mesh coord this service covers.
    //
    // Streams the per-coord shards through the sockets verbatim. Returns once
    // all bytes are in the socket FIFOs; the caller must `barrier()` (or
    // destruct the service) to know the kernels have drained them.
    // `metadata` follows the same per-call contract as the bytes overload.
    void forward_to_tensor(
        const Tensor& host_tensor,
        ttsl::Span<const std::byte> metadata = {});

    // Block until every in-flight host->socket write has been ACKed by the
    // device-side kernel. Call before reading the backing tensor, before
    // destruction, or any time a caller needs flow-control synchronisation.
    void barrier();

    const Tensor& get_backing_tensor() const;

    // The per-shard TensorSpec produced by the mapper. This is the single source
    // of truth for the device tensor's per-coord spec; same as
    // `get_backing_tensor().tensor_spec()`.
    const TensorSpec& get_per_shard_spec() const;

    std::vector<distributed::H2DSocket*> get_sockets() const;

    // ===== Worker-sync handshake accessors =====
    // Only meaningful when `Config::worker_cores` was set. Both address getters
    // TT_FATAL if worker-sync wasn't enabled at construction.

    // L1 address of the data-ready GlobalSemaphore on every worker core in
    // Config::worker_cores. Same value across (device, worker core) by
    // mesh-wide GlobalSemaphore construction. Workers poll their local copy
    // here; the persistent service kernel multicasts atomic-inc to it after
    // each transfer.
    DeviceAddr get_data_ready_sem_addr() const;

    // L1 address of the consumed-counter on this coord's service core.
    // Workers send NoC atomic-incs here (one per consumed iteration); the
    // persistent kernel polls it locally to know when all workers have
    // acknowledged the iteration. Per-coord because each device's service
    // core is independent (possibly different cores per coord).
    DeviceAddr get_consumed_counter_addr(const distributed::MeshCoordinate& coord) const;

    // Logical CoreCoord of the service core on this coord's device. Combine
    // with `get_consumed_counter_addr` to build the NoC destination workers
    // atomic-inc into; the caller converts logical -> physical via the mesh
    // device's `worker_core_from_logical_core` at workload setup time.
    CoreCoord get_service_core(const distributed::MeshCoordinate& coord) const;

    // L1 address of the metadata destination on every worker core in
    // `Config::worker_cores`. Same address across (device, worker core) by
    // mesh-wide L1-sharded Buffer construction. Workers read their local
    // copy; the service kernel multicasts the first `metadata_size_bytes`
    // of the trailing metadata page here each transfer. TT_FATALs if
    // `Config::metadata_size_bytes` was 0.
    DeviceAddr get_metadata_addr() const;

    // ===== Cross-process attachment =====

    // Export a service descriptor to /dev/shm/ so a remote process can attach
    // via `H2DStreamService::connect(service_id, ...)` and drive
    // `forward_to_tensor` calls into the same backing tensor. The descriptor
    // bundles the per-coord socket descriptors inline — the remote does a
    // single file read and reconstructs every socket in-memory via
    // `H2DSocket::connect_from_descriptor`, avoiding any race between
    // service- and socket-level descriptor files becoming visible on disk.
    //
    // The descriptor also carries the chunk plan, the mapper config, the mesh
    // shape, the global tensor spec, the metadata size, and the socket-level
    // config so the connector reconstructs an equivalent service handle
    // without needing a `MeshDevice` of its own.
    //
    // Returns the descriptor file path. Owner-only; TT_FATALs in connector
    // mode.
    std::string export_descriptor(const std::string& service_id);

    // Attach to an exported H2DStreamService from another process.
    //
    // Waits for the descriptor file at the conventional `/dev/shm/` path
    // (`tt_h2d_stream_service_<service_id>.bin`), reads it, reconstructs the
    // mapper via the shape-only `create_mesh_mapper(mesh_shape, mapper_config)`
    // overload, and attaches every per-coord H2DSocket inline from the
    // embedded socket descriptors. NO `MeshDevice` handle is acquired — the
    // connector talks to the device only through the existing PCIeCoreWriter
    // path inside each H2DSocket.
    //
    // The returned service supports `forward_to_tensor`,
    // `forward_to_tensor_bytes`, `barrier`, and `get_per_shard_spec`. Owner-
    // only methods (e.g. `get_backing_tensor`, the worker-sync getters,
    // `export_descriptor`) TT_FATAL on the returned instance.
    //
    // @param service_id Identifier the owner passed to `export_descriptor`.
    // @param timeout_ms Max wait time for the descriptor file (default 10s).
    static std::unique_ptr<H2DStreamService> connect(
        const std::string& service_id, std::optional<uint32_t> timeout_ms = std::nullopt);

private:
    // Connector-mode ctor. Called by the static `connect()` factory after it
    // has:
    //   * read the exported service descriptor,
    //   * built a Config-equivalent payload (global_spec reconstructed from
    //     the descriptor, mapper built via the shape-only
    //     `create_mesh_mapper(mesh_shape, mapper_config)`),
    //   * attached every per-coord socket via `H2DSocket::connect_from_descriptor`.
    //
    // The ctor stitches everything together: claims ownership of `cfg.mapper`,
    // installs the connected sockets, sets the cached chunk plan, derives
    // `per_shard_spec_` by running the mapper on a zero host tensor, and
    // allocates the per-call metadata scratch buffer.
    //
    // `mesh_device_` stays null on the connector — no device handle is held.
    // Arity disambiguates this from the public owner ctor (2 args vs 4 args).
    H2DStreamService(
        Config cfg,
        std::vector<std::unique_ptr<distributed::H2DSocket>> sockets,
        uint32_t socket_page_size,
        uint32_t num_socket_pages);

    // Flip the termination semaphore from 0 to 1, kicking every persistent
    // receiver kernel out of its socket-wait poll loop on the next iteration.
    // Idempotent — safe to call multiple times.
    void signal_termination();

    // True for services constructed by the public `Config`-based ctor (the
    // process that owns the device tensor, the service-core claim, the
    // persistent kernel, and the worker-sync / metadata allocations).
    // False for services produced by the future `connect()` factory — those
    // attach to an exported descriptor via shared memory and do NOT own
    // device-side resources. The dtor branches on this flag to skip owner-only
    // teardown when the service is a connector.
    bool is_owner_ = true;

    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    Config cfg_;

    std::unique_ptr<ttnn::distributed::TensorToMesh> mapper_;
    Tensor device_tensor_;

    // Per-shard tensor spec produced by the mapper. Cached so owner and
    // connector run the same Tensor-overload validation. Populated:
    //   * Owner: from `device_tensor_.tensor_spec()` once the device tensor
    //     is allocated (B3).
    //   * Connector: by running the mapper on a zero host tensor sized to
    //     `cfg_.global_spec` — the same trick the owner ctor uses to derive
    //     the per-shard spec before allocating its device tensor (B2).
    std::optional<TensorSpec> per_shard_spec_;

    std::vector<std::unique_ptr<distributed::H2DSocket>> sockets_;

    // Per-coord service core claimed via ServiceCoreManager at construction.
    // The receiver kernel + socket FIFO + termination semaphore for that coord
    // all live on this core. Different coords may have different cores (each
    // device has its own free dispatch-column cores).
    std::map<distributed::MeshCoordinate, CoreCoord> service_cores_;

    // Per-coord termination signal for the persistent receiver kernels. One
    // uint32 in L1 per device, at an address allocated from that device's
    // service core (via ServiceCoreManager::allocate_l1). Initialised to 0
    // in the ctor (raw `WriteToDeviceL1`); flipped to 1 in
    // `signal_termination`; deallocated in the dtor. No GlobalSemaphore
    // wrapper — `GlobalSemaphore::reset_semaphore_value` requires a
    // MeshBuffer-backed AnyBuffer and crashes on the single-IDevice path we
    // need here (per-coord service cores can differ across devices, so we
    // can't use a single mesh-wide semaphore).
    std::map<distributed::MeshCoordinate, DeviceAddr> termination_addrs_;

    // Optional worker-sync state. All populated together; all empty when
    // `cfg_.worker_cores` is unset.
    //
    // * `data_ready_sem_` — single mesh-wide GlobalSemaphore on the worker
    //   CoreRangeSet derived from cfg_.worker_cores. Lives on the worker grid
    //   (BankManager-backed); same L1 address on every (device, worker core).
    //   The service kernel multicasts an atomic-inc each iteration; workers
    //   poll their local copy.
    // * `consumed_addrs_` — per-coord L1 word on the service core (allocated
    //   via ServiceCoreManager::allocate_l1, zero-init via WriteToDeviceL1).
    //   Each worker NoC-atomic-incs this once per consumed iteration; the
    //   service kernel polls it locally. Deallocated in the dtor.
    // * `num_workers_` — count of cores in cfg_.worker_cores. Uniform across
    //   the mesh by design (asserted).
    std::optional<GlobalSemaphore> data_ready_sem_;
    std::map<distributed::MeshCoordinate, DeviceAddr> consumed_addrs_;
    uint32_t num_workers_ = 0;

    // Metadata multicast state. Populated only when `cfg_.metadata_size_bytes > 0`.
    //
    // * `metadata_buffer_` — owning mesh-wide L1-sharded Buffer allocated
    //   across `cfg_.worker_cores`. REPLICATED across the mesh (every device
    //   gets its own backing allocation at the same L1 address) and
    //   HEIGHT_SHARDED across the worker_cores CoreRangeSet (every worker
    //   core gets one shard at the same in-core offset). Destruction
    //   deallocates the L1 region.
    // * `metadata_l1_addr_` — cached `metadata_buffer_->address()` so the
    //   kernel CT args and multicast destination don't need to redereference
    //   the Buffer per call. 0 when metadata is disabled.
    std::shared_ptr<distributed::MeshBuffer> metadata_buffer_;
    DeviceAddr metadata_l1_addr_ = 0;

    // Path to the exported service descriptor (set by `export_descriptor`,
    // empty otherwise). The dtor mirrors `H2DSocket::~H2DSocket`: when non-
    // empty, the file is unlinked and untracked so it does not linger in
    // `ShmResourceTracker` until process exit.
    std::string descriptor_path_;

    // Per-service host scratch buffer for the trailing metadata page. Sized
    // to socket_page_size_ at construction (when metadata is enabled), reused
    // across every forward_to_tensor call: each call copies the caller's
    // metadata into the head and pushes the whole page through every socket.
    // Empty when metadata is disabled.
    std::vector<std::byte> metadata_scratch_;

    // Persistent receiver workload — built and enqueued once in the ctor,
    // drained in the dtor after termination is signalled.
    // Persistent receiver workload. Held by unique_ptr so the connector path
    // doesn't pay for default-constructing a `MeshWorkload` at member-init
    // time — `MeshWorkloadImpl`'s ctor calls `MetalContext::instance()` to
    // size kernel/kernel-group tables, which lazy-initializes the Cluster and
    // acquires the exclusive PCIe chip lock. On the connector that collides
    // with the owner's hold of the lock and deadlocks. Owner-only construction
    // via `make_unique<MeshWorkload>()` in B8; connector leaves this null.
    std::unique_ptr<distributed::MeshWorkload> workload_;

    // Chunk plan, cached in the ctor and consumed by every `forward_to_tensor`
    // call. The same values are baked into the kernels' CT args so they must
    // stay constant for the service's lifetime.
    uint32_t socket_page_size_ = 0;
    uint32_t num_socket_pages_ = 0;
};

}  // namespace tt::tt_metal
