// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <vector>

#include <tt_stl/span.hpp>

#include <optional>

#include <map>

#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/experimental/sockets/h2d_socket.hpp>
#include <tt-metalium/experimental/sockets/d2h_socket.hpp>
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
// Builds persistent receiver kernels once; forward_to_tensor calls only push
// bytes into per-coord socket FIFOs (no per-call dispatch).
class H2DStreamService {
public:
    struct Config {
        // Logical shape & layout of the un-sharded source tensor.
        TensorSpec global_spec;

        // TensorToMesh describing how the global tensor is split/replicated.
        // Ownership transferred at construction. Optional: defaults to
        // replicate-on-every-mesh-dim when null; sharded distributions must
        // supply one (build via ttnn::distributed::create_mesh_mapper).
        std::unique_ptr<ttnn::distributed::TensorToMesh> mapper;

        // Socket / scratch CB sizing. All required.
        BufferType socket_buffer_type = BufferType::L1;
        uint32_t fifo_size_bytes = 0;
        uint32_t scratch_cb_size_bytes = 0;
        distributed::H2DMode socket_mode = distributed::H2DMode::DEVICE_PULL;

        // Optional worker-core sync handshake. When set, after each transfer the
        // kernel multicasts a data-ready inc to a GlobalSemaphore on these cores
        // and waits for every worker to ack via a per-coord L1 counter. The
        // CoreRange must be uniform across every participating device.
        std::optional<CoreRange> worker_cores;

        // Optional inline metadata multicast. When non-zero, every transfer ships
        // these extra bytes to a fixed L1 address (get_metadata_addr()) on every
        // worker core before flipping data_ready. Requires worker_cores; must be
        // <= the socket page size. Per-call metadata must be exactly this size.
        uint32_t metadata_size_bytes = 0;

        // Optional host-side hook applied in place to a copy of `bytes` before the
        // mapper runs (raw-bytes overload only). Must be length-preserving.
        std::function<void(ttsl::Span<std::byte> bytes,
                           ttsl::Span<const std::byte> metadata)> preprocessor;
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

    // Distributed host tensor path. `host_tensor` must be a host tensor with
    // `tensor_spec() == get_per_shard_spec()` and a populated shard at every
    // covered coord. Streams the per-coord shards through the sockets verbatim;
    // `metadata` follows the same per-call contract as the bytes overload.
    void forward_to_tensor(
        const Tensor& host_tensor,
        ttsl::Span<const std::byte> metadata = {});

    // Block until every in-flight host->socket write has been ACKed by the
    // device-side kernel.
    void barrier();

    const Tensor& get_backing_tensor() const;

    // The per-shard TensorSpec produced by the mapper; same as
    // `get_backing_tensor().tensor_spec()`.
    const TensorSpec& get_per_shard_spec() const;

    // Size accessors let bytes-only callers size their forward_to_tensor(bytes)
    // arguments without pulling TensorSpec into their compile unit.

    // Bytes the caller must hand to `forward_to_tensor(bytes[, metadata])` per
    // call — the packed size of one full global tensor.
    std::size_t payload_size_bytes() const;

    // Bytes of metadata that must be attached to each call. Zero means the
    // metadata path is disabled and the single-arg `forward_to_tensor(bytes)`
    // overload must be used.
    std::size_t metadata_size_bytes() const;

    std::vector<distributed::H2DSocket*> get_sockets() const;

    // Worker-sync handshake accessors. Only meaningful when Config::worker_cores
    // was set; each TT_FATALs otherwise.

    // Worker CoreRange the service synchronizes with (== Config::worker_cores).
    CoreRange get_worker_cores() const;

    // L1 address of the data-ready GlobalSemaphore, uniform across every worker
    // core. Workers poll it; the service kernel multicasts an inc each transfer.
    DeviceAddr get_data_ready_sem_addr() const;

    // L1 address of the consumed-counter on this coord's service core. Workers
    // atomic-inc it once per consumed iteration; the kernel polls it locally.
    // Per-coord because each device's service core may differ.
    DeviceAddr get_consumed_counter_addr(const distributed::MeshCoordinate& coord) const;

    // Logical CoreCoord of the service core on this coord's device. Combine with
    // get_consumed_counter_addr to build the NoC destination workers inc into.
    CoreCoord get_service_core(const distributed::MeshCoordinate& coord) const;

    // L1 address of the metadata destination, uniform across every worker core.
    DeviceAddr get_metadata_addr() const;

    // Export a service descriptor to /dev/shm/ so a remote process can connect()
    // and drive forward_to_tensor into the same backing tensor. Returns the
    // descriptor file path. Owner-only; TT_FATALs in connector mode.
    std::string export_descriptor(const std::string& service_id);

    // Attach to an exported H2DStreamService from another process; reconstructs
    // the mapper and every per-coord H2DSocket without acquiring a MeshDevice.
    // The returned service supports forward_to_tensor, barrier, and
    // get_per_shard_spec; owner-only methods TT_FATAL on it.
    //
    // @param service_id Identifier the owner passed to `export_descriptor`.
    // @param timeout_ms Max wait time for the descriptor file (default 10s).
    // @param preprocessor Optional process-local hook; same contract as
    //     `Config::preprocessor`.
    static std::unique_ptr<H2DStreamService> connect(
        const std::string& service_id,
        std::optional<uint32_t> timeout_ms = std::nullopt,
        std::function<void(ttsl::Span<std::byte> bytes,
                           ttsl::Span<const std::byte> metadata)> preprocessor = nullptr);

private:
    // Connector-mode ctor used by connect(): `mesh_device_` stays null; arity
    // disambiguates it from the public owner ctor.
    H2DStreamService(
        Config cfg,
        std::vector<std::unique_ptr<distributed::H2DSocket>> sockets,
        uint32_t socket_page_size,
        uint32_t num_socket_pages);

    // Flip the termination signal 0 -> 1 so each persistent receiver kernel exits
    // on its next poll. Idempotent.
    void signal_termination();

    // True for owner services (own all device-side resources), false for
    // connector services. The dtor branches on it.
    bool is_owner_ = true;

    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    Config cfg_;

    std::unique_ptr<ttnn::distributed::TensorToMesh> mapper_;
    Tensor device_tensor_;

    // Per-shard tensor spec produced by the mapper. Cached so owner and
    // connector run the same Tensor-overload validation.
    std::optional<TensorSpec> per_shard_spec_;

    std::vector<std::unique_ptr<distributed::H2DSocket>> sockets_;

    // Per-coord service core claimed via ServiceCoreManager. May differ per coord.
    std::map<distributed::MeshCoordinate, CoreCoord> service_cores_;

    // Per-coord termination signal (one uint32 in L1 on the service core),
    // written directly rather than via a GlobalSemaphore (which can't target
    // per-coord service cores).
    std::map<distributed::MeshCoordinate, DeviceAddr> termination_addrs_;

    // Optional worker-sync state; all empty when cfg_.worker_cores is unset.
    std::optional<GlobalSemaphore> data_ready_sem_;
    std::map<distributed::MeshCoordinate, DeviceAddr> consumed_addrs_;
    uint32_t num_workers_ = 0;

    // Metadata multicast state; populated only when cfg_.metadata_size_bytes > 0.
    std::shared_ptr<distributed::MeshBuffer> metadata_buffer_;
    DeviceAddr metadata_l1_addr_ = 0;

    // Path to the exported service descriptor (empty otherwise); the dtor unlinks it.
    std::string descriptor_path_;

    // Host scratch reused across calls; empty when metadata is disabled.
    std::vector<std::byte> metadata_scratch_;

    // Host scratch reused across calls; empty when no preprocessor is registered.
    std::vector<std::byte> preprocess_scratch_;

    // Held by unique_ptr so the connector (which never builds one) avoids
    // default-constructing a MeshWorkload, whose ctor would acquire the PCIe chip
    // lock and deadlock against the owner.
    std::unique_ptr<distributed::MeshWorkload> workload_;

    // Chunk plan baked into the kernels' CT args, so it must stay constant for
    // the service's lifetime.
    uint32_t socket_page_size_ = 0;
    uint32_t num_socket_pages_ = 0;
};

// Persistent device-to-host streaming service backed by a fixed device tensor.
//
// Worker cores write a fixed backing device tensor; a persistent sender kernel
// on each service core reads DRAM pages and PCIe-writes them into host FIFOs via
// D2HSocket; the host calls read_from_tensor() to drain the FIFOs.
//
// Optional worker sync (Config::worker_cores):
//   1. Service core multicasts transfer_done_sem → workers may write backing.
//   2. Workers write slices; when metadata is enabled the compile-time master
//      forwarder reads its own replicated metadata copy from worker L1 and
//      writes it to the service-core staging region after all writes (fan-in),
//      then each worker acks the service-core write_ack counter.
//   3. Service core waits for num_workers acks, streams backing (+ metadata) to
//      host; host calls read_from_tensor().
// Config::master_forwarder_core is required only when metadata_size_bytes > 0.
class D2HStreamService {
public:
    struct Config {
        TensorSpec global_spec;
        std::unique_ptr<ttnn::distributed::TensorToMesh> mapper;
        std::optional<distributed::MeshComposerConfig> composer_config;
        uint32_t fifo_size_bytes = 0;
        uint32_t scratch_cb_size_bytes = 0;
        // Worker grid that writes the backing tensor before each D2H transfer.
        std::optional<CoreRange> worker_cores;
        // Fixed metadata-forwarder core within worker_cores. Required only when
        // metadata_size_bytes > 0; omitted when metadata is disabled (no master
        // kernel — every worker writes + acks directly).
        std::optional<CoreCoord> master_forwarder_core;
        // Opt-in inline metadata shipped host-ward with each transfer (0 disables;
        // single-arg read_from_tensor is used). When > 0: requires worker_cores +
        // master_forwarder_core, >= 2 workers, and metadata_size_bytes <= the derived
        // socket_page_size (single trailing page; multi-page is a future extension).
        // The master forwarder fans the replicated per-worker metadata in to the
        // service core and the sender ships it as the trailing socket page.
        uint32_t metadata_size_bytes = 0;
    };

    D2HStreamService(const std::shared_ptr<distributed::MeshDevice>& mesh_device, Config cfg);
    ~D2HStreamService();

    D2HStreamService(const D2HStreamService&) = delete;
    D2HStreamService& operator=(const D2HStreamService&) = delete;
    D2HStreamService(D2HStreamService&&) = delete;
    D2HStreamService& operator=(D2HStreamService&&) = delete;

    void read_from_tensor(ttsl::Span<std::byte> bytes, ttsl::Span<std::byte> metadata = {});
    void read_from_tensor(Tensor& host_tensor, ttsl::Span<std::byte> metadata = {});
    // Owner-only: bump the per-device write_ack counter so the persistent sender
    // reads freshly-written backing tensor pages. Used when Config::worker_cores
    // is unset; workers ack directly (and via master metadata path) when worker
    // sync is configured.
    void notify_backing_ready();
    void barrier();

    const Tensor& get_backing_tensor() const;
    const TensorSpec& get_per_shard_spec() const;
    std::size_t payload_size_bytes() const;
    std::size_t metadata_size_bytes() const;

    std::vector<distributed::D2HSocket*> get_sockets() const;

    // Owner-only getters for wiring up the device workload (the single worker
    // kernel) that produces the backing tensor and, when enabled, the metadata.
    // The metadata fan-in uses:
    //   * get_worker_metadata_addr()       — per-worker L1 holding the replicated
    //     metadata; the master reads its own copy from here (the source).
    //   * get_metadata_input_addr(coord)   — per-coord service-core L1 the master
    //     writes the metadata to and the sender ships from (the destination).
    // The master role is a worker runtime arg; there is no inter-worker handshake.
    CoreRange get_worker_cores() const;
    CoreCoord get_master_forwarder_core() const;
    DeviceAddr get_write_ack_counter_addr(const distributed::MeshCoordinate& coord) const;
    DeviceAddr get_transfer_done_sem_addr() const;
    DeviceAddr get_worker_metadata_addr() const;
    CoreCoord get_service_core(const distributed::MeshCoordinate& coord) const;
    DeviceAddr get_metadata_input_addr(const distributed::MeshCoordinate& coord) const;
    DeviceAddr get_metadata_addr(const distributed::MeshCoordinate& coord) const;

    std::string export_descriptor(const std::string& service_id);
    static std::unique_ptr<D2HStreamService> connect(
        const std::string& service_id, std::optional<uint32_t> timeout_ms = std::nullopt);

private:
    D2HStreamService(
        Config cfg,
        std::vector<std::unique_ptr<distributed::D2HSocket>> sockets,
        uint32_t socket_page_size,
        uint32_t num_socket_pages);

    void signal_termination();

    bool is_owner_ = true;

    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    Config cfg_;

    std::unique_ptr<ttnn::distributed::TensorToMesh> mapper_;
    std::unique_ptr<ttnn::distributed::MeshToTensor> composer_;
    Tensor device_tensor_;
    std::optional<TensorSpec> per_shard_spec_;

    std::vector<std::unique_ptr<distributed::D2HSocket>> sockets_;
    std::map<distributed::MeshCoordinate, CoreCoord> service_cores_;
    std::map<distributed::MeshCoordinate, DeviceAddr> termination_addrs_;

    // Service cores also claimed under the MeshDevice id for CB validation across
    // repeated mesh opens (see ctor). Released in the destructor.
    std::vector<CoreCoord> mesh_id_claimed_cores_;

    // Worker-sync state (populated when Config::worker_cores is set).
    std::optional<GlobalSemaphore> transfer_done_sem_;                   // service core -> workers: backing unlocked
    std::map<distributed::MeshCoordinate, DeviceAddr> write_ack_addrs_;  // workers -> service core: write done
    // Metadata fan-in state (additionally populated when metadata_size_bytes > 0).
    // Per-worker L1 holding the replicated metadata (the master reads its own copy).
    std::shared_ptr<distributed::MeshBuffer> metadata_worker_buffer_;
    DeviceAddr metadata_worker_l1_addr_ = 0;
    // Per-coord service-core L1 the master fans metadata into and the sender ships from.
    std::map<distributed::MeshCoordinate, DeviceAddr> metadata_input_addrs_;
    uint32_t num_workers_ = 0;
    CoreCoord master_forwarder_core_{0, 0};

    std::string descriptor_path_;
    // Host-side scratch page: read each socket's trailing metadata page into here
    // for the cross-socket consistency check in read_from_tensor().
    std::vector<std::byte> metadata_scratch_;
    std::unique_ptr<distributed::MeshWorkload> workload_;

    uint32_t socket_page_size_ = 0;
    uint32_t num_socket_pages_ = 0;
};

}  // namespace tt::tt_metal
