// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once

#include <cstddef>
#include <cstdint>
#include <condition_variable>
#include <exception>
#include <functional>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
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
class NamedShm;
}  // namespace tt::tt_metal::distributed

namespace tt::tt_metal::experimental {
class PinnedMemory;
}  // namespace tt::tt_metal::experimental

namespace tt::tt_metal {

// Persistent host-to-device streaming service backed by a fixed device tensor.
// Builds persistent receiver kernels once; forward_to_tensor calls only push
// bytes into per-coord socket FIFOs (no per-call dispatch).
class H2DStreamService {
public:
    // Tuned auto worker count used when parallel_host_push is enabled and host_push_thread_count == 0.
    static constexpr uint32_t kAutoHostPushThreadCount = 8;

    struct Config {
        // Logical shape & layout of the un-sharded source tensor.
        TensorSpec global_spec;

        // TensorToMesh describing how the global tensor is split/replicated.
        // Ownership transferred at construction. Optional: defaults to
        // replicate-on-every-mesh-dim when null; sharded distributions must
        // supply one (build via ttnn::distributed::create_mesh_mapper).
        std::unique_ptr<ttnn::distributed::TensorToMesh> mapper;

        // Buffer type backing the socket FIFO. The service is DEVICE_PULL-only: the data FIFO lives in
        // host pinned memory and the reader pulls each socket page over PCIe (the persistent reader has
        // no local-L1 / HOST_PUSH read path), so the socket mode is not configurable here.
        BufferType socket_buffer_type = BufferType::L1;
        // Host FIFO size in bytes. 0 = auto: the service sizes it to a few socket pages of host
        // headroom (denominated in socket pages, not tensor pages). An explicit value must be >= the
        // derived socket page size.
        uint32_t fifo_size_bytes = 0;
        // Optional upper bound on the socket page size (read-coalescing granularity), in bytes. 0 =
        // auto (burst-derived default). NOT a total scratch-CB size -- the data-CB slot depth is
        // auto-sized to fill the service-core L1 regardless. The effective page may be smaller (capped
        // by L1 and by divisibility of the tensor page count).
        uint32_t max_socket_page_size_bytes = 0;

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
        std::function<void(ttsl::Span<std::byte> bytes, ttsl::Span<const std::byte> metadata)> preprocessor;

        // Experimental host-side feeder parallelism. When enabled (and there is more than one
        // socket), the service runs a persistent pool of host worker threads; each forward_to_tensor
        // call fans the per-socket writes out to that pool and blocks until all workers finish.
        // Disabled by default.
        bool parallel_host_push = false;
        // Optional explicit host worker count. Only used when parallel_host_push is enabled. 0 = auto:
        // start the service's tuned default worker count, clamped by num_sockets. 1 forces serial.
        // N > 1 starts up to N grouped workers, each writing a contiguous socket range page-major.
        uint32_t host_push_thread_count = 0;
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
    void forward_to_tensor(ttsl::Span<const std::byte> bytes, ttsl::Span<const std::byte> metadata = {});

    // Distributed host tensor path. `host_tensor` must be a host tensor with
    // `tensor_spec() == get_per_shard_spec()` and a populated shard at every
    // covered coord. Streams the per-coord shards through the sockets verbatim;
    // `metadata` follows the same per-call contract as the bytes overload.
    void forward_to_tensor(const ttnn::Tensor& host_tensor, ttsl::Span<const std::byte> metadata = {});

    // Block until every in-flight transfer has fully landed in the backing tensor.
    // The reader acks each socket page as soon as it is staged in L1 (recycling the
    // host FIFO slot early), so the socket ack no longer implies the DRAM write is
    // done; barrier() therefore also waits on the per-coord writer DRAM-completion
    // counters (pushed to shared host pinned memory) before returning. Safe to read
    // the backing tensor afterward from the owner.
    void barrier();

    const ttnn::Tensor& get_backing_tensor() const;

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

    // Data-CB depth (full socket-page slots) the service derived from service-core L1.
    uint32_t get_slot_count() const;

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
    // @param parallel_host_push Process-local feeder choice (same as
    //     `Config::parallel_host_push`); not carried by the descriptor.
    // @param host_push_thread_count Process-local explicit worker count (same as
    //     `Config::host_push_thread_count`); not carried by the descriptor.
    static std::unique_ptr<H2DStreamService> connect(
        const std::string& service_id,
        std::optional<uint32_t> timeout_ms = std::nullopt,
        std::function<void(ttsl::Span<std::byte> bytes, ttsl::Span<const std::byte> metadata)> preprocessor = nullptr,
        bool parallel_host_push = false,
        uint32_t host_push_thread_count = 0);

private:
    // Connector-mode ctor used by connect(): `mesh_device_` stays null; arity
    // disambiguates it from the public owner ctor.
    H2DStreamService(
        Config cfg,
        std::vector<std::unique_ptr<distributed::H2DSocket>> sockets,
        uint32_t socket_page_size,
        uint32_t num_socket_pages,
        const std::string& completion_shm_name,
        uint64_t completion_shm_size,
        uint32_t completion_issued_offset,
        uint32_t completion_completed_offset,
        uint32_t completion_completed_stride);

    // Flip the termination signal 0 -> 1 so each persistent receiver kernel exits
    // on its next poll. Idempotent.
    void signal_termination();

    // Block until the reader has ACKed every in-flight socket write (data staged in
    // L1). This is the socket-only half of barrier(); the destructor uses it rather
    // than the full barrier() because wait_done() -- not the writer DRAM-completion
    // counters -- is what guarantees the DRAM scatter finished at teardown, and
    // waiting on those counters during teardown would hang if a worker-sync wait is
    // still outstanding.
    void drain_socket_acks();

    enum class HostPushJobKind { None, Payload, Metadata, Stop };

    struct alignas(64) HostPushWorkerState {
        std::mutex mutex;
        std::condition_variable cv;
        HostPushJobKind job = HostPushJobKind::None;
        const std::vector<std::byte*>* payload_bases = nullptr;
        size_t socket_begin = 0;
        size_t socket_end = 0;
        bool done = true;
        std::exception_ptr error;
    };

    size_t effective_host_push_worker_count() const;
    void start_host_push_workers();
    void stop_host_push_workers();
    void write_payload_with_host_push_workers(const std::vector<std::byte*>& bases);
    void write_metadata_with_host_push_workers();
    void submit_host_push_job(
        size_t worker_index, HostPushJobKind job, const std::vector<std::byte*>* payload_bases = nullptr);
    void wait_host_push_jobs();
    void host_push_worker_loop(size_t worker_index);

    // True for owner services (own all device-side resources), false for
    // connector services. The dtor branches on it.
    bool is_owner_ = true;

    std::shared_ptr<distributed::MeshDevice> mesh_device_;
    Config cfg_;

    std::unique_ptr<ttnn::distributed::TensorToMesh> mapper_;
    ttnn::Tensor device_tensor_;

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
    uint32_t slot_count_ = 0;  // data-CB depth derived from service-core L1 (owner only)

    // Writer DRAM-completion tracking. The owner creates one shared completion
    // region and maps it to every participating coord; connectors map the same
    // SHM through the exported descriptor. Every slot is padded to host PCIe
    // alignment (device PCIe writes have stricter target alignment than host
    // reads), so the layout is:
    //
    //   offset 0                 -> issued:       uint32, incremented once per logical forward_to_tensor
    //   offset PCIe_align*(i+1)   -> completed[i]: uint32, pushed by writer i after its DRAM commit
    //
    // i.e. both completed_offset and completed_stride are the PCIe alignment, NOT
    // sizeof(uint32_t). uint32_t modulo equality is intentional: barrier correctness
    // requires fewer than 2^32 uncompleted logical transfers outstanding.
    std::unique_ptr<distributed::NamedShm> completion_shm_;
    std::shared_ptr<uint32_t[]> completion_host_mem_;
    std::shared_ptr<experimental::PinnedMemory> completion_pinned_;
    volatile uint32_t* completion_issued_ = nullptr;
    std::vector<volatile uint32_t*> completion_counters_;
    uint64_t completion_shm_size_ = 0;
    // Assigned at construction (owner: make_completion_layout; connector: from the descriptor) to the
    // PCIe alignment; these in-class values are placeholders, not the runtime layout.
    uint32_t completion_issued_offset_ = 0;
    uint32_t completion_completed_offset_ = 0;
    uint32_t completion_completed_stride_ = 0;
    // Per-coord L1 scratch word the writer stages the count in before pushing it.
    std::map<distributed::MeshCoordinate, DeviceAddr> completion_src_addrs_;

    std::vector<std::unique_ptr<HostPushWorkerState>> host_push_worker_states_;
    std::vector<std::thread> host_push_workers_;
};
}  // namespace tt::tt_metal
