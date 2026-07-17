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
class D2HStreamService {
public:
    // Tuned auto worker count used when parallel_host_read is enabled and host_read_thread_count == 0.
    static constexpr uint32_t kAutoHostReadThreadCount = 32;

    struct Config {
        // Payload spec streamed device -> host. std::nullopt = metadata-only mode: no DRAM
        // backing tensor, only the metadata record is sent (metadata_size_bytes must be > 0),
        // payload_size_bytes()==0, and reads go through read_metadata() (read_from_tensor throws).
        std::optional<TensorSpec> global_spec;
        std::unique_ptr<ttnn::distributed::TensorToMesh> mapper;
        std::optional<distributed::MeshComposerConfig> composer_config;
        uint32_t fifo_size_bytes = 0;
        // Optional upper bound on the socket page size (read-coalescing granularity), in bytes. 0 =
        // auto (burst-derived default). NOT a total scratch-CB size -- the data-CB slot depth is
        // auto-sized to fill the service-core L1 regardless. The effective page may be smaller
        // (capped by L1 and by divisibility of the tensor page count).
        uint32_t max_socket_page_size_bytes = 0;
        std::optional<CoreRange> worker_cores;
        std::optional<CoreCoord> metadata_master_core;
        uint32_t metadata_size_bytes = 0;
        // Host-side drain parallelism. D2H historically used one host thread per socket per
        // read_from_tensor() call; this keeps the default parallel behavior but uses a persistent,
        // grouped worker pool to avoid per-call thread creation.
        bool parallel_host_read = true;
        // Optional explicit host worker count. Only used when parallel_host_read is enabled.
        // 0 = auto: start the tuned default worker count, clamped by num_sockets. 1 forces serial.
        // N > 1 starts up to N grouped workers, each reading a contiguous socket range.
        uint32_t host_read_thread_count = 0;
    };

    D2HStreamService(const std::shared_ptr<distributed::MeshDevice>& mesh_device, Config cfg);
    ~D2HStreamService();

    D2HStreamService(const D2HStreamService&) = delete;
    D2HStreamService& operator=(const D2HStreamService&) = delete;
    D2HStreamService(D2HStreamService&&) = delete;
    D2HStreamService& operator=(D2HStreamService&&) = delete;

    void read_metadata(ttsl::Span<std::byte> metadata);

    void read_from_tensor(ttsl::Span<std::byte> bytes, ttsl::Span<std::byte> metadata = {});
    void read_from_tensor(Tensor& host_tensor, ttsl::Span<std::byte> metadata = {});
    void notify_backing_ready();
    void barrier();

    const Tensor& get_backing_tensor() const;
    const TensorSpec& get_per_shard_spec() const;
    std::size_t payload_size_bytes() const;
    std::size_t metadata_size_bytes() const;

    // Data-CB depth (full socket-page slots) the service derived from service-core L1.
    uint32_t get_slot_count() const;

    std::vector<distributed::D2HSocket*> get_sockets() const;

    CoreRange get_worker_cores() const;
    CoreCoord get_metadata_master_core() const;
    DeviceAddr get_write_ack_counter_addr(const distributed::MeshCoordinate& coord) const;
    DeviceAddr get_data_ready_counter_addr(const distributed::MeshCoordinate& coord) const;
    DeviceAddr get_transfer_done_sem_addr() const;
    DeviceAddr get_worker_metadata_addr() const;
    CoreCoord get_service_core(const distributed::MeshCoordinate& coord) const;
    DeviceAddr get_metadata_input_addr(const distributed::MeshCoordinate& coord) const;
    DeviceAddr get_metadata_addr(const distributed::MeshCoordinate& coord) const;

    std::string export_descriptor(const std::string& service_id);
    static std::unique_ptr<D2HStreamService> connect(
        const std::string& service_id,
        std::optional<uint32_t> timeout_ms = std::nullopt,
        bool parallel_host_read = true,
        uint32_t host_read_thread_count = 0);

private:
    D2HStreamService(
        Config cfg,
        std::vector<std::unique_ptr<distributed::D2HSocket>> sockets,
        uint32_t socket_page_size,
        uint32_t num_socket_pages);

    void signal_termination();
    bool tensor_enabled() const;
    void read_metadata_from_sockets(ttsl::Span<std::byte> metadata);

    enum class HostReadJobKind { None, Payload, Stop };

    struct alignas(64) HostReadWorkerState {
        std::mutex mutex;
        std::condition_variable cv;
        HostReadJobKind job = HostReadJobKind::None;
        const std::vector<std::byte*>* payload_bases = nullptr;
        size_t socket_begin = 0;
        size_t socket_end = 0;
        bool done = true;
        std::exception_ptr error;
    };

    size_t effective_host_read_worker_count() const;
    void start_host_read_workers();
    void stop_host_read_workers();
    void read_payload_with_host_read_workers(const std::vector<std::byte*>& bases);
    void submit_host_read_job(
        size_t worker_index, HostReadJobKind job, const std::vector<std::byte*>* payload_bases = nullptr);
    void wait_host_read_jobs();
    void host_read_worker_loop(size_t worker_index);

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

    std::vector<CoreCoord> mesh_id_claimed_cores_;

    std::optional<GlobalSemaphore> transfer_done_sem_;
    std::map<distributed::MeshCoordinate, DeviceAddr> write_ack_addrs_;
    std::shared_ptr<distributed::MeshBuffer> metadata_worker_buffer_;
    DeviceAddr metadata_worker_l1_addr_ = 0;
    std::map<distributed::MeshCoordinate, DeviceAddr> metadata_input_addrs_;
    uint32_t num_workers_ = 0;
    CoreCoord metadata_master_core_{0, 0};

    std::string descriptor_path_;
    std::vector<std::byte> metadata_scratch_;
    std::unique_ptr<distributed::MeshWorkload> workload_;

    uint32_t socket_page_size_ = 0;
    uint32_t num_socket_pages_ = 0;
    uint32_t slot_count_ = 0;  // data-CB depth derived from service-core L1 (owner only)

    std::vector<std::unique_ptr<HostReadWorkerState>> host_read_worker_states_;
    std::vector<std::thread> host_read_workers_;
};
}  // namespace tt::tt_metal
