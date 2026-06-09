// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/d2d_stream_service.hpp"

#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <tt-metalium/internal/service/service_core_manager.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/global_semaphore.hpp"
#include "ttnn/tensor/tensor_ops.hpp"

namespace tt::tt_metal {

namespace CMAKE_UNIQUE_NAMESPACE {

// Build a single-shard host tensor with zero-initialised data of size `spec`.
// Used purely to feed the mapper at construction time so we can extract a
// TensorTopology + per-shard spec before any user data exists. The bytes are
// never read. Mirrors the helper in socket_services.cpp (H2D).
// TODO: replace with a direct "topology from MeshMapperConfig + global shape"
Tensor make_zero_host_tensor(const TensorSpec& spec) {
    const size_t bytes = spec.compute_packed_buffer_size_bytes();
    switch (spec.data_type()) {
        case DataType::BFLOAT16:
            return Tensor::from_vector<bfloat16>(std::vector<bfloat16>(bytes / sizeof(bfloat16)), spec);
        case DataType::FLOAT32: return Tensor::from_vector<float>(std::vector<float>(bytes / sizeof(float)), spec);
        case DataType::INT32: return Tensor::from_vector<int32_t>(std::vector<int32_t>(bytes / sizeof(int32_t)), spec);
        case DataType::UINT8: return Tensor::from_vector<uint8_t>(std::vector<uint8_t>(bytes / sizeof(uint8_t)), spec);
        case DataType::UINT16:
            return Tensor::from_vector<uint16_t>(std::vector<uint16_t>(bytes / sizeof(uint16_t)), spec);
        case DataType::BFLOAT4_B:
        case DataType::BFLOAT8_B:
        case DataType::UINT32:
            return Tensor::from_vector<uint32_t>(std::vector<uint32_t>(bytes / sizeof(uint32_t)), spec);
        case DataType::INVALID: TT_THROW("D2DStreamService: invalid global_spec data type");
    }
    TT_THROW("Unreachable");
}

// Claim one service core per participating coord on `mesh`, recording the
// choice per coord. Mirrors the service-core claim in socket_services.cpp (H2D).
std::map<distributed::MeshCoordinate, CoreCoord> claim_service_cores(
    const std::shared_ptr<distributed::MeshDevice>& mesh,
    const std::vector<distributed::MeshCoordinate>& coords,
    const char* side) {
    auto& svc = tt::tt_metal::internal::ServiceCoreManager::get();
    std::map<distributed::MeshCoordinate, CoreCoord> service_cores;
    for (const auto& coord : coords) {
        auto* d = mesh->get_device(coord);
        auto claimable = svc.get_claimable_cores(d);
        TT_FATAL(
            !claimable.empty(), "D2DStreamService: no claimable {} service core on device at coord {}", side, coord);
        const CoreCoord chosen = claimable.front();
        svc.claim(d, {chosen});
        service_cores.emplace(coord, chosen);
    }
    return service_cores;
}

// Chunking plan shared by both sides. Identical to the H2D derivation in
// socket_services.cpp: pack as many whole tensor pages into a socket page as
// the scratch CB budget allows, reduced to a divisor of the page count so a
// transfer is an integer number of socket pages.
struct ChunkPlan {
    uint32_t socket_page_size;  // bytes per socket page (== pages_per_chunk * tensor_page_size)
    uint32_t num_socket_pages;  // socket pages per full transfer (== tensor_num_pages / pages_per_chunk)
    uint32_t pages_per_chunk;   // tensor pages drained per socket page
};

ChunkPlan derive_chunk_plan(uint32_t tensor_page_size, uint32_t tensor_num_pages, uint32_t scratch_cb_size_bytes) {
    TT_FATAL(tensor_page_size > 0, "D2DStreamService: tensor page size must be > 0");
    TT_FATAL(tensor_num_pages > 0, "D2DStreamService: backing tensor must have at least one page");
    TT_FATAL(
        scratch_cb_size_bytes >= tensor_page_size,
        "D2DStreamService: scratch_cb_size_bytes ({} B) must be >= tensor page size ({} B)",
        scratch_cb_size_bytes,
        tensor_page_size);

    const uint32_t max_pages_per_chunk_by_cb = scratch_cb_size_bytes / tensor_page_size;
    uint32_t pages_per_chunk = std::min(tensor_num_pages, max_pages_per_chunk_by_cb);
    while (pages_per_chunk > 1 && (tensor_num_pages % pages_per_chunk) != 0) {
        --pages_per_chunk;
    }
    TT_FATAL(
        pages_per_chunk > 0,
        "D2DStreamService: derived pages_per_chunk == 0 (tensor_page_size={}, tensor_num_pages={}, "
        "scratch_cb_size_bytes={}); the socket FIFO must hold at least one tensor page",
        tensor_page_size,
        tensor_num_pages,
        scratch_cb_size_bytes);
    return ChunkPlan{
        .socket_page_size = pages_per_chunk * tensor_page_size,
        .num_socket_pages = tensor_num_pages / pages_per_chunk,
        .pages_per_chunk = pages_per_chunk,
    };
}

// Number of cores in an inclusive CoreRange (worker-grid mcast destination count
// + sync-arithmetic target).
uint32_t core_range_size(const CoreRange& range) {
    return (range.end_coord.x - range.start_coord.x + 1) * (range.end_coord.y - range.start_coord.y + 1);
}

// Allocate one zero-initialised uint32 L1 word on a service core, recording the
// address per coord. Used for the per-coord counters and termination words that
// can't be GlobalSemaphores (service cores differ per device, and
// GlobalSemaphore's reset path requires a MeshBuffer-backed buffer).
std::map<distributed::MeshCoordinate, DeviceAddr> allocate_service_core_words(
    const std::shared_ptr<distributed::MeshDevice>& mesh,
    const std::map<distributed::MeshCoordinate, CoreCoord>& service_cores) {
    auto& svc = tt::tt_metal::internal::ServiceCoreManager::get();
    std::vector<uint32_t> zero_word{0};
    std::map<distributed::MeshCoordinate, DeviceAddr> addrs;
    for (const auto& [coord, core] : service_cores) {
        auto* d = mesh->get_device(coord);
        const DeviceAddr addr = svc.allocate_l1(d, core, sizeof(uint32_t));
        tt::tt_metal::detail::WriteToDeviceL1(d, core, static_cast<uint32_t>(addr), zero_word);
        addrs.emplace(coord, addr);
    }
    return addrs;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

// ===========================================================================
// Sender / Receiver Impl state
// ===========================================================================
//
// Everything create_pair builds for one side: the backing tensor + per-shard
// spec, the claimed service core per coord, the MeshSocket endpoint, the
// worker-sync resources (counters / semaphores / termination words), and the
// persistent MeshWorkload. The handle destructor tears all of it down.

struct D2DStreamServiceSender::Impl {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    TensorSpec per_shard_spec;
    Tensor backing_tensor;
    CoreRange worker_cores;
    std::map<distributed::MeshCoordinate, CoreCoord> service_cores;
    // Sender endpoint of the MeshSocket pair (no data buffer — sender only owns
    // the config buffer). std::optional because MeshSocket has no default ctor.
    std::optional<distributed::MeshSocket> socket;

    // Chunk plan + worker-sync resources.
    uint32_t socket_page_size = 0;
    uint32_t num_socket_pages = 0;
    uint32_t pages_per_chunk = 0;
    uint32_t num_workers = 0;
    // Per-coord service-core L1 words (service cores differ per device).
    std::map<distributed::MeshCoordinate, DeviceAddr> data_ready_counter_addrs;
    std::map<distributed::MeshCoordinate, DeviceAddr> termination_addrs;
    // Fabric-link lease. share_fabric_links mirrors Config; when true the kernel
    // holds no fabric connection until granted a turn. link_grant is a single per-
    // coord service-core L1 word forming a strict host<->kernel ping-pong:
    //   0 = service idle/done (holds no connection, links free for the model graph)
    //   1 = granted (the service's turn for exactly one transfer)
    // release_fabric_links() writes 1; the kernel writes 0 after its transfer;
    // wait_for_fabric_links() polls for 0. Reader/writer roles flip at the transfer
    // boundary, so one word is race-free (writers never overlap). Always set from
    // Config in create_pair (this member default never takes effect).
    bool share_fabric_links = true;
    std::map<distributed::MeshCoordinate, DeviceAddr> link_grant_addrs;
    // Mesh-wide GlobalSemaphore on sender_worker_cores; the service kernel
    // multicast-incs it once per drained iteration.
    std::optional<GlobalSemaphore> consumed_sem;

    // Optional inline metadata. Per-coord L1 buffer on the sender service core
    // (allocated via ServiceCoreManager, AFTER the socket reservation); the
    // designated worker writes the blob here before acking. Empty when disabled.
    uint32_t metadata_size_bytes = 0;
    std::map<distributed::MeshCoordinate, DeviceAddr> metadata_addrs;

    // Persistent sender workload, launched once at create_pair.
    std::unique_ptr<distributed::MeshWorkload> workload;
    bool launched = false;
};

struct D2DStreamServiceReceiver::Impl {
    std::shared_ptr<distributed::MeshDevice> mesh_device;
    TensorSpec per_shard_spec;
    Tensor backing_tensor;
    CoreRange worker_cores;
    std::map<distributed::MeshCoordinate, CoreCoord> service_cores;
    // Receiver endpoint of the MeshSocket pair (owns the data FIFO + config
    // buffer). std::optional because MeshSocket has no default ctor.
    std::optional<distributed::MeshSocket> socket;

    // Chunk plan + worker-sync resources.
    uint32_t socket_page_size = 0;
    uint32_t num_socket_pages = 0;
    uint32_t pages_per_chunk = 0;
    uint32_t num_workers = 0;
    // Per-coord service-core L1 words (service cores differ per device).
    std::map<distributed::MeshCoordinate, DeviceAddr> consumed_counter_addrs;
    std::map<distributed::MeshCoordinate, DeviceAddr> termination_addrs;
    // Fabric-link lease — mirror of the sender. Single per-coord link_grant word
    // (0 = idle/done, 1 = granted one drain); same ping-pong protocol. Always set
    // from Config in create_pair (this member default never takes effect).
    bool share_fabric_links = true;
    std::map<distributed::MeshCoordinate, DeviceAddr> link_grant_addrs;
    // Mesh-wide GlobalSemaphore on receiver_worker_cores; the service kernel
    // multicast-incs it after the transfer has landed.
    std::optional<GlobalSemaphore> data_ready_sem;

    // Optional inline metadata. L1 MeshBuffer sharded across receiver_worker_cores
    // (uniform address mesh-wide), mirroring H2D; the service kernel multicasts
    // the blob here on every receiver worker core. Null/0 when disabled.
    uint32_t metadata_size_bytes = 0;
    std::shared_ptr<distributed::MeshBuffer> metadata_buffer;
    DeviceAddr metadata_l1_addr = 0;

    // Persistent receiver workload, launched once at create_pair.
    std::unique_ptr<distributed::MeshWorkload> workload;
    bool launched = false;
};

// ===========================================================================
// Sender handle
// ===========================================================================

D2DStreamServiceSender::D2DStreamServiceSender(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

D2DStreamServiceSender::~D2DStreamServiceSender() {
    // Flip this side's termination word so the persistent kernel exits at its
    // next data_ready poll, drain the kernel, free service-core L1, drop the
    // socket, and release the cores. The two handles tear down independently.
    //
    // This assumes the data path has quiesced (no transfer in flight) — the
    // sender kernel only observes termination at the top of its loop, not while
    // draining a transfer. Callers Finish() their worker workloads before
    // destroying the service, which leaves both kernels parked at that point.
    try {
        if (impl_ == nullptr) {
            return;
        }
        auto& svc = tt::tt_metal::internal::ServiceCoreManager::get();
        auto* mesh = impl_->mesh_device.get();

        // 1. Signal termination + drain the persistent sender kernel.
        if (impl_->launched && mesh != nullptr) {
            std::vector<uint32_t> one_word{1};
            for (const auto& [coord, addr] : impl_->termination_addrs) {
                tt::tt_metal::detail::WriteToDeviceL1(
                    mesh->get_device(coord), impl_->service_cores.at(coord), static_cast<uint32_t>(addr), one_word);
            }
            distributed::Finish(mesh->mesh_command_queue());
            for (const auto& [coord, core] : impl_->service_cores) {
                svc.wait_done(mesh->get_device(coord), core);
            }
        }
        impl_->workload.reset();

        // 2. Free the service-core L1 words (termination + data_ready_counter).
        if (mesh != nullptr) {
            for (const auto& [coord, addr] : impl_->termination_addrs) {
                svc.deallocate_l1(mesh->get_device(coord), impl_->service_cores.at(coord), addr);
            }
            for (const auto& [coord, addr] : impl_->data_ready_counter_addrs) {
                svc.deallocate_l1(mesh->get_device(coord), impl_->service_cores.at(coord), addr);
            }
            for (const auto& [coord, addr] : impl_->link_grant_addrs) {
                svc.deallocate_l1(mesh->get_device(coord), impl_->service_cores.at(coord), addr);
            }
            for (const auto& [coord, addr] : impl_->metadata_addrs) {
                svc.deallocate_l1(mesh->get_device(coord), impl_->service_cores.at(coord), addr);
            }
        }

        // 3. Drop the socket (frees its config buffer), then release the cores.
        impl_->socket.reset();
        if (mesh != nullptr) {
            for (const auto& [coord, core] : impl_->service_cores) {
                svc.release(mesh->get_device(coord), {core});
            }
        }
    } catch (const std::exception& e) {
        log_warning(tt::LogOp, "D2DStreamServiceSender: shutdown failed: {}", e.what());
    } catch (...) {
        log_warning(tt::LogOp, "D2DStreamServiceSender: shutdown failed with unknown exception");
    }
}

const Tensor& D2DStreamServiceSender::get_backing_tensor() const { return impl_->backing_tensor; }

const TensorSpec& D2DStreamServiceSender::get_per_shard_spec() const { return impl_->per_shard_spec; }

CoreRange D2DStreamServiceSender::get_worker_cores() const { return impl_->worker_cores; }

CoreCoord D2DStreamServiceSender::get_service_core(const distributed::MeshCoordinate& coord) const {
    auto it = impl_->service_cores.find(coord);
    TT_FATAL(
        it != impl_->service_cores.end(),
        "D2DStreamServiceSender::get_service_core: no service core claimed at coord {}",
        coord);
    return it->second;
}

DeviceAddr D2DStreamServiceSender::get_data_ready_counter_addr(const distributed::MeshCoordinate& coord) const {
    auto it = impl_->data_ready_counter_addrs.find(coord);
    TT_FATAL(
        it != impl_->data_ready_counter_addrs.end(),
        "D2DStreamServiceSender::get_data_ready_counter_addr: no counter at coord {}",
        coord);
    return it->second;
}

DeviceAddr D2DStreamServiceSender::get_consumed_sem_addr() const {
    TT_FATAL(
        impl_->consumed_sem.has_value(),
        "D2DStreamServiceSender::get_consumed_sem_addr: consumed_sem not allocated (sender_worker_cores empty?)");
    return impl_->consumed_sem->address();
}

DeviceAddr D2DStreamServiceSender::get_metadata_addr(const distributed::MeshCoordinate& coord) const {
    TT_FATAL(
        impl_->metadata_size_bytes > 0,
        "D2DStreamServiceSender::get_metadata_addr: metadata not configured (Config::metadata_size_bytes == 0)");
    auto it = impl_->metadata_addrs.find(coord);
    TT_FATAL(
        it != impl_->metadata_addrs.end(),
        "D2DStreamServiceSender::get_metadata_addr: no metadata buffer at coord {}",
        coord);
    return it->second;
}

void D2DStreamServiceSender::wait_for_fabric_links() {
    TT_FATAL(
        impl_->share_fabric_links,
        "D2DStreamServiceSender::wait_for_fabric_links: service was created with share_fabric_links == false "
        "(it owns the link for its lifetime); the lease API is a no-op in that mode");
    auto* mesh = impl_->mesh_device.get();
    TT_FATAL(mesh != nullptr, "D2DStreamServiceSender::wait_for_fabric_links: mesh device is null");
    // Block until every sender service core is off the link, i.e. link_grant == 0
    // (idle/done — any granted transfer has completed and the connection closed).
    // The kernel is the only writer of 0, so this is the read half of the ping-pong.
    std::vector<uint32_t> rb;
    for (const auto& [coord, addr] : impl_->link_grant_addrs) {
        auto* d = mesh->get_device(coord);
        const CoreCoord core = impl_->service_cores.at(coord);
        do {
            rb.clear();
            tt::tt_metal::detail::ReadFromDeviceL1(d, core, static_cast<uint32_t>(addr), sizeof(uint32_t), rb);
        } while (rb.empty() || rb[0] != 0u);
    }
}

void D2DStreamServiceSender::release_fabric_links() {
    TT_FATAL(
        impl_->share_fabric_links,
        "D2DStreamServiceSender::release_fabric_links: service was created with share_fabric_links == false");
    auto* mesh = impl_->mesh_device.get();
    TT_FATAL(mesh != nullptr, "D2DStreamServiceSender::release_fabric_links: mesh device is null");
    // Grant every sender service core ONE transfer: link_grant = 1. The host is the
    // only writer of 1, and only writes it after wait_for_fabric_links() observed 0,
    // so writers never overlap. Must be called only AFTER the fabric op's CQ has
    // been Finish()-ed (unordered host write).
    std::vector<uint32_t> one_word{1};
    for (const auto& [coord, addr] : impl_->link_grant_addrs) {
        tt::tt_metal::detail::WriteToDeviceL1(
            mesh->get_device(coord), impl_->service_cores.at(coord), static_cast<uint32_t>(addr), one_word);
    }
}

// ===========================================================================
// Receiver handle
// ===========================================================================

D2DStreamServiceReceiver::D2DStreamServiceReceiver(std::unique_ptr<Impl> impl) : impl_(std::move(impl)) {}

D2DStreamServiceReceiver::~D2DStreamServiceReceiver() {
    // Mirror of the sender dtor. The receiver kernel observes termination at its
    // socket-wait poll and its consumed-counter poll (its idle states between
    // transfers), so teardown is clean once outstanding transfers have drained.
    try {
        if (impl_ == nullptr) {
            return;
        }
        auto& svc = tt::tt_metal::internal::ServiceCoreManager::get();
        auto* mesh = impl_->mesh_device.get();

        // 1. Signal termination + drain the persistent receiver kernel.
        if (impl_->launched && mesh != nullptr) {
            std::vector<uint32_t> one_word{1};
            for (const auto& [coord, addr] : impl_->termination_addrs) {
                tt::tt_metal::detail::WriteToDeviceL1(
                    mesh->get_device(coord), impl_->service_cores.at(coord), static_cast<uint32_t>(addr), one_word);
            }
            distributed::Finish(mesh->mesh_command_queue());
            for (const auto& [coord, core] : impl_->service_cores) {
                svc.wait_done(mesh->get_device(coord), core);
            }
        }
        impl_->workload.reset();

        // 2. Free the service-core L1 words (termination + consumed_counter).
        if (mesh != nullptr) {
            for (const auto& [coord, addr] : impl_->termination_addrs) {
                svc.deallocate_l1(mesh->get_device(coord), impl_->service_cores.at(coord), addr);
            }
            for (const auto& [coord, addr] : impl_->consumed_counter_addrs) {
                svc.deallocate_l1(mesh->get_device(coord), impl_->service_cores.at(coord), addr);
            }
            for (const auto& [coord, addr] : impl_->link_grant_addrs) {
                svc.deallocate_l1(mesh->get_device(coord), impl_->service_cores.at(coord), addr);
            }
        }

        // 3. Drop the socket (frees its config + data FIFO buffers), then
        //    release the cores.
        impl_->socket.reset();
        if (mesh != nullptr) {
            for (const auto& [coord, core] : impl_->service_cores) {
                svc.release(mesh->get_device(coord), {core});
            }
        }
    } catch (const std::exception& e) {
        log_warning(tt::LogOp, "D2DStreamServiceReceiver: shutdown failed: {}", e.what());
    } catch (...) {
        log_warning(tt::LogOp, "D2DStreamServiceReceiver: shutdown failed with unknown exception");
    }
}

const Tensor& D2DStreamServiceReceiver::get_backing_tensor() const { return impl_->backing_tensor; }

const TensorSpec& D2DStreamServiceReceiver::get_per_shard_spec() const { return impl_->per_shard_spec; }

CoreRange D2DStreamServiceReceiver::get_worker_cores() const { return impl_->worker_cores; }

CoreCoord D2DStreamServiceReceiver::get_service_core(const distributed::MeshCoordinate& coord) const {
    auto it = impl_->service_cores.find(coord);
    TT_FATAL(
        it != impl_->service_cores.end(),
        "D2DStreamServiceReceiver::get_service_core: no service core claimed at coord {}",
        coord);
    return it->second;
}

DeviceAddr D2DStreamServiceReceiver::get_data_ready_sem_addr() const {
    TT_FATAL(
        impl_->data_ready_sem.has_value(),
        "D2DStreamServiceReceiver::get_data_ready_sem_addr: data_ready_sem not allocated (receiver_worker_cores "
        "empty?)");
    return impl_->data_ready_sem->address();
}

DeviceAddr D2DStreamServiceReceiver::get_consumed_counter_addr(const distributed::MeshCoordinate& coord) const {
    auto it = impl_->consumed_counter_addrs.find(coord);
    TT_FATAL(
        it != impl_->consumed_counter_addrs.end(),
        "D2DStreamServiceReceiver::get_consumed_counter_addr: no counter at coord {}",
        coord);
    return it->second;
}

DeviceAddr D2DStreamServiceReceiver::get_metadata_addr() const {
    TT_FATAL(
        impl_->metadata_size_bytes > 0,
        "D2DStreamServiceReceiver::get_metadata_addr: metadata not configured (Config::metadata_size_bytes == 0)");
    return impl_->metadata_l1_addr;
}

void D2DStreamServiceReceiver::wait_for_fabric_links() {
    TT_FATAL(
        impl_->share_fabric_links,
        "D2DStreamServiceReceiver::wait_for_fabric_links: service was created with share_fabric_links == false");
    auto* mesh = impl_->mesh_device.get();
    TT_FATAL(mesh != nullptr, "D2DStreamServiceReceiver::wait_for_fabric_links: mesh device is null");
    // Mirror of the sender: block until every receiver service core is off the link
    // (link_grant == 0 — its granted drain, if any, completed and the credit-return
    // connection closed).
    std::vector<uint32_t> rb;
    for (const auto& [coord, addr] : impl_->link_grant_addrs) {
        auto* d = mesh->get_device(coord);
        const CoreCoord core = impl_->service_cores.at(coord);
        do {
            rb.clear();
            tt::tt_metal::detail::ReadFromDeviceL1(d, core, static_cast<uint32_t>(addr), sizeof(uint32_t), rb);
        } while (rb.empty() || rb[0] != 0u);
    }
}

void D2DStreamServiceReceiver::release_fabric_links() {
    TT_FATAL(
        impl_->share_fabric_links,
        "D2DStreamServiceReceiver::release_fabric_links: service was created with share_fabric_links == false");
    auto* mesh = impl_->mesh_device.get();
    TT_FATAL(mesh != nullptr, "D2DStreamServiceReceiver::release_fabric_links: mesh device is null");
    // Grant every receiver service core ONE drain (link_grant = 1). Must be called
    // only AFTER the fabric op's CQ has been Finish()-ed (unordered host write).
    std::vector<uint32_t> one_word{1};
    for (const auto& [coord, addr] : impl_->link_grant_addrs) {
        tt::tt_metal::detail::WriteToDeviceL1(
            mesh->get_device(coord), impl_->service_cores.at(coord), static_cast<uint32_t>(addr), one_word);
    }
}

// ===========================================================================
// Persistent program builders
// ===========================================================================

namespace CMAKE_UNIQUE_NAMESPACE {

// Both kernels run on the single service core, RISCV_0. CB indices are private
// to each program so the sender (scratch + headers) and receiver (headers only)
// don't need to agree.
constexpr tt::CBIndex kScratchCbIndex = tt::CBIndex::c_0;
constexpr tt::CBIndex kPacketHeaderCbIndex = tt::CBIndex::c_1;

// Per-coord worker-sync CT-arg block. `sem_addr` is the mesh-wide worker-grid
// GlobalSemaphore (consumed_sem on the sender, data_ready_sem on the receiver);
// `counter_addr` is the per-coord service-core L1 word (data_ready_counter on
// the sender, consumed_counter on the receiver). All zero when disabled.
struct WorkerSyncArgs {
    bool enabled = false;
    uint32_t sem_addr = 0;
    uint32_t counter_addr = 0;
    uint32_t mcast_noc_x_start = 0;
    uint32_t mcast_noc_y_start = 0;
    uint32_t mcast_noc_x_end = 0;
    uint32_t mcast_noc_y_end = 0;
    uint32_t num_workers = 0;
};

// Always populates the per-coord addresses, mcast bbox, and num_workers (the
// sender's data_ready_counter trigger needs counter_addr + num_workers even
// when the consumed_sem multicast is off). `enabled` only toggles the gated
// blocks: the consumed_sem multicast on the sender, the whole data_ready_sem /
// consumed_counter handshake on the receiver.
WorkerSyncArgs make_worker_sync_args(
    IDevice* device,
    const CoreRange& worker_cores,
    uint32_t num_workers,
    uint32_t sem_addr,
    uint32_t counter_addr,
    bool enabled) {
    WorkerSyncArgs ws;
    const auto start_phys = device->worker_core_from_logical_core(worker_cores.start_coord);
    const auto end_phys = device->worker_core_from_logical_core(worker_cores.end_coord);
    ws.enabled = enabled;
    ws.sem_addr = sem_addr;
    ws.counter_addr = counter_addr;
    ws.mcast_noc_x_start = static_cast<uint32_t>(start_phys.x);
    ws.mcast_noc_y_start = static_cast<uint32_t>(start_phys.y);
    ws.mcast_noc_x_end = static_cast<uint32_t>(end_phys.x);
    ws.mcast_noc_y_end = static_cast<uint32_t>(end_phys.y);
    ws.num_workers = num_workers;
    return ws;
}

// CT-arg layout must stay in sync with persistent_d2d_receiver.cpp.
Program build_receiver_program(
    const Buffer& output_buffer,
    const CoreCoord& service_core,
    uint32_t socket_config_addr,
    uint32_t termination_addr,
    const ChunkPlan& plan,
    uint32_t tensor_page_size,
    const WorkerSyncArgs& ws,
    bool metadata_enabled,
    uint32_t metadata_size_bytes,
    uint32_t metadata_l1_addr,
    bool share_fabric_links,
    uint32_t link_grant_addr,
    const tt::tt_fabric::FabricNodeId& receiver_node,
    const tt::tt_fabric::FabricNodeId& sender_node,
    uint32_t link_index) {
    auto program = CreateProgram();

    // One packet header is enough for the receiver (control-flow notify only);
    // allocate two to mirror the sender and leave headroom.
    const uint32_t header_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    auto ph_cfg = CircularBufferConfig(2 * header_size, {{kPacketHeaderCbIndex, tt::DataFormat::UInt32}})
                      .set_page_size(kPacketHeaderCbIndex, header_size);
    CreateCircularBuffer(program, service_core, ph_cfg);

    auto accessor_ct = TensorAccessorArgs(output_buffer).get_compile_time_args();

    std::vector<uint32_t> ct_args = {
        socket_config_addr,
        termination_addr,
        plan.socket_page_size,
        plan.num_socket_pages,
        plan.pages_per_chunk,
        tensor_page_size,
        static_cast<uint32_t>(output_buffer.address()),
        static_cast<uint32_t>(kPacketHeaderCbIndex),
        ws.enabled ? 1u : 0u,
        ws.sem_addr,      // data_ready_sem_addr
        ws.counter_addr,  // consumed_counter_addr
        ws.mcast_noc_x_start,
        ws.mcast_noc_y_start,
        ws.mcast_noc_x_end,
        ws.mcast_noc_y_end,
        ws.num_workers,
        metadata_enabled ? 1u : 0u,    // [16]
        metadata_size_bytes,           // [17]
        metadata_l1_addr,              // [18] receiver worker-grid L1 (uniform)
        share_fabric_links ? 1u : 0u,  // [19] fabric-link lease mode
        link_grant_addr,               // [20] service-core L1 (lease ping-pong word)
    };
    ct_args.insert(ct_args.end(), accessor_ct.begin(), accessor_ct.end());

    auto kernel = CreateKernel(
        program,
        "ttnn/core/tensor/kernels/persistent_d2d_receiver.cpp",
        service_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = ct_args,
        });

    // The receiver returns socket credits to the sender, so its fabric
    // connection runs receiver -> sender.
    std::vector<uint32_t> rt_args;
    tt::tt_fabric::append_fabric_connection_rt_args(
        receiver_node, sender_node, link_index, program, service_core, rt_args);
    SetRuntimeArgs(program, kernel, service_core, rt_args);
    return program;
}

// CT-arg layout must stay in sync with persistent_d2d_sender.cpp.
Program build_sender_program(
    const Buffer& input_buffer,
    const CoreCoord& service_core,
    uint32_t socket_config_addr,
    uint32_t termination_addr,
    const ChunkPlan& plan,
    uint32_t tensor_page_size,
    uint32_t fabric_max_payload_size,
    DataType dtype,
    const WorkerSyncArgs& ws,
    bool metadata_enabled,
    uint32_t metadata_size_bytes,
    uint32_t sender_metadata_l1_addr,
    bool share_fabric_links,
    uint32_t link_grant_addr,
    const tt::tt_fabric::FabricNodeId& sender_node,
    const tt::tt_fabric::FabricNodeId& receiver_node,
    uint32_t link_index) {
    auto program = CreateProgram();

    // Single-slot scratch CB sized to one socket page (DRAM -> CB staging).
    auto scratch_cfg =
        CircularBufferConfig(plan.socket_page_size, {{kScratchCbIndex, datatype_to_dataformat_converter(dtype)}})
            .set_page_size(kScratchCbIndex, plan.socket_page_size);
    CreateCircularBuffer(program, service_core, scratch_cfg);

    const uint32_t header_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    auto ph_cfg = CircularBufferConfig(2 * header_size, {{kPacketHeaderCbIndex, tt::DataFormat::UInt32}})
                      .set_page_size(kPacketHeaderCbIndex, header_size);
    CreateCircularBuffer(program, service_core, ph_cfg);

    auto accessor_ct = TensorAccessorArgs(input_buffer).get_compile_time_args();

    std::vector<uint32_t> ct_args = {
        socket_config_addr,
        termination_addr,
        plan.socket_page_size,
        plan.num_socket_pages,
        plan.pages_per_chunk,
        tensor_page_size,
        static_cast<uint32_t>(input_buffer.address()),
        static_cast<uint32_t>(kScratchCbIndex),
        static_cast<uint32_t>(kPacketHeaderCbIndex),
        fabric_max_payload_size,
        ws.enabled ? 1u : 0u,
        ws.counter_addr,  // data_ready_counter_addr
        ws.sem_addr,      // consumed_sem_addr
        ws.mcast_noc_x_start,
        ws.mcast_noc_y_start,
        ws.mcast_noc_x_end,
        ws.mcast_noc_y_end,
        ws.num_workers,
        metadata_enabled ? 1u : 0u,    // [18]
        metadata_size_bytes,           // [19]
        sender_metadata_l1_addr,       // [20] sender service-core L1 (per-coord)
        share_fabric_links ? 1u : 0u,  // [21] fabric-link lease mode
        link_grant_addr,               // [22] service-core L1 (lease ping-pong word)
    };
    ct_args.insert(ct_args.end(), accessor_ct.begin(), accessor_ct.end());

    auto kernel = CreateKernel(
        program,
        "ttnn/core/tensor/kernels/persistent_d2d_sender.cpp",
        service_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = ct_args,
        });

    // The sender writes bulk data downstream, so its fabric connection runs
    // sender -> receiver. The receiver NoC coords come from the socket's
    // downstream encoding at runtime, so the fabric-connection args are the only
    // runtime args.
    std::vector<uint32_t> rt_args;
    tt::tt_fabric::append_fabric_connection_rt_args(
        sender_node, receiver_node, link_index, program, service_core, rt_args);
    SetRuntimeArgs(program, kernel, service_core, rt_args);
    return program;
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

// ===========================================================================
// Factory
// ===========================================================================

std::pair<std::unique_ptr<D2DStreamServiceSender>, std::unique_ptr<D2DStreamServiceReceiver>>
D2DStreamService::create_pair(
    const std::shared_ptr<distributed::MeshDevice>& sender_mesh,
    const std::shared_ptr<distributed::MeshDevice>& receiver_mesh,
    D2DStreamConfig cfg) {
    // --- (a) validate shapes --------------------------------------------------
    TT_FATAL(sender_mesh != nullptr, "D2DStreamService: sender_mesh must not be null");
    TT_FATAL(receiver_mesh != nullptr, "D2DStreamService: receiver_mesh must not be null");
    TT_FATAL(
        sender_mesh->shape() == receiver_mesh->shape(),
        "D2DStreamService: sender_mesh shape {} must equal receiver_mesh shape {} (1:1 coord mapping)",
        sender_mesh->shape(),
        receiver_mesh->shape());

    // --- (b) validate mapper --------------------------------------------------
    TT_FATAL(cfg.mapper != nullptr, "D2DStreamService: cfg.mapper must not be null");

    // --- (c) run the mapper on a zero host tensor to derive per-shard spec +
    //         topology shared by both sides ------------------------------------
    auto mapper = std::move(cfg.mapper);
    const auto distributed_dummy = (*mapper)(CMAKE_UNIQUE_NAMESPACE::make_zero_host_tensor(cfg.global_spec));
    const auto& per_shard_spec = distributed_dummy.tensor_spec();
    const auto& topology = distributed_dummy.tensor_topology();
    const auto& coords = topology.mesh_coords();

    // --- (d) allocate sender + receiver backing tensors -----------------------
    // The per-shard spec is identical on both sides (symmetric-mapping
    // invariant); only the target mesh differs.
    Tensor sender_backing = create_device_tensor(per_shard_spec, sender_mesh.get(), topology);
    Tensor receiver_backing = create_device_tensor(per_shard_spec, receiver_mesh.get(), topology);

    // --- (e) claim one service core per coord on each side --------------------
    auto sender_service_cores = CMAKE_UNIQUE_NAMESPACE::claim_service_cores(sender_mesh, coords, "sender");
    auto receiver_service_cores = CMAKE_UNIQUE_NAMESPACE::claim_service_cores(receiver_mesh, coords, "receiver");

    log_info(tt::LogMetal, "D2DStreamService: claimed service cores");
    log_info(tt::LogMetal, "D2DStreamService: sender service cores: {}", sender_service_cores);
    log_info(tt::LogMetal, "D2DStreamService: receiver service cores: {}", receiver_service_cores);

    // If create_pair throws before the handles take ownership, release the
    // claimed service cores (and the L1 they back) so a later create_pair on the
    // same process can re-claim them — the claims live in a process-global
    // manager, so leaking them would cascade into subsequent failures. On
    // success the handle destructors own teardown; `committed` disarms the guard.
    // (Both maps are moved into the handles before commit, so on a post-commit
    // throw the guard sees empty maps and the handle destructors do the release.)
    bool committed = false;
    struct ClaimReleaseGuard {
        const std::shared_ptr<distributed::MeshDevice>& sender_mesh;
        const std::shared_ptr<distributed::MeshDevice>& receiver_mesh;
        const std::map<distributed::MeshCoordinate, CoreCoord>& sender_cores;
        const std::map<distributed::MeshCoordinate, CoreCoord>& receiver_cores;
        const bool& committed;
        ~ClaimReleaseGuard() {
            if (committed) {
                return;
            }
            auto& svc = tt::tt_metal::internal::ServiceCoreManager::get();
            for (const auto& [coord, core] : sender_cores) {
                svc.release(sender_mesh->get_device(coord), {core});
            }
            for (const auto& [coord, core] : receiver_cores) {
                svc.release(receiver_mesh->get_device(coord), {core});
            }
        }
    } claim_release_guard{sender_mesh, receiver_mesh, sender_service_cores, receiver_service_cores, committed};

    // --- (f) build the SocketConnection list + create the MeshSocket pair -----
    // One connection per PARTICIPATING coord (topology.mesh_coords(), NOT the
    // full mesh range): a stray connection on an unparticipating coord asserts
    // inside MeshSocket::create_socket_pair. Each connection wires sender coord
    // (x, y) on its service core 1:1 to receiver coord (x, y) on its service
    // core.
    std::vector<distributed::SocketConnection> connections;
    connections.reserve(coords.size());
    for (const auto& coord : coords) {
        connections.emplace_back(
            distributed::MeshCoreCoord(coord, sender_service_cores.at(coord)),
            distributed::MeshCoreCoord(coord, receiver_service_cores.at(coord)));
    }

    // socket_mem_config is forwarded verbatim (socket_storage_type, fifo_size,
    // sub-device fields). create_socket_pair allocates the sender config buffer,
    // the receiver config buffer, and the receiver data FIFO, then runs the
    // same-mesh / cross-mesh handshake.
    distributed::SocketConfig socket_config(connections, cfg.socket_mem_config);
    auto socket_pair = distributed::MeshSocket::create_socket_pair(sender_mesh, receiver_mesh, socket_config);
    auto& sender_socket = socket_pair.first;
    auto& receiver_socket = socket_pair.second;

    // The active cores on each endpoint are the claimed service cores, 1:1 per
    // coord; log them at debug level for socket setup diagnosis.
    for (const auto& active : sender_socket.get_active_cores()) {
        log_debug(
            tt::LogMetal,
            "D2DStreamService: sender socket active core coord={} logical=({}, {})",
            active.device_coord,
            active.core_coord.x,
            active.core_coord.y);
    }
    for (const auto& active : receiver_socket.get_active_cores()) {
        log_debug(
            tt::LogMetal,
            "D2DStreamService: receiver socket active core coord={} logical=({}, {})",
            active.device_coord,
            active.core_coord.x,
            active.core_coord.y);
    }

    // --- (g) derive the shared chunk plan -------------------------------------
    // V0 supports L1 socket storage only — the sender's fabric write targets the
    // receiver service core's L1 bank (get_noc_addr_from_bank_id<false>).
    TT_FATAL(
        cfg.socket_mem_config.socket_storage_type == BufferType::L1,
        "D2DStreamService: V0 supports socket_storage_type == L1 only");

    const uint32_t l1_alignment = hal::get_l1_alignment();
    const uint32_t tensor_page_size = sender_backing.buffer()->aligned_page_size();
    const uint32_t tensor_num_pages = sender_backing.buffer()->num_pages();
    TT_FATAL(
        tensor_page_size % l1_alignment == 0,
        "D2DStreamService: tensor page size {} must be L1-aligned ({}). V0 supports UINT32 ROW_MAJOR DRAM where this "
        "holds.",
        tensor_page_size,
        l1_alignment);

    // The socket FIFO doubles as the scratch budget: pack as many tensor pages
    // into a socket page as the FIFO holds. socket_page_size <= fifo_size by
    // construction, so the receiver FIFO always holds a whole socket page.
    const auto plan =
        CMAKE_UNIQUE_NAMESPACE::derive_chunk_plan(tensor_page_size, tensor_num_pages, cfg.socket_mem_config.fifo_size);

    const auto fabric_max_payload_size = static_cast<uint32_t>(
        tt::round_down(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(), static_cast<size_t>(l1_alignment)));
    TT_FATAL(fabric_max_payload_size > 0, "D2DStreamService: fabric max payload size rounded to zero");

    // Optional inline metadata ships as one trailing socket page, so the blob
    // must fit a single socket page (same single-page rule as H2D).
    const uint32_t metadata_size_bytes = cfg.metadata_size_bytes;
    const bool metadata_enabled = metadata_size_bytes > 0;
    TT_FATAL(
        !metadata_enabled || metadata_size_bytes <= plan.socket_page_size,
        "D2DStreamService: metadata_size_bytes ({}) exceeds socket_page_size ({}); metadata must fit one socket page",
        metadata_size_bytes,
        plan.socket_page_size);

    // --- (h) allocate the per-side worker-sync resources ----------------------
    const uint32_t sender_num_workers = CMAKE_UNIQUE_NAMESPACE::core_range_size(cfg.sender_worker_cores);
    const uint32_t receiver_num_workers = CMAKE_UNIQUE_NAMESPACE::core_range_size(cfg.receiver_worker_cores);

    // Reserve the MeshSocket L1 buffers' footprint in each service core's ServiceCoreManager
    // allocator BEFORE allocating any counter/termination words. The socket config buffer (both
    // sides) and the receiver-side data FIFO were placed by the device's L1 allocator, which grows
    // top-down from L1_END; the ServiceCoreManager per-core allocator also grows top-down from
    // L1_END with no awareness of them. Without this reservation the counter/termination words
    // alias the socket buffers and get clobbered by inbound payload — the receiver's
    // consumed_counter in particular lands inside the data FIFO and hangs the handshake. See
    // notes/d2d_reuse_hang_investigation.md.
    {
        auto& svc = tt::tt_metal::internal::ServiceCoreManager::get();
        // Lowest L1 address occupied by any of the given socket buffers (they all sit at the top of
        // the service core's L1); reserving from there to the top covers them all in one shot.
        const auto lowest_socket_addr = [](std::initializer_list<std::shared_ptr<distributed::MeshBuffer>> bufs) {
            DeviceAddr lo = std::numeric_limits<DeviceAddr>::max();
            for (const auto& b : bufs) {
                if (b != nullptr) {
                    lo = std::min(lo, b->address());
                }
            }
            return lo;
        };
        // Only the receiver socket owns a data FIFO; get_data_buffer() TT_FATALs on the sender.
        for (const auto& coord : coords) {
            svc.reserve_l1_to_top(
                sender_mesh->get_device(coord),
                sender_service_cores.at(coord),
                lowest_socket_addr({sender_socket.get_config_buffer()}));
            svc.reserve_l1_to_top(
                receiver_mesh->get_device(coord),
                receiver_service_cores.at(coord),
                lowest_socket_addr({receiver_socket.get_config_buffer(), receiver_socket.get_data_buffer()}));
        }
    }

    auto sender_termination_addrs =
        CMAKE_UNIQUE_NAMESPACE::allocate_service_core_words(sender_mesh, sender_service_cores);
    auto receiver_termination_addrs =
        CMAKE_UNIQUE_NAMESPACE::allocate_service_core_words(receiver_mesh, receiver_service_cores);
    auto sender_data_ready_counter_addrs =
        CMAKE_UNIQUE_NAMESPACE::allocate_service_core_words(sender_mesh, sender_service_cores);
    auto receiver_consumed_counter_addrs =
        CMAKE_UNIQUE_NAMESPACE::allocate_service_core_words(receiver_mesh, receiver_service_cores);

    // Fabric-link lease word (one per coord, zero-initialised = idle/done, allocated
    // AFTER the socket reservation so it can't alias the FIFO). Always allocated; the
    // kernel only consults it when share_fabric_links is on, so OWN mode is unchanged.
    auto sender_link_grant_addrs =
        CMAKE_UNIQUE_NAMESPACE::allocate_service_core_words(sender_mesh, sender_service_cores);
    auto receiver_link_grant_addrs =
        CMAKE_UNIQUE_NAMESPACE::allocate_service_core_words(receiver_mesh, receiver_service_cores);

    // Optional sender-side inline-metadata buffer: a per-coord L1 region on the
    // SENDER service core (not the worker grid — the service is agnostic to which
    // worker produced the blob). Allocated via ServiceCoreManager AFTER the socket
    // reservation above, so it can't alias the socket buffers. The designated
    // sender worker writes the blob here before acking; the sender service ships
    // it after the data drain (step 3).
    std::map<distributed::MeshCoordinate, DeviceAddr> sender_metadata_addrs;
    if (metadata_enabled) {
        auto& svc = tt::tt_metal::internal::ServiceCoreManager::get();
        const uint32_t aligned_md = tt::align(metadata_size_bytes, l1_alignment);
        std::vector<uint32_t> zero(aligned_md / sizeof(uint32_t), 0u);
        for (const auto& coord : coords) {
            auto* d = sender_mesh->get_device(coord);
            const CoreCoord core = sender_service_cores.at(coord);
            const DeviceAddr addr = svc.allocate_l1(d, core, aligned_md);
            tt::tt_metal::detail::WriteToDeviceL1(d, core, static_cast<uint32_t>(addr), zero);
            sender_metadata_addrs.emplace(coord, addr);
        }
    }

    // Mesh-wide worker-grid GlobalSemaphores (same L1 address on every
    // (device, worker core)), exposed to the user's worker kernels via the
    // getters.
    auto consumed_sem = ttnn::global_semaphore::create_global_semaphore(
        sender_mesh.get(), CoreRangeSet(cfg.sender_worker_cores), /*initial_value=*/0, BufferType::L1);
    auto data_ready_sem = ttnn::global_semaphore::create_global_semaphore(
        receiver_mesh.get(), CoreRangeSet(cfg.receiver_worker_cores), /*initial_value=*/0, BufferType::L1);

    // Optional receiver-side inline-metadata buffer: L1, HEIGHT_SHARDED across the
    // receiver worker grid (one shard per worker), REPLICATED across the mesh so
    // the in-core L1 address is uniform. Mirrors H2DStreamService's metadata
    // buffer (socket_services.cpp B7.6). The receiver service multicasts the blob
    // here on every receiver worker core after each transfer lands. No service-
    // core involvement, so no reserve_l1_to_top interaction.
    std::shared_ptr<distributed::MeshBuffer> receiver_metadata_buffer;
    DeviceAddr receiver_metadata_l1_addr = 0;
    if (metadata_enabled) {
        const DeviceAddr aligned_shard_size =
            tt::align(static_cast<DeviceAddr>(metadata_size_bytes), static_cast<DeviceAddr>(l1_alignment));
        distributed::DeviceLocalBufferConfig device_local = {
            .page_size = aligned_shard_size,
            .buffer_type = BufferType::L1,
            .sharding_args = BufferShardingArgs(
                ShardSpecBuffer(
                    CoreRangeSet(cfg.receiver_worker_cores),
                    {1, 1},
                    ShardOrientation::ROW_MAJOR,
                    {1, 1},
                    {receiver_num_workers, 1}),
                TensorMemoryLayout::HEIGHT_SHARDED),
            .bottom_up = std::nullopt,
            .sub_device_id = std::nullopt,
        };
        distributed::MeshBufferConfig mesh_config = distributed::ReplicatedBufferConfig{
            .size = aligned_shard_size * static_cast<DeviceAddr>(receiver_num_workers),
        };
        receiver_metadata_buffer = distributed::MeshBuffer::create(mesh_config, device_local, receiver_mesh.get());
        receiver_metadata_l1_addr = receiver_metadata_buffer->address();
    }

    // --- (i) build the persistent receiver + sender workloads -----------------
    // Worker sync is always on: the receiver multicast-incs data_ready_sem after
    // each transfer lands and waits for num_workers acks on consumed_counter;
    // the sender multicast-incs consumed_sem after each drain and gates each
    // iteration on num_workers increments of data_ready_counter. The flag is a
    // compile-time arg so a future "service-only" mode could disable it; tests
    // that drive the handshake from host (no real workers) rely on the service
    // kernels' termination poll to tear down cleanly.
    constexpr bool worker_sync_enabled = true;

    auto receiver_workload = std::make_unique<distributed::MeshWorkload>();
    auto sender_workload = std::make_unique<distributed::MeshWorkload>();

    const uint32_t sender_socket_config_addr = static_cast<uint32_t>(sender_socket.get_config_buffer()->address());
    const uint32_t receiver_socket_config_addr = static_cast<uint32_t>(receiver_socket.get_config_buffer()->address());

    for (const auto& coord : coords) {
        // ---- receiver program ----
        const Buffer* recv_buf = receiver_backing.mesh_buffer().get_device_buffer(coord);
        TT_FATAL(recv_buf != nullptr, "D2DStreamService: receiver device buffer missing for coord {}", coord);
        auto* recv_device = receiver_mesh->get_device(coord);
        const CoreCoord recv_service_core = receiver_service_cores.at(coord);

        const auto receiver_node = receiver_mesh->get_fabric_node_id(coord);
        const auto recv_upstream_sender_node =
            receiver_socket.get_fabric_node_id(distributed::SocketEndpoint::SENDER, coord);
        const auto recv_links = tt::tt_fabric::get_forwarding_link_indices(receiver_node, recv_upstream_sender_node);
        TT_FATAL(!recv_links.empty(), "D2DStreamService: no fabric link receiver->sender at coord {}", coord);

        const auto recv_ws = CMAKE_UNIQUE_NAMESPACE::make_worker_sync_args(
            recv_device,
            cfg.receiver_worker_cores,
            receiver_num_workers,
            static_cast<uint32_t>(data_ready_sem.address()),
            static_cast<uint32_t>(receiver_consumed_counter_addrs.at(coord)),
            worker_sync_enabled);

        receiver_workload->add_program(
            distributed::MeshCoordinateRange(coord),
            CMAKE_UNIQUE_NAMESPACE::build_receiver_program(
                *recv_buf,
                recv_service_core,
                receiver_socket_config_addr,
                static_cast<uint32_t>(receiver_termination_addrs.at(coord)),
                plan,
                tensor_page_size,
                recv_ws,
                metadata_enabled,
                metadata_size_bytes,
                static_cast<uint32_t>(receiver_metadata_l1_addr),
                cfg.share_fabric_links,
                static_cast<uint32_t>(receiver_link_grant_addrs.at(coord)),
                receiver_node,
                recv_upstream_sender_node,
                recv_links.front()));

        // ---- sender program ----
        const Buffer* send_buf = sender_backing.mesh_buffer().get_device_buffer(coord);
        TT_FATAL(send_buf != nullptr, "D2DStreamService: sender device buffer missing for coord {}", coord);
        auto* send_device = sender_mesh->get_device(coord);
        const CoreCoord send_service_core = sender_service_cores.at(coord);

        const auto sender_node = sender_mesh->get_fabric_node_id(coord);
        const auto send_downstream_receiver_node =
            sender_socket.get_fabric_node_id(distributed::SocketEndpoint::RECEIVER, coord);
        const auto send_links = tt::tt_fabric::get_forwarding_link_indices(sender_node, send_downstream_receiver_node);
        TT_FATAL(!send_links.empty(), "D2DStreamService: no fabric link sender->receiver at coord {}", coord);

        const auto send_ws = CMAKE_UNIQUE_NAMESPACE::make_worker_sync_args(
            send_device,
            cfg.sender_worker_cores,
            sender_num_workers,
            static_cast<uint32_t>(consumed_sem.address()),
            static_cast<uint32_t>(sender_data_ready_counter_addrs.at(coord)),
            worker_sync_enabled);

        sender_workload->add_program(
            distributed::MeshCoordinateRange(coord),
            CMAKE_UNIQUE_NAMESPACE::build_sender_program(
                *send_buf,
                send_service_core,
                sender_socket_config_addr,
                static_cast<uint32_t>(sender_termination_addrs.at(coord)),
                plan,
                tensor_page_size,
                fabric_max_payload_size,
                sender_backing.dtype(),
                send_ws,
                metadata_enabled,
                metadata_size_bytes,
                metadata_enabled ? static_cast<uint32_t>(sender_metadata_addrs.at(coord)) : 0u,
                cfg.share_fabric_links,
                static_cast<uint32_t>(sender_link_grant_addrs.at(coord)),
                sender_node,
                send_downstream_receiver_node,
                send_links.front()));
    }

    // --- (j) stash everything into the two Impls ------------------------------
    // CoreRange / TensorSpec aren't default-constructible, so build each Impl
    // via aggregate init (all fields at once).
    auto sender_impl = std::make_unique<D2DStreamServiceSender::Impl>(D2DStreamServiceSender::Impl{
        .mesh_device = sender_mesh,
        .per_shard_spec = per_shard_spec,
        .backing_tensor = sender_backing,
        .worker_cores = cfg.sender_worker_cores,
        .service_cores = std::move(sender_service_cores),
        .socket = std::move(sender_socket),
        .socket_page_size = plan.socket_page_size,
        .num_socket_pages = plan.num_socket_pages,
        .pages_per_chunk = plan.pages_per_chunk,
        .num_workers = sender_num_workers,
        .data_ready_counter_addrs = std::move(sender_data_ready_counter_addrs),
        .termination_addrs = std::move(sender_termination_addrs),
        .share_fabric_links = cfg.share_fabric_links,
        .link_grant_addrs = std::move(sender_link_grant_addrs),
        .consumed_sem = std::move(consumed_sem),
        .metadata_size_bytes = metadata_size_bytes,
        .metadata_addrs = std::move(sender_metadata_addrs),
        .workload = std::move(sender_workload),
        .launched = false,
    });

    auto receiver_impl = std::make_unique<D2DStreamServiceReceiver::Impl>(D2DStreamServiceReceiver::Impl{
        .mesh_device = receiver_mesh,
        .per_shard_spec = per_shard_spec,
        .backing_tensor = receiver_backing,
        .worker_cores = cfg.receiver_worker_cores,
        .service_cores = std::move(receiver_service_cores),
        .socket = std::move(receiver_socket),
        .socket_page_size = plan.socket_page_size,
        .num_socket_pages = plan.num_socket_pages,
        .pages_per_chunk = plan.pages_per_chunk,
        .num_workers = receiver_num_workers,
        .consumed_counter_addrs = std::move(receiver_consumed_counter_addrs),
        .termination_addrs = std::move(receiver_termination_addrs),
        .share_fabric_links = cfg.share_fabric_links,
        .link_grant_addrs = std::move(receiver_link_grant_addrs),
        .data_ready_sem = std::move(data_ready_sem),
        .metadata_size_bytes = metadata_size_bytes,
        .metadata_buffer = std::move(receiver_metadata_buffer),
        .metadata_l1_addr = receiver_metadata_l1_addr,
        .workload = std::move(receiver_workload),
        .launched = false,
    });

    auto sender_handle = std::unique_ptr<D2DStreamServiceSender>(new D2DStreamServiceSender(std::move(sender_impl)));
    auto receiver_handle =
        std::unique_ptr<D2DStreamServiceReceiver>(new D2DStreamServiceReceiver(std::move(receiver_impl)));

    // --- (k) launch the persistent kernels (non-blocking) ---------------------
    // Receiver first so it's parked on its socket wait before the sender starts
    // pushing pages.
    EnqueueMeshWorkload(
        receiver_handle->impl_->mesh_device->mesh_command_queue(),
        *receiver_handle->impl_->workload,
        /*blocking=*/false);
    receiver_handle->impl_->launched = true;
    EnqueueMeshWorkload(
        sender_handle->impl_->mesh_device->mesh_command_queue(), *sender_handle->impl_->workload, /*blocking=*/false);
    sender_handle->impl_->launched = true;

    // Ownership of the service-core claims now lives in the handles.
    committed = true;
    return {std::move(sender_handle), std::move(receiver_handle)};
}

}  // namespace tt::tt_metal
