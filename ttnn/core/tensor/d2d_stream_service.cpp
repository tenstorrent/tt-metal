// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/d2d_stream_service.hpp"

#include <chrono>
#include <limits>
#include <map>
#include <memory>
#include <optional>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt_stl/span.hpp>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <internal/service/service_core_manager.hpp>
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

#include "stream_service_common.hpp"

namespace ttnn {

// Metal names that ADL provided when this TU lived in tt::tt_metal.
// Keep only high-frequency symbols; rare ones stay qualified at use sites.
// Do not alias `distributed` at this scope — it collides with ttnn::distributed.
using tt::tt_metal::Buffer;
using tt::tt_metal::BufferType;
using tt::tt_metal::CircularBufferConfig;
using tt::tt_metal::CoreCoord;
using tt::tt_metal::CoreRange;
using tt::tt_metal::CoreRangeSet;
using tt::tt_metal::CreateCircularBuffer;
using tt::tt_metal::CreateKernel;
using tt::tt_metal::CreateProgram;
using tt::tt_metal::DataMovementConfig;
using tt::tt_metal::DataMovementProcessor;
using tt::tt_metal::DeviceAddr;
using tt::tt_metal::GlobalSemaphore;
using tt::tt_metal::NOC;
using tt::tt_metal::Program;
using tt::tt_metal::SetRuntimeArgs;
using tt::tt_metal::TensorAccessorArgs;
using tt::tt_metal::TensorSpec;
using tt::tt_metal::TensorTopology;

namespace CMAKE_UNIQUE_NAMESPACE {
namespace distributed = tt::tt_metal::distributed;
namespace stream_service_common = tt::tt_metal::stream_service_common;

// make_zero_host_tensor / ChunkPlan / derive_chunk_plan / core_range_size are
// shared verbatim with the H2D service (socket_services.cpp); they live in
// stream_service_common.hpp so the two services can't drift on the socket
// wire-format or accepted dtypes.

using stream_service_common::ChunkPlan;
using stream_service_common::claim_service_cores;
using stream_service_common::core_range_size;
using stream_service_common::derive_chunk_plan;
using stream_service_common::make_worker_sync_args;
using stream_service_common::WorkerSyncArgs;

// Allocate one zero-initialised uint32 L1 word on a service core, recording the
// address per coord. Used for the per-coord counters and termination words that
// can't be GlobalSemaphores (service cores differ per device, and
// GlobalSemaphore's reset path requires a MeshBuffer-backed buffer).
std::map<distributed::MeshCoordinate, DeviceAddr> allocate_service_core_words(
    const std::shared_ptr<distributed::MeshDevice>& mesh,
    const std::map<distributed::MeshCoordinate, CoreCoord>& service_cores) {
    auto& svc = tt::tt_metal::internal::service_core_manager();
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

// Build the SocketConnection list: one connection per participating coord, wiring
// sender coord (x, y) on its service core 1:1 to receiver coord (x, y) on its
// service core. Both sides must build the IDENTICAL list (the MeshSocket handshake
// asserts sender_core + receiver_core match across endpoints), which is why the
// multi-host path exchanges service cores first (see exchange_service_cores).
std::vector<distributed::SocketConnection> build_connections(
    const std::vector<distributed::MeshCoordinate>& coords,
    const std::map<distributed::MeshCoordinate, CoreCoord>& sender_cores,
    const std::map<distributed::MeshCoordinate, CoreCoord>& receiver_cores) {
    std::vector<distributed::SocketConnection> connections;
    connections.reserve(coords.size());
    for (const auto& coord : coords) {
        connections.emplace_back(
            distributed::MeshCoreCoord(coord, sender_cores.at(coord)),
            distributed::MeshCoreCoord(coord, receiver_cores.at(coord)));
    }
    return connections;
}

// Per-coord base address of a backing tensor's device buffer. The sender uses the
// receiver's map as the direct fabric-write destination per coord.
std::map<distributed::MeshCoordinate, DeviceAddr> collect_backing_addrs(
    const Tensor& backing, const std::vector<distributed::MeshCoordinate>& coords) {
    std::map<distributed::MeshCoordinate, DeviceAddr> addrs;
    for (const auto& coord : coords) {
        const Buffer* buf = backing.mesh_buffer().get_device_buffer(coord);
        TT_FATAL(buf != nullptr, "D2DStreamService: backing device buffer missing for coord {}", coord);
        addrs.emplace(coord, buf->address());
    }
    return addrs;
}

// DistributedContext tag for the multi-host service-core exchange. Distinct from
// any MeshSocket descriptor-exchange tag; the exchange is fully sequential before
// the handshake, so there is no concurrent (source, tag) collision.
constexpr int kServiceCoreExchangeTag = 0x44324443;  // 'D2DC'

// Result of the pre-handshake exchange: the peer's per-coord service cores and
// per-coord backing tensor base addresses. The sender uses the receiver's backing
// addresses as the direct fabric-write destinations — Step 1 lands data straight in
// the receiver DRAM tensor (no L1 FIFO copy). The receiver ignores the sender's.
struct ExchangedEndpoint {
    std::map<distributed::MeshCoordinate, CoreCoord> service_cores;
    std::map<distributed::MeshCoordinate, DeviceAddr> backing_addrs;
};

// Multi-host pre-handshake: trade the locally-claimed service cores AND the local
// backing tensor base addresses with the peer process so each side can build the
// identical SocketConnection list and the sender can target the receiver's DRAM
// tensor directly. The payload is [tensor_page_size, num_pages, socket_page_size,
//     num_socket_pages, metadata_size_bytes, share_fabric_links,
//     x0, y0, addr0, x1, y1, addr1, ...] in `coords` order. The six fingerprint
// fields catch any peer mismatch (global_spec, mapper, fifo_size, metadata,
// share_fabric_links) before the fabric hangs silently. is_sender orders send/recv
// to avoid a symmetric-blocking deadlock.
ExchangedEndpoint exchange_service_cores(
    const std::shared_ptr<distributed::multihost::DistributedContext>& ctx,
    bool is_sender,
    distributed::multihost::Rank peer_rank,
    const std::vector<distributed::MeshCoordinate>& coords,
    const std::map<distributed::MeshCoordinate, CoreCoord>& local_cores,
    const std::map<distributed::MeshCoordinate, DeviceAddr>& local_backing_addrs,
    uint32_t tensor_page_size,
    uint32_t num_pages,
    uint32_t socket_page_size,
    uint32_t num_socket_pages,
    uint32_t metadata_size_bytes,
    bool share_fabric_links) {
    std::vector<uint32_t> out;

    // Push 6 * 3 * coords.size() arguments to out
    const uint32_t kNumOutputArgs = 6 + 3 * coords.size();
    out.reserve(kNumOutputArgs);
    out.push_back(tensor_page_size);
    out.push_back(num_pages);
    out.push_back(socket_page_size);
    out.push_back(num_socket_pages);
    out.push_back(metadata_size_bytes);
    out.push_back(share_fabric_links ? 1u : 0u);
    for (const auto& c : coords) {
        const auto core = local_cores.at(c);
        out.push_back(static_cast<uint32_t>(core.x));
        out.push_back(static_cast<uint32_t>(core.y));
        out.push_back(static_cast<uint32_t>(local_backing_addrs.at(c)));
    }
    std::vector<uint32_t> in(out.size(), 0u);

    const distributed::multihost::Tag tag{kServiceCoreExchangeTag};
    const auto as_bytes = [](std::vector<uint32_t>& v) {
        return ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(v.data()), v.size() * sizeof(uint32_t));
    };
    auto out_bytes = as_bytes(out);
    auto in_bytes = as_bytes(in);
    if (is_sender) {
        ctx->send(out_bytes, peer_rank, tag);
        ctx->recv(in_bytes, peer_rank, tag);
    } else {
        ctx->recv(in_bytes, peer_rank, tag);
        ctx->send(out_bytes, peer_rank, tag);
    }

    const uint32_t in_tensor_page_size = in[0];
    const uint32_t in_num_pages = in[1];
    const uint32_t in_socket_page_size = in[2];
    const uint32_t in_num_socket_pages = in[3];
    const uint32_t in_metadata_size_bytes = in[4];
    const bool in_share_fabric_links = static_cast<bool>(in[5]);

    const bool fp_match = in_tensor_page_size == tensor_page_size && in_num_pages == num_pages &&
                          in_socket_page_size == socket_page_size && in_num_socket_pages == num_socket_pages &&
                          in_metadata_size_bytes == metadata_size_bytes && in_share_fabric_links == share_fabric_links;
    TT_FATAL(
        fp_match,
        "D2DStreamService: peer chunk-plan mismatch — "
        "local (page_size={} num_pages={} socket_page_size={} num_socket_pages={} "
        "metadata_size_bytes={} share_fabric_links={}) vs "
        "peer (page_size={} num_pages={} socket_page_size={} num_socket_pages={} "
        "metadata_size_bytes={} share_fabric_links={}); "
        "sender and receiver must use the same global_spec, mapper, fifo_size, "
        "metadata_size_bytes, and share_fabric_links",
        tensor_page_size,
        num_pages,
        socket_page_size,
        num_socket_pages,
        metadata_size_bytes,
        share_fabric_links,
        in_tensor_page_size,
        in_num_pages,
        in_socket_page_size,
        in_num_socket_pages,
        in_metadata_size_bytes,
        in_share_fabric_links);

    ExchangedEndpoint peer;
    for (size_t i = 0; i < coords.size(); ++i) {
        peer.service_cores.emplace(coords[i], CoreCoord{in[6 + 3 * i], in[6 + 3 * i + 1]});
        peer.backing_addrs.emplace(coords[i], static_cast<DeviceAddr>(in[6 + 3 * i + 2]));
    }
    return peer;
}

// Output of running the mapper once: the per-shard spec + topology shared by both
// sides, and the participating coords (topology.mesh_coords()).
struct MapperOutput {
    TensorSpec per_shard_spec;
    TensorTopology topology;
    std::vector<distributed::MeshCoordinate> coords;
};

// Run cfg.mapper on a zero host tensor to derive the per-shard spec + topology.
// Consumes cfg.mapper (moved out). Deterministic: both sides run it independently
// on their identically-shaped local mesh and get the same result.
MapperOutput run_mapper(D2DStreamConfig& cfg) {
    TT_FATAL(cfg.mapper != nullptr, "D2DStreamService: cfg.mapper must not be null");
    auto mapper = std::move(cfg.mapper);
    // Qualify explicitly: an unqualified call would, via ADL on the
    // TensorSpec argument, also find tt::tt_metal::make_zero_host_tensor
    // from socket_service_common.hpp (same Unity translation unit) and be ambiguous.
    const auto distributed_dummy = (*mapper)(stream_service_common::make_zero_host_tensor(cfg.global_spec));
    return MapperOutput{
        .per_shard_spec = distributed_dummy.tensor_spec(),
        .topology = distributed_dummy.tensor_topology(),
        .coords = distributed_dummy.tensor_topology().mesh_coords(),
    };
}

// Chunk plan + sizing shared by both sides, derived from an allocated backing
// tensor (identical on both sides under the symmetric-mapping invariant).
struct CommonPlan {
    ChunkPlan plan;
    uint32_t tensor_page_size;
    uint32_t fabric_max_payload_size;
    uint32_t metadata_size_bytes;
    bool metadata_enabled;
    uint32_t l1_alignment;
};

CommonPlan derive_common_plan(const D2DStreamConfig& cfg, const Tensor& backing) {
    // V0 supports L1 socket storage only — the sender's fabric write targets the
    // receiver service core's L1 bank.
    TT_FATAL(
        cfg.socket_mem_config.socket_storage_type == BufferType::L1,
        "D2DStreamService: V0 supports socket_storage_type == L1 only");

    const uint32_t l1_alignment = tt::tt_metal::hal::get_l1_alignment();
    const uint32_t tensor_page_size = backing.buffer()->aligned_page_size();
    const uint32_t tensor_num_pages = backing.buffer()->num_pages();
    TT_FATAL(
        tensor_page_size % l1_alignment == 0,
        "D2DStreamService: tensor page size {} must be L1-aligned ({}). V0 supports UINT32 ROW_MAJOR DRAM where this "
        "holds.",
        tensor_page_size,
        l1_alignment);

    const auto plan = derive_chunk_plan(tensor_page_size, tensor_num_pages, cfg.socket_mem_config.fifo_size);

    const auto fabric_max_payload_size = static_cast<uint32_t>(
        tt::round_down(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(), static_cast<size_t>(l1_alignment)));
    TT_FATAL(fabric_max_payload_size > 0, "D2DStreamService: fabric max payload size rounded to zero");

    const uint32_t metadata_size_bytes = cfg.metadata_size_bytes;
    const bool metadata_enabled = metadata_size_bytes > 0;
    TT_FATAL(
        !metadata_enabled || metadata_size_bytes <= plan.socket_page_size,
        "D2DStreamService: metadata_size_bytes ({}) exceeds socket_page_size ({}); metadata must fit one socket page",
        metadata_size_bytes,
        plan.socket_page_size);

    return CommonPlan{
        .plan = plan,
        .tensor_page_size = tensor_page_size,
        .fabric_max_payload_size = fabric_max_payload_size,
        .metadata_size_bytes = metadata_size_bytes,
        .metadata_enabled = metadata_enabled,
        .l1_alignment = l1_alignment,
    };
}

// Releases a side's claimed service cores if the enclosing factory throws before
// `committed` is set. The cores map is moved into the handle's Impl on success, so
// after a successful build this guard iterates an empty map and releases nothing
// (the handle destructor then owns teardown). One instance per side.
struct ServiceCoreReleaseGuard {
    const std::shared_ptr<distributed::MeshDevice>& mesh;
    const std::map<distributed::MeshCoordinate, CoreCoord>& cores;
    const bool& committed;
    ~ServiceCoreReleaseGuard() {
        if (committed) {
            return;
        }
        auto& svc = tt::tt_metal::internal::service_core_manager();
        for (const auto& [coord, core] : cores) {
            svc.release(mesh->get_device(coord), {core});
        }
    }
};

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
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device;
    TensorSpec per_shard_spec;
    Tensor backing_tensor;
    CoreRange worker_cores;
    std::map<tt::tt_metal::distributed::MeshCoordinate, CoreCoord> service_cores;
    // Sender endpoint of the MeshSocket pair (no data buffer — sender only owns
    // the config buffer). std::optional because MeshSocket has no default ctor.
    std::optional<tt::tt_metal::distributed::MeshSocket> socket;

    // Chunk plan + worker-sync resources.
    uint32_t socket_page_size = 0;
    uint32_t num_socket_pages = 0;
    uint32_t pages_per_chunk = 0;
    uint32_t num_workers = 0;
    // Per-coord service-core L1 words (service cores differ per device).
    std::map<tt::tt_metal::distributed::MeshCoordinate, DeviceAddr> data_ready_counter_addrs;
    std::map<tt::tt_metal::distributed::MeshCoordinate, DeviceAddr> termination_addrs;
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
    std::map<tt::tt_metal::distributed::MeshCoordinate, DeviceAddr> link_grant_addrs;
    // Multi-lane sender (Step 2a) lane-sync words: master bumps go_count to release
    // the sub for a transfer; sub bumps done_count when its half is shipped. One pair
    // per coord in service-core L1; unused (but allocated) when single-lane.
    std::map<tt::tt_metal::distributed::MeshCoordinate, DeviceAddr> go_count_addrs;
    std::map<tt::tt_metal::distributed::MeshCoordinate, DeviceAddr> done_count_addrs;
    // Mesh-wide GlobalSemaphore on sender_worker_cores; the service kernel
    // multicast-incs it once per drained iteration.
    std::optional<GlobalSemaphore> consumed_sem;

    // Optional inline metadata. Per-coord L1 buffer on the sender service core
    // (allocated via ServiceCoreManager, AFTER the socket reservation); the
    // designated worker writes the blob here before acking. Empty when disabled.
    uint32_t metadata_size_bytes = 0;
    std::map<tt::tt_metal::distributed::MeshCoordinate, DeviceAddr> metadata_addrs;

    // Persistent sender workload, launched once at create_pair.
    std::unique_ptr<tt::tt_metal::distributed::MeshWorkload> workload;
    bool launched = false;

    // Cached lease workloads (LEASE mode only): release_fabric_links enqueues the RELEASE
    // (write grant=1) BEFORE the producer; wait_for_fabric_links enqueues the WAIT (spin
    // grant==0) AFTER. Both CQ-ordered — no host PCIe poke. Null in OWN mode.
    std::unique_ptr<tt::tt_metal::distributed::MeshWorkload> lease_wait_workload;
    std::unique_ptr<tt::tt_metal::distributed::MeshWorkload> lease_release_workload;
};

struct D2DStreamServiceReceiver::Impl {
    std::shared_ptr<tt::tt_metal::distributed::MeshDevice> mesh_device;
    TensorSpec per_shard_spec;
    Tensor backing_tensor;
    CoreRange worker_cores;
    std::map<tt::tt_metal::distributed::MeshCoordinate, CoreCoord> service_cores;
    // Receiver endpoint of the MeshSocket pair (owns the data FIFO + config
    // buffer). std::optional because MeshSocket has no default ctor.
    std::optional<tt::tt_metal::distributed::MeshSocket> socket;

    // Chunk plan + worker-sync resources.
    uint32_t socket_page_size = 0;
    uint32_t num_socket_pages = 0;
    uint32_t pages_per_chunk = 0;
    uint32_t num_workers = 0;
    // Per-coord service-core L1 words (service cores differ per device).
    std::map<tt::tt_metal::distributed::MeshCoordinate, DeviceAddr> consumed_counter_addrs;
    std::map<tt::tt_metal::distributed::MeshCoordinate, DeviceAddr> termination_addrs;
    // Fabric-link lease — mirror of the sender. Single per-coord link_grant word
    // (0 = idle/done, 1 = granted one drain); same ping-pong protocol. Always set
    // from Config in create_pair (this member default never takes effect).
    bool share_fabric_links = true;
    std::map<tt::tt_metal::distributed::MeshCoordinate, DeviceAddr> link_grant_addrs;
    // Mesh-wide GlobalSemaphore on receiver_worker_cores; the service kernel
    // multicast-incs it after the transfer has landed.
    std::optional<GlobalSemaphore> data_ready_sem;

    // Optional inline metadata. L1 MeshBuffer sharded across receiver_worker_cores
    // (uniform address mesh-wide), mirroring H2D; the service kernel multicasts
    // the blob here on every receiver worker core. Null/0 when disabled.
    uint32_t metadata_size_bytes = 0;
    std::shared_ptr<tt::tt_metal::distributed::MeshBuffer> metadata_buffer;
    DeviceAddr metadata_l1_addr = 0;

    // Persistent receiver workload, launched once at create_pair.
    std::unique_ptr<tt::tt_metal::distributed::MeshWorkload> workload;
    bool launched = false;

    // Cached lease workloads (LEASE mode only), mirror of the sender's. Null in OWN mode.
    std::unique_ptr<tt::tt_metal::distributed::MeshWorkload> lease_wait_workload;
    std::unique_ptr<tt::tt_metal::distributed::MeshWorkload> lease_release_workload;
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
        auto& svc = tt::tt_metal::internal::service_core_manager();
        auto* mesh = impl_->mesh_device.get();

        // 1. Signal termination + drain the persistent sender kernel.
        if (impl_->launched && mesh != nullptr) {
            std::vector<uint32_t> one_word{1};
            for (const auto& [coord, addr] : impl_->termination_addrs) {
                tt::tt_metal::detail::WriteToDeviceL1(
                    mesh->get_device(coord), impl_->service_cores.at(coord), static_cast<uint32_t>(addr), one_word);
            }
            tt::tt_metal::distributed::Finish(mesh->mesh_command_queue());
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
            for (const auto& [coord, addr] : impl_->go_count_addrs) {
                svc.deallocate_l1(mesh->get_device(coord), impl_->service_cores.at(coord), addr);
            }
            for (const auto& [coord, addr] : impl_->done_count_addrs) {
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

CoreCoord D2DStreamServiceSender::get_service_core(const tt::tt_metal::distributed::MeshCoordinate& coord) const {
    auto it = impl_->service_cores.find(coord);
    TT_FATAL(
        it != impl_->service_cores.end(),
        "D2DStreamServiceSender::get_service_core: no service core claimed at coord {}",
        coord);
    return it->second;
}

DeviceAddr D2DStreamServiceSender::get_data_ready_counter_addr(
    const tt::tt_metal::distributed::MeshCoordinate& coord) const {
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

DeviceAddr D2DStreamServiceSender::get_metadata_addr(const tt::tt_metal::distributed::MeshCoordinate& coord) const {
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
    TT_FATAL(
        impl_->lease_wait_workload != nullptr,
        "D2DStreamServiceSender::wait_for_fabric_links: lease wait workload not built");
    // Enqueue a kernel that spins until every sender service core is off the link
    // (link_grant == 0 — any granted transfer finished and the connection closed).
    // CQ-ordered AFTER the producer, so whatever the caller enqueues next is fenced until
    // the link is free. No host Finish: CQ order already provides that ordering, and
    // blocking the host here would be redundant. Non-blocking.
    EnqueueMeshWorkload(mesh->mesh_command_queue(), *impl_->lease_wait_workload, /*blocking=*/false);
}

void D2DStreamServiceSender::release_fabric_links() {
    TT_FATAL(
        impl_->share_fabric_links,
        "D2DStreamServiceSender::release_fabric_links: service was created with share_fabric_links == false");
    auto* mesh = impl_->mesh_device.get();
    TT_FATAL(mesh != nullptr, "D2DStreamServiceSender::release_fabric_links: mesh device is null");
    TT_FATAL(
        impl_->lease_release_workload != nullptr,
        "D2DStreamServiceSender::release_fabric_links: lease release workload not built");
    // Grant every sender service core ONE transfer (link_grant = 1) via a CQ-enqueued
    // kernel. Enqueue this BEFORE the producer workload so the grant is already in place
    // when the producer runs — the service then completes its handshake with the
    // producer in the same window. CQ order keeps the grant (1, here) and the service's
    // own reset (0, on its hot path) strictly alternating. Non-blocking.
    EnqueueMeshWorkload(mesh->mesh_command_queue(), *impl_->lease_release_workload, /*blocking=*/false);
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
        auto& svc = tt::tt_metal::internal::service_core_manager();
        auto* mesh = impl_->mesh_device.get();

        // 1. Signal termination + drain the persistent receiver kernel.
        if (impl_->launched && mesh != nullptr) {
            std::vector<uint32_t> one_word{1};
            for (const auto& [coord, addr] : impl_->termination_addrs) {
                tt::tt_metal::detail::WriteToDeviceL1(
                    mesh->get_device(coord), impl_->service_cores.at(coord), static_cast<uint32_t>(addr), one_word);
            }
            tt::tt_metal::distributed::Finish(mesh->mesh_command_queue());
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

CoreCoord D2DStreamServiceReceiver::get_service_core(const tt::tt_metal::distributed::MeshCoordinate& coord) const {
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

DeviceAddr D2DStreamServiceReceiver::get_consumed_counter_addr(
    const tt::tt_metal::distributed::MeshCoordinate& coord) const {
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
    TT_FATAL(
        impl_->lease_wait_workload != nullptr,
        "D2DStreamServiceReceiver::wait_for_fabric_links: lease wait workload not built");
    // Mirror of the sender: CQ-enqueue a kernel that spins until every receiver service
    // core is off the link (link_grant == 0). CQ-ordered, no host Finish. Non-blocking.
    EnqueueMeshWorkload(mesh->mesh_command_queue(), *impl_->lease_wait_workload, /*blocking=*/false);
}

void D2DStreamServiceReceiver::release_fabric_links() {
    TT_FATAL(
        impl_->share_fabric_links,
        "D2DStreamServiceReceiver::release_fabric_links: service was created with share_fabric_links == false");
    auto* mesh = impl_->mesh_device.get();
    TT_FATAL(mesh != nullptr, "D2DStreamServiceReceiver::release_fabric_links: mesh device is null");
    TT_FATAL(
        impl_->lease_release_workload != nullptr,
        "D2DStreamServiceReceiver::release_fabric_links: lease release workload not built");
    // Mirror of the sender: CQ-enqueue the grant (link_grant = 1) BEFORE the consumer
    // workload so the receiver service can drain + complete its handshake in the same
    // window. Non-blocking.
    EnqueueMeshWorkload(mesh->mesh_command_queue(), *impl_->lease_release_workload, /*blocking=*/false);
}

// ===========================================================================
// Persistent program builders
// ===========================================================================

namespace CMAKE_UNIQUE_NAMESPACE {
namespace distributed = tt::tt_metal::distributed;

// Both kernels run on the single service core, RISCV_0. CB indices are private
// to each program so the sender (scratch + headers) and receiver (headers only)
// don't need to agree.
constexpr tt::CBIndex kPacketHeaderCbIndex = tt::CBIndex::c_1;

// Sender read-pipeline trid-ring depth (power of 2): each lane stages this many tensor
// pages and keeps that many DRAM reads in flight while fabric-writing. Decoupled from
// the (vestigial) socket chunk plan so the pipeline runs at full depth regardless of
// pages_per_chunk. Sizes the per-lane scratch CB (kSenderReadRingSlots * tensor page);
// two lanes consume 2*kSenderReadRingSlots trids, which must fit the 0..15 id space.
constexpr uint32_t kSenderReadRingSlots = 4;

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
    uint32_t num_lanes,
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
        ws.data_ready_sem_addr,  // data_ready_sem_addr
        ws.counter_addr,         // consumed_counter_addr
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
        num_lanes,                     // [21] sender lanes = data-landed incs awaited per transfer
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
//
// Builds num_lanes sender kernels into ONE program on the service core, one RISC +
// one fabric link per lane: lane 0 = master (RISCV_0/NOC_0, scratch c_0 / header c_1),
// lane 1 = sub (RISCV_1/NOC_1, scratch c_2 / header c_3). Distinct NoCs keep the
// per-lane trid rings independent; distinct links keep the EDM flow-control
// independent. The lanes split the tensor pages into contiguous halves (lane 0 takes
// the odd page) and sync over the shared go/done L1 words. num_lanes == 1 builds only
// the master, streaming the whole tensor. The SAME full CT layout goes to both
// kernels; per-lane fields differ.
Program build_sender_program(
    const Buffer& input_buffer,
    const CoreCoord& service_core,
    uint32_t socket_config_addr,
    uint32_t termination_addr,
    const ChunkPlan& plan,
    uint32_t tensor_page_size,
    uint32_t fabric_max_payload_size,
    tt::tt_metal::DataType dtype,
    const WorkerSyncArgs& ws,
    bool metadata_enabled,
    uint32_t metadata_size_bytes,
    uint32_t sender_metadata_l1_addr,
    bool share_fabric_links,
    uint32_t link_grant_addr,
    uint32_t receiver_tensor_addr,
    uint32_t go_count_addr,
    uint32_t done_count_addr,
    uint32_t num_lanes,
    const tt::tt_fabric::FabricNodeId& sender_node,
    const tt::tt_fabric::FabricNodeId& receiver_node,
    const std::vector<uint32_t>& link_indices) {
    auto program = CreateProgram();

    const uint32_t total_pages = plan.num_socket_pages * plan.pages_per_chunk;
    const uint32_t half = (total_pages + 1) / 2;  // lane 0 takes the odd page
    const uint32_t header_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    auto accessor_ct = TensorAccessorArgs(input_buffer).get_compile_time_args();

    // Per-lane CB indices, RISC, and NoC.
    const tt::CBIndex scratch_cbs[2] = {tt::CBIndex::c_0, tt::CBIndex::c_2};
    const tt::CBIndex header_cbs[2] = {tt::CBIndex::c_1, tt::CBIndex::c_3};
    const DataMovementProcessor procs[2] = {DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1};
    const NOC nocs[2] = {NOC::RISCV_0_default, NOC::RISCV_1_default};

    for (uint32_t lane = 0; lane < num_lanes; ++lane) {
        const tt::CBIndex scratch_cb = scratch_cbs[lane];
        const tt::CBIndex header_cb = header_cbs[lane];

        // This lane's scratch CB: the trid-ring staging buffer, sized to the ring
        // (kSenderReadRingSlots tensor pages) — NOT the socket page. The kernel indexes
        // it manually by slot, so the whole CB is one allocation page.
        const uint32_t scratch_cb_bytes = kSenderReadRingSlots * tensor_page_size;
        auto scratch_cfg =
            CircularBufferConfig(scratch_cb_bytes, {{scratch_cb, datatype_to_dataformat_converter(dtype)}})
                .set_page_size(scratch_cb, scratch_cb_bytes);
        CreateCircularBuffer(program, service_core, scratch_cfg);
        auto ph_cfg = CircularBufferConfig(2 * header_size, {{header_cb, tt::DataFormat::UInt32}})
                          .set_page_size(header_cb, header_size);
        CreateCircularBuffer(program, service_core, ph_cfg);

        const uint32_t lane_start = lane * half;
        const uint32_t lane_end = (lane + 1 == num_lanes) ? total_pages : (lane + 1) * half;

        std::vector<uint32_t> ct_args = {
            socket_config_addr,
            termination_addr,
            plan.socket_page_size,
            plan.num_socket_pages,
            plan.pages_per_chunk,
            tensor_page_size,
            static_cast<uint32_t>(input_buffer.address()),
            static_cast<uint32_t>(scratch_cb),
            static_cast<uint32_t>(header_cb),
            fabric_max_payload_size,
            ws.enabled ? 1u : 0u,
            ws.counter_addr,         // data_ready_counter_addr
            ws.data_ready_sem_addr,  // consumed_sem_addr
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
            lane == 0 ? 1u : 0u,           // [23] is_master
            num_lanes,                     // [24]
            lane_start,                    // [25]
            lane_end,                      // [26]
            go_count_addr,                 // [27] shared L1: master -> sub
            done_count_addr,               // [28] shared L1: sub -> master
            kSenderReadRingSlots,          // [29] trid-ring depth = scratch CB slot capacity
        };
        ct_args.insert(ct_args.end(), accessor_ct.begin(), accessor_ct.end());

        auto kernel = CreateKernel(
            program,
            "ttnn/core/tensor/kernels/persistent_d2d_sender.cpp",
            service_core,
            DataMovementConfig{
                .processor = procs[lane],
                .noc = nocs[lane],
                .compile_args = ct_args,
            });

        // The sender writes bulk data downstream, so its fabric connection runs
        // sender -> receiver on THIS lane's link. The fabric-connection args come
        // first (the kernel's build_from_args consumes them), followed by the per-coord
        // receiver backing tensor base address (the direct fabric-write destination).
        std::vector<uint32_t> rt_args;
        tt::tt_fabric::append_fabric_connection_rt_args(
            sender_node, receiver_node, link_indices[lane], program, service_core, rt_args);
        rt_args.push_back(receiver_tensor_addr);
        SetRuntimeArgs(program, kernel, service_core, rt_args);
    }
    return program;
}

// Build a cached fabric-link lease workload: one program per coord, a d2d_lease kernel
// on a single worker core that NoC-accesses that coord's service-core link_grant word.
// is_wait -> WAIT (spin until 0); else RELEASE (write 1). Enqueued on demand by
// wait_for_fabric_links / release_fabric_links so the lease is CQ-ordered: RELEASE is
// enqueued BEFORE the producer workload (grant in place when the producer runs), WAIT
// AFTER (fences the next op). Built only in LEASE mode (share_fabric_links).
std::unique_ptr<distributed::MeshWorkload> build_lease_workload(
    const std::shared_ptr<distributed::MeshDevice>& mesh,
    const std::map<distributed::MeshCoordinate, CoreCoord>& service_cores,
    const std::map<distributed::MeshCoordinate, DeviceAddr>& link_grant_addrs,
    const CoreRange& worker_cores,
    bool is_wait) {
    // One worker core per coord drives the lease (NoC to the service core). It runs
    // between model ops, when the worker grid is otherwise idle.
    const CoreCoord lease_core = worker_cores.start_coord;
    const CoreRange lease_core_range{lease_core, lease_core};
    constexpr tt::CBIndex kLeaseScratchCb = tt::CBIndex::c_0;
    const uint32_t scratch_bytes = tt::tt_metal::hal::get_l1_alignment();

    auto workload = std::make_unique<distributed::MeshWorkload>();
    for (const auto& [coord, svc_core] : service_cores) {
        auto program = CreateProgram();
        // WAIT needs a 1-word L1 staging slot for the NoC read; RELEASE doesn't.
        if (is_wait) {
            auto cb_cfg = CircularBufferConfig(scratch_bytes, {{kLeaseScratchCb, tt::DataFormat::UInt32}})
                              .set_page_size(kLeaseScratchCb, scratch_bytes);
            CreateCircularBuffer(program, lease_core_range, cb_cfg);
        }
        const std::vector<uint32_t> ct_args = {static_cast<uint32_t>(kLeaseScratchCb)};
        auto kernel = CreateKernel(
            program,
            "ttnn/core/tensor/kernels/d2d_lease.cpp",
            lease_core_range,
            DataMovementConfig{
                .processor = DataMovementProcessor::RISCV_0,
                .noc = NOC::RISCV_0_default,
                .compile_args = ct_args,
                .defines = {{"LEASE_MODE", is_wait ? "0" : "1"}}});

        auto* device = mesh->get_device(coord);
        const auto svc_phys = device->worker_core_from_logical_core(svc_core);
        const std::vector<uint32_t> rt_args = {
            static_cast<uint32_t>(svc_phys.x),
            static_cast<uint32_t>(svc_phys.y),
            static_cast<uint32_t>(link_grant_addrs.at(coord)),
        };
        SetRuntimeArgs(program, kernel, lease_core, rt_args);
        workload->add_program(distributed::MeshCoordinateRange(coord), std::move(program));
    }
    return workload;
}

// ===========================================================================
// Per-side resource builders
// ===========================================================================
//
// Everything for one side that depends on the already-constructed MeshSocket
// endpoint: the L1 reservation, the per-coord counter / termination / link-grant
// words, the optional metadata buffer, the mesh-wide worker semaphore, and the
// persistent workload. Called by D2DStreamService::finalize_sender /
// finalize_receiver, which then assemble the handle Impl. They take the socket +
// service cores by const ref (finalize_* moves them into the Impl afterwards).

struct SenderSideResources {
    std::map<distributed::MeshCoordinate, DeviceAddr> data_ready_counter_addrs;
    std::map<distributed::MeshCoordinate, DeviceAddr> termination_addrs;
    std::map<distributed::MeshCoordinate, DeviceAddr> link_grant_addrs;
    std::map<distributed::MeshCoordinate, DeviceAddr> metadata_addrs;
    // Lane-sync words (multi-lane sender): master bumps go_count, sub bumps done_count.
    // Allocated even when single-lane (unused then); freed by the handle dtor.
    std::map<distributed::MeshCoordinate, DeviceAddr> go_count_addrs;
    std::map<distributed::MeshCoordinate, DeviceAddr> done_count_addrs;
    std::optional<GlobalSemaphore> consumed_sem;
    std::unique_ptr<distributed::MeshWorkload> workload;
    // Cached lease workloads (LEASE mode only): release_fabric_links enqueues the RELEASE
    // (write grant=1) before the producer; wait_for_fabric_links enqueues the WAIT (spin
    // grant==0) after. Null in OWN mode.
    std::unique_ptr<distributed::MeshWorkload> lease_wait_workload;
    std::unique_ptr<distributed::MeshWorkload> lease_release_workload;
};

SenderSideResources build_sender_side(
    const std::shared_ptr<distributed::MeshDevice>& mesh,
    const distributed::MeshSocket& sender_socket,
    const std::map<distributed::MeshCoordinate, CoreCoord>& service_cores,
    const std::map<distributed::MeshCoordinate, DeviceAddr>& receiver_tensor_addrs,
    const Tensor& backing,
    const std::vector<distributed::MeshCoordinate>& coords,
    const CommonPlan& common,
    const D2DStreamConfig& cfg) {
    auto& svc = tt::tt_metal::internal::service_core_manager();
    const uint32_t num_workers = core_range_size(cfg.sender_worker_cores);

    // Reserve the socket config buffer's L1 footprint in each service core's
    // per-core allocator BEFORE allocating any words (both grow top-down from
    // L1_END; without this the words alias the socket buffers). The sender owns
    // only a config buffer — no data FIFO.
    const DeviceAddr socket_lo = sender_socket.get_config_buffer()->address();
    for (const auto& coord : coords) {
        svc.reserve_l1_to_top(mesh->get_device(coord), service_cores.at(coord), socket_lo);
    }

    auto termination_addrs = allocate_service_core_words(mesh, service_cores);
    auto data_ready_counter_addrs = allocate_service_core_words(mesh, service_cores);
    auto link_grant_addrs = allocate_service_core_words(mesh, service_cores);
    // Lane-sync words (multi-lane sender). Zero-initialised, like the other words.
    auto go_count_addrs = allocate_service_core_words(mesh, service_cores);
    auto done_count_addrs = allocate_service_core_words(mesh, service_cores);

    std::map<distributed::MeshCoordinate, DeviceAddr> metadata_addrs;
    if (common.metadata_enabled) {
        // Size the metadata L1 buffer to a full socket page, not just
        // metadata_size_bytes. The sender kernel ships the metadata as one
        // trailing socket page and fabric_write_socket_page() always reads
        // socket_page_size bytes from this address; a buffer sized only to the
        // metadata would make it over-read adjacent service-core L1 (counters /
        // reserved socket region) and ship those bytes over fabric. The worker
        // writes metadata_size_bytes into the front; the zero-init keeps the
        // padding tail clean. socket_page_size is already L1-aligned (it is a
        // multiple of the L1-aligned tensor page), so the align() is a no-op
        // guard.
        const uint32_t aligned_md = tt::align(common.plan.socket_page_size, common.l1_alignment);
        std::vector<uint32_t> zero(aligned_md / sizeof(uint32_t), 0u);
        for (const auto& coord : coords) {
            auto* d = mesh->get_device(coord);
            const CoreCoord core = service_cores.at(coord);
            const DeviceAddr addr = svc.allocate_l1(d, core, aligned_md);
            tt::tt_metal::detail::WriteToDeviceL1(d, core, static_cast<uint32_t>(addr), zero);
            metadata_addrs.emplace(coord, addr);
        }
    }

    auto consumed_sem = ttnn::global_semaphore::create_global_semaphore(
        mesh.get(), CoreRangeSet(cfg.sender_worker_cores), /*initial_value=*/0, BufferType::L1);

    constexpr bool worker_sync_enabled = true;
    auto workload = std::make_unique<distributed::MeshWorkload>();
    const uint32_t socket_config_addr = static_cast<uint32_t>(sender_socket.get_config_buffer()->address());

    for (const auto& coord : coords) {
        const Buffer* send_buf = backing.mesh_buffer().get_device_buffer(coord);
        TT_FATAL(send_buf != nullptr, "D2DStreamService: sender device buffer missing for coord {}", coord);
        auto* device = mesh->get_device(coord);
        const CoreCoord service_core = service_cores.at(coord);

        const auto sender_node = mesh->get_fabric_node_id(coord);
        const auto receiver_node = sender_socket.get_fabric_node_id(distributed::SocketEndpoint::RECEIVER, coord);
        const auto links = tt::tt_fabric::get_forwarding_link_indices(sender_node, receiver_node);
        TT_FATAL(!links.empty(), "D2DStreamService: no fabric link sender->receiver at coord {}", coord);

        // Lane count: one RISC + one link per lane, capped by config and by the
        // available forwarding links. The receiver derives the same count from its
        // (symmetric) receiver->sender link set, so it awaits the matching number of
        // data-landed increments.
        const uint32_t num_lanes = std::min<uint32_t>(cfg.max_sender_lanes, static_cast<uint32_t>(links.size()));
        TT_FATAL(num_lanes >= 1, "D2DStreamService: num_lanes must be >= 1 at coord {}", coord);
        const std::vector<uint32_t> lane_links(links.begin(), links.begin() + num_lanes);

        const auto ws = make_worker_sync_args(
            device,
            cfg.sender_worker_cores,
            num_workers,
            static_cast<uint32_t>(consumed_sem.address()),
            static_cast<uint32_t>(data_ready_counter_addrs.at(coord)),
            worker_sync_enabled);

        workload->add_program(
            distributed::MeshCoordinateRange(coord),
            build_sender_program(
                *send_buf,
                service_core,
                socket_config_addr,
                static_cast<uint32_t>(termination_addrs.at(coord)),
                common.plan,
                common.tensor_page_size,
                common.fabric_max_payload_size,
                backing.dtype(),
                ws,
                common.metadata_enabled,
                common.metadata_size_bytes,
                common.metadata_enabled ? static_cast<uint32_t>(metadata_addrs.at(coord)) : 0u,
                cfg.share_fabric_links,
                cfg.share_fabric_links ? static_cast<uint32_t>(link_grant_addrs.at(coord)) : 0u,
                static_cast<uint32_t>(receiver_tensor_addrs.at(coord)),
                static_cast<uint32_t>(go_count_addrs.at(coord)),
                static_cast<uint32_t>(done_count_addrs.at(coord)),
                num_lanes,
                sender_node,
                receiver_node,
                lane_links));
    }

    // LEASE mode: cache the per-coord RELEASE / WAIT lease workloads so the host can
    // enqueue them (CQ-ordered) without rebuilding. OWN mode leaves them null. Built
    // before link_grant_addrs is moved into the result.
    std::unique_ptr<distributed::MeshWorkload> lease_release_workload;
    std::unique_ptr<distributed::MeshWorkload> lease_wait_workload;
    if (cfg.share_fabric_links) {
        lease_release_workload =
            build_lease_workload(mesh, service_cores, link_grant_addrs, cfg.sender_worker_cores, /*is_wait=*/false);
        lease_wait_workload =
            build_lease_workload(mesh, service_cores, link_grant_addrs, cfg.sender_worker_cores, /*is_wait=*/true);
    }

    return SenderSideResources{
        .data_ready_counter_addrs = std::move(data_ready_counter_addrs),
        .termination_addrs = std::move(termination_addrs),
        .link_grant_addrs = std::move(link_grant_addrs),
        .metadata_addrs = std::move(metadata_addrs),
        .go_count_addrs = std::move(go_count_addrs),
        .done_count_addrs = std::move(done_count_addrs),
        .consumed_sem = std::move(consumed_sem),
        .workload = std::move(workload),
        .lease_wait_workload = std::move(lease_wait_workload),
        .lease_release_workload = std::move(lease_release_workload),
    };
}

struct ReceiverSideResources {
    std::map<distributed::MeshCoordinate, DeviceAddr> consumed_counter_addrs;
    std::map<distributed::MeshCoordinate, DeviceAddr> termination_addrs;
    std::map<distributed::MeshCoordinate, DeviceAddr> link_grant_addrs;
    std::optional<GlobalSemaphore> data_ready_sem;
    std::shared_ptr<distributed::MeshBuffer> metadata_buffer;
    DeviceAddr metadata_l1_addr = 0;
    std::unique_ptr<distributed::MeshWorkload> workload;
    // Cached lease workloads (LEASE mode only), mirror of the sender's. Null in OWN mode.
    std::unique_ptr<distributed::MeshWorkload> lease_wait_workload;
    std::unique_ptr<distributed::MeshWorkload> lease_release_workload;
};

ReceiverSideResources build_receiver_side(
    const std::shared_ptr<distributed::MeshDevice>& mesh,
    const distributed::MeshSocket& receiver_socket,
    const std::map<distributed::MeshCoordinate, CoreCoord>& service_cores,
    const Tensor& backing,
    const std::vector<distributed::MeshCoordinate>& coords,
    const CommonPlan& common,
    const D2DStreamConfig& cfg) {
    auto& svc = tt::tt_metal::internal::service_core_manager();
    const uint32_t num_workers = core_range_size(cfg.receiver_worker_cores);

    // Reserve the socket config buffer + data FIFO L1 footprint. They sit at the
    // top of L1, so the lower of the two addresses covers both in one reservation.
    const DeviceAddr socket_lo =
        std::min(receiver_socket.get_config_buffer()->address(), receiver_socket.get_data_buffer()->address());
    for (const auto& coord : coords) {
        svc.reserve_l1_to_top(mesh->get_device(coord), service_cores.at(coord), socket_lo);
    }

    auto termination_addrs = allocate_service_core_words(mesh, service_cores);
    auto consumed_counter_addrs = allocate_service_core_words(mesh, service_cores);
    auto link_grant_addrs = allocate_service_core_words(mesh, service_cores);

    auto data_ready_sem = ttnn::global_semaphore::create_global_semaphore(
        mesh.get(), CoreRangeSet(cfg.receiver_worker_cores), /*initial_value=*/0, BufferType::L1);

    // Optional receiver-side inline-metadata buffer: L1, HEIGHT_SHARDED across the
    // receiver worker grid (one shard per worker), REPLICATED across the mesh so the
    // in-core L1 address is uniform. The receiver service multicasts the blob here
    // on every receiver worker core after each transfer lands.
    std::shared_ptr<distributed::MeshBuffer> metadata_buffer;
    DeviceAddr metadata_l1_addr = 0;
    if (common.metadata_enabled) {
        const DeviceAddr aligned_shard_size = tt::align(
            static_cast<DeviceAddr>(common.metadata_size_bytes), static_cast<DeviceAddr>(common.l1_alignment));
        distributed::DeviceLocalBufferConfig device_local = {
            .page_size = aligned_shard_size,
            .buffer_type = BufferType::L1,
            .sharding_args = tt::tt_metal::BufferShardingArgs(
                tt::tt_metal::ShardSpecBuffer(
                    CoreRangeSet(cfg.receiver_worker_cores),
                    {1, 1},
                    ShardOrientation::ROW_MAJOR,
                    {1, 1},
                    {num_workers, 1}),
                tt::tt_metal::TensorMemoryLayout::HEIGHT_SHARDED),
            .bottom_up = std::nullopt,
            .sub_device_id = std::nullopt,
        };
        distributed::MeshBufferConfig mesh_config = distributed::ReplicatedBufferConfig{
            .size = aligned_shard_size * static_cast<DeviceAddr>(num_workers),
        };
        metadata_buffer = distributed::MeshBuffer::create(mesh_config, device_local, mesh.get());
        metadata_l1_addr = metadata_buffer->address();
    }

    constexpr bool worker_sync_enabled = true;
    auto workload = std::make_unique<distributed::MeshWorkload>();
    const uint32_t socket_config_addr = static_cast<uint32_t>(receiver_socket.get_config_buffer()->address());

    for (const auto& coord : coords) {
        const Buffer* recv_buf = backing.mesh_buffer().get_device_buffer(coord);
        TT_FATAL(recv_buf != nullptr, "D2DStreamService: receiver device buffer missing for coord {}", coord);
        auto* device = mesh->get_device(coord);
        const CoreCoord service_core = service_cores.at(coord);

        const auto receiver_node = mesh->get_fabric_node_id(coord);
        const auto sender_node = receiver_socket.get_fabric_node_id(distributed::SocketEndpoint::SENDER, coord);
        const auto links = tt::tt_fabric::get_forwarding_link_indices(receiver_node, sender_node);
        TT_FATAL(!links.empty(), "D2DStreamService: no fabric link receiver->sender at coord {}", coord);

        // Sender lane count, derived identically to the sender (min(max_sender_lanes,
        // forwarding links)) — symmetric link topology ⇒ same value. This is how many
        // data-landed increments the receiver awaits per transfer (one per sender lane).
        const uint32_t num_lanes = std::min<uint32_t>(cfg.max_sender_lanes, static_cast<uint32_t>(links.size()));

        const auto ws = make_worker_sync_args(
            device,
            cfg.receiver_worker_cores,
            num_workers,
            static_cast<uint32_t>(data_ready_sem.address()),
            static_cast<uint32_t>(consumed_counter_addrs.at(coord)),
            worker_sync_enabled);

        workload->add_program(
            distributed::MeshCoordinateRange(coord),
            build_receiver_program(
                *recv_buf,
                service_core,
                socket_config_addr,
                static_cast<uint32_t>(termination_addrs.at(coord)),
                common.plan,
                common.tensor_page_size,
                ws,
                common.metadata_enabled,
                common.metadata_size_bytes,
                static_cast<uint32_t>(metadata_l1_addr),
                cfg.share_fabric_links,
                cfg.share_fabric_links ? static_cast<uint32_t>(link_grant_addrs.at(coord)) : 0u,
                num_lanes,
                receiver_node,
                sender_node,
                links.front()));
    }

    // LEASE mode: cache the per-coord RELEASE / WAIT lease workloads (mirror of the
    // sender). OWN mode leaves them null. Built before link_grant_addrs is moved out.
    std::unique_ptr<distributed::MeshWorkload> lease_release_workload;
    std::unique_ptr<distributed::MeshWorkload> lease_wait_workload;
    if (cfg.share_fabric_links) {
        lease_release_workload =
            build_lease_workload(mesh, service_cores, link_grant_addrs, cfg.receiver_worker_cores, /*is_wait=*/false);
        lease_wait_workload =
            build_lease_workload(mesh, service_cores, link_grant_addrs, cfg.receiver_worker_cores, /*is_wait=*/true);
    }

    return ReceiverSideResources{
        .consumed_counter_addrs = std::move(consumed_counter_addrs),
        .termination_addrs = std::move(termination_addrs),
        .link_grant_addrs = std::move(link_grant_addrs),
        .data_ready_sem = std::move(data_ready_sem),
        .metadata_buffer = std::move(metadata_buffer),
        .metadata_l1_addr = metadata_l1_addr,
        .workload = std::move(workload),
        .lease_wait_workload = std::move(lease_wait_workload),
        .lease_release_workload = std::move(lease_release_workload),
    };
}

}  // namespace CMAKE_UNIQUE_NAMESPACE

// ===========================================================================
// Per-side handle assembly (shared by all three factories)
// ===========================================================================

std::unique_ptr<D2DStreamServiceSender> D2DStreamService::finalize_sender(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh,
    tt::tt_metal::distributed::MeshSocket socket,
    std::map<tt::tt_metal::distributed::MeshCoordinate, CoreCoord> service_cores,
    const std::map<tt::tt_metal::distributed::MeshCoordinate, DeviceAddr>& receiver_tensor_addrs,
    const Tensor& backing,
    const D2DStreamConfig& cfg) {
    // Release the claimed cores if we throw before the handle owns them; on success
    // `ok` disarms the guard and the handle destructor owns teardown.
    bool ok = false;
    CMAKE_UNIQUE_NAMESPACE::ServiceCoreReleaseGuard guard{mesh, service_cores, ok};

    const auto common = CMAKE_UNIQUE_NAMESPACE::derive_common_plan(cfg, backing);
    const auto& coords = backing.tensor_topology().mesh_coords();
    auto res = CMAKE_UNIQUE_NAMESPACE::build_sender_side(
        mesh, socket, service_cores, receiver_tensor_addrs, backing, coords, common, cfg);

    auto impl = std::make_unique<D2DStreamServiceSender::Impl>(D2DStreamServiceSender::Impl{
        .mesh_device = mesh,
        .per_shard_spec = backing.tensor_spec(),
        .backing_tensor = backing,
        .worker_cores = cfg.sender_worker_cores,
        .service_cores = std::move(service_cores),
        .socket = std::move(socket),
        .socket_page_size = common.plan.socket_page_size,
        .num_socket_pages = common.plan.num_socket_pages,
        .pages_per_chunk = common.plan.pages_per_chunk,
        .num_workers = CMAKE_UNIQUE_NAMESPACE::core_range_size(cfg.sender_worker_cores),
        .data_ready_counter_addrs = std::move(res.data_ready_counter_addrs),
        .termination_addrs = std::move(res.termination_addrs),
        .share_fabric_links = cfg.share_fabric_links,
        .link_grant_addrs = std::move(res.link_grant_addrs),
        .go_count_addrs = std::move(res.go_count_addrs),
        .done_count_addrs = std::move(res.done_count_addrs),
        .consumed_sem = std::move(res.consumed_sem),
        .metadata_size_bytes = common.metadata_size_bytes,
        .metadata_addrs = std::move(res.metadata_addrs),
        .workload = std::move(res.workload),
        .launched = false,
        .lease_wait_workload = std::move(res.lease_wait_workload),
        .lease_release_workload = std::move(res.lease_release_workload),
    });
    auto handle = std::unique_ptr<D2DStreamServiceSender>(new D2DStreamServiceSender(std::move(impl)));
    ok = true;
    return handle;
}

std::unique_ptr<D2DStreamServiceReceiver> D2DStreamService::finalize_receiver(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& mesh,
    tt::tt_metal::distributed::MeshSocket socket,
    std::map<tt::tt_metal::distributed::MeshCoordinate, CoreCoord> service_cores,
    const Tensor& backing,
    const D2DStreamConfig& cfg) {
    bool ok = false;
    CMAKE_UNIQUE_NAMESPACE::ServiceCoreReleaseGuard guard{mesh, service_cores, ok};

    const auto common = CMAKE_UNIQUE_NAMESPACE::derive_common_plan(cfg, backing);
    const auto& coords = backing.tensor_topology().mesh_coords();
    auto res = CMAKE_UNIQUE_NAMESPACE::build_receiver_side(mesh, socket, service_cores, backing, coords, common, cfg);

    auto impl = std::make_unique<D2DStreamServiceReceiver::Impl>(D2DStreamServiceReceiver::Impl{
        .mesh_device = mesh,
        .per_shard_spec = backing.tensor_spec(),
        .backing_tensor = backing,
        .worker_cores = cfg.receiver_worker_cores,
        .service_cores = std::move(service_cores),
        .socket = std::move(socket),
        .socket_page_size = common.plan.socket_page_size,
        .num_socket_pages = common.plan.num_socket_pages,
        .pages_per_chunk = common.plan.pages_per_chunk,
        .num_workers = CMAKE_UNIQUE_NAMESPACE::core_range_size(cfg.receiver_worker_cores),
        .consumed_counter_addrs = std::move(res.consumed_counter_addrs),
        .termination_addrs = std::move(res.termination_addrs),
        .share_fabric_links = cfg.share_fabric_links,
        .link_grant_addrs = std::move(res.link_grant_addrs),
        .data_ready_sem = std::move(res.data_ready_sem),
        .metadata_size_bytes = common.metadata_size_bytes,
        .metadata_buffer = std::move(res.metadata_buffer),
        .metadata_l1_addr = res.metadata_l1_addr,
        .workload = std::move(res.workload),
        .launched = false,
        .lease_wait_workload = std::move(res.lease_wait_workload),
        .lease_release_workload = std::move(res.lease_release_workload),
    });
    auto handle = std::unique_ptr<D2DStreamServiceReceiver>(new D2DStreamServiceReceiver(std::move(impl)));
    ok = true;
    return handle;
}

// ===========================================================================
// Factory
// ===========================================================================

std::pair<std::unique_ptr<D2DStreamServiceSender>, std::unique_ptr<D2DStreamServiceReceiver>>
D2DStreamService::create_pair(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& sender_mesh,
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& receiver_mesh,
    D2DStreamConfig cfg) {
    // --- validate shapes + mapper ---------------------------------------------
    TT_FATAL(sender_mesh != nullptr, "D2DStreamService: sender_mesh must not be null");
    TT_FATAL(receiver_mesh != nullptr, "D2DStreamService: receiver_mesh must not be null");
    TT_FATAL(
        sender_mesh->shape() == receiver_mesh->shape(),
        "D2DStreamService: sender_mesh shape {} must equal receiver_mesh shape {} (1:1 coord mapping)",
        sender_mesh->shape(),
        receiver_mesh->shape());

    // Run the mapper once; both sides share the per-shard spec + topology.
    auto mo = CMAKE_UNIQUE_NAMESPACE::run_mapper(cfg);

    // Allocate both backing tensors (identical per-shard spec, different mesh).
    Tensor sender_backing = create_device_tensor(mo.per_shard_spec, sender_mesh.get(), mo.topology);
    Tensor receiver_backing = create_device_tensor(mo.per_shard_spec, receiver_mesh.get(), mo.topology);

    // Claim one service core per coord on each side.
    auto sender_service_cores = CMAKE_UNIQUE_NAMESPACE::claim_service_cores(sender_mesh, mo.coords, "sender");
    auto receiver_service_cores = CMAKE_UNIQUE_NAMESPACE::claim_service_cores(receiver_mesh, mo.coords, "receiver");
    log_debug(tt::LogMetal, "D2DStreamService: sender service cores: {}", sender_service_cores);
    log_debug(tt::LogMetal, "D2DStreamService: receiver service cores: {}", receiver_service_cores);

    // Release the claimed cores if we throw before the handles take ownership (the
    // claims live in a process-global manager; leaking them cascades into later
    // failures). On success the cores are moved into the handle Impls, so each
    // guard then iterates an empty map and the handle destructors own teardown.
    bool committed = false;
    CMAKE_UNIQUE_NAMESPACE::ServiceCoreReleaseGuard sender_guard{sender_mesh, sender_service_cores, committed};
    CMAKE_UNIQUE_NAMESPACE::ServiceCoreReleaseGuard receiver_guard{receiver_mesh, receiver_service_cores, committed};

    // Single-host: this process knows both sides' cores, so create_socket_pair
    // builds both endpoints with no network rendezvous. One connection per
    // PARTICIPATING coord (mo.coords, NOT the full mesh range).
    auto connections =
        CMAKE_UNIQUE_NAMESPACE::build_connections(mo.coords, sender_service_cores, receiver_service_cores);
    tt::tt_metal::distributed::SocketConfig socket_config(connections, cfg.socket_mem_config);
    auto socket_pair =
        tt::tt_metal::distributed::MeshSocket::create_socket_pair(sender_mesh, receiver_mesh, socket_config);
    auto& sender_socket = socket_pair.first;
    auto& receiver_socket = socket_pair.second;

    // Build + assemble each side from its socket endpoint. finalize_* is shared
    // with the multi-host factories and does NOT launch (so we control the order
    // below). The service cores move into the handle Impls here, emptying the maps
    // the guards above reference. Single-host knows the receiver tensor addresses
    // directly (both meshes are local) — no exchange needed.
    auto receiver_tensor_addrs = CMAKE_UNIQUE_NAMESPACE::collect_backing_addrs(receiver_backing, mo.coords);
    auto sender_handle = finalize_sender(
        sender_mesh,
        std::move(sender_socket),
        std::move(sender_service_cores),
        receiver_tensor_addrs,
        sender_backing,
        cfg);
    auto receiver_handle = finalize_receiver(
        receiver_mesh, std::move(receiver_socket), std::move(receiver_service_cores), receiver_backing, cfg);

    // Launch the persistent kernels (non-blocking). Receiver first so it's parked
    // on its socket wait before the sender starts pushing pages.
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

std::unique_ptr<D2DStreamServiceSender> D2DStreamService::create_sender(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& sender_mesh,
    D2DStreamConfig cfg,
    const D2DEndpointConfig& endpoints) {
    TT_FATAL(sender_mesh != nullptr, "D2DStreamService::create_sender: sender_mesh must not be null");
    const auto ctx = endpoints.distributed_context
                         ? endpoints.distributed_context
                         : tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    TT_FATAL(
        *ctx->rank() == *endpoints.sender_rank,
        "D2DStreamService::create_sender must run on the sender rank (local rank {}, endpoints.sender_rank {})",
        *ctx->rank(),
        *endpoints.sender_rank);

    // Run the mapper locally; the receiver process runs the same mapper on its
    // identically-shaped mesh and derives the same per-shard spec.
    auto mo = CMAKE_UNIQUE_NAMESPACE::run_mapper(cfg);
    Tensor sender_backing = create_device_tensor(mo.per_shard_spec, sender_mesh.get(), mo.topology);
    auto sender_service_cores = CMAKE_UNIQUE_NAMESPACE::claim_service_cores(sender_mesh, mo.coords, "sender");
    log_debug(tt::LogMetal, "D2DStreamService: sender service cores: {}", sender_service_cores);

    bool committed = false;
    CMAKE_UNIQUE_NAMESPACE::ServiceCoreReleaseGuard sender_guard{sender_mesh, sender_service_cores, committed};

    const auto common = CMAKE_UNIQUE_NAMESPACE::derive_common_plan(cfg, sender_backing);

    // Trade claimed service cores AND backing tensor addresses with the receiver
    // process so both sides build the identical SocketConnection list (the MeshSocket
    // handshake asserts they match) and the sender learns the receiver's per-coord
    // DRAM tensor addresses (its direct fabric-write destinations).
    auto sender_backing_addrs = CMAKE_UNIQUE_NAMESPACE::collect_backing_addrs(sender_backing, mo.coords);
    auto peer = CMAKE_UNIQUE_NAMESPACE::exchange_service_cores(
        ctx,
        /*is_sender=*/true,
        endpoints.receiver_rank,
        mo.coords,
        sender_service_cores,
        sender_backing_addrs,
        common.tensor_page_size,
        sender_backing.buffer()->num_pages(),
        common.plan.socket_page_size,
        common.plan.num_socket_pages,
        common.metadata_size_bytes,
        cfg.share_fabric_links);
    auto receiver_service_cores = std::move(peer.service_cores);
    auto receiver_tensor_addrs = std::move(peer.backing_addrs);

    // Build the SENDER endpoint. The per-endpoint MeshSocket ctor derives the
    // sender/receiver mesh-ids from the ranks, detects this process as the sender,
    // and runs the cross-process descriptor handshake (blocks until the receiver
    // process reaches its matching MeshSocket ctor, subject to the socket timeout).
    auto connections =
        CMAKE_UNIQUE_NAMESPACE::build_connections(mo.coords, sender_service_cores, receiver_service_cores);
    tt::tt_metal::distributed::SocketConfig socket_config(
        connections, cfg.socket_mem_config, endpoints.sender_rank, endpoints.receiver_rank, ctx);
    tt::tt_metal::distributed::MeshSocket sender_socket(sender_mesh, socket_config);

    // Build + assemble (shared with create_pair). The service cores move into the
    // handle Impl here, emptying the map sender_guard references.
    auto sender_handle = finalize_sender(
        sender_mesh,
        std::move(sender_socket),
        std::move(sender_service_cores),
        receiver_tensor_addrs,
        sender_backing,
        cfg);
    EnqueueMeshWorkload(
        sender_handle->impl_->mesh_device->mesh_command_queue(), *sender_handle->impl_->workload, /*blocking=*/false);
    sender_handle->impl_->launched = true;

    committed = true;
    return sender_handle;
}

std::unique_ptr<D2DStreamServiceReceiver> D2DStreamService::create_receiver(
    const std::shared_ptr<tt::tt_metal::distributed::MeshDevice>& receiver_mesh,
    D2DStreamConfig cfg,
    const D2DEndpointConfig& endpoints) {
    TT_FATAL(receiver_mesh != nullptr, "D2DStreamService::create_receiver: receiver_mesh must not be null");
    const auto ctx = endpoints.distributed_context
                         ? endpoints.distributed_context
                         : tt::tt_metal::distributed::multihost::DistributedContext::get_current_world();
    TT_FATAL(
        *ctx->rank() == *endpoints.receiver_rank,
        "D2DStreamService::create_receiver must run on the receiver rank (local rank {}, endpoints.receiver_rank {})",
        *ctx->rank(),
        *endpoints.receiver_rank);

    auto mo = CMAKE_UNIQUE_NAMESPACE::run_mapper(cfg);
    Tensor receiver_backing = create_device_tensor(mo.per_shard_spec, receiver_mesh.get(), mo.topology);
    auto receiver_service_cores = CMAKE_UNIQUE_NAMESPACE::claim_service_cores(receiver_mesh, mo.coords, "receiver");
    log_debug(tt::LogMetal, "D2DStreamService: receiver service cores: {}", receiver_service_cores);

    bool committed = false;
    CMAKE_UNIQUE_NAMESPACE::ServiceCoreReleaseGuard receiver_guard{receiver_mesh, receiver_service_cores, committed};

    const auto common = CMAKE_UNIQUE_NAMESPACE::derive_common_plan(cfg, receiver_backing);

    // Trade service cores + backing addresses. The receiver sends its own backing
    // addresses (the sender needs them as fabric-write destinations) and ignores the
    // sender's in return.
    auto receiver_backing_addrs = CMAKE_UNIQUE_NAMESPACE::collect_backing_addrs(receiver_backing, mo.coords);
    auto peer = CMAKE_UNIQUE_NAMESPACE::exchange_service_cores(
        ctx,
        /*is_sender=*/false,
        endpoints.sender_rank,
        mo.coords,
        receiver_service_cores,
        receiver_backing_addrs,
        common.tensor_page_size,
        receiver_backing.buffer()->num_pages(),
        common.plan.socket_page_size,
        common.plan.num_socket_pages,
        common.metadata_size_bytes,
        cfg.share_fabric_links);
    auto sender_service_cores = std::move(peer.service_cores);

    auto connections =
        CMAKE_UNIQUE_NAMESPACE::build_connections(mo.coords, sender_service_cores, receiver_service_cores);
    tt::tt_metal::distributed::SocketConfig socket_config(
        connections, cfg.socket_mem_config, endpoints.sender_rank, endpoints.receiver_rank, ctx);
    tt::tt_metal::distributed::MeshSocket receiver_socket(receiver_mesh, socket_config);

    // Build + assemble (shared with create_pair). The service cores move into the
    // handle Impl here, emptying the map receiver_guard references.
    auto receiver_handle = finalize_receiver(
        receiver_mesh, std::move(receiver_socket), std::move(receiver_service_cores), receiver_backing, cfg);
    EnqueueMeshWorkload(receiver_mesh->mesh_command_queue(), *receiver_handle->impl_->workload, /*blocking=*/false);
    receiver_handle->impl_->launched = true;

    committed = true;
    return receiver_handle;
}

}  // namespace ttnn
