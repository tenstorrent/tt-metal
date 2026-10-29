// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "tensor/d2d_stream_service.hpp"

#include <algorithm>
#include <chrono>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <optional>
#include <set>
#include <string>
#include <utility>
#include <vector>

#include <tt_stl/assert.hpp>
#include <tt_stl/indestructible.hpp>
#include <tt_stl/span.hpp>
#include <tt_stl/fmt.hpp>
#include <tt-logger/tt-logger.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/distributed_context.hpp>
#include <tt-metalium/experimental/fabric/fabric.hpp>
#include <tt-metalium/experimental/fabric/pipeline_builder.hpp>
#include <tt-metalium/experimental/sockets/mesh_socket.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <internal/service/service_core_manager.hpp>
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tt_metal.hpp>
#include <tt-metalium/math.hpp>
#include <internal/mesh_device.hpp>

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

// Value the host writes to a service's termination word; passed to the kernels.
constexpr uint32_t kTerminateValue = 1;

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

// Allocate `num_words` zero-initialised uint32 L1 words on a service core, recording the
// address per coord. Used for the per-coord counters and termination words that
// can't be GlobalSemaphores (service cores differ per device, and
// GlobalSemaphore's reset path requires a MeshBuffer-backed buffer). Words are
// word_stride_bytes apart.
std::map<distributed::MeshCoordinate, DeviceAddr> allocate_service_core_words(
    const std::shared_ptr<distributed::MeshDevice>& mesh,
    const std::map<distributed::MeshCoordinate, CoreCoord>& service_cores,
    uint32_t num_words = 1,
    uint32_t word_stride_bytes = sizeof(uint32_t)) {
    auto& svc = tt::tt_metal::internal::service_core_manager(*mesh);
    std::vector<uint32_t> zero_words(num_words * word_stride_bytes / sizeof(uint32_t), 0);
    std::map<distributed::MeshCoordinate, DeviceAddr> addrs;
    for (const auto& [coord, core] : service_cores) {
        auto* d = tt::tt_metal::internal::get_device(*mesh, coord);
        const DeviceAddr addr = svc.allocate_l1(d, core, num_words * word_stride_bytes);
        tt::tt_metal::detail::WriteToDeviceL1(d, core, static_cast<uint32_t>(addr), zero_words);
        addrs.emplace(coord, addr);
    }
    return addrs;
}
static_assert(
    static_cast<uint32_t>(d2d_stream::GateState::Closed) == 0,
    "allocate_service_core_words zero-fills the stage-gate array");

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
//     num_socket_pages, metadata_size_bytes, share_fabric_links, relay,
//     x0, y0, addr0, x1, y1, addr1, ...] in `coords` order. The seven fingerprint
// fields catch any peer mismatch (global_spec, mapper, fifo_size, metadata,
// share_fabric_links, relay) before the fabric hangs silently. is_sender orders send/recv
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
    bool share_fabric_links,
    bool relay) {
    std::vector<uint32_t> out;

    // Push 7 + 3 * coords.size() arguments to out
    constexpr uint32_t kNumFingerprintArgs = 7;
    const uint32_t kNumOutputArgs = kNumFingerprintArgs + 3 * coords.size();
    out.reserve(kNumOutputArgs);
    out.push_back(tensor_page_size);
    out.push_back(num_pages);
    out.push_back(socket_page_size);
    out.push_back(num_socket_pages);
    out.push_back(metadata_size_bytes);
    out.push_back(share_fabric_links ? 1u : 0u);
    out.push_back(relay ? 1u : 0u);
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
    const bool in_relay = static_cast<bool>(in[6]);

    const bool fp_match = in_tensor_page_size == tensor_page_size && in_num_pages == num_pages &&
                          in_socket_page_size == socket_page_size && in_num_socket_pages == num_socket_pages &&
                          in_metadata_size_bytes == metadata_size_bytes &&
                          in_share_fabric_links == share_fabric_links && in_relay == relay;
    TT_FATAL(
        fp_match,
        "D2DStreamService: peer chunk-plan mismatch — "
        "local (page_size={} num_pages={} socket_page_size={} num_socket_pages={} "
        "metadata_size_bytes={} share_fabric_links={} relay={}) vs "
        "peer (page_size={} num_pages={} socket_page_size={} num_socket_pages={} "
        "metadata_size_bytes={} share_fabric_links={} relay={}); "
        "sender and receiver must use the same global_spec, mapper, fifo_size, "
        "metadata_size_bytes, share_fabric_links, and relay (TT_METAL_FABRIC_DISABLE_INTERMESH_VC)",
        tensor_page_size,
        num_pages,
        socket_page_size,
        num_socket_pages,
        metadata_size_bytes,
        share_fabric_links,
        relay,
        in_tensor_page_size,
        in_num_pages,
        in_socket_page_size,
        in_num_socket_pages,
        in_metadata_size_bytes,
        in_share_fabric_links,
        in_relay);

    ExchangedEndpoint peer;
    for (size_t i = 0; i < coords.size(); ++i) {
        const size_t b = kNumFingerprintArgs + 3 * i;
        peer.service_cores.emplace(coords[i], CoreCoord{in[b], in[b + 1]});
        peer.backing_addrs.emplace(coords[i], static_cast<DeviceAddr>(in[b + 2]));
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

// The stage gate, or nullopt when it's off (unset or num_gates == 0).
std::optional<D2DStageGateConfig> enabled_stage_gate(const D2DStreamConfig& cfg) {
    if (cfg.stage_gate && cfg.stage_gate->num_gates > 0) {
        return cfg.stage_gate;
    }
    return std::nullopt;
}

// Most gates a receiver can have. Each takes L1-alignment bytes of service-core L1.
constexpr uint32_t kMaxStageGates = 1024;

// With the intermesh VC disabled, a landed packet can only go N or S, so every stream needs the relay.
void apply_intermesh_vc_disabled(D2DStreamConfig& cfg) {
    if (tt::tt_fabric::IsIntermeshVCDisabled()) {
        cfg.relay = true;
    }
}

// Checks the receiver's stage-gate config. Runs before anything is allocated or
// exchanged with the peer, so a bad config throws before the sender launches.
void validate_stage_gate(const D2DStreamConfig& cfg) {
    const auto gate = enabled_stage_gate(cfg);
    if (!gate) {
        return;
    }
    TT_FATAL(
        gate->num_gates <= kMaxStageGates,
        "D2DStreamService: stage_gate num_gates ({}) must be at most {}",
        gate->num_gates,
        kMaxStageGates);
    TT_FATAL(cfg.metadata_size_bytes > 0, "D2DStreamService: stage_gate requires metadata_size_bytes > 0");
    for (const uint32_t offset : {gate->slot_id_offset_bytes, gate->gate_mode_offset_bytes}) {
        TT_FATAL(
            offset % sizeof(uint32_t) == 0 && offset + sizeof(uint32_t) <= cfg.metadata_size_bytes,
            "D2DStreamService: stage-gate metadata offset {} must be 4-byte aligned and within "
            "metadata_size_bytes ({})",
            offset,
            cfg.metadata_size_bytes);
    }
    TT_FATAL(
        gate->slot_id_offset_bytes != gate->gate_mode_offset_bytes,
        "D2DStreamService: stage-gate slot_id and gate_mode must be distinct metadata words (both at offset {})",
        gate->slot_id_offset_bytes);
}

// Chunk plan + sizing shared by both sides, derived from an allocated backing
// tensor (identical on both sides under the symmetric-mapping invariant).
struct CommonPlan {
    ChunkPlan plan;
    uint32_t tensor_page_size;
    uint32_t fabric_max_payload_size;
    uint32_t metadata_size_bytes;
    // Bytes actually staged + shipped for the metadata blob: metadata_size_bytes
    // rounded up to the L1 alignment (0 when metadata is disabled). Sized off the
    // metadata itself, NOT off socket_page_size — see derive_common_plan.
    uint32_t metadata_span_bytes;
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

    // Socket page is flow control only -- payload bypasses the FIFO (see "no L1 FIFO copy" above) -- so
    // it is sized to the write alignment, not to a tensor page. Page as both args => pages_per_chunk 1.
    constexpr uint32_t kSocketPageSizeBytes = 64;
    const uint32_t socket_page_size = tt::align(kSocketPageSizeBytes, l1_alignment);
    const auto plan = derive_chunk_plan(socket_page_size, tensor_num_pages, socket_page_size);

    const auto fabric_max_payload_size = static_cast<uint32_t>(
        tt::round_down(tt::tt_fabric::get_tt_fabric_max_payload_size_bytes(), static_cast<size_t>(l1_alignment)));
    TT_FATAL(fabric_max_payload_size > 0, "D2DStreamService: fabric max payload size rounded to zero");

    const uint32_t metadata_size_bytes = cfg.metadata_size_bytes;
    const bool metadata_enabled = metadata_size_bytes > 0;
    // Metadata capacity is bounded by fifo_size, NOT by socket_page_size. The blob is
    // staged in sender service-core L1 and fabric-written to the receiver's FIFO base
    // (the receiver never pops, so its read_ptr stays at that base), so the FIFO is the
    // only region it has to fit in — the 64 B flow-control page is independent of how
    // much metadata a transfer carries. The shipped span is L1-aligned so both the
    // staging allocation and the fabric write stay alignment-legal; the padding tail is
    // zero-init (see the staging buffer below) and the receiver mcasts only the
    // un-padded metadata_size_bytes.
    const uint32_t metadata_span_bytes = metadata_enabled ? tt::align(metadata_size_bytes, l1_alignment) : 0u;
    const uint32_t min_fifo_size = std::max(socket_page_size, metadata_span_bytes);
    TT_FATAL(
        cfg.socket_mem_config.fifo_size >= min_fifo_size,
        "D2DStreamService: fifo_size ({} B) must be >= {} B — max of socket_page_size ({} B) and the L1-aligned "
        "metadata span ({} B, for metadata_size_bytes={})",
        cfg.socket_mem_config.fifo_size,
        min_fifo_size,
        socket_page_size,
        metadata_span_bytes,
        metadata_size_bytes);

    return CommonPlan{
        .plan = plan,
        .tensor_page_size = tensor_page_size,
        .fabric_max_payload_size = fabric_max_payload_size,
        .metadata_size_bytes = metadata_size_bytes,
        .metadata_span_bytes = metadata_span_bytes,
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
        auto& svc = tt::tt_metal::internal::service_core_manager(*mesh);
        for (const auto& [coord, core] : cores) {
            svc.release(tt::tt_metal::internal::get_device(*mesh, coord), {core});
        }
    }
};

// Fabric routers held by persistent D2D services, process-wide. A Tensix worker
// connects to an ethernet router through the router's one local worker channel,
// which serves exactly one worker: a second connection overwrites the first's
// worker record, the router returns credits to the second only, and the first
// waits forever for a free slot. A chip that hosts both a D2D receiver and a D2D
// sender has two such workers, and when both of its routes leave in the same
// direction they would otherwise pick the same router (the first forwarding link).
// Each service claims its routers here, keyed by fabric node and ethernet core,
// before its kernels are built; a claim is held until the service is destroyed.
// Claims apply in LEASE mode too: the sender and receiver leases are independent,
// so their connection windows can overlap. Services in one lease group
// (Config::fabric_link_group) may share a router: their caller serializes their
// connections, so the router's worker channel only ever has one of them.
class FabricLinkRegistry {
public:
    // Claims (node, eth_core) for `holder`, in lease group `group` (empty: none).
    // A router already held in the same non-empty group is shared; otherwise this
    // returns the current holder instead.
    std::optional<std::string> try_claim(
        const tt::tt_fabric::FabricNodeId& node,
        CoreCoord eth_core,
        const std::string& holder,
        const std::string& group) {
        std::lock_guard lock(mutex_);
        auto [it, inserted] = claims_.try_emplace({node, eth_core}, Claim{.holder = holder, .group = group});
        if (inserted) {
            return std::nullopt;
        }
        if (!group.empty() && it->second.group == group) {
            ++it->second.services;
            return std::nullopt;
        }
        return it->second.holder;
    }

    void release(const tt::tt_fabric::FabricNodeId& node, CoreCoord eth_core) {
        std::lock_guard lock(mutex_);
        auto it = claims_.find({node, eth_core});
        if (it != claims_.end() && --it->second.services == 0) {
            claims_.erase(it);
        }
    }

private:
    struct Claim {
        std::string holder;  // the first service to claim it, for messages
        std::string group;
        uint32_t services = 1;
    };
    std::mutex mutex_;
    std::map<std::pair<tt::tt_fabric::FabricNodeId, CoreCoord>, Claim> claims_;
};

// Never destroyed: a service handle owned by another static may release its claims
// during static teardown, after a destructible registry would be gone.
FabricLinkRegistry& fabric_link_registry() {
    static ttsl::Indestructible<FabricLinkRegistry> registry;
    return registry.get();
}

// One service's router claims; released on destruction (handle teardown, or a
// factory that throws part way). Move-only.
class FabricLinkClaims {
public:
    FabricLinkClaims() = default;
    FabricLinkClaims(const FabricLinkClaims&) = delete;
    FabricLinkClaims& operator=(const FabricLinkClaims&) = delete;
    FabricLinkClaims(FabricLinkClaims&& other) noexcept : held_(std::exchange(other.held_, {})) {}
    FabricLinkClaims& operator=(FabricLinkClaims&& other) noexcept {
        if (this != &other) {
            release_all();
            held_ = std::exchange(other.held_, {});
        }
        return *this;
    }
    ~FabricLinkClaims() { release_all(); }

    std::optional<std::string> try_claim(
        const tt::tt_fabric::FabricNodeId& node,
        CoreCoord eth_core,
        const std::string& holder,
        const std::string& group) {
        auto other = fabric_link_registry().try_claim(node, eth_core, holder, group);
        if (!other) {
            held_.emplace_back(node, eth_core);
        }
        return other;
    }

private:
    void release_all() {
        for (const auto& [node, eth_core] : held_) {
            fabric_link_registry().release(node, eth_core);
        }
        held_.clear();
    }

    std::vector<std::pair<tt::tt_fabric::FabricNodeId, CoreCoord>> held_;
};

// A lease group lets services share routers because their caller takes turns between
// them; in OWN mode a service holds its connection for life, so no turn ever ends.
void validate_fabric_link_group(const D2DStreamConfig& cfg) {
    TT_FATAL(
        cfg.fabric_link_group.empty() || cfg.share_fabric_links,
        "D2DStreamService: fabric_link_group '{}' needs LEASE mode (share_fabric_links == true); an OWN-mode "
        "service holds its fabric connections for its whole life, so it cannot share a router",
        cfg.fabric_link_group);
}

// Which end of a D2D stream a service is, for its router claims and their messages.
enum class D2DServiceRole { Sender, Receiver };

constexpr const char* role_name(D2DServiceRole role) {
    switch (role) {
        case D2DServiceRole::Sender: return "sender";
        case D2DServiceRole::Receiver: return "receiver";
    }
    return "unknown";
}

// Claim up to `max_links` free routers on src toward dst and return their link
// indices (from get_forwarding_link_indices), in order: the sender's lanes, or the
// receiver's single credit-return link. Fails when none is free, and warns when
// fewer than min(max_links, forwarding links) are.
std::vector<uint32_t> claim_forwarding_links(
    const distributed::MeshDevice& mesh,
    const distributed::MeshCoordinate& coord,
    const tt::tt_fabric::FabricNodeId& src,
    const tt::tt_fabric::FabricNodeId& dst,
    uint32_t max_links,
    D2DServiceRole service_role,
    const std::string& group,
    FabricLinkClaims& claims) {
    const char* role = role_name(service_role);
    const auto links = tt::tt_fabric::get_forwarding_link_indices(mesh, src, dst);
    TT_FATAL(!links.empty(), "D2DStreamService: no fabric link from the {} at coord {} toward its peer", role, coord);
    const std::string holder = fmt::format(
        "D2D {} at coord {} ({} -> {}){}",
        role,
        coord,
        src,
        dst,
        group.empty() ? std::string{} : fmt::format(", lease group '{}'", group));
    std::vector<uint32_t> claimed;
    std::string taken;  // "; "-separated, for the messages below
    for (const uint32_t link : links) {
        if (claimed.size() == max_links) {
            break;
        }
        const CoreCoord eth = tt::tt_fabric::get_forwarding_eth_core(mesh, src, dst, link);
        if (auto other = claims.try_claim(src, eth, holder, group)) {
            taken += fmt::format("{}link {} (eth {}) held by the {}", taken.empty() ? "" : "; ", link, eth, *other);
            continue;
        }
        claimed.push_back(link);
    }
    TT_FATAL(
        !claimed.empty(),
        "D2DStreamService: the {} at coord {} has no free fabric router toward its peer: every forwarding "
        "link from {} toward {} is held by another D2D service on {} ({}). A router's worker channel "
        "serves one worker, so the two services cannot share one. Create a chip's D2D receivers before its "
        "senders, so a sender's lanes are chosen around the receiver's link.",
        role,
        coord,
        src,
        dst,
        src,
        taken);
    const size_t wanted = std::min<size_t>(links.size(), max_links);
    if (claimed.size() < wanted) {
        log_warning(
            tt::LogMetal,
            "D2DStreamService: the {} at coord {} ({}) uses {} of {} forwarding links toward its peer; skipped {}",
            role,
            coord,
            src,
            claimed.size(),
            wanted,
            taken);
    }
    return claimed;
}

// DistributedContext tag for the sender -> receiver lane-count message, sent once
// both MeshSocket endpoints exist (the sender's routers, and so its lane count, are
// chosen only then).
constexpr int kLaneCountExchangeTag = 0x4432444C;  // 'D2DL'

// Per-coord sender lane links (the links each lane holds, see claim_forwarding_links).
using LaneLinks = std::map<distributed::MeshCoordinate, std::vector<uint32_t>>;

// Lanes per coord: one per lane link.
std::map<distributed::MeshCoordinate, uint32_t> lane_counts(const LaneLinks& lane_links) {
    std::map<distributed::MeshCoordinate, uint32_t> counts;
    for (const auto& [coord, links] : lane_links) {
        counts.emplace(coord, static_cast<uint32_t>(links.size()));
    }
    return counts;
}

// Multi-rank: the sender tells the receiver how many lanes it runs per coord, in
// `coords` order. The receiver awaits one data-landed increment per sender lane,
// and a sender may run fewer lanes than its direction has links (some held by
// another service on its chip), so the receiver cannot derive the count itself.
// A sender that fails to build sends zeros (`lane_links` null), so its receiver
// fails too instead of waiting here forever.
void send_sender_lanes(
    const std::shared_ptr<distributed::multihost::DistributedContext>& ctx,
    distributed::multihost::Rank receiver_rank,
    const std::vector<distributed::MeshCoordinate>& coords,
    const LaneLinks* lane_links) {
    std::vector<uint32_t> words(coords.size(), 0u);
    if (lane_links != nullptr) {
        for (size_t i = 0; i < coords.size(); ++i) {
            words[i] = static_cast<uint32_t>(lane_links->at(coords[i]).size());
        }
    }
    ctx->send(
        ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(words.data()), words.size() * sizeof(uint32_t)),
        receiver_rank,
        distributed::multihost::Tag{kLaneCountExchangeTag});
}

std::map<distributed::MeshCoordinate, uint32_t> recv_sender_lanes(
    const std::shared_ptr<distributed::multihost::DistributedContext>& ctx,
    distributed::multihost::Rank sender_rank,
    const std::vector<distributed::MeshCoordinate>& coords) {
    std::vector<uint32_t> words(coords.size(), 0u);
    ctx->recv(
        ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(words.data()), words.size() * sizeof(uint32_t)),
        sender_rank,
        distributed::multihost::Tag{kLaneCountExchangeTag});
    std::map<distributed::MeshCoordinate, uint32_t> lanes;
    for (size_t i = 0; i < coords.size(); ++i) {
        TT_FATAL(
            words[i] >= 1,
            "D2DStreamService: the sender (rank {}) failed to build its service at coord {}; see its log",
            *sender_rank,
            coords[i]);
        lanes.emplace(coords[i], words[i]);
    }
    return lanes;
}

// Relay mode (D2DStreamConfig::relay). A cable links an exit chip (sender mesh) to an entry chip
// (receiver mesh); each coord uses one cable, and slot k is its k-th coord on both ends.
// Both sides build the same plan from the fabric nodes known after the MeshSocket handshake.

struct RelayCable {
    distributed::MeshCoordinate exit_coord;
    distributed::MeshCoordinate entry_coord;
    std::vector<distributed::MeshCoordinate> slots;
};

struct RelayPlan {
    std::vector<RelayCable> cables;
    // coord -> (cable index, slot index)
    std::map<distributed::MeshCoordinate, std::pair<uint32_t, uint32_t>> slot_of;
    uint32_t max_slots = 0;
};

// Hops between two coords of one mesh, with wrap-around on the axes the fabric closes into a
// ring (torus). Only used to pick a cable, so a rough count is enough.
uint32_t relay_hops(
    const distributed::MeshCoordinate& a, const distributed::MeshCoordinate& b, const distributed::MeshShape& shape) {
    const auto fabric = tt::tt_fabric::GetFabricConfig();
    const bool wrap_y = fabric == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_Y ||
                        fabric == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY;
    const bool wrap_x = fabric == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_X ||
                        fabric == tt::tt_fabric::FabricConfig::FABRIC_2D_TORUS_XY;
    uint32_t hops = 0;
    for (size_t dim = 0; dim < a.dims(); ++dim) {
        const uint32_t n = shape[dim];
        const uint32_t d = a[dim] > b[dim] ? a[dim] - b[dim] : b[dim] - a[dim];
        const bool wrap = dim == 0 ? wrap_y : wrap_x;  // dim 0 = rows (Y), dim 1 = columns (X)
        hops += wrap ? std::min(d, n - d) : d;
    }
    return hops;
}

// is_sender: this process owns the sender nodes (it asks the control plane about its own chips).
RelayPlan make_relay_plan(
    const std::vector<distributed::MeshCoordinate>& coords,
    const std::map<distributed::MeshCoordinate, tt::tt_fabric::FabricNodeId>& sender_nodes,
    const std::map<distributed::MeshCoordinate, tt::tt_fabric::FabricNodeId>& receiver_nodes,
    const distributed::MeshShape& shape,
    bool is_sender) {
    using tt::tt_fabric::RoutingDirection;
    const auto& local_nodes = is_sender ? sender_nodes : receiver_nodes;
    const auto& peer_nodes = is_sender ? receiver_nodes : sender_nodes;

    // 1. Find every direct link between the two meshes, as (exit coord, entry coord) pairs seen from this side's
    //    chips. Two links between the same two chips give one pair: one cable, whose links become its lanes.
    std::set<std::pair<distributed::MeshCoordinate, distributed::MeshCoordinate>> candidates;
    for (const auto& c : coords) {
        const auto node = local_nodes.at(c);
        for (const auto dir :
             {RoutingDirection::N,
              RoutingDirection::E,
              RoutingDirection::S,
              RoutingDirection::W,
              RoutingDirection::Z}) {
            const auto neighbors = tt::tt_fabric::pipeline_get_chip_neighbors(node, dir);
            for (const auto& [peer_coord, peer_node] : peer_nodes) {
                auto it = neighbors.find(*peer_node.mesh_id);
                if (it == neighbors.end() ||
                    std::find(it->second.begin(), it->second.end(), peer_node.chip_id) == it->second.end()) {
                    continue;
                }
                candidates.emplace(is_sender ? c : peer_coord, is_sender ? peer_coord : c);
            }
        }
    }

    // 2. Choose the cables: at most one cable per chip, so a chip linked to two chips of the other mesh keeps
    //    only the first. A coord may not be the exit of one cable and the entry of another: its data would
    //    have to use both.
    RelayPlan plan;
    std::set<distributed::MeshCoordinate> exits;
    std::set<distributed::MeshCoordinate> entries;
    for (const auto& [exit_coord, entry_coord] : candidates) {
        if (exits.contains(exit_coord) || entries.contains(entry_coord) ||
            (entries.contains(exit_coord) && exit_coord != entry_coord) ||
            (exits.contains(entry_coord) && exit_coord != entry_coord)) {
            continue;
        }
        exits.insert(exit_coord);
        entries.insert(entry_coord);
        plan.cables.push_back(RelayCable{exit_coord, entry_coord, {}});
    }
    TT_FATAL(!plan.cables.empty(), "D2DStreamService relay: no direct link between the sender and receiver meshes");

    // 3. Give every coord to a cable and a slot on it. Slot k is DRAM staging tensor k on the cable's exit and
    //    entry chips: the exit chip collects that coord's data there before the cable, and the entry chip
    //    receives it there before forwarding it to the coord. A cable chip's own coord skips the staging.
    const uint32_t num_cables = static_cast<uint32_t>(plan.cables.size());
    const uint32_t cap = std::max<uint32_t>(2, (coords.size() + num_cables - 1) / num_cables);
    std::vector<uint32_t> load(num_cables, 0);
    std::map<distributed::MeshCoordinate, uint32_t> cable_of;
    // 3a. A cable chip's own coord goes to its cable: its data is already at the cable on that side.
    for (uint32_t p = 0; p < num_cables; ++p) {
        for (const auto& c : {plan.cables[p].exit_coord, plan.cables[p].entry_coord}) {
            if (cable_of.emplace(c, p).second) {
                load[p] += 1;
            }
        }
    }
    // 3b. Every other coord goes to the cable with the fewest hops (exit side + entry side), at most
    //     cap coords per cable (ceil(coords / cables), at least 2); on a tie, the cable with fewer coords.
    for (const auto& c : coords) {
        if (cable_of.contains(c)) {
            continue;
        }
        uint32_t best = num_cables;
        uint32_t best_cost = std::numeric_limits<uint32_t>::max();
        for (uint32_t p = 0; p < num_cables; ++p) {
            if (load[p] >= cap) {
                continue;
            }
            const uint32_t cost =
                relay_hops(c, plan.cables[p].exit_coord, shape) + relay_hops(plan.cables[p].entry_coord, c, shape);
            if (cost < best_cost || (cost == best_cost && load[p] < load[best])) {
                best = p;
                best_cost = cost;
            }
        }
        TT_FATAL(best < num_cables, "D2DStreamService relay: no cable left for coord {}", c);
        cable_of.emplace(c, best);
        load[best] += 1;
    }
    // 3c. Number each cable's slots in sorted coord order, so both sides number them the same way.
    for (const auto& [c, p] : cable_of) {  // std::map order = sorted coords
        plan.slot_of[c] = {p, static_cast<uint32_t>(plan.cables[p].slots.size())};
        plan.cables[p].slots.push_back(c);
    }
    // 3d. Check the slot count per cable, size the staging tensors (max_slots) and log each cable.
    for (const auto& cable : plan.cables) {
        TT_FATAL(
            cable.slots.size() <= 32, "D2DStreamService relay: {} coords on one cable (max 32)", cable.slots.size());
        plan.max_slots = std::max<uint32_t>(plan.max_slots, cable.slots.size());
        log_info(
            tt::LogMetal,
            "D2DStreamService relay: cable {} -> {} carries {} coords",
            cable.exit_coord,
            cable.entry_coord,
            cable.slots.size());
    }
    return plan;
}

// This side's relay state: the plan, the staging tensors (max_slots of them, the same
// address on every chip) and, per cable, the relay words on this side's relay core.
struct RelayCableWords {
    DeviceAddr arrived_base = 0;  // [slot], word_stride apart
    DeviceAddr acked_base = 0;    // exit: cable_acked[slot]; entry: recv_acked[slot]
    DeviceAddr md_base = 0;       // [slot], metadata_span apart (0 = no metadata)
    DeviceAddr go_slot_addr = 0;
    // Lane sync of this relay (master -> sub, sub -> master). The receiver side has no
    // go/done words of its own, so both sides allocate these here.
    DeviceAddr go_count_addr = 0;
    DeviceAddr done_count_addr = 0;
};

struct RelayLocal {
    RelayPlan plan;
    std::map<distributed::MeshCoordinate, tt::tt_fabric::FabricNodeId> sender_nodes;
    std::map<distributed::MeshCoordinate, tt::tt_fabric::FabricNodeId> receiver_nodes;
    std::vector<Tensor> staging;
    std::vector<RelayCableWords> words;  // [cable]
    uint32_t word_stride = 0;
    // Service-core L1 the relay words use; freed by the handle destructor.
    std::vector<std::pair<distributed::MeshCoordinate, DeviceAddr>> l1_allocs;
    // What build_relay_* needs from the config (it runs after finalize_*).
    CommonPlan common;
    uint32_t max_lanes = 0;
    std::map<distributed::MeshCoordinate, DeviceAddr> receiver_tensor_addrs;  // sender side only
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

    // Relay mode only (null otherwise). The workload above stays empty until build_relay_sender.
    std::shared_ptr<CMAKE_UNIQUE_NAMESPACE::RelayLocal> relay;
    // The forwarding links this coord's lanes hold, one per lane: min(max_sender_lanes,
    // free forwarding links), so fewer than the direction has when another D2D service
    // on the chip holds one (see FabricLinkRegistry). The receiver learns the counts
    // from the sender.
    CMAKE_UNIQUE_NAMESPACE::LaneLinks lane_links;
    // The routers this service holds. Last member, so the claims are released only
    // after the handle destructor has stopped the kernels that use them.
    CMAKE_UNIQUE_NAMESPACE::FabricLinkClaims link_claims;
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

    std::optional<D2DStageGateConfig> stage_gate;
    std::map<tt::tt_metal::distributed::MeshCoordinate, DeviceAddr> stage_gate_addrs;

    DeviceAddr stage_gate_addr(
        const tt::tt_metal::distributed::MeshCoordinate& coord, uint32_t gate, const char* fn) const {
        TT_FATAL(stage_gate.has_value(), "D2DStreamServiceReceiver::{}: stage gate not configured", fn);
        TT_FATAL(
            gate < stage_gate->num_gates,
            "D2DStreamServiceReceiver::{}: gate {} out of range (num_gates = {})",
            fn,
            gate,
            stage_gate->num_gates);
        auto it = stage_gate_addrs.find(coord);
        TT_FATAL(it != stage_gate_addrs.end(), "D2DStreamServiceReceiver::{}: no stage gate at coord {}", fn, coord);
        return it->second + (gate * tt::tt_metal::hal::get_l1_alignment());
    }

    // Persistent receiver workload, launched once at create_pair.
    std::unique_ptr<tt::tt_metal::distributed::MeshWorkload> workload;
    bool launched = false;

    // Cached lease workloads (LEASE mode only), mirror of the sender's. Null in OWN mode.
    std::unique_ptr<tt::tt_metal::distributed::MeshWorkload> lease_wait_workload;
    std::unique_ptr<tt::tt_metal::distributed::MeshWorkload> lease_release_workload;

    // Relay mode only (null otherwise). The workload above stays empty until build_relay_receiver.
    std::shared_ptr<CMAKE_UNIQUE_NAMESPACE::RelayLocal> relay;
    // The router this service's credit-return connection holds. Last member, as on
    // the sender.
    CMAKE_UNIQUE_NAMESPACE::FabricLinkClaims link_claims;
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
        auto* mesh = impl_->mesh_device.get();
        if (mesh == nullptr) {
            // No device-side state to tear down; just drop the host-side objects.
            impl_->workload.reset();
            impl_->socket.reset();
            return;
        }
        auto& svc = tt::tt_metal::internal::service_core_manager(*mesh);

        // 1. Signal termination + drain the persistent sender kernel.
        if (impl_->launched) {
            std::vector<uint32_t> terminate_word{kTerminateValue};
            for (const auto& [coord, addr] : impl_->termination_addrs) {
                tt::tt_metal::detail::WriteToDeviceL1(
                    tt::tt_metal::internal::get_device(*mesh, coord),
                    impl_->service_cores.at(coord),
                    static_cast<uint32_t>(addr),
                    terminate_word);
            }
            tt::tt_metal::distributed::Finish(mesh->mesh_command_queue());
            for (const auto& [coord, core] : impl_->service_cores) {
                svc.wait_done(tt::tt_metal::internal::get_device(*mesh, coord), core);
            }
        }
        impl_->workload.reset();

        // 2. Free the service-core L1 words (termination + data_ready_counter).
        if (mesh != nullptr) {
            for (const auto& [coord, addr] : impl_->termination_addrs) {
                svc.deallocate_l1(
                    tt::tt_metal::internal::get_device(*mesh, coord), impl_->service_cores.at(coord), addr);
            }
            for (const auto& [coord, addr] : impl_->data_ready_counter_addrs) {
                svc.deallocate_l1(
                    tt::tt_metal::internal::get_device(*mesh, coord), impl_->service_cores.at(coord), addr);
            }
            for (const auto& [coord, addr] : impl_->link_grant_addrs) {
                svc.deallocate_l1(
                    tt::tt_metal::internal::get_device(*mesh, coord), impl_->service_cores.at(coord), addr);
            }
            for (const auto& [coord, addr] : impl_->go_count_addrs) {
                svc.deallocate_l1(
                    tt::tt_metal::internal::get_device(*mesh, coord), impl_->service_cores.at(coord), addr);
            }
            for (const auto& [coord, addr] : impl_->done_count_addrs) {
                svc.deallocate_l1(
                    tt::tt_metal::internal::get_device(*mesh, coord), impl_->service_cores.at(coord), addr);
            }
            for (const auto& [coord, addr] : impl_->metadata_addrs) {
                svc.deallocate_l1(
                    tt::tt_metal::internal::get_device(*mesh, coord), impl_->service_cores.at(coord), addr);
            }
            if (impl_->relay) {
                for (const auto& [coord, addr] : impl_->relay->l1_allocs) {
                    svc.deallocate_l1(mesh->get_device(coord), impl_->service_cores.at(coord), addr);
                }
            }
        }

        // 3. Drop the socket (frees its config buffer), then release the cores.
        impl_->socket.reset();
        if (mesh != nullptr) {
            for (const auto& [coord, core] : impl_->service_cores) {
                svc.release(tt::tt_metal::internal::get_device(*mesh, coord), {core});
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

uint32_t D2DStreamServiceSender::get_num_lanes(const tt::tt_metal::distributed::MeshCoordinate& coord) const {
    auto it = impl_->lane_links.find(coord);
    TT_FATAL(it != impl_->lane_links.end(), "D2DStreamServiceSender::get_num_lanes: no sender at coord {}", coord);
    return static_cast<uint32_t>(it->second.size());
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
        auto* mesh = impl_->mesh_device.get();
        if (mesh == nullptr) {
            // No device-side state to tear down; just drop the host-side objects.
            impl_->workload.reset();
            impl_->socket.reset();
            return;
        }
        auto& svc = tt::tt_metal::internal::service_core_manager(*mesh);

        // 1. Signal termination + drain the persistent receiver kernel.
        if (impl_->launched) {
            std::vector<uint32_t> terminate_word{kTerminateValue};
            for (const auto& [coord, addr] : impl_->termination_addrs) {
                tt::tt_metal::detail::WriteToDeviceL1(
                    tt::tt_metal::internal::get_device(*mesh, coord),
                    impl_->service_cores.at(coord),
                    static_cast<uint32_t>(addr),
                    terminate_word);
            }
            tt::tt_metal::distributed::Finish(mesh->mesh_command_queue());
            for (const auto& [coord, core] : impl_->service_cores) {
                svc.wait_done(tt::tt_metal::internal::get_device(*mesh, coord), core);
            }
        }
        impl_->workload.reset();

        // 2. Free the service-core L1 words (termination + consumed_counter).
        if (mesh != nullptr) {
            for (const auto& [coord, addr] : impl_->termination_addrs) {
                svc.deallocate_l1(
                    tt::tt_metal::internal::get_device(*mesh, coord), impl_->service_cores.at(coord), addr);
            }
            for (const auto& [coord, addr] : impl_->consumed_counter_addrs) {
                svc.deallocate_l1(
                    tt::tt_metal::internal::get_device(*mesh, coord), impl_->service_cores.at(coord), addr);
            }
            for (const auto& [coord, addr] : impl_->stage_gate_addrs) {
                svc.deallocate_l1(
                    tt::tt_metal::internal::get_device(*mesh, coord), impl_->service_cores.at(coord), addr);
            }
            for (const auto& [coord, addr] : impl_->link_grant_addrs) {
                svc.deallocate_l1(
                    tt::tt_metal::internal::get_device(*mesh, coord), impl_->service_cores.at(coord), addr);
            }
            if (impl_->relay) {
                for (const auto& [coord, addr] : impl_->relay->l1_allocs) {
                    svc.deallocate_l1(mesh->get_device(coord), impl_->service_cores.at(coord), addr);
                }
            }
        }

        // 3. Drop the socket (frees its config + data FIFO buffers), then
        //    release the cores.
        impl_->socket.reset();
        if (mesh != nullptr) {
            for (const auto& [coord, core] : impl_->service_cores) {
                svc.release(tt::tt_metal::internal::get_device(*mesh, coord), {core});
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

// Gates are spaced by the L1 alignment so any NoC write can open one.
DeviceAddr D2DStageGateDescriptor::gate_addr(uint32_t gate) const {
    return gate_base_addr + (gate * tt::tt_metal::hal::get_l1_alignment());
}

uint32_t D2DStreamServiceReceiver::get_num_stage_gates() const {
    return impl_->stage_gate ? impl_->stage_gate->num_gates : 0;
}

D2DStageGateDescriptor D2DStreamServiceReceiver::get_stage_gate_descriptor(
    const tt::tt_metal::distributed::MeshCoordinate& coord) const {
    const DeviceAddr base = impl_->stage_gate_addr(coord, 0, "get_stage_gate_descriptor");
    auto* mesh = impl_->mesh_device.get();
    return D2DStageGateDescriptor{
        .fabric_node_id = mesh->get_fabric_node_id(coord),
        .service_core_noc = tt::tt_metal::internal::get_device(*mesh, coord)
                                ->worker_core_from_logical_core(impl_->service_cores.at(coord)),
        .gate_base_addr = base,
    };
}

void D2DStreamServiceReceiver::set_stage_gate(
    const tt::tt_metal::distributed::MeshCoordinate& coord, uint32_t gate, bool open) {
    const DeviceAddr addr = impl_->stage_gate_addr(coord, gate, "set_stage_gate");
    std::vector<uint32_t> word{
        static_cast<uint32_t>(open ? d2d_stream::GateState::Open : d2d_stream::GateState::Closed)};
    tt::tt_metal::detail::WriteToDeviceL1(
        tt::tt_metal::internal::get_device(*impl_->mesh_device, coord),
        impl_->service_cores.at(coord),
        static_cast<uint32_t>(addr),
        word);
}

bool D2DStreamServiceReceiver::is_stage_gate_open(
    const tt::tt_metal::distributed::MeshCoordinate& coord, uint32_t gate) const {
    const DeviceAddr addr = impl_->stage_gate_addr(coord, gate, "is_stage_gate_open");
    std::vector<uint32_t> word;
    tt::tt_metal::detail::ReadFromDeviceL1(
        tt::tt_metal::internal::get_device(*impl_->mesh_device, coord),
        impl_->service_cores.at(coord),
        static_cast<uint32_t>(addr),
        sizeof(uint32_t),
        word);
    return word.at(0) == static_cast<uint32_t>(d2d_stream::GateState::Open);
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
    const std::optional<D2DStageGateConfig>& stage_gate,
    uint32_t stage_gate_addr,
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
            .named_compile_args =
                {
                    {"terminate_value", kTerminateValue},
                    {"num_stage_gates", stage_gate ? stage_gate->num_gates : 0},
                    {"stage_gate_addr", stage_gate_addr},
                    {"stage_gate_stride", tt::tt_metal::hal::get_l1_alignment()},
                    {"stage_gate_slot_offset", stage_gate ? stage_gate->slot_id_offset_bytes : 0},
                    {"stage_gate_mode_offset", stage_gate ? stage_gate->gate_mode_offset_bytes : 0},
                    {"gate_open_value", static_cast<uint32_t>(d2d_stream::GateState::Open)},
                    {"gate_closed_value", static_cast<uint32_t>(d2d_stream::GateState::Closed)},
                    {"gate_close_on_transit_mode", static_cast<uint32_t>(d2d_stream::GateMode::CloseOnTransit)},
                },
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
    uint32_t metadata_span_bytes,
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
            metadata_span_bytes,           // [19] L1-aligned bytes shipped from the staging buffer
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
                .named_compile_args = {{"terminate_value", kTerminateValue}},
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

        auto* device = tt::tt_metal::internal::get_device(*mesh, coord);
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

// Relay mode, both sides: build the plan and allocate this side's relay state. Called by
// build_*_side after the socket region is reserved, so the words cannot alias it.
std::shared_ptr<RelayLocal> allocate_relay(
    const std::shared_ptr<distributed::MeshDevice>& mesh,
    const distributed::MeshSocket& socket,
    const std::map<distributed::MeshCoordinate, CoreCoord>& service_cores,
    const Tensor& backing,
    const std::vector<distributed::MeshCoordinate>& coords,
    const CommonPlan& common,
    const D2DStreamConfig& cfg,
    bool is_sender) {
    TT_FATAL(
        cfg.share_fabric_links, "D2DStreamService relay: needs share_fabric_links (the relays hold links per lease)");
    TT_FATAL(!enabled_stage_gate(cfg), "D2DStreamService relay: stage_gate is not supported");
    TT_FATAL(
        tt::tt_fabric::is_2d_fabric_config(tt::tt_fabric::GetFabricConfig()),
        "D2DStreamService relay: needs a 2D fabric (the relays route by chip id)");

    auto relay = std::make_shared<RelayLocal>();
    for (const auto& c : coords) {
        relay->sender_nodes.emplace(
            c,
            is_sender ? mesh->get_fabric_node_id(c)
                      : socket.get_fabric_node_id(distributed::SocketEndpoint::SENDER, c));
        relay->receiver_nodes.emplace(
            c,
            is_sender ? socket.get_fabric_node_id(distributed::SocketEndpoint::RECEIVER, c)
                      : mesh->get_fabric_node_id(c));
    }
    relay->plan = make_relay_plan(coords, relay->sender_nodes, relay->receiver_nodes, mesh->shape(), is_sender);
    relay->common = common;
    relay->max_lanes = cfg.max_sender_lanes;
    relay->word_stride = common.l1_alignment;

    for (uint32_t k = 0; k < relay->plan.max_slots; ++k) {
        relay->staging.push_back(create_device_tensor(backing.tensor_spec(), mesh.get(), backing.tensor_topology()));
    }

    auto& svc = tt::tt_metal::internal::service_core_manager(*mesh);
    for (const auto& cable : relay->plan.cables) {
        const auto coord = is_sender ? cable.exit_coord : cable.entry_coord;
        auto* device = mesh->get_device(coord);
        const CoreCoord core = service_cores.at(coord);
        auto alloc = [&](uint32_t bytes) {
            const DeviceAddr addr = svc.allocate_l1(device, core, bytes);
            std::vector<uint32_t> zero(bytes / sizeof(uint32_t), 0u);
            tt::tt_metal::detail::WriteToDeviceL1(device, core, static_cast<uint32_t>(addr), zero);
            relay->l1_allocs.emplace_back(coord, addr);
            return addr;
        };
        const auto num_slots = static_cast<uint32_t>(cable.slots.size());
        RelayCableWords words;
        words.arrived_base = alloc(num_slots * relay->word_stride);
        words.acked_base = alloc(num_slots * relay->word_stride);
        words.go_slot_addr = alloc(relay->word_stride);
        words.go_count_addr = alloc(relay->word_stride);
        words.done_count_addr = alloc(relay->word_stride);
        words.md_base = common.metadata_enabled ? alloc(num_slots * common.metadata_span_bytes) : 0;
        relay->words.push_back(words);
    }
    return relay;
}

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
    LaneLinks lane_links;
    std::shared_ptr<RelayLocal> relay;  // relay mode only
};

SenderSideResources build_sender_side(
    const std::shared_ptr<distributed::MeshDevice>& mesh,
    const distributed::MeshSocket& sender_socket,
    const std::map<distributed::MeshCoordinate, CoreCoord>& service_cores,
    const std::map<distributed::MeshCoordinate, DeviceAddr>& receiver_tensor_addrs,
    const Tensor& backing,
    const std::vector<distributed::MeshCoordinate>& coords,
    const CommonPlan& common,
    const D2DStreamConfig& cfg,
    FabricLinkClaims& link_claims) {
    auto& svc = tt::tt_metal::internal::service_core_manager(*mesh);
    const uint32_t num_workers = core_range_size(cfg.sender_worker_cores);

    // Choose each coord's lanes first, before any allocation, so a chip with no free
    // router fails cleanly. One RISC + one link per lane, capped by config and by the
    // forwarding links no other D2D service on the chip holds. Relay mode claims none:
    // its kernels pick their own links in build_relay_sender.
    TT_FATAL(cfg.max_sender_lanes >= 1, "D2DStreamService: max_sender_lanes must be >= 1");
    validate_fabric_link_group(cfg);
    LaneLinks lane_links_per_coord;
    const std::vector<distributed::MeshCoordinate> no_claim_coords;
    for (const auto& coord : cfg.relay ? no_claim_coords : coords) {
        const auto sender_node = mesh->get_fabric_node_id(coord);
        const auto receiver_node = sender_socket.get_fabric_node_id(distributed::SocketEndpoint::RECEIVER, coord);
        auto lanes = claim_forwarding_links(
            *mesh,
            coord,
            sender_node,
            receiver_node,
            cfg.max_sender_lanes,
            D2DServiceRole::Sender,
            cfg.fabric_link_group,
            link_claims);
        lane_links_per_coord.emplace(coord, std::move(lanes));
    }

    // Reserve the socket config buffer's L1 footprint in each service core's
    // per-core allocator BEFORE allocating any words (both grow top-down from
    // L1_END; without this the words alias the socket buffers). The sender owns
    // only a config buffer — no data FIFO.
    const DeviceAddr socket_lo = sender_socket.get_config_buffer()->address();
    for (const auto& coord : coords) {
        svc.reserve_l1_to_top(tt::tt_metal::internal::get_device(*mesh, coord), service_cores.at(coord), socket_lo);
    }

    auto termination_addrs = allocate_service_core_words(mesh, service_cores);
    auto data_ready_counter_addrs = allocate_service_core_words(mesh, service_cores);
    auto link_grant_addrs = allocate_service_core_words(mesh, service_cores);
    // Lane-sync words (multi-lane sender). Zero-initialised, like the other words.
    auto go_count_addrs = allocate_service_core_words(mesh, service_cores);
    auto done_count_addrs = allocate_service_core_words(mesh, service_cores);

    std::map<distributed::MeshCoordinate, DeviceAddr> metadata_addrs;
    if (common.metadata_enabled) {
        // Size the metadata L1 buffer to the shipped span (metadata_size_bytes rounded
        // up to the L1 alignment), NOT to a socket page: the sender kernel fabric-writes
        // exactly metadata_span_bytes from this address, so the allocation must cover the
        // span or the write would over-read adjacent service-core L1 (counters /
        // reserved socket region) and ship those bytes over fabric. The worker writes
        // metadata_size_bytes into the front; the zero-init keeps the alignment padding
        // tail clean.
        const uint32_t aligned_md = common.metadata_span_bytes;
        std::vector<uint32_t> zero(aligned_md / sizeof(uint32_t), 0u);
        for (const auto& coord : coords) {
            auto* d = tt::tt_metal::internal::get_device(*mesh, coord);
            const CoreCoord core = service_cores.at(coord);
            const DeviceAddr addr = svc.allocate_l1(d, core, aligned_md);
            tt::tt_metal::detail::WriteToDeviceL1(d, core, static_cast<uint32_t>(addr), zero);
            metadata_addrs.emplace(coord, addr);
        }
    }

    auto consumed_sem = ttnn::global_semaphore::create_global_semaphore(
        mesh.get(), CoreRangeSet(cfg.sender_worker_cores), /*initial_value=*/0, BufferType::L1);

    // Relay mode: allocate the relay state; build_relay_sender builds the programs later.
    std::shared_ptr<RelayLocal> relay;
    if (cfg.relay) {
        relay = allocate_relay(mesh, sender_socket, service_cores, backing, coords, common, cfg, /*is_sender=*/true);
        relay->receiver_tensor_addrs = receiver_tensor_addrs;
    }
    const std::vector<distributed::MeshCoordinate> no_coords;

    constexpr bool worker_sync_enabled = true;
    auto workload = std::make_unique<distributed::MeshWorkload>();
    const uint32_t socket_config_addr = static_cast<uint32_t>(sender_socket.get_config_buffer()->address());

    for (const auto& coord : cfg.relay ? no_coords : coords) {
        const Buffer* send_buf = backing.mesh_buffer().get_device_buffer(coord);
        TT_FATAL(send_buf != nullptr, "D2DStreamService: sender device buffer missing for coord {}", coord);
        auto* device = tt::tt_metal::internal::get_device(*mesh, coord);
        const CoreCoord service_core = service_cores.at(coord);

        const auto sender_node = mesh->get_fabric_node_id(coord);
        const auto receiver_node = sender_socket.get_fabric_node_id(distributed::SocketEndpoint::RECEIVER, coord);
        const std::vector<uint32_t>& lane_links = lane_links_per_coord.at(coord);
        const uint32_t num_lanes = static_cast<uint32_t>(lane_links.size());

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
                common.metadata_span_bytes,
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
        .lane_links = std::move(lane_links_per_coord),
        .relay = std::move(relay),
    };
}

struct ReceiverSideResources {
    std::map<distributed::MeshCoordinate, DeviceAddr> consumed_counter_addrs;
    std::map<distributed::MeshCoordinate, DeviceAddr> termination_addrs;
    std::map<distributed::MeshCoordinate, DeviceAddr> link_grant_addrs;
    std::optional<GlobalSemaphore> data_ready_sem;
    std::shared_ptr<distributed::MeshBuffer> metadata_buffer;
    DeviceAddr metadata_l1_addr = 0;
    std::map<distributed::MeshCoordinate, DeviceAddr> stage_gate_addrs;
    std::unique_ptr<distributed::MeshWorkload> workload;
    // Cached lease workloads (LEASE mode only), mirror of the sender's. Null in OWN mode.
    std::unique_ptr<distributed::MeshWorkload> lease_wait_workload;
    std::unique_ptr<distributed::MeshWorkload> lease_release_workload;
    std::shared_ptr<RelayLocal> relay;  // relay mode only
};

ReceiverSideResources build_receiver_side(
    const std::shared_ptr<distributed::MeshDevice>& mesh,
    const distributed::MeshSocket& receiver_socket,
    const std::map<distributed::MeshCoordinate, CoreCoord>& service_cores,
    const Tensor& backing,
    const std::vector<distributed::MeshCoordinate>& coords,
    const CommonPlan& common,
    const D2DStreamConfig& cfg,
    const std::map<distributed::MeshCoordinate, uint32_t>& sender_lanes,
    FabricLinkClaims& link_claims) {
    auto& svc = tt::tt_metal::internal::service_core_manager(*mesh);
    const uint32_t num_workers = core_range_size(cfg.receiver_worker_cores);

    // Choose each coord's credit-return router first, before any allocation: one
    // forwarding link toward the sender that no other D2D service on the chip holds.
    // Relay mode claims none, as on the sender.
    validate_fabric_link_group(cfg);
    std::map<distributed::MeshCoordinate, uint32_t> ack_link_per_coord;
    const std::vector<distributed::MeshCoordinate> no_claim_coords;
    for (const auto& coord : cfg.relay ? no_claim_coords : coords) {
        const auto receiver_node = mesh->get_fabric_node_id(coord);
        const auto sender_node = receiver_socket.get_fabric_node_id(distributed::SocketEndpoint::SENDER, coord);
        const auto ack = claim_forwarding_links(
            *mesh,
            coord,
            receiver_node,
            sender_node,
            /*max_links=*/1,
            D2DServiceRole::Receiver,
            cfg.fabric_link_group,
            link_claims);
        ack_link_per_coord.emplace(coord, ack.front());
    }

    // Reserve the socket config buffer + data FIFO L1 footprint. They sit at the
    // top of L1, so the lower of the two addresses covers both in one reservation.
    const DeviceAddr socket_lo =
        std::min(receiver_socket.get_config_buffer()->address(), receiver_socket.get_data_buffer()->address());
    for (const auto& coord : coords) {
        svc.reserve_l1_to_top(tt::tt_metal::internal::get_device(*mesh, coord), service_cores.at(coord), socket_lo);
    }

    auto termination_addrs = allocate_service_core_words(mesh, service_cores);
    auto consumed_counter_addrs = allocate_service_core_words(mesh, service_cores);
    auto link_grant_addrs = allocate_service_core_words(mesh, service_cores);

    const auto gate = enabled_stage_gate(cfg);
    std::map<distributed::MeshCoordinate, DeviceAddr> stage_gate_addrs;
    if (gate) {
        stage_gate_addrs =
            allocate_service_core_words(mesh, service_cores, gate->num_gates, tt::tt_metal::hal::get_l1_alignment());
    }

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

    // Relay mode: allocate the relay state; build_relay_receiver builds the programs later.
    std::shared_ptr<RelayLocal> relay;
    if (cfg.relay) {
        relay = allocate_relay(mesh, receiver_socket, service_cores, backing, coords, common, cfg, /*is_sender=*/false);
    }
    const std::vector<distributed::MeshCoordinate> no_coords;

    constexpr bool worker_sync_enabled = true;
    auto workload = std::make_unique<distributed::MeshWorkload>();
    const uint32_t socket_config_addr = static_cast<uint32_t>(receiver_socket.get_config_buffer()->address());

    for (const auto& coord : cfg.relay ? no_coords : coords) {
        const Buffer* recv_buf = backing.mesh_buffer().get_device_buffer(coord);
        TT_FATAL(recv_buf != nullptr, "D2DStreamService: receiver device buffer missing for coord {}", coord);
        auto* device = tt::tt_metal::internal::get_device(*mesh, coord);
        const CoreCoord service_core = service_cores.at(coord);

        const auto receiver_node = mesh->get_fabric_node_id(coord);
        const auto sender_node = receiver_socket.get_fabric_node_id(distributed::SocketEndpoint::SENDER, coord);

        // The sender's lane count, as the sender chose it: how many data-landed
        // increments the receiver awaits per transfer (one per sender lane).
        const uint32_t num_lanes = sender_lanes.at(coord);

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
                gate,
                gate ? static_cast<uint32_t>(stage_gate_addrs.at(coord)) : 0u,
                receiver_node,
                sender_node,
                ack_link_per_coord.at(coord)));
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
        .stage_gate_addrs = std::move(stage_gate_addrs),
        .workload = std::move(workload),
        .lease_wait_workload = std::move(lease_wait_workload),
        .lease_release_workload = std::move(lease_release_workload),
        .relay = std::move(relay),
    };
}

// ---------------------------------------------------------------------------
// Relay mode: what the two sides trade, socket config patches, relay programs
// ---------------------------------------------------------------------------

constexpr int kRelayExchangeTag = 0x44325252;  // 'D2RR'

// What one side tells the other about its relay cores. Both sides build the same plan, so
// both know the cable and slot counts, and the packed words need no length header.
struct RelayPeerCable {
    uint32_t exit_row = 0, exit_col = 0, entry_row = 0, entry_col = 0;  // to check both plans match
    uint32_t noc_x = 0, noc_y = 0;                                      // relay service core, virtual NoC
    uint32_t arrived_base = 0, md_base = 0, acked_base = 0;
    uint32_t z_lanes = 0;  // cable lanes; only the sender side sets it
};
constexpr uint32_t kRelayCableWords = 10;

struct RelayPeer {
    std::vector<RelayPeerCable> cables;
    std::vector<uint32_t> staging_addrs;  // [slot], the same on every chip of that side
};

CoreCoord relay_noc(
    const std::shared_ptr<distributed::MeshDevice>& mesh,
    const std::map<distributed::MeshCoordinate, CoreCoord>& service_cores,
    const distributed::MeshCoordinate& c) {
    return mesh->get_device(c)->worker_core_from_logical_core(service_cores.at(c));
}

// Lanes of the unchanged sender kernel at `src` toward its exit relay. Same rule as
// build_sender_side; the exit relay waits for this many arrived increments.
uint32_t relay_source_lanes(
    const distributed::MeshDevice& mesh, const RelayLocal& relay, const distributed::MeshCoordinate& src, uint32_t p) {
    const auto links = tt::tt_fabric::get_forwarding_link_indices(
        mesh, relay.sender_nodes.at(src), relay.sender_nodes.at(relay.plan.cables[p].exit_coord));
    TT_FATAL(!links.empty(), "D2DStreamService relay: no fabric link from coord {} to its exit chip", src);
    return std::min<uint32_t>(relay.max_lanes, links.size());
}

// Lanes from the entry relay to target `t`. The relay kernels have at most two lanes.
uint32_t relay_target_lanes(
    const distributed::MeshDevice& mesh, const RelayLocal& relay, const distributed::MeshCoordinate& t, uint32_t p) {
    const auto links = tt::tt_fabric::get_forwarding_link_indices(
        mesh, relay.receiver_nodes.at(relay.plan.cables[p].entry_coord), relay.receiver_nodes.at(t));
    TT_FATAL(!links.empty(), "D2DStreamService relay: no fabric link from the entry chip to coord {}", t);
    return std::min<uint32_t>({relay.max_lanes, static_cast<uint32_t>(links.size()), 2u});
}

// Lanes over cable p (sender side only; the receiver side gets it from the peer).
uint32_t relay_cable_lanes(const distributed::MeshDevice& mesh, const RelayLocal& relay, uint32_t p) {
    const auto& cable = relay.plan.cables[p];
    const auto links = tt::tt_fabric::get_forwarding_link_indices(
        mesh, relay.sender_nodes.at(cable.exit_coord), relay.receiver_nodes.at(cable.entry_coord));
    TT_FATAL(!links.empty(), "D2DStreamService relay: no fabric link on cable {}", p);
    return std::min<uint32_t>({relay.max_lanes, static_cast<uint32_t>(links.size()), 2u});
}

RelayPeer make_relay_peer(
    const RelayLocal& relay,
    const std::shared_ptr<distributed::MeshDevice>& mesh,
    const std::map<distributed::MeshCoordinate, CoreCoord>& service_cores,
    bool is_sender) {
    RelayPeer peer;
    for (uint32_t p = 0; p < relay.plan.cables.size(); ++p) {
        const auto& cable = relay.plan.cables[p];
        const auto noc = relay_noc(mesh, service_cores, is_sender ? cable.exit_coord : cable.entry_coord);
        const auto& words = relay.words[p];
        peer.cables.push_back(RelayPeerCable{
            .exit_row = cable.exit_coord[0],
            .exit_col = cable.exit_coord[1],
            .entry_row = cable.entry_coord[0],
            .entry_col = cable.entry_coord[1],
            .noc_x = static_cast<uint32_t>(noc.x),
            .noc_y = static_cast<uint32_t>(noc.y),
            .arrived_base = static_cast<uint32_t>(words.arrived_base),
            .md_base = static_cast<uint32_t>(words.md_base),
            .acked_base = static_cast<uint32_t>(words.acked_base),
            .z_lanes = is_sender ? relay_cable_lanes(*mesh, relay, p) : 0u,
        });
    }
    for (const auto& t : relay.staging) {
        peer.staging_addrs.push_back(static_cast<uint32_t>(t.mesh_buffer().address()));
    }
    return peer;
}

// Both plans must name the same cables in the same order, or slot k means two different coords.
void check_relay_peer(const RelayPeer& peer, const RelayLocal& relay) {
    TT_FATAL(
        peer.cables.size() == relay.plan.cables.size() && peer.staging_addrs.size() == relay.plan.max_slots,
        "D2DStreamService relay: the peer plan has {} cables / {} slots, this side has {} / {}",
        peer.cables.size(),
        peer.staging_addrs.size(),
        relay.plan.cables.size(),
        relay.plan.max_slots);
    for (uint32_t p = 0; p < peer.cables.size(); ++p) {
        const auto& a = peer.cables[p];
        const auto& b = relay.plan.cables[p];
        TT_FATAL(
            a.exit_row == b.exit_coord[0] && a.exit_col == b.exit_coord[1] && a.entry_row == b.entry_coord[0] &&
                a.entry_col == b.entry_coord[1],
            "D2DStreamService relay: cable {} differs between the two sides ({} -> {} here)",
            p,
            b.exit_coord,
            b.entry_coord);
    }
}

// Trade RelayPeer with the other process, in the same send/recv order as exchange_service_cores.
RelayPeer exchange_relay_peer(
    const std::shared_ptr<distributed::multihost::DistributedContext>& ctx,
    bool is_sender,
    distributed::multihost::Rank peer_rank,
    const RelayPeer& local,
    const RelayLocal& relay) {
    std::vector<uint32_t> out;
    for (const auto& c : local.cables) {
        out.insert(
            out.end(),
            {c.exit_row,
             c.exit_col,
             c.entry_row,
             c.entry_col,
             c.noc_x,
             c.noc_y,
             c.arrived_base,
             c.md_base,
             c.acked_base,
             c.z_lanes});
    }
    out.insert(out.end(), local.staging_addrs.begin(), local.staging_addrs.end());
    std::vector<uint32_t> in(out.size(), 0u);

    const distributed::multihost::Tag tag{kRelayExchangeTag};
    const auto as_bytes = [](std::vector<uint32_t>& v) {
        return ttsl::Span<std::byte>(reinterpret_cast<std::byte*>(v.data()), v.size() * sizeof(uint32_t));
    };
    if (is_sender) {
        ctx->send(as_bytes(out), peer_rank, tag);
        ctx->recv(as_bytes(in), peer_rank, tag);
    } else {
        ctx->recv(as_bytes(in), peer_rank, tag);
        ctx->send(as_bytes(out), peer_rank, tag);
    }

    RelayPeer peer;
    const size_t n = local.cables.size();
    for (size_t p = 0; p < n; ++p) {
        const uint32_t* w = in.data() + p * kRelayCableWords;
        peer.cables.push_back(RelayPeerCable{w[0], w[1], w[2], w[3], w[4], w[5], w[6], w[7], w[8], w[9]});
    }
    peer.staging_addrs.assign(in.begin() + n * kRelayCableWords, in.end());
    check_relay_peer(peer, relay);
    return peer;
}

// Sender config (tt_metal/hw/inc/hostdev/socket.h): sender_socket_md (7 words, padded to the
// L1 alignment), then bytes_acked[1] (one aligned word), then the downstream encoding
// {mesh_id, chip_id, noc_y, noc_x}.
constexpr uint32_t kSenderSocketMdBytes = 7 * sizeof(uint32_t);

uint32_t sender_bytes_acked_addr(uint32_t config_addr, uint32_t l1_alignment) {
    return config_addr + tt::align(kSenderSocketMdBytes, l1_alignment);
}

// Point the sender at coord `core` to its exit relay: its bytes_sent increments go to the
// relay's arrived[k], its metadata to md slot k, and its route to the exit chip.
void patch_sender_config(
    tt::tt_metal::IDevice* device,
    const CoreCoord& core,
    uint32_t config_addr,
    uint32_t l1_alignment,
    uint32_t bytes_sent_addr,
    uint32_t fifo_addr,
    const tt::tt_fabric::FabricNodeId& relay_node,
    const CoreCoord& relay_noc) {
    const uint32_t enc_word =
        (tt::align(kSenderSocketMdBytes, l1_alignment) + tt::align(sizeof(uint32_t), l1_alignment)) / sizeof(uint32_t);
    std::vector<uint32_t> w;
    tt::tt_metal::detail::ReadFromDeviceL1(device, core, config_addr, (enc_word + 4) * sizeof(uint32_t), w);
    w[3] = bytes_sent_addr;  // downstream_bytes_sent_addr
    w[4] = fifo_addr;        // downstream_fifo_addr (where the metadata goes)
    w[enc_word + 0] = *relay_node.mesh_id;
    w[enc_word + 1] = static_cast<uint32_t>(relay_node.chip_id);
    w[enc_word + 2] = static_cast<uint32_t>(relay_noc.y);
    w[enc_word + 3] = static_cast<uint32_t>(relay_noc.x);
    tt::tt_metal::detail::WriteToDeviceL1(device, core, config_addr, w);
}

// Receiver config: receiver_socket_md is 6 words, then D2DSocketRecvInterface
// {upstream_mesh_id, upstream_chip_id, noc_y, noc_x, upstream_bytes_acked_addr}. Point the
// receiver's credits at the entry relay's recv_acked[k].
void patch_receiver_config(
    tt::tt_metal::IDevice* device,
    const CoreCoord& core,
    uint32_t config_addr,
    uint32_t bytes_acked_addr,
    const tt::tt_fabric::FabricNodeId& relay_node,
    const CoreCoord& relay_noc) {
    std::vector<uint32_t> w;
    tt::tt_metal::detail::ReadFromDeviceL1(device, core, config_addr, 11 * sizeof(uint32_t), w);
    w[6] = *relay_node.mesh_id;
    w[7] = static_cast<uint32_t>(relay_node.chip_id);
    w[8] = static_cast<uint32_t>(relay_noc.y);
    w[9] = static_cast<uint32_t>(relay_noc.x);
    w[10] = bytes_acked_addr;
    tt::tt_metal::detail::WriteToDeviceL1(device, core, config_addr, w);
}

// Both relays: one scratch CB (each lane uses its own ring of kSenderReadRingSlots pages) and
// one header CB (two headers per lane). The kernels add the sub lane's offsets themselves.
void create_relay_cbs(Program& program, const CoreCoord& core, uint32_t page_size, tt::tt_metal::DataType dtype) {
    const uint32_t scratch_bytes = 2 * kSenderReadRingSlots * page_size;
    auto scratch_cfg =
        CircularBufferConfig(scratch_bytes, {{tt::CBIndex::c_0, datatype_to_dataformat_converter(dtype)}})
            .set_page_size(tt::CBIndex::c_0, scratch_bytes);
    CreateCircularBuffer(program, core, scratch_cfg);
    const uint32_t header_size = tt::tt_fabric::get_tt_fabric_packet_header_size_bytes();
    auto ph_cfg = CircularBufferConfig(4 * header_size, {{kPacketHeaderCbIndex, tt::DataFormat::UInt32}})
                      .set_page_size(kPacketHeaderCbIndex, header_size);
    CreateCircularBuffer(program, core, ph_cfg);
}

// The words of the relay coord's own service (the relay takes over its job).
struct RelayCoordWords {
    uint32_t termination_addr = 0;
    uint32_t link_grant_addr = 0;
    uint32_t metadata_addr = 0;  // exit: own sender md buffer; entry: worker md buffer
};

const DataMovementProcessor kRelayProcs[2] = {DataMovementProcessor::RISCV_0, DataMovementProcessor::RISCV_1};
const NOC kRelayNocs[2] = {NOC::RISCV_0_default, NOC::RISCV_1_default};

// The exit relay of cable p. CT / RT layout must stay in sync with persistent_d2d_exit_relay.cpp.
Program build_exit_relay_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh,
    const RelayLocal& relay,
    uint32_t p,
    const RelayPeer& peer,
    const std::map<distributed::MeshCoordinate, CoreCoord>& service_cores,
    const Tensor& backing,
    const WorkerSyncArgs& ws,
    const RelayCoordWords& own,
    uint32_t socket_config_addr) {
    const auto& cable = relay.plan.cables[p];
    const auto& words = relay.words[p];
    const auto& common = relay.common;
    const auto e = cable.exit_coord;
    const CoreCoord core = service_cores.at(e);
    const auto exit_node = relay.sender_nodes.at(e);
    const auto entry_node = relay.receiver_nodes.at(cable.entry_coord);
    const auto num_slots = static_cast<uint32_t>(cable.slots.size());
    const uint32_t own_slot = relay.plan.slot_of.at(e).second;

    const auto z_links = tt::tt_fabric::get_forwarding_link_indices(*mesh, exit_node, entry_node);
    const uint32_t z_lanes = relay_cable_lanes(*mesh, relay, p);
    const auto cable_dir = tt::tt_fabric::pipeline_get_forwarding_direction(exit_node, entry_node);

    // Credit connections: one per first-hop direction toward the remote sources.
    std::vector<std::pair<tt::tt_fabric::FabricNodeId, uint32_t>> credit_conns;  // (a node that way, link)
    std::map<tt::tt_fabric::RoutingDirection, uint32_t> conn_of_dir;
    std::vector<uint32_t> credit_conn(num_slots, 0);
    for (uint32_t k = 0; k < num_slots; ++k) {
        if (k == own_slot) {
            continue;
        }
        const auto src_node = relay.sender_nodes.at(cable.slots[k]);
        const auto dir = tt::tt_fabric::pipeline_get_forwarding_direction(exit_node, src_node);
        TT_FATAL(dir.has_value(), "D2DStreamService relay: no route from exit chip {} to coord {}", e, cable.slots[k]);
        TT_FATAL(dir != cable_dir, "D2DStreamService relay: a credit route at {} leaves over the cable", e);
        const auto [it, added] = conn_of_dir.emplace(*dir, static_cast<uint32_t>(credit_conns.size()));
        if (added) {
            credit_conns.emplace_back(
                src_node, tt::tt_fabric::get_forwarding_link_indices(*mesh, exit_node, src_node).front());
        }
        credit_conn[k] = it->second;
    }

    auto program = CreateProgram();
    create_relay_cbs(program, core, common.tensor_page_size, backing.dtype());
    const auto accessor_ct = TensorAccessorArgs(*backing.mesh_buffer().get_device_buffer(e)).get_compile_time_args();
    const uint32_t total_pages = common.plan.num_socket_pages * common.plan.pages_per_chunk;
    const uint32_t half = (total_pages + 1) / 2;
    const auto backing_addr = static_cast<uint32_t>(backing.mesh_buffer().address());
    const uint32_t acked_addr = sender_bytes_acked_addr(socket_config_addr, common.l1_alignment);

    for (uint32_t lane = 0; lane < z_lanes; ++lane) {
        const bool master = lane == 0;
        const std::unordered_map<std::string, uint32_t> named = {
            {"terminate_value", kTerminateValue},
            {"termination_addr", own.termination_addr},
            {"link_grant_addr", own.link_grant_addr},
            {"is_master", master ? 1u : 0u},
            {"z_lanes", z_lanes},
            {"num_slots", num_slots},
            {"own_slot", own_slot},
            {"page_size", common.tensor_page_size},
            {"max_payload", common.fabric_max_payload_size},
            {"page_start", master ? 0u : half},
            {"page_end", (master && z_lanes > 1) ? half : total_pages},
            {"ring_slots", kSenderReadRingSlots},
            {"scratch_cb", static_cast<uint32_t>(tt::CBIndex::c_0)},
            {"header_cb", static_cast<uint32_t>(kPacketHeaderCbIndex)},
            {"metadata_enabled", common.metadata_enabled ? 1u : 0u},
            {"metadata_span", common.metadata_span_bytes},
            {"own_md_addr", own.metadata_addr},
            {"arrived_base", static_cast<uint32_t>(words.arrived_base)},
            {"md_base", static_cast<uint32_t>(words.md_base)},
            {"cable_acked_base", static_cast<uint32_t>(words.acked_base)},
            {"word_stride", relay.word_stride},
            {"go_count_addr", static_cast<uint32_t>(words.go_count_addr)},
            {"go_slot_addr", static_cast<uint32_t>(words.go_slot_addr)},
            {"done_count_addr", static_cast<uint32_t>(words.done_count_addr)},
            {"data_ready_counter_addr", ws.counter_addr},
            {"consumed_sem_addr", ws.data_ready_sem_addr},
            {"mcast_x_start", ws.mcast_noc_x_start},
            {"mcast_y_start", ws.mcast_noc_y_start},
            {"mcast_x_end", ws.mcast_noc_x_end},
            {"mcast_y_end", ws.mcast_noc_y_end},
            {"num_workers", ws.num_workers},
            {"entry_chip", static_cast<uint32_t>(entry_node.chip_id)},
            {"entry_mesh", *entry_node.mesh_id},
            {"entry_noc_x", peer.cables[p].noc_x},
            {"entry_noc_y", peer.cables[p].noc_y},
            {"entry_arrived_base", peer.cables[p].arrived_base},
            {"entry_md_base", peer.cables[p].md_base},
            {"num_credit_conns", master ? static_cast<uint32_t>(credit_conns.size()) : 0u},
        };
        auto kernel = CreateKernel(
            program,
            "ttnn/core/tensor/kernels/persistent_d2d_exit_relay.cpp",
            core,
            DataMovementConfig{
                .processor = kRelayProcs[lane],
                .noc = kRelayNocs[lane],
                .compile_args = accessor_ct,
                .named_compile_args = named,
            });

        // RT args: the cable connection on this lane's link, the credit connections
        // (master only), then the per-slot words.
        std::vector<uint32_t> rt;
        tt::tt_fabric::append_fabric_connection_rt_args(exit_node, entry_node, z_links[lane], program, core, rt);
        if (master) {
            for (const auto& [node, link] : credit_conns) {
                tt::tt_fabric::append_fabric_connection_rt_args(exit_node, node, link, program, core, rt);
            }
        }
        for (uint32_t k = 0; k < num_slots; ++k) {
            const auto& s = cable.slots[k];
            // Source: this chip's own backing, or staging k. Destination: the entry chip's
            // backing when slot k is the entry coord, else the entry side's staging k.
            rt.push_back(
                k == own_slot ? backing_addr : static_cast<uint32_t>(relay.staging[k].mesh_buffer().address()));
            rt.push_back(
                s == cable.entry_coord ? static_cast<uint32_t>(relay.receiver_tensor_addrs.at(s))
                                       : peer.staging_addrs[k]);
            if (!master) {
                continue;
            }
            if (k == own_slot) {
                rt.insert(rt.end(), {0u, 0u, 0u, 0u, 0u, 0u, 0u});
                continue;
            }
            const auto src_node = relay.sender_nodes.at(s);
            const auto src_noc = relay_noc(mesh, service_cores, s);
            rt.insert(
                rt.end(),
                {relay_source_lanes(*mesh, relay, s, p),
                 static_cast<uint32_t>(src_node.chip_id),
                 *src_node.mesh_id,
                 static_cast<uint32_t>(src_noc.x),
                 static_cast<uint32_t>(src_noc.y),
                 acked_addr,
                 credit_conn[k]});
        }
        SetRuntimeArgs(program, kernel, core, rt);
    }
    return program;
}

// The entry relay of cable p. CT / RT layout must stay in sync with persistent_d2d_entry_relay.cpp.
Program build_entry_relay_program(
    const std::shared_ptr<distributed::MeshDevice>& mesh,
    const RelayLocal& relay,
    uint32_t p,
    const RelayPeer& peer,
    const std::map<distributed::MeshCoordinate, CoreCoord>& service_cores,
    const Tensor& backing,
    const WorkerSyncArgs& ws,
    const RelayCoordWords& own,
    uint32_t socket_config_addr,
    uint32_t socket_fifo_addr) {
    const auto& cable = relay.plan.cables[p];
    const auto& words = relay.words[p];
    const auto& common = relay.common;
    const auto e = cable.entry_coord;
    const CoreCoord core = service_cores.at(e);
    const auto entry_node = relay.receiver_nodes.at(e);
    const auto exit_node = relay.sender_nodes.at(cable.exit_coord);
    const auto num_slots = static_cast<uint32_t>(cable.slots.size());
    const uint32_t own_slot = relay.plan.slot_of.at(e).second;
    const uint32_t z_lanes = peer.cables[p].z_lanes;
    TT_FATAL(z_lanes >= 1 && z_lanes <= 2, "D2DStreamService relay: bad cable lane count {} from the peer", z_lanes);
    const auto z_link = tt::tt_fabric::get_forwarding_link_indices(*mesh, entry_node, exit_node).front();

    // Per lane: one connection per first-hop direction toward the targets that use that lane.
    std::vector<std::pair<tt::tt_fabric::FabricNodeId, uint32_t>> conns[2];
    std::map<tt::tt_fabric::RoutingDirection, uint32_t> conn_of_dir[2];
    std::vector<uint32_t> lanes(num_slots, 1);
    std::vector<uint32_t> conn_idx[2] = {std::vector<uint32_t>(num_slots, 0), std::vector<uint32_t>(num_slots, 0)};
    for (uint32_t k = 0; k < num_slots; ++k) {
        if (k == own_slot) {
            continue;
        }
        const auto& t = cable.slots[k];
        const auto t_node = relay.receiver_nodes.at(t);
        const auto dir = tt::tt_fabric::pipeline_get_forwarding_direction(entry_node, t_node);
        TT_FATAL(dir.has_value(), "D2DStreamService relay: no route from entry chip {} to coord {}", e, t);
        const auto links = tt::tt_fabric::get_forwarding_link_indices(*mesh, entry_node, t_node);
        lanes[k] = relay_target_lanes(*mesh, relay, t, p);
        for (uint32_t lane = 0; lane < lanes[k]; ++lane) {
            const auto [it, added] = conn_of_dir[lane].emplace(*dir, static_cast<uint32_t>(conns[lane].size()));
            if (added) {
                conns[lane].emplace_back(t_node, links[lane]);
            }
            conn_idx[lane][k] = it->second;
        }
    }
    const uint32_t num_lanes = conns[1].empty() ? 1 : 2;

    auto program = CreateProgram();
    create_relay_cbs(program, core, common.tensor_page_size, backing.dtype());
    const auto accessor_ct = TensorAccessorArgs(*backing.mesh_buffer().get_device_buffer(e)).get_compile_time_args();
    const uint32_t total_pages = common.plan.num_socket_pages * common.plan.pages_per_chunk;
    const auto backing_addr = static_cast<uint32_t>(backing.mesh_buffer().address());

    for (uint32_t lane = 0; lane < num_lanes; ++lane) {
        const bool master = lane == 0;
        const std::unordered_map<std::string, uint32_t> named = {
            {"terminate_value", kTerminateValue},
            {"termination_addr", own.termination_addr},
            {"link_grant_addr", own.link_grant_addr},
            {"is_master", master ? 1u : 0u},
            {"z_lanes", z_lanes},
            {"num_slots", num_slots},
            {"own_slot", own_slot},
            {"page_size", common.tensor_page_size},
            {"max_payload", common.fabric_max_payload_size},
            {"total_pages", total_pages},
            {"half_pages", (total_pages + 1) / 2},
            {"ring_slots", kSenderReadRingSlots},
            {"scratch_cb", static_cast<uint32_t>(tt::CBIndex::c_0)},
            {"header_cb", static_cast<uint32_t>(kPacketHeaderCbIndex)},
            {"metadata_enabled", common.metadata_enabled ? 1u : 0u},
            {"metadata_span", common.metadata_span_bytes},
            {"metadata_size", common.metadata_size_bytes},
            {"worker_md_addr", own.metadata_addr},
            {"arrived_base", static_cast<uint32_t>(words.arrived_base)},
            {"md_base", static_cast<uint32_t>(words.md_base)},
            {"recv_acked_base", static_cast<uint32_t>(words.acked_base)},
            {"word_stride", relay.word_stride},
            {"go_count_addr", static_cast<uint32_t>(words.go_count_addr)},
            {"go_slot_addr", static_cast<uint32_t>(words.go_slot_addr)},
            {"done_count_addr", static_cast<uint32_t>(words.done_count_addr)},
            {"data_ready_sem_addr", ws.data_ready_sem_addr},
            {"consumed_counter_addr", ws.counter_addr},
            {"mcast_x_start", ws.mcast_noc_x_start},
            {"mcast_y_start", ws.mcast_noc_y_start},
            {"mcast_x_end", ws.mcast_noc_x_end},
            {"mcast_y_end", ws.mcast_noc_y_end},
            {"num_workers", ws.num_workers},
            {"exit_chip", static_cast<uint32_t>(exit_node.chip_id)},
            {"exit_mesh", *exit_node.mesh_id},
            {"exit_noc_x", peer.cables[p].noc_x},
            {"exit_noc_y", peer.cables[p].noc_y},
            {"exit_cable_acked_base", peer.cables[p].acked_base},
            {"num_dir_conns", static_cast<uint32_t>(conns[lane].size())},
        };
        auto kernel = CreateKernel(
            program,
            "ttnn/core/tensor/kernels/persistent_d2d_entry_relay.cpp",
            core,
            DataMovementConfig{
                .processor = kRelayProcs[lane],
                .noc = kRelayNocs[lane],
                .compile_args = accessor_ct,
                .named_compile_args = named,
            });

        // RT args: the cable connection (master only, for the credits), this lane's
        // direction connections, then 11 words per slot (the same for both lanes).
        std::vector<uint32_t> rt;
        if (master) {
            tt::tt_fabric::append_fabric_connection_rt_args(entry_node, exit_node, z_link, program, core, rt);
        }
        for (const auto& [node, link] : conns[lane]) {
            tt::tt_fabric::append_fabric_connection_rt_args(entry_node, node, link, program, core, rt);
        }
        for (uint32_t k = 0; k < num_slots; ++k) {
            if (k == own_slot) {
                rt.insert(rt.end(), {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u, 1u, 0u, 0u});
                continue;
            }
            const auto& t = cable.slots[k];
            const auto t_node = relay.receiver_nodes.at(t);
            const auto t_noc = relay_noc(mesh, service_cores, t);
            rt.insert(
                rt.end(),
                {static_cast<uint32_t>(relay.staging[k].mesh_buffer().address()),
                 backing_addr,
                 static_cast<uint32_t>(t_node.chip_id),
                 *t_node.mesh_id,
                 static_cast<uint32_t>(t_noc.x),
                 static_cast<uint32_t>(t_noc.y),
                 socket_fifo_addr,    // the receiver reads its metadata from its FIFO base
                 socket_config_addr,  // receiver_socket_md::bytes_sent is word 0
                 lanes[k],
                 conn_idx[0][k],
                 conn_idx[1][k]});
        }
        SetRuntimeArgs(program, kernel, core, rt);
    }
    return program;
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
    CMAKE_UNIQUE_NAMESPACE::FabricLinkClaims link_claims;
    auto res = CMAKE_UNIQUE_NAMESPACE::build_sender_side(
        mesh, socket, service_cores, receiver_tensor_addrs, backing, coords, common, cfg, link_claims);

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
        .relay = std::move(res.relay),
        .lane_links = std::move(res.lane_links),
        .link_claims = std::move(link_claims),
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
    const D2DStreamConfig& cfg,
    const std::map<tt::tt_metal::distributed::MeshCoordinate, uint32_t>& sender_lanes) {
    bool ok = false;
    CMAKE_UNIQUE_NAMESPACE::ServiceCoreReleaseGuard guard{mesh, service_cores, ok};

    const auto common = CMAKE_UNIQUE_NAMESPACE::derive_common_plan(cfg, backing);
    const auto& coords = backing.tensor_topology().mesh_coords();
    CMAKE_UNIQUE_NAMESPACE::FabricLinkClaims link_claims;
    auto res = CMAKE_UNIQUE_NAMESPACE::build_receiver_side(
        mesh, socket, service_cores, backing, coords, common, cfg, sender_lanes, link_claims);

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
        .stage_gate = CMAKE_UNIQUE_NAMESPACE::enabled_stage_gate(cfg),
        .stage_gate_addrs = std::move(res.stage_gate_addrs),
        .workload = std::move(res.workload),
        .launched = false,
        .lease_wait_workload = std::move(res.lease_wait_workload),
        .lease_release_workload = std::move(res.lease_release_workload),
        .relay = std::move(res.relay),
        .link_claims = std::move(link_claims),
    });
    auto handle = std::unique_ptr<D2DStreamServiceReceiver>(new D2DStreamServiceReceiver(std::move(impl)));
    ok = true;
    return handle;
}

// ===========================================================================
// Relay mode: build the workloads (after both sides know each other's relay words)
// ===========================================================================

struct D2DStreamService::RelayInfo : CMAKE_UNIQUE_NAMESPACE::RelayPeer {};

void D2DStreamService::build_relay_sender(D2DStreamServiceSender& sender, const RelayInfo& peer) {
    namespace tm_distributed = tt::tt_metal::distributed;
    auto& impl = *sender.impl_;
    const auto& relay = *impl.relay;
    const auto& common = relay.common;
    const auto& plan = relay.plan;
    const auto& mesh = impl.mesh_device;
    CMAKE_UNIQUE_NAMESPACE::check_relay_peer(peer, relay);
    const auto config_addr = static_cast<uint32_t>(impl.socket->get_config_buffer()->address());
    const auto ws_for = [&](const tm_distributed::MeshCoordinate& c) {
        return CMAKE_UNIQUE_NAMESPACE::make_worker_sync_args(
            mesh->get_device(c),
            impl.worker_cores,
            impl.num_workers,
            static_cast<uint32_t>(impl.consumed_sem->address()),
            static_cast<uint32_t>(impl.data_ready_counter_addrs.at(c)),
            /*enabled=*/true);
    };

    // The socket constructor wrote the configs through the command queue. Patch them only
    // after that write has landed, or it would overwrite the patch.
    tm_distributed::Finish(mesh->mesh_command_queue());

    auto workload = std::make_unique<tm_distributed::MeshWorkload>();
    // 1. Every coord except an exit chip: the unchanged sender kernel, aimed at its exit relay.
    for (const auto& [c, pk] : plan.slot_of) {
        const auto [p, k] = pk;
        const auto& cable = plan.cables[p];
        if (c == cable.exit_coord) {
            continue;
        }
        const auto& words = relay.words[p];
        auto* device = mesh->get_device(c);
        const CoreCoord core = impl.service_cores.at(c);
        const auto src_node = relay.sender_nodes.at(c);
        const auto exit_node = relay.sender_nodes.at(cable.exit_coord);
        CMAKE_UNIQUE_NAMESPACE::patch_sender_config(
            device,
            core,
            config_addr,
            common.l1_alignment,
            static_cast<uint32_t>(words.arrived_base + k * relay.word_stride),
            common.metadata_enabled ? static_cast<uint32_t>(words.md_base + k * common.metadata_span_bytes) : 0u,
            exit_node,
            CMAKE_UNIQUE_NAMESPACE::relay_noc(mesh, impl.service_cores, cable.exit_coord));
        const auto links = tt::tt_fabric::get_forwarding_link_indices(*mesh, src_node, exit_node);
        const uint32_t lanes = CMAKE_UNIQUE_NAMESPACE::relay_source_lanes(*mesh, relay, c, p);
        const std::vector<uint32_t> lane_links(links.begin(), links.begin() + lanes);
        workload->add_program(
            tm_distributed::MeshCoordinateRange(c),
            CMAKE_UNIQUE_NAMESPACE::build_sender_program(
                *impl.backing_tensor.mesh_buffer().get_device_buffer(c),
                core,
                config_addr,
                static_cast<uint32_t>(impl.termination_addrs.at(c)),
                common.plan,
                common.tensor_page_size,
                common.fabric_max_payload_size,
                impl.backing_tensor.dtype(),
                ws_for(c),
                common.metadata_enabled,
                common.metadata_span_bytes,
                common.metadata_enabled ? static_cast<uint32_t>(impl.metadata_addrs.at(c)) : 0u,
                /*share_fabric_links=*/true,
                static_cast<uint32_t>(impl.link_grant_addrs.at(c)),
                static_cast<uint32_t>(relay.staging[k].mesh_buffer().address()),
                static_cast<uint32_t>(impl.go_count_addrs.at(c)),
                static_cast<uint32_t>(impl.done_count_addrs.at(c)),
                lanes,
                src_node,
                exit_node,
                lane_links));
    }
    // 2. Exit chips: the exit relay, in place of their sender service.
    for (uint32_t p = 0; p < plan.cables.size(); ++p) {
        const auto e = plan.cables[p].exit_coord;
        workload->add_program(
            tm_distributed::MeshCoordinateRange(e),
            CMAKE_UNIQUE_NAMESPACE::build_exit_relay_program(
                mesh,
                relay,
                p,
                peer,
                impl.service_cores,
                impl.backing_tensor,
                ws_for(e),
                {.termination_addr = static_cast<uint32_t>(impl.termination_addrs.at(e)),
                 .link_grant_addr = static_cast<uint32_t>(impl.link_grant_addrs.at(e)),
                 .metadata_addr = common.metadata_enabled ? static_cast<uint32_t>(impl.metadata_addrs.at(e)) : 0u},
                config_addr));
    }
    impl.workload = std::move(workload);
}

void D2DStreamService::build_relay_receiver(D2DStreamServiceReceiver& receiver, const RelayInfo& peer) {
    namespace tm_distributed = tt::tt_metal::distributed;
    auto& impl = *receiver.impl_;
    const auto& relay = *impl.relay;
    const auto& common = relay.common;
    const auto& plan = relay.plan;
    const auto& mesh = impl.mesh_device;
    CMAKE_UNIQUE_NAMESPACE::check_relay_peer(peer, relay);
    const auto config_addr = static_cast<uint32_t>(impl.socket->get_config_buffer()->address());
    const auto fifo_addr = static_cast<uint32_t>(impl.socket->get_data_buffer()->address());
    const auto ws_for = [&](const tm_distributed::MeshCoordinate& c) {
        return CMAKE_UNIQUE_NAMESPACE::make_worker_sync_args(
            mesh->get_device(c),
            impl.worker_cores,
            impl.num_workers,
            static_cast<uint32_t>(impl.data_ready_sem->address()),
            static_cast<uint32_t>(impl.consumed_counter_addrs.at(c)),
            /*enabled=*/true);
    };

    tm_distributed::Finish(mesh->mesh_command_queue());

    auto workload = std::make_unique<tm_distributed::MeshWorkload>();
    // 1. Every coord except an entry chip: the unchanged receiver kernel. Its data now
    //    comes from the entry relay, and its credits go back there.
    for (const auto& [c, pk] : plan.slot_of) {
        const auto [p, k] = pk;
        const auto& cable = plan.cables[p];
        if (c == cable.entry_coord) {
            continue;
        }
        auto* device = mesh->get_device(c);
        const CoreCoord core = impl.service_cores.at(c);
        const auto t_node = relay.receiver_nodes.at(c);
        const auto entry_node = relay.receiver_nodes.at(cable.entry_coord);
        CMAKE_UNIQUE_NAMESPACE::patch_receiver_config(
            device,
            core,
            config_addr,
            static_cast<uint32_t>(relay.words[p].acked_base + k * relay.word_stride),
            entry_node,
            CMAKE_UNIQUE_NAMESPACE::relay_noc(mesh, impl.service_cores, cable.entry_coord));
        const auto links = tt::tt_fabric::get_forwarding_link_indices(*mesh, t_node, entry_node);
        TT_FATAL(!links.empty(), "D2DStreamService relay: no fabric link from coord {} to its entry chip", c);
        workload->add_program(
            tm_distributed::MeshCoordinateRange(c),
            CMAKE_UNIQUE_NAMESPACE::build_receiver_program(
                *impl.backing_tensor.mesh_buffer().get_device_buffer(c),
                core,
                config_addr,
                static_cast<uint32_t>(impl.termination_addrs.at(c)),
                common.plan,
                common.tensor_page_size,
                ws_for(c),
                common.metadata_enabled,
                common.metadata_size_bytes,
                static_cast<uint32_t>(impl.metadata_l1_addr),
                /*share_fabric_links=*/true,
                static_cast<uint32_t>(impl.link_grant_addrs.at(c)),
                // The entry relay sends one bytes_sent increment per lane it uses for c.
                CMAKE_UNIQUE_NAMESPACE::relay_target_lanes(*mesh, relay, c, p),
                std::nullopt,
                0u,
                t_node,
                entry_node,
                links.front()));
    }
    // 2. Entry chips: the entry relay, in place of their receiver service.
    for (uint32_t p = 0; p < plan.cables.size(); ++p) {
        const auto e = plan.cables[p].entry_coord;
        workload->add_program(
            tm_distributed::MeshCoordinateRange(e),
            CMAKE_UNIQUE_NAMESPACE::build_entry_relay_program(
                mesh,
                relay,
                p,
                peer,
                impl.service_cores,
                impl.backing_tensor,
                ws_for(e),
                {.termination_addr = static_cast<uint32_t>(impl.termination_addrs.at(e)),
                 .link_grant_addr = static_cast<uint32_t>(impl.link_grant_addrs.at(e)),
                 .metadata_addr = static_cast<uint32_t>(impl.metadata_l1_addr)},
                config_addr,
                fifo_addr));
    }
    impl.workload = std::move(workload);
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
    CMAKE_UNIQUE_NAMESPACE::apply_intermesh_vc_disabled(cfg);
    CMAKE_UNIQUE_NAMESPACE::validate_stage_gate(cfg);

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
    // The receiver awaits one data-landed increment per sender lane, as the sender chose them.
    auto receiver_handle = finalize_receiver(
        receiver_mesh,
        std::move(receiver_socket),
        std::move(receiver_service_cores),
        receiver_backing,
        cfg,
        CMAKE_UNIQUE_NAMESPACE::lane_counts(sender_handle->impl_->lane_links));

    // Relay mode: both sides are local, so each side's relay words are known directly.
    if (cfg.relay) {
        const auto& s = *sender_handle->impl_;
        const auto& r = *receiver_handle->impl_;
        const RelayInfo sender_info{
            CMAKE_UNIQUE_NAMESPACE::make_relay_peer(*s.relay, s.mesh_device, s.service_cores, /*is_sender=*/true)};
        const RelayInfo receiver_info{
            CMAKE_UNIQUE_NAMESPACE::make_relay_peer(*r.relay, r.mesh_device, r.service_cores, /*is_sender=*/false)};
        build_relay_receiver(*receiver_handle, sender_info);
        build_relay_sender(*sender_handle, receiver_info);
    }

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

    CMAKE_UNIQUE_NAMESPACE::apply_intermesh_vc_disabled(cfg);

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
        cfg.share_fabric_links,
        cfg.relay);
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
    // The sender's routers (and so its lane count) are chosen in finalize_sender; the
    // receiver process waits for the counts before building its kernels, so it hears
    // about a failure here too.
    std::unique_ptr<D2DStreamServiceSender> sender_handle;
    try {
        sender_handle = finalize_sender(
            sender_mesh,
            std::move(sender_socket),
            std::move(sender_service_cores),
            receiver_tensor_addrs,
            sender_backing,
            cfg);
    } catch (...) {
        if (!cfg.relay) {
            CMAKE_UNIQUE_NAMESPACE::send_sender_lanes(ctx, endpoints.receiver_rank, mo.coords, /*lane_links=*/nullptr);
        }
        throw;
    }
    // Relay mode has no sender lanes: the relay kernels choose their own links.
    if (!cfg.relay) {
        CMAKE_UNIQUE_NAMESPACE::send_sender_lanes(
            ctx, endpoints.receiver_rank, mo.coords, &sender_handle->impl_->lane_links);
    }
    if (cfg.relay) {
        const auto& s = *sender_handle->impl_;
        const auto local =
            CMAKE_UNIQUE_NAMESPACE::make_relay_peer(*s.relay, s.mesh_device, s.service_cores, /*is_sender=*/true);
        build_relay_sender(
            *sender_handle,
            RelayInfo{CMAKE_UNIQUE_NAMESPACE::exchange_relay_peer(
                ctx, /*is_sender=*/true, endpoints.receiver_rank, local, *s.relay)});
    }
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
    CMAKE_UNIQUE_NAMESPACE::apply_intermesh_vc_disabled(cfg);
    CMAKE_UNIQUE_NAMESPACE::validate_stage_gate(cfg);

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
        cfg.share_fabric_links,
        cfg.relay);
    auto sender_service_cores = std::move(peer.service_cores);

    auto connections =
        CMAKE_UNIQUE_NAMESPACE::build_connections(mo.coords, sender_service_cores, receiver_service_cores);
    tt::tt_metal::distributed::SocketConfig socket_config(
        connections, cfg.socket_mem_config, endpoints.sender_rank, endpoints.receiver_rank, ctx);
    tt::tt_metal::distributed::MeshSocket receiver_socket(receiver_mesh, socket_config);
    // The sender's lane count per coord, sent once its routers are chosen. Relay mode sends none.
    const auto sender_lanes = cfg.relay
                                  ? std::map<tt::tt_metal::distributed::MeshCoordinate, uint32_t>{}
                                  : CMAKE_UNIQUE_NAMESPACE::recv_sender_lanes(ctx, endpoints.sender_rank, mo.coords);

    // Build + assemble (shared with create_pair). The service cores move into the
    // handle Impl here, emptying the map receiver_guard references.
    auto receiver_handle = finalize_receiver(
        receiver_mesh,
        std::move(receiver_socket),
        std::move(receiver_service_cores),
        receiver_backing,
        cfg,
        sender_lanes);
    if (cfg.relay) {
        const auto& r = *receiver_handle->impl_;
        const auto local =
            CMAKE_UNIQUE_NAMESPACE::make_relay_peer(*r.relay, r.mesh_device, r.service_cores, /*is_sender=*/false);
        build_relay_receiver(
            *receiver_handle,
            RelayInfo{CMAKE_UNIQUE_NAMESPACE::exchange_relay_peer(
                ctx, /*is_sender=*/false, endpoints.sender_rank, local, *r.relay)});
    }
    EnqueueMeshWorkload(receiver_mesh->mesh_command_queue(), *receiver_handle->impl_->workload, /*blocking=*/false);
    receiver_handle->impl_->launched = true;

    committed = true;
    return receiver_handle;
}

}  // namespace ttnn
