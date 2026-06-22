// SPDX-FileCopyrightText: © 2026 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "ttnn/services/h2d_socket_service.hpp"

#include <algorithm>
#include <cerrno>
#include <cstdio>
#include <cstring>
#include <exception>
#include <thread>
#include <unordered_set>

#include <tt_stl/assert.hpp>
#include <tt_stl/reflection.hpp>

#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/data_types.hpp>
#include <tt-metalium/distributed.hpp>
#include <tt-metalium/hal.hpp>
#include <tt-metalium/host_api.hpp>
#include <internal/service/service_core_manager.hpp>
#include <tt-metalium/memory_pin.hpp>
#include <tt-metalium/mesh_buffer.hpp>
#include <tt-metalium/mesh_coord.hpp>
#include <tt-metalium/mesh_device.hpp>
#include <tt-metalium/program.hpp>
#include <tt-metalium/tensor_accessor_args.hpp>
#include <tt-metalium/tt_align.hpp>
#include <tt-metalium/tt_metal.hpp>

#include "socket_service_common.hpp"
#include "tensor/tensor_ops.hpp"
#include "tt_metal/distributed/h2d_stream_service_descriptor.hpp"
#include "tt_metal/distributed/hd_socket_descriptor.hpp"
#include "tt_metal/distributed/shm_resource_tracker.hpp"
#include "ttnn/distributed/distributed_tensor.hpp"
#include "ttnn/global_semaphore.hpp"

namespace tt::tt_metal {

namespace {

// Zero-copy wrap of caller-provided raw bytes into a host Tensor matching `spec`.
Tensor make_borrowed_host_tensor(ttsl::Span<const std::byte> bytes, const TensorSpec& spec) {
    auto* raw = const_cast<std::byte*>(bytes.data());
    const auto& shape = spec.logical_shape();
    switch (spec.data_type()) {
        case DataType::BFLOAT16:
            return Tensor::from_borrowed_data<bfloat16>(
                ttsl::Span<bfloat16>(reinterpret_cast<bfloat16*>(raw), bytes.size() / sizeof(bfloat16)),
                shape,
                MemoryPin{});
        case DataType::FLOAT32:
            return Tensor::from_borrowed_data<float>(
                ttsl::Span<float>(reinterpret_cast<float*>(raw), bytes.size() / sizeof(float)), shape, MemoryPin{});
        case DataType::INT32:
            return Tensor::from_borrowed_data<int32_t>(
                ttsl::Span<int32_t>(reinterpret_cast<int32_t*>(raw), bytes.size() / sizeof(int32_t)),
                shape,
                MemoryPin{});
        case DataType::UINT8:
            return Tensor::from_borrowed_data<uint8_t>(
                ttsl::Span<uint8_t>(reinterpret_cast<uint8_t*>(raw), bytes.size() / sizeof(uint8_t)),
                shape,
                MemoryPin{});
        case DataType::UINT16:
            return Tensor::from_borrowed_data<uint16_t>(
                ttsl::Span<uint16_t>(reinterpret_cast<uint16_t*>(raw), bytes.size() / sizeof(uint16_t)),
                shape,
                MemoryPin{});
        case DataType::BFLOAT4_B:
        case DataType::BFLOAT8_B:
        case DataType::UINT32:
            return Tensor::from_borrowed_data<uint32_t>(
                ttsl::Span<uint32_t>(reinterpret_cast<uint32_t*>(raw), bytes.size() / sizeof(uint32_t)),
                shape,
                MemoryPin{});
        case DataType::FP8_E4M3: TT_THROW("H2DStreamService: FP8_E4M3 is not supported");
        case DataType::INVALID: TT_THROW("H2DStreamService: invalid global_spec data type");
    }
    TT_THROW("Unreachable");
}

// Worker-sync CT-arg block. Populated when Config::worker_cores is set; all
// fields zero when disabled (the kernel's `if constexpr (worker_sync_enabled)`
// gate skips the block entirely).
struct H2DWorkerSyncArgs {
    bool enabled = false;
    uint32_t data_ready_sem_addr = 0;    // worker-grid L1 (mesh-wide GlobalSemaphore)
    uint32_t consumed_counter_addr = 0;  // service-core L1 (per-coord, allocated via ServiceCoreManager)
    uint32_t mcast_noc_x_start = 0;      // physical NoC bbox of worker_cores on this device
    uint32_t mcast_noc_y_start = 0;
    uint32_t mcast_noc_x_end = 0;
    uint32_t mcast_noc_y_end = 0;
    uint32_t num_workers = 0;  // mcast destination count + sync arithmetic target
};

// Metadata multicast CT-arg block. Populated when Config::metadata_size_bytes > 0.
struct MetadataArgs {
    bool enabled = false;
    uint32_t metadata_size_bytes = 0;  // user-specified size; <= socket_page_size
    uint32_t metadata_l1_addr = 0;     // worker-grid L1 (mesh-wide L1-sharded Buffer)
};

// Builds the single-core persistent H2D program for one socket / device buffer.
//
// CT-arg layout (must stay in sync with persistent_h2d_receiver.cpp):
//   [0]  socket_config_addr
//   [1]  termination_semaphore_addr
//   [2]  socket_page_size
//   [3]  num_socket_pages
//   [4]  output_tensor_addr
//   [5]  output_tensor_page_size
//   [6]  pages_per_chunk
//   [7]  scratch_buffer_cb_index
//   [8]  worker_sync_enabled            (uint32 0/1)
//   [9]  data_ready_sem_addr            (uint32, worker-grid L1)
//   [10] consumed_counter_addr          (uint32, local service-core L1)
//   [11] worker_mcast_noc_x_start
//   [12] worker_mcast_noc_y_start
//   [13] worker_mcast_noc_x_end
//   [14] worker_mcast_noc_y_end
//   [15] num_workers
//   [16] metadata_enabled                 (uint32 0/1)
//   [17] metadata_size_bytes              (uint32, un-padded user size)
//   [18] metadata_l1_addr                 (uint32, worker-grid L1)
//   [19..] TensorAccessorArgs
Program build_persistent_h2d_program(
    const Buffer& device_buffer,
    const CoreCoord& recv_core,
    uint32_t socket_config_buffer_address,
    uint32_t termination_semaphore_addr,
    const ChunkPlan& plan,
    uint32_t tensor_page_size,
    DataType dtype,
    const H2DWorkerSyncArgs& worker_sync,
    const MetadataArgs& metadata) {
    auto program = CreateProgram();

    constexpr tt::CBIndex scratch_cb_index = tt::CBIndex::c_0;
    auto cb_cfg =
        CircularBufferConfig(plan.socket_page_size, {{scratch_cb_index, datatype_to_dataformat_converter(dtype)}})
            .set_page_size(scratch_cb_index, plan.socket_page_size);
    CreateCircularBuffer(program, recv_core, cb_cfg);

    auto tensor_accessor_args = TensorAccessorArgs(device_buffer);
    auto tensor_accessor_compile_args = tensor_accessor_args.get_compile_time_args();

    std::vector<uint32_t> ct_args = {
        socket_config_buffer_address,
        termination_semaphore_addr,
        plan.socket_page_size,
        plan.num_socket_pages,
        static_cast<uint32_t>(device_buffer.address()),
        tensor_page_size,
        plan.pages_per_chunk,
        static_cast<uint32_t>(scratch_cb_index),
        // Worker-sync block (indices 8..15); all zero when disabled.
        static_cast<uint32_t>(worker_sync.enabled ? 1u : 0u),
        worker_sync.data_ready_sem_addr,
        worker_sync.consumed_counter_addr,
        worker_sync.mcast_noc_x_start,
        worker_sync.mcast_noc_y_start,
        worker_sync.mcast_noc_x_end,
        worker_sync.mcast_noc_y_end,
        worker_sync.num_workers,
        // Metadata block (indices 16..18); all zero when disabled.
        static_cast<uint32_t>(metadata.enabled ? 1u : 0u),
        metadata.metadata_size_bytes,
        metadata.metadata_l1_addr,
    };
    ct_args.insert(ct_args.end(), tensor_accessor_compile_args.begin(), tensor_accessor_compile_args.end());

    CreateKernel(
        program,
        "models/demos/deepseek_v3_b1/micro_ops/host_io/kernels/persistent_h2d_receiver.cpp",
        recv_core,
        DataMovementConfig{
            .processor = DataMovementProcessor::RISCV_0,
            .noc = NOC::RISCV_0_default,
            .compile_args = ct_args,
        });

    return program;
}

}  // namespace

H2DStreamService::H2DStreamService(const std::shared_ptr<distributed::MeshDevice>& mesh_device, Config cfg) :
    mesh_device_(mesh_device), cfg_(std::move(cfg)) {
    TT_FATAL(mesh_device_ != nullptr, "H2DStreamService: mesh_device must not be null");
    TT_FATAL(cfg_.fifo_size_bytes > 0, "H2DStreamService: fifo_size_bytes must be > 0");
    TT_FATAL(cfg_.scratch_cb_size_bytes > 0, "H2DStreamService: scratch_cb_size_bytes must be > 0");
    TT_FATAL(
        cfg_.metadata_size_bytes == 0 || cfg_.worker_cores.has_value(),
        "H2DStreamService: metadata_size_bytes={} requires Config::worker_cores to be set "
        "(no workers to multicast metadata to)",
        cfg_.metadata_size_bytes);

    // Default to replicate-on-every-mesh-dim when no mapper is supplied.
    if (cfg_.mapper == nullptr) {
        ttsl::SmallVector<distributed::MeshMapperConfig::Placement> replicate_all(
            mesh_device_->shape().dims(), distributed::MeshMapperConfig::Replicate{});
        cfg_.mapper = ttnn::distributed::create_mesh_mapper(
            *mesh_device_, distributed::MeshMapperConfig{.placements = std::move(replicate_all)});
    }
    mapper_ = std::move(cfg_.mapper);

    const auto distributed_dummy = (*mapper_)(make_zero_host_tensor(cfg_.global_spec));
    const auto& per_shard_spec = distributed_dummy.tensor_spec();
    const auto& topology = distributed_dummy.tensor_topology();

    device_tensor_ = create_device_tensor(per_shard_spec, mesh_device_.get(), topology);
    per_shard_spec_ = device_tensor_.tensor_spec();

    // Each device may resolve a different free service core; record it per coord.
    auto& svc = tt::tt_metal::internal::service_core_manager();
    const auto& coords = topology.mesh_coords();
    for (const auto& coord : coords) {
        auto* d = mesh_device_->get_device(coord);
        auto claimable = svc.get_claimable_cores(d);
        TT_FATAL(!claimable.empty(), "H2DStreamService: no claimable service core on device at coord {}", coord);
        const CoreCoord chosen = claimable.front();
        svc.claim(d, {chosen});
        service_cores_.emplace(coord, chosen);
    }

    // Iterate participating coords (not the full mesh shape) so replication-
    // collapsed or shape-overridden mappings stay correct.
    sockets_.reserve(coords.size());
    for (const auto& coord : coords) {
        sockets_.push_back(std::make_unique<distributed::H2DSocket>(
            mesh_device_,
            distributed::MeshCoreCoord(coord, service_cores_.at(coord)),
            cfg_.socket_buffer_type,
            cfg_.fifo_size_bytes,
            cfg_.socket_mode));
    }

    // Every per-device buffer shares the same spec, so this buffer is representative.
    const uint32_t tensor_page_size = device_tensor_.buffer()->page_size();
    const uint32_t tensor_num_pages = device_tensor_.buffer()->num_pages();
    const ChunkPlan plan = derive_chunk_plan(tensor_page_size, tensor_num_pages, cfg_.scratch_cb_size_bytes);
    socket_page_size_ = plan.socket_page_size;
    num_socket_pages_ = plan.num_socket_pages;

    // Metadata travels as exactly one trailing socket page, so it must fit.
    TT_FATAL(
        cfg_.metadata_size_bytes <= socket_page_size_,
        "H2DStreamService: metadata_size_bytes={} exceeds derived socket_page_size={} "
        "(single-metadata-page constraint). Either reduce metadata or increase "
        "scratch_cb_size_bytes / per-shard page size.",
        cfg_.metadata_size_bytes,
        socket_page_size_);

    for (auto& s : sockets_) {
        s->set_page_size(plan.socket_page_size);
    }

    // Per-device termination signal: one uint32 in L1, written directly rather
    // than via a GlobalSemaphore (which can't target per-coord service cores).
    std::vector<uint32_t> zero_word{0};
    for (const auto& coord : coords) {
        auto* d = mesh_device_->get_device(coord);
        const CoreCoord chosen = service_cores_.at(coord);
        const DeviceAddr sem_addr = svc.allocate_l1(d, chosen, sizeof(uint32_t));
        termination_addrs_.emplace(coord, sem_addr);
        tt::tt_metal::detail::WriteToDeviceL1(d, chosen, static_cast<uint32_t>(sem_addr), zero_word);
    }

    if (cfg_.worker_cores.has_value()) {
        const auto& worker_range = cfg_.worker_cores.value();
        num_workers_ = worker_range.size();
        TT_FATAL(num_workers_ > 0, "H2DStreamService: cfg.worker_cores must contain at least one core");

        // Mesh-wide, same address on every (device, worker core). Workers poll
        // their local copy; the service kernel multicasts atomic-inc into it.
        data_ready_sem_.emplace(ttnn::global_semaphore::create_global_semaphore(
            mesh_device_.get(),
            CoreRangeSet(worker_range),
            /*initial_value=*/0,
            BufferType::L1));

        // Consumed counter: per-coord L1 word on the service core.
        for (const auto& coord : coords) {
            auto* d = mesh_device_->get_device(coord);
            const CoreCoord chosen = service_cores_.at(coord);
            const DeviceAddr addr = svc.allocate_l1(d, chosen, sizeof(uint32_t));
            consumed_addrs_.emplace(coord, addr);
            tt::tt_metal::detail::WriteToDeviceL1(d, chosen, static_cast<uint32_t>(addr), zero_word);
        }
    }

    // Mesh-wide L1-sharded Buffer across worker_cores: REPLICATED per device,
    // HEIGHT_SHARDED so each worker core gets one L1-aligned shard.
    if (cfg_.metadata_size_bytes > 0) {
        const uint32_t l1_align = hal::get_l1_alignment();
        const DeviceAddr aligned_shard_size =
            tt::align(static_cast<DeviceAddr>(cfg_.metadata_size_bytes), static_cast<DeviceAddr>(l1_align));

        const CoreRangeSet shard_grid(cfg_.worker_cores.value());

        // Mirrors the L1-sharded allocation pattern in h2d_socket.cpp.
        distributed::DeviceLocalBufferConfig device_local = {
            .page_size = aligned_shard_size,
            .buffer_type = BufferType::L1,
            .sharding_args = BufferShardingArgs(
                ShardSpecBuffer(shard_grid, {1, 1}, ShardOrientation::ROW_MAJOR, {1, 1}, {num_workers_, 1}),
                TensorMemoryLayout::HEIGHT_SHARDED),
            .bottom_up = std::nullopt,
            .sub_device_id = std::nullopt,
        };
        distributed::MeshBufferConfig mesh_config = distributed::ReplicatedBufferConfig{
            .size = aligned_shard_size * static_cast<DeviceAddr>(num_workers_),
        };

        metadata_buffer_ = distributed::MeshBuffer::create(mesh_config, device_local, mesh_device_.get());
        metadata_l1_addr_ = metadata_buffer_->address();

        // Zero-init so trailing padding is deterministic; only the leading
        // metadata_size_bytes are overwritten per call.
        metadata_scratch_.assign(socket_page_size_, std::byte{0});
    }

    if (cfg_.preprocessor) {
        preprocess_scratch_.assign(cfg_.global_spec.compute_packed_buffer_size_bytes(), std::byte{0});
    }

    workload_ = std::make_unique<distributed::MeshWorkload>();
    for (auto& s : sockets_) {
        const auto core = s->get_active_cores()[0];
        const Buffer* dbuf = device_tensor_.mesh_buffer().get_device_buffer(core.device_coord);
        TT_FATAL(dbuf != nullptr, "H2DStreamService: device buffer missing for coord {}", core.device_coord);
        const uint32_t term_addr = static_cast<uint32_t>(termination_addrs_.at(core.device_coord));

        // Per-coord worker-sync args. Populated only when cfg.worker_cores is
        // set; otherwise everything stays zero and the kernel skips the sync
        // block via `if constexpr (worker_sync_enabled == 0)`.
        H2DWorkerSyncArgs worker_sync;
        if (cfg_.worker_cores.has_value()) {
            const auto& worker_range = cfg_.worker_cores.value();
            auto* d = mesh_device_->get_device(core.device_coord);
            const auto start_phys = d->worker_core_from_logical_core(worker_range.start_coord);
            const auto end_phys = d->worker_core_from_logical_core(worker_range.end_coord);
            worker_sync.enabled = true;
            worker_sync.data_ready_sem_addr = static_cast<uint32_t>(data_ready_sem_->address());
            worker_sync.consumed_counter_addr = static_cast<uint32_t>(consumed_addrs_.at(core.device_coord));
            worker_sync.mcast_noc_x_start = static_cast<uint32_t>(start_phys.x);
            worker_sync.mcast_noc_y_start = static_cast<uint32_t>(start_phys.y);
            worker_sync.mcast_noc_x_end = static_cast<uint32_t>(end_phys.x);
            worker_sync.mcast_noc_y_end = static_cast<uint32_t>(end_phys.y);
            worker_sync.num_workers = num_workers_;
        }

        MetadataArgs metadata;
        if (cfg_.metadata_size_bytes > 0) {
            metadata.enabled = true;
            metadata.metadata_size_bytes = cfg_.metadata_size_bytes;
            metadata.metadata_l1_addr = static_cast<uint32_t>(metadata_l1_addr_);
        }

        auto program = build_persistent_h2d_program(
            *dbuf,
            core.core_coord,
            s->get_config_buffer_address(),
            term_addr,
            plan,
            tensor_page_size,
            device_tensor_.dtype(),
            worker_sync,
            metadata);
        workload_->add_program(distributed::MeshCoordinateRange(core.device_coord), std::move(program));
    }

    EnqueueMeshWorkload(mesh_device_->mesh_command_queue(), *workload_, /*blocking=*/false);
}

H2DStreamService::H2DStreamService(
    Config cfg,
    std::vector<std::unique_ptr<distributed::H2DSocket>> sockets,
    uint32_t socket_page_size,
    uint32_t num_socket_pages) :
    is_owner_(false), cfg_(std::move(cfg)) {
    TT_FATAL(!sockets.empty(), "H2DStreamService(connector): sockets vector must not be empty");
    TT_FATAL(cfg_.mapper != nullptr, "H2DStreamService(connector): mapper must be pre-built and supplied");
    TT_FATAL(socket_page_size > 0, "H2DStreamService(connector): socket_page_size must be > 0");
    TT_FATAL(num_socket_pages > 0, "H2DStreamService(connector): num_socket_pages must be > 0");

    mapper_ = std::move(cfg_.mapper);

    socket_page_size_ = socket_page_size;
    num_socket_pages_ = num_socket_pages;
    sockets_ = std::move(sockets);
    for (auto& s : sockets_) {
        s->set_page_size(socket_page_size_);
    }

    const auto distributed_dummy = (*mapper_)(make_zero_host_tensor(cfg_.global_spec));
    per_shard_spec_ = distributed_dummy.tensor_spec();

    if (cfg_.metadata_size_bytes > 0) {
        metadata_scratch_.assign(socket_page_size_, std::byte{0});
    }

    if (cfg_.preprocessor) {
        preprocess_scratch_.assign(cfg_.global_spec.compute_packed_buffer_size_bytes(), std::byte{0});
    }
}

H2DStreamService::~H2DStreamService() {
    // try/catch so a teardown failure (e.g. mesh device already gone) never
    // escapes the destructor.
    try {
        if (!is_owner_) {
            // Connector owns no device-side resources; sockets free their own SHM.
            sockets_.clear();
            return;
        }

        // Drain in-flight writes, then flip the per-device termination signals so
        // each persistent kernel exits on its next poll.
        barrier();
        signal_termination();

        if (mesh_device_) {
            distributed::Finish(mesh_device_->mesh_command_queue());
        }

        // Wait for each kernel to actually return (RUN_MSG_DONE), not just for
        // dispatch to drain, or a later instance finds the service core occupied.
        auto& svc = tt::tt_metal::internal::service_core_manager();
        if (mesh_device_) {
            for (const auto& [coord, core] : service_cores_) {
                auto* d = mesh_device_->get_device(coord);
                svc.wait_done(d, core);
            }
        }

        if (mesh_device_) {
            for (const auto& [coord, addr] : termination_addrs_) {
                auto* d = mesh_device_->get_device(coord);
                svc.deallocate_l1(d, service_cores_.at(coord), addr);
            }
            termination_addrs_.clear();

            for (const auto& [coord, addr] : consumed_addrs_) {
                auto* d = mesh_device_->get_device(coord);
                svc.deallocate_l1(d, service_cores_.at(coord), addr);
            }
            consumed_addrs_.clear();

            // Destroy sockets before releasing the service-core claims: H2DSocket
            // dtors deallocate their own L1 and TT_FATAL if the core is already
            // released.
            sockets_.clear();

            for (const auto& [coord, core] : service_cores_) {
                auto* d = mesh_device_->get_device(coord);
                svc.release(d, {core});
            }
        }

        // Unlink + untrack the exported descriptor so it doesn't linger in
        // ShmResourceTracker until process exit.
        if (!descriptor_path_.empty()) {
            if (std::remove(descriptor_path_.c_str()) == 0 || errno == ENOENT) {
                distributed::ShmResourceTracker::instance().untrack_file(descriptor_path_);
            }
            descriptor_path_.clear();
        }
    } catch (const std::exception& e) {
        log_warning(tt::LogOp, "H2DStreamService: shutdown failed: {}", e.what());
    } catch (...) {
        log_warning(tt::LogOp, "H2DStreamService: shutdown failed with unknown exception");
    }
}

void H2DStreamService::barrier() {
    for (auto& s : sockets_) {
        s->barrier();
    }
}

void H2DStreamService::signal_termination() {
    if (mesh_device_ == nullptr) {
        return;
    }
    std::vector<uint32_t> one_word{1};
    for (const auto& [coord, addr] : termination_addrs_) {
        auto* d = mesh_device_->get_device(coord);
        const CoreCoord chosen = service_cores_.at(coord);
        tt::tt_metal::detail::WriteToDeviceL1(d, chosen, static_cast<uint32_t>(addr), one_word);
    }
}

std::vector<distributed::H2DSocket*> H2DStreamService::get_sockets() const {
    std::vector<distributed::H2DSocket*> out;
    out.reserve(sockets_.size());
    for (const auto& s : sockets_) {
        out.push_back(s.get());
    }
    return out;
}

// The owner-only getters below return state for wiring up a consumer workload;
// the connector has no MeshDevice, so calling them there is a programming error.
namespace {
inline void require_owner(bool is_owner, const char* api) {
    TT_FATAL(
        is_owner,
        "{}: this getter is owner-only. The connector-mode service (built via "
        "H2DStreamService::connect) has no MeshDevice and cannot dispatch consumer "
        "workloads, so the worker-sync / metadata addresses it would return are "
        "meaningless. Call this from the owner process instead.",
        api);
}
}  // namespace

CoreRange H2DStreamService::get_worker_cores() const {
    TT_FATAL(
        cfg_.worker_cores.has_value(),
        "H2DStreamService::get_worker_cores: worker-sync was not configured (Config::worker_cores unset).");
    return *cfg_.worker_cores;
}

DeviceAddr H2DStreamService::get_data_ready_sem_addr() const {
    require_owner(is_owner_, "H2DStreamService::get_data_ready_sem_addr");
    TT_FATAL(
        data_ready_sem_.has_value(),
        "H2DStreamService::get_data_ready_sem_addr: worker-sync was not configured (Config::worker_cores unset).");
    return data_ready_sem_->address();
}

DeviceAddr H2DStreamService::get_consumed_counter_addr(const distributed::MeshCoordinate& coord) const {
    require_owner(is_owner_, "H2DStreamService::get_consumed_counter_addr");
    auto it = consumed_addrs_.find(coord);
    TT_FATAL(
        it != consumed_addrs_.end(),
        "H2DStreamService::get_consumed_counter_addr: no consumed-counter at coord {} (worker-sync was not "
        "configured or the coord does not participate in this service).",
        coord);
    return it->second;
}

CoreCoord H2DStreamService::get_service_core(const distributed::MeshCoordinate& coord) const {
    require_owner(is_owner_, "H2DStreamService::get_service_core");
    auto it = service_cores_.find(coord);
    TT_FATAL(
        it != service_cores_.end(),
        "H2DStreamService::get_service_core: no service core claimed at coord {} (does this coord participate "
        "in this service?).",
        coord);
    return it->second;
}

DeviceAddr H2DStreamService::get_metadata_addr() const {
    require_owner(is_owner_, "H2DStreamService::get_metadata_addr");
    TT_FATAL(
        cfg_.metadata_size_bytes > 0,
        "H2DStreamService::get_metadata_addr: metadata multicast was not configured "
        "(Config::metadata_size_bytes is 0).");
    return metadata_l1_addr_;
}

const TensorSpec& H2DStreamService::get_per_shard_spec() const {
    TT_FATAL(per_shard_spec_.has_value(), "H2DStreamService::get_per_shard_spec: per-shard spec not derived");
    return *per_shard_spec_;
}

const Tensor& H2DStreamService::get_backing_tensor() const {
    require_owner(is_owner_, "H2DStreamService::get_backing_tensor");
    return device_tensor_;
}

std::size_t H2DStreamService::payload_size_bytes() const { return cfg_.global_spec.compute_packed_buffer_size_bytes(); }

std::size_t H2DStreamService::metadata_size_bytes() const { return cfg_.metadata_size_bytes; }

std::string H2DStreamService::export_descriptor(const std::string& service_id) {
    TT_FATAL(is_owner_, "H2DStreamService::export_descriptor: only owner-side services can be exported");
    TT_FATAL(mesh_device_ != nullptr, "H2DStreamService::export_descriptor: mesh device unavailable");
    TT_FATAL(mapper_ != nullptr, "H2DStreamService::export_descriptor: mapper unavailable");

    distributed::H2DStreamServiceDescriptor desc;
    desc.global_shape = cfg_.global_spec.logical_shape();
    desc.global_dtype = cfg_.global_spec.data_type();
    desc.mesh_shape = mesh_device_->shape();
    desc.mapper_config = mapper_->config();
    desc.socket_page_size = socket_page_size_;
    desc.num_socket_pages = num_socket_pages_;
    desc.metadata_size_bytes = cfg_.metadata_size_bytes;
    desc.socket_buffer_type = cfg_.socket_buffer_type;
    desc.socket_mode = cfg_.socket_mode;

    // Embed each socket's descriptor inline so the whole service is one file;
    // avoids a visibility race between service- and socket-level descriptors.
    desc.per_coord_entries.reserve(sockets_.size());
    for (auto& s : sockets_) {
        const auto coord = s->get_active_cores()[0].device_coord;
        desc.per_coord_entries.emplace_back(coord, s->populate_descriptor());
    }

    auto path = distributed::descriptor_path_for_service(service_id);
    desc.write_to_file(path);
    distributed::ShmResourceTracker::instance().track_file(path);
    descriptor_path_ = path;
    return path;
}

std::unique_ptr<H2DStreamService> H2DStreamService::connect(
    const std::string& service_id,
    std::optional<uint32_t> timeout_ms,
    std::function<void(ttsl::Span<std::byte> bytes, ttsl::Span<const std::byte> metadata)> preprocessor) {
    auto desc = distributed::H2DStreamServiceDescriptor::wait_and_read(
        distributed::descriptor_path_for_service(service_id), timeout_ms.value_or(10000));

    const TensorLayout tensor_layout(
        desc.global_dtype,
        PageConfig(Layout::ROW_MAJOR),
        MemoryConfig{TensorMemoryLayout::INTERLEAVED, BufferType::DRAM, std::nullopt});
    TensorSpec global_spec(desc.global_shape, tensor_layout);

    auto mapper = ttnn::distributed::create_mesh_mapper(desc.mesh_shape, desc.mapper_config);

    std::vector<std::unique_ptr<distributed::H2DSocket>> sockets;
    sockets.reserve(desc.per_coord_entries.size());
    for (const auto& [coord, socket_desc] : desc.per_coord_entries) {
        // socket_desc already carries the owner-side coord; the entry key is unused here.
        (void)coord;
        sockets.push_back(distributed::H2DSocket::connect_from_descriptor(socket_desc));
    }

    Config cfg{
        .global_spec = std::move(global_spec),
        .mapper = std::move(mapper),
        .socket_buffer_type = desc.socket_buffer_type,
        .fifo_size_bytes = 0,
        .scratch_cb_size_bytes = 0,
        .socket_mode = desc.socket_mode,
        .worker_cores = std::nullopt,
        .metadata_size_bytes = desc.metadata_size_bytes,
        // Preprocessor is process-local and not carried by the descriptor.
        .preprocessor = std::move(preprocessor),
    };

    return std::unique_ptr<H2DStreamService>(
        new H2DStreamService(std::move(cfg), std::move(sockets), desc.socket_page_size, desc.num_socket_pages));
}

void H2DStreamService::forward_to_tensor(ttsl::Span<const std::byte> bytes, ttsl::Span<const std::byte> metadata) {
    // Partial transfers aren't supported: the kernel's chunk count is baked into its CT args.
    const size_t expected = cfg_.global_spec.compute_packed_buffer_size_bytes();
    TT_FATAL(
        bytes.size() == expected,
        "H2DStreamService::forward_to_tensor: span size {} B does not match global_spec packed size {} B",
        bytes.size(),
        expected);

    TT_FATAL(
        metadata.size() == cfg_.metadata_size_bytes,
        "H2DStreamService::forward_to_tensor: metadata span size {} B does not match "
        "Config::metadata_size_bytes={} (must be exact match, including 0 when metadata is disabled)",
        metadata.size(),
        cfg_.metadata_size_bytes);

    // ROW_MAJOR + default MemoryConfig is what keeps the borrowed-data wrap and
    // the mapper's zero-copy fast path engaged.
    TT_FATAL(
        cfg_.global_spec.layout() == Layout::ROW_MAJOR,
        "H2DStreamService::forward_to_tensor(span): global_spec must be ROW_MAJOR (got {}). "
        "Use the Tensor overload with a pre-distributed host tensor for other layouts.",
        cfg_.global_spec.layout());

    ttsl::Span<const std::byte> mapper_input = bytes;
    if (cfg_.preprocessor) {
        std::memcpy(preprocess_scratch_.data(), bytes.data(), bytes.size());
        cfg_.preprocessor(ttsl::Span<std::byte>(preprocess_scratch_.data(), preprocess_scratch_.size()), metadata);
        mapper_input = ttsl::Span<const std::byte>(preprocess_scratch_.data(), preprocess_scratch_.size());
    }

    Tensor borrowed = make_borrowed_host_tensor(mapper_input, cfg_.global_spec);
    Tensor distributed = (*mapper_)(borrowed);
    forward_to_tensor(distributed, metadata);
}

void H2DStreamService::forward_to_tensor(const Tensor& host_tensor, ttsl::Span<const std::byte> metadata) {
    TT_FATAL(
        host_tensor.storage_type() == StorageType::HOST,
        "H2DStreamService::forward_to_tensor: expected host tensor, got storage_type={}",
        host_tensor.storage_type());
    TT_FATAL(
        metadata.size() == cfg_.metadata_size_bytes,
        "H2DStreamService::forward_to_tensor: metadata span size {} B does not match "
        "Config::metadata_size_bytes={} (must be exact match, including 0 when metadata is disabled)",
        metadata.size(),
        cfg_.metadata_size_bytes);

    const auto& host_mesh_tensor = host_tensor.host_storage().host_tensor();
    TT_FATAL(per_shard_spec_.has_value(), "H2DStreamService::forward_to_tensor: per-shard spec not derived");
    TT_FATAL(
        host_mesh_tensor.tensor_spec() == *per_shard_spec_,
        "H2DStreamService::forward_to_tensor: host tensor per-shard spec ({}) does not match expected "
        "per-shard spec ({}). Did you distribute the host tensor with a different mapper config?",
        host_mesh_tensor.tensor_spec(),
        *per_shard_spec_);

    const auto& dhb = host_mesh_tensor.buffer();
    const uint64_t expected_shard_bytes =
        static_cast<uint64_t>(num_socket_pages_) * static_cast<uint64_t>(socket_page_size_);

    std::vector<std::byte*> bases;
    bases.reserve(sockets_.size());

    for (auto& s : sockets_) {
        // get_active_cores() returns by value; copy the coord so it outlives the temporary.
        const auto coord = s->get_active_cores()[0].device_coord;
        TT_FATAL(
            dhb.is_local(coord),
            "H2DStreamService::forward_to_tensor: host tensor has no local shard for coord {}",
            coord);
        auto shard_opt = dhb.get_shard(coord);
        TT_FATAL(
            shard_opt.has_value(),
            "H2DStreamService::forward_to_tensor: host shard for coord {} is not populated",
            coord);

        auto shard_span = shard_opt->view_bytes();
        TT_FATAL(
            shard_span.size() == expected_shard_bytes,
            "H2DStreamService::forward_to_tensor: host shard at coord {} has {} B, expected {} "
            "({} socket pages * {} B). Layout drift between mapper output and backing tensor.",
            coord,
            shard_span.size(),
            expected_shard_bytes,
            num_socket_pages_,
            socket_page_size_);

        bases.push_back(shard_span.data());
    }

    // Page-major: send page `i` to every socket before `i+1` so every kernel can progress.
    for (uint32_t i = 0; i < num_socket_pages_; ++i) {
        const size_t offset = static_cast<size_t>(i) * socket_page_size_;
        for (size_t s = 0; s < sockets_.size(); ++s) {
            sockets_[s]->write(bases[s] + offset, /*num_pages=*/1);
        }
    }

    // Trailing metadata page: the kernel multicasts the leading metadata_size_bytes
    // to every worker. Same bytes to every device.
    if (cfg_.metadata_size_bytes > 0) {
        std::memcpy(metadata_scratch_.data(), metadata.data(), metadata.size());
        for (auto& s : sockets_) {
            s->write(metadata_scratch_.data(), /*num_pages=*/1);
        }
    }
}

}  // namespace tt::tt_metal
